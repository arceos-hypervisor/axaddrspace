use alloc::vec::Vec;
use core::fmt;

use axerrno::{AxError, AxResult, ax_err};
use memory_addr::{MemoryAddr, PhysAddr, is_aligned_4k};
use memory_set::{MemoryArea, MemorySet};
use page_table_multiarch::PagingHandler;

use crate::npt::NestedPageTable as PageTable;
use crate::{GuestPhysAddr, GuestPhysAddrRange, mapping_err_to_ax_err};

mod backend;

pub use backend::Backend;
pub use page_table_entry::MappingFlags;

/// The virtual memory address space.
pub struct AddrSpace<H: PagingHandler> {
    va_range: GuestPhysAddrRange,
    areas: MemorySet<Backend<H>>,
    pt: PageTable<H>,
}

impl<H: PagingHandler> AddrSpace<H> {
    /// Returns the address space base.
    pub const fn base(&self) -> GuestPhysAddr {
        self.va_range.start
    }

    /// Returns the address space end.
    pub const fn end(&self) -> GuestPhysAddr {
        self.va_range.end
    }

    /// Returns the address space size.
    pub fn size(&self) -> usize {
        self.va_range.size()
    }

    /// Returns the reference to the inner page table.
    pub const fn page_table(&self) -> &PageTable<H> {
        &self.pt
    }

    /// Returns the root physical address of the inner page table.
    pub const fn page_table_root(&self) -> PhysAddr {
        self.pt.root_paddr()
    }

    /// Checks if the address space contains the given address range.
    pub fn contains_range(&self, start: GuestPhysAddr, size: usize) -> bool {
        self.va_range
            .contains_range(GuestPhysAddrRange::from_start_size(start, size))
    }

    /// Creates a new empty address space.
    pub fn new_empty(base: GuestPhysAddr, size: usize) -> AxResult<Self> {
        Ok(Self {
            va_range: GuestPhysAddrRange::from_start_size(base, size),
            areas: MemorySet::new(),
            pt: PageTable::try_new().map_err(|_| AxError::NoMemory)?,
        })
    }

    /// Add a new linear mapping.
    ///
    /// See [`Backend`] for more details about the mapping backends.
    ///
    /// The `flags` parameter indicates the mapping permissions and attributes.
    pub fn map_linear(
        &mut self,
        start_vaddr: GuestPhysAddr,
        start_paddr: PhysAddr,
        size: usize,
        flags: MappingFlags,
    ) -> AxResult {
        if !self.contains_range(start_vaddr, size) {
            return ax_err!(InvalidInput, "address out of range");
        }
        if !start_vaddr.is_aligned_4k() || !start_paddr.is_aligned_4k() || !is_aligned_4k(size) {
            return ax_err!(InvalidInput, "address not aligned");
        }

        let offset = start_vaddr.as_usize() - start_paddr.as_usize();
        let area = MemoryArea::new(start_vaddr, size, flags, Backend::new_linear(offset));
        self.areas
            .map(area, &mut self.pt, false)
            .map_err(mapping_err_to_ax_err)?;
        Ok(())
    }

    /// Add a new allocation mapping.
    ///
    /// See [`Backend`] for more details about the mapping backends.
    ///
    /// The `flags` parameter indicates the mapping permissions and attributes.
    pub fn map_alloc(
        &mut self,
        start: GuestPhysAddr,
        size: usize,
        flags: MappingFlags,
        populate: bool,
    ) -> AxResult {
        if !self.contains_range(start, size) {
            return ax_err!(
                InvalidInput,
                alloc::format!("address [{:?}~{:?}] out of range", start, start + size).as_str()
            );
        }
        if !start.is_aligned_4k() || !is_aligned_4k(size) {
            return ax_err!(InvalidInput, "address not aligned");
        }

        let area = MemoryArea::new(start, size, flags, Backend::new_alloc(populate));
        self.areas
            .map(area, &mut self.pt, false)
            .map_err(mapping_err_to_ax_err)?;
        Ok(())
    }

    /// Removes mappings within the specified virtual address range.
    pub fn unmap(&mut self, start: GuestPhysAddr, size: usize) -> AxResult {
        if !self.contains_range(start, size) {
            return ax_err!(InvalidInput, "address out of range");
        }
        if !start.is_aligned_4k() || !is_aligned_4k(size) {
            return ax_err!(InvalidInput, "address not aligned");
        }

        self.areas
            .unmap(start, size, &mut self.pt)
            .map_err(mapping_err_to_ax_err)?;
        Ok(())
    }

    /// Removes all mappings in the address space.
    pub fn clear(&mut self) {
        self.areas.clear(&mut self.pt).unwrap();
    }

    /// Handles a page fault at the given address.
    ///
    /// `access_flags` indicates the access type that caused the page fault.
    ///
    /// Returns `true` if the page fault is handled successfully (not a real
    /// fault).
    pub fn handle_page_fault(&mut self, vaddr: GuestPhysAddr, access_flags: MappingFlags) -> bool {
        if !self.va_range.contains(vaddr) {
            return false;
        }
        if let Some(area) = self.areas.find(vaddr) {
            let orig_flags = area.flags();
            if !orig_flags.contains(access_flags) {
                return false;
            }
            area.backend()
                .handle_page_fault(vaddr, orig_flags, &mut self.pt)
        } else {
            false
        }
    }

    /// Translates the given `VirtAddr` into `PhysAddr`.
    ///
    /// Returns `None` if the virtual address is out of range or not mapped.
    pub fn translate(&self, vaddr: GuestPhysAddr) -> Option<PhysAddr> {
        if !self.va_range.contains(vaddr) {
            return None;
        }
        self.pt
            .query(vaddr)
            .map(|(phys_addr, _, _)| {
                debug!("vaddr {:?} translate to {:?}", vaddr, phys_addr);
                phys_addr
            })
            .ok()
    }

    /// Translate&Copy the given `VirtAddr` with LENGTH len to a mutable u8 Vec through page table.
    ///
    /// Returns `None` if the virtual address is out of range or not mapped.
    pub fn translated_byte_buffer(
        &self,
        vaddr: GuestPhysAddr,
        len: usize,
    ) -> Option<Vec<&'static mut [u8]>> {
        if !self.va_range.contains(vaddr) {
            return None;
        }
        if let Some(area) = self.areas.find(vaddr) {
            if len > area.size() {
                warn!(
                    "AddrSpace translated_byte_buffer len {:#x} exceeds area length {:#x}",
                    len,
                    area.size()
                );
                return None;
            }

            let mut start = vaddr;
            let end = start + len;

            debug!(
                "start {:?} end {:?} area size {:#x}",
                start,
                end,
                area.size()
            );

            let mut v = Vec::new();
            while start < end {
                let (start_paddr, _, page_size) = self.page_table().query(start).unwrap();
                let mut end_va = start.align_down(page_size) + page_size.into();
                end_va = end_va.min(end);

                v.push(unsafe {
                    core::slice::from_raw_parts_mut(
                        H::phys_to_virt(start_paddr).as_mut_ptr(),
                        (end_va - start.as_usize()).into(),
                    )
                });
                start = end_va;
            }
            Some(v)
        } else {
            None
        }
    }

    /// Translates the given `VirtAddr` into `PhysAddr`,
    /// and returns the size of the `MemoryArea` corresponding to the target vaddr.
    ///
    /// Returns `None` if the virtual address is out of range or not mapped.
    pub fn translate_and_get_limit(&self, vaddr: GuestPhysAddr) -> Option<(PhysAddr, usize)> {
        if !self.va_range.contains(vaddr) {
            return None;
        }
        if let Some(area) = self.areas.find(vaddr) {
            self.pt
                .query(vaddr)
                .map(|(phys_addr, _, _)| (phys_addr, area.size()))
                .ok()
        } else {
            None
        }
    }
}

impl<H: PagingHandler> fmt::Debug for AddrSpace<H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("AddrSpace")
            .field("va_range", &self.va_range)
            .field("page_table_root", &self.pt.root_paddr())
            .field("areas", &self.areas)
            .finish()
    }
}

impl<H: PagingHandler> Drop for AddrSpace<H> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use memory_addr::VirtAddr;
    use spin::Mutex;

    // Mock physical memory environment - ensure alignment
    const MEMORY_SIZE: usize = 0x10000; // 64KB
    const MEMORY_START_PADDR: usize = 0x1000;
    
    // Use #[repr(align(4096))] to ensure 4KB alignment
    #[repr(align(4096))]
    struct AlignedMemory([u8; MEMORY_SIZE]);
    
    static mut MOCK_MEMORY: AlignedMemory = AlignedMemory([0; MEMORY_SIZE]);
    // Mock physical frame allocator state
    static NEXT_PADDR: AtomicUsize = AtomicUsize::new(MEMORY_START_PADDR);
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    struct MockHandler;

    impl PagingHandler for MockHandler {
        fn alloc_frame() -> Option<PhysAddr> {
            let paddr = NEXT_PADDR.fetch_add(0x1000, Ordering::SeqCst);
            if paddr >= MEMORY_START_PADDR + MEMORY_SIZE {
                return None;
            }
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
            Some(PhysAddr::from(paddr))
        }

        fn dealloc_frame(_paddr: PhysAddr) {
            DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }

        fn phys_to_virt(paddr: PhysAddr) -> VirtAddr {
            let paddr_usize = paddr.as_usize();
            assert!(
                paddr_usize >= MEMORY_START_PADDR && paddr_usize < MEMORY_START_PADDR + MEMORY_SIZE,
                "Physical address {:#x} out of bounds", paddr_usize
            );
            
            let offset = paddr_usize - MEMORY_START_PADDR;
            unsafe {
                let base_ptr = core::ptr::addr_of_mut!(MOCK_MEMORY.0).cast::<u8>();
                let vaddr = base_ptr.add(offset);
                VirtAddr::from_mut_ptr_of(vaddr)
            }
        }
    }

    fn reset_mock_env() {
        NEXT_PADDR.store(MEMORY_START_PADDR, Ordering::SeqCst);
        ALLOC_COUNT.store(0, Ordering::SeqCst);
        DEALLOC_COUNT.store(0, Ordering::SeqCst);
        unsafe {
            let ptr = core::ptr::addr_of_mut!(MOCK_MEMORY.0).cast::<u8>();
            core::ptr::write_bytes(ptr, 0, MEMORY_SIZE);
        }
    }

    #[test]
    fn test_addrspace_creation() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let base = GuestPhysAddr::from_usize(0x10000);
        let size = 0x1000;
        {
            let addr_space = AddrSpace::<MockHandler>::new_empty(base, size).unwrap();
            assert_eq!(addr_space.base(), base);
            assert_eq!(addr_space.size(), size);
            assert_eq!(addr_space.end(), base + size);
            assert_eq!(ALLOC_COUNT.load(Ordering::SeqCst), 1); // Allocated root page table
        }
        assert_eq!(DEALLOC_COUNT.load(Ordering::SeqCst), 1); // Deallocated root page table
    }

    #[test]
    fn test_contains_range() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let base = GuestPhysAddr::from_usize(0x10000);
        let size = 0x4000; // 16KB
        let addr_space = AddrSpace::<MockHandler>::new_empty(base, size).unwrap();

        // Within range
        assert!(addr_space.contains_range(base, 0x1000));
        assert!(addr_space.contains_range(base + 0x1000, 0x2000));
        assert!(addr_space.contains_range(base, size));

        // Out of range
        assert!(!addr_space.contains_range(base - 0x1000, 0x1000));
        assert!(!addr_space.contains_range(base + size, 0x1000));
        assert!(!addr_space.contains_range(base, size + 0x1000));

        // Partially out of range
        assert!(!addr_space.contains_range(base + 0x3000, 0x2000));
    }
    
    // This test is temporarily disabled because flush_tlb is not implemented for x86.
    #[ignore]
    #[test]
    fn test_map_linear() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x18000);
        let paddr = PhysAddr::from_usize(0x10000);
        let size = 0x8000; // 32KB
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Linear mapping
        addr_space.map_linear(vaddr, paddr, size, flags).unwrap();

        assert_eq!(addr_space.translate(vaddr).unwrap(), paddr);
        assert_eq!(addr_space.translate(vaddr + 0x1000).unwrap(), paddr + 0x1000);
    }

    #[test]
    fn test_map_alloc_populate() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x10000);
        let size = 0x2000; // 8KB
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Frame count before allocation: 1 root page table
        let initial_allocs = ALLOC_COUNT.load(Ordering::SeqCst);
        assert_eq!(initial_allocs, 1);

        // Allocate physical frames immediately
        addr_space.map_alloc(vaddr, size, flags, true).unwrap();

        // Verify additional frames were allocated
        let final_allocs = ALLOC_COUNT.load(Ordering::SeqCst);
        assert!(final_allocs > initial_allocs);

        // Verify mappings exist and addresses are valid
        let paddr1 = addr_space.translate(vaddr).unwrap();
        let paddr2 = addr_space.translate(vaddr + 0x1000).unwrap();
        
        // Verify physical addresses are within valid range
        assert!(paddr1.as_usize() >= MEMORY_START_PADDR && paddr1.as_usize() < MEMORY_START_PADDR + MEMORY_SIZE);
        assert!(paddr2.as_usize() >= MEMORY_START_PADDR && paddr2.as_usize() < MEMORY_START_PADDR + MEMORY_SIZE);

        // Verify two pages have different physical addresses
        assert_ne!(paddr1, paddr2);
    }

    #[test]
    fn test_map_alloc_lazy() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x13000);
        let size = 0x1000;
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        let initial_allocs = ALLOC_COUNT.load(Ordering::SeqCst);

        // Lazy allocation - don't allocate physical frames immediately
        addr_space.map_alloc(vaddr, size, flags, false).unwrap();

        // Frame count should only increase for page table structure, not data pages
        let after_map_allocs = ALLOC_COUNT.load(Ordering::SeqCst);
        assert!(after_map_allocs >= initial_allocs); // May have allocated intermediate page tables
        assert!(addr_space.translate(vaddr).is_none());
    }

    #[test]
    fn test_page_fault_handling() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x14000);
        let size = 0x1000;
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Create lazy allocation mapping
        addr_space.map_alloc(vaddr, size, flags, false).unwrap();

        let before_pf_allocs = ALLOC_COUNT.load(Ordering::SeqCst);

        // Simulate page fault
        let handled = addr_space.handle_page_fault(vaddr, MappingFlags::READ);

        // Page fault should be handled
        assert!(handled);

        // Should have allocated physical frames
        let after_pf_allocs = ALLOC_COUNT.load(Ordering::SeqCst);
        assert!(after_pf_allocs > before_pf_allocs);

        // Translation should succeed now
        let paddr = addr_space.translate(vaddr);
        assert!(paddr.is_some());
    }

    #[test]
    fn test_unmap() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x15000);
        let size = 0x2000;
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Create mapping
        addr_space.map_alloc(vaddr, size, flags, true).unwrap();

        // Verify mapping exists
        assert!(addr_space.translate(vaddr).is_some());
        assert!(addr_space.translate(vaddr + 0x1000).is_some());

        let before_unmap_deallocs = DEALLOC_COUNT.load(Ordering::SeqCst);

        // Unmap
        addr_space.unmap(vaddr, size).unwrap();

        // Verify mapping is removed
        assert!(addr_space.translate(vaddr).is_none());
        assert!(addr_space.translate(vaddr + 0x1000).is_none());

        // Verify frames were deallocated
        let after_unmap_deallocs = DEALLOC_COUNT.load(Ordering::SeqCst);
        assert!(after_unmap_deallocs > before_unmap_deallocs);
    }

    #[test]
    fn test_clear() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr1 = GuestPhysAddr::from_usize(0x16000);
        let vaddr2 = GuestPhysAddr::from_usize(0x17000);
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Create multiple mappings
        addr_space.map_alloc(vaddr1, 0x1000, flags, true).unwrap();
        addr_space.map_alloc(vaddr2, 0x1000, flags, true).unwrap();

        // Verify mappings exist
        assert!(addr_space.translate(vaddr1).is_some());
        assert!(addr_space.translate(vaddr2).is_some());

        let before_clear_deallocs = DEALLOC_COUNT.load(Ordering::SeqCst);

        // Clear all mappings
        addr_space.clear();

        // Verify all mappings are removed
        assert!(addr_space.translate(vaddr1).is_none());
        assert!(addr_space.translate(vaddr2).is_none());

        // Verify frames were deallocated
        let after_clear_deallocs = DEALLOC_COUNT.load(Ordering::SeqCst);
        assert!(after_clear_deallocs > before_clear_deallocs);
    }
    
    #[test]
    fn test_translate() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x18000);
        let size = 0x1000;
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Create mapping
        addr_space.map_alloc(vaddr, size, flags, true).unwrap();

        // Verify translation succeeds
        let paddr = addr_space.translate(vaddr);
        assert!(paddr.is_some());
        let paddr = paddr.unwrap();
        assert!(paddr.as_usize() >= MEMORY_START_PADDR);
        assert!(paddr.as_usize() < MEMORY_START_PADDR + MEMORY_SIZE);

        // Verify unmapped address translation fails
        let unmapped_vaddr = GuestPhysAddr::from_usize(0x19000);
        assert!(addr_space.translate(unmapped_vaddr).is_none());

        // Verify out-of-range address translation fails
        let out_of_range = GuestPhysAddr::from_usize(0x30000);
        assert!(addr_space.translate(out_of_range).is_none());
    }

    #[test]
    fn test_translated_byte_buffer() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x19000);
        let size = 0x2000; // 8KB
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Create mapping
        addr_space.map_alloc(vaddr, size, flags, true).unwrap();

        // Verify byte buffer can be obtained
        let buffer = addr_space.translated_byte_buffer(vaddr, 0x100);
        assert!(buffer.is_some());
        let mut buffer = buffer.unwrap();
        assert!(!buffer.is_empty());

        // Verify data write and read
        let test_bytes = [0xAA, 0xBB, 0xCC, 0xDD];
        if !buffer.is_empty() && buffer[0].len() >= test_bytes.len() {
            for (i, &byte) in test_bytes.iter().enumerate() {
                buffer[0][i] = byte;
            }

            // Verify data write was successful
            let verify_buffer = addr_space.translated_byte_buffer(vaddr, test_bytes.len()).unwrap();
            for (i, &expected) in test_bytes.iter().enumerate() {
                assert_eq!(verify_buffer[0][i], expected);
            }
        }

        // Verify exceeding area size returns None
        assert!(addr_space.translated_byte_buffer(vaddr, size + 0x1000).is_none());

        // Verify unmapped address returns None
        let unmapped_vaddr = GuestPhysAddr::from_usize(0x1D000);
        assert!(addr_space.translated_byte_buffer(unmapped_vaddr, 0x100).is_none());
    }

    #[test]
    fn test_translate_and_get_limit() {
        let _guard = TEST_MUTEX.lock();
        reset_mock_env();

        let mut addr_space = AddrSpace::<MockHandler>::new_empty(0x10000.into(), 0x10000).unwrap();
        let vaddr = GuestPhysAddr::from_usize(0x1A000);
        let size = 0x3000; // 12KB
        let flags = MappingFlags::READ | MappingFlags::WRITE;

        // Create mapping
        addr_space.map_alloc(vaddr, size, flags, true).unwrap();

        // Verify translation and area size retrieval
        let result = addr_space.translate_and_get_limit(vaddr);
        assert!(result.is_some());
        let (paddr, area_size) = result.unwrap();
        assert!(paddr.as_usize() >= MEMORY_START_PADDR && paddr.as_usize() < MEMORY_START_PADDR + MEMORY_SIZE);
        assert_eq!(area_size, size);

        // Verify unmapped address returns None
        let unmapped_vaddr = GuestPhysAddr::from_usize(0x1E000);
        assert!(addr_space.translate_and_get_limit(unmapped_vaddr).is_none());

        // Verify out-of-range address returns None
        let out_of_range = GuestPhysAddr::from_usize(0x30000);
        assert!(addr_space.translate_and_get_limit(out_of_range).is_none());
    }
}