use alloc::vec::Vec;
use core::fmt;

use axerrno::{AxError, AxResult, ax_err};
use memory_addr::{AddrRange, MemoryAddr, PAGE_SIZE_4K, PhysAddr, is_aligned_4k};
use memory_set::{MemoryArea, MemorySet};
use page_table_multiarch::{
    GenericPTE, PageSize, PageTable64, PagingError, PagingHandler, PagingMetaData,
};

use crate::mapping_err_to_ax_err;

mod backend;

pub use backend::Backend;
pub use page_table_entry::MappingFlags;

/// The virtual memory address space.
pub struct AddrSpace<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> {
    va_range: AddrRange<M::VirtAddr>,
    areas: MemorySet<Backend<M, PTE, H>>,
    pt: PageTable64<M, PTE, H>,
}

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> AddrSpace<M, PTE, H> {
    /// Returns the address space base.
    pub const fn base(&self) -> M::VirtAddr {
        self.va_range.start
    }

    /// Returns the address space end.
    pub const fn end(&self) -> M::VirtAddr {
        self.va_range.end
    }

    /// Returns the address space size.
    pub fn size(&self) -> usize {
        self.va_range.size()
    }

    /// Returns the reference to the inner page table.
    pub const fn page_table(&self) -> &PageTable64<M, PTE, H> {
        &self.pt
    }

    /// Returns the root physical address of the inner page table.
    pub const fn page_table_root(&self) -> PhysAddr {
        self.pt.root_paddr()
    }

    /// Checks if the address space contains the given address range.
    pub fn contains_range(&self, start: M::VirtAddr, size: usize) -> bool {
        self.va_range
            .contains_range(AddrRange::from_start_size(start, size))
    }

    /// Creates a new empty address space.
    pub fn new_empty(base: M::VirtAddr, size: usize) -> AxResult<Self> {
        Ok(Self {
            va_range: AddrRange::from_start_size(base, size),
            areas: MemorySet::new(),
            pt: PageTable64::<M, PTE, H>::try_new().map_err(|_| AxError::NoMemory)?,
        })
    }

    /// Add a new linear mapping.
    ///
    /// See [`Backend`] for more details about the mapping backends.
    ///
    /// The `flags` parameter indicates the mapping permissions and attributes.
    pub fn map_linear(
        &mut self,
        start_vaddr: M::VirtAddr,
        start_paddr: PhysAddr,
        size: usize,
        flags: MappingFlags,
        allow_huge: bool,
    ) -> AxResult {
        if !self.contains_range(start_vaddr, size) {
            return ax_err!(
                InvalidInput,
                alloc::format!(
                    "[{:?} {:?}]address out of range",
                    start_vaddr,
                    start_vaddr.add(size)
                )
            );
        }
        if !start_vaddr.is_aligned_4k() || !start_paddr.is_aligned_4k() || !is_aligned_4k(size) {
            return ax_err!(InvalidInput, "address not aligned");
        }

        let offset = start_vaddr.into() - start_paddr.as_usize();
        let area = MemoryArea::new(
            start_vaddr,
            size,
            flags,
            Backend::new_linear(offset, allow_huge),
        );
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
        start: M::VirtAddr,
        size: usize,
        flags: MappingFlags,
        populate: bool,
    ) -> AxResult {
        if !self.contains_range(start, size) {
            return ax_err!(
                InvalidInput,
                alloc::format!("address [{:?}~{:?}] out of range", start, start.add(size)).as_str()
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

    pub fn protect(&mut self, start: M::VirtAddr, size: usize, flags: MappingFlags) -> AxResult {
        if !self.contains_range(start, size) {
            return ax_err!(InvalidInput, "address out of range");
        }
        if !start.is_aligned_4k() || !is_aligned_4k(size) {
            return ax_err!(InvalidInput, "address not aligned");
        }

        let update_flags = |new_flags: MappingFlags| {
            move |old_flags: MappingFlags| -> Option<MappingFlags> {
                if old_flags == new_flags {
                    return None;
                }
                Some(new_flags)
            }
        };

        self.areas
            .protect(start, size, update_flags(flags), &mut self.pt)
            .map_err(mapping_err_to_ax_err)?;
        Ok(())
    }

    /// Removes mappings within the specified virtual address range.
    pub fn unmap(&mut self, start: M::VirtAddr, size: usize) -> AxResult {
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
    pub fn handle_page_fault(&mut self, vaddr: M::VirtAddr, access_flags: MappingFlags) -> bool {
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
    pub fn translate(&self, vaddr: M::VirtAddr) -> Option<(PhysAddr, MappingFlags, PageSize)> {
        if !self.va_range.contains(vaddr) {
            return None;
        }
        self.pt.query(vaddr).ok()
    }

    /// Translates the given virtual address range into a vector of physical address and size tuples.
    ///
    /// Each tuple represents a contiguous physical memory region.
    /// (PhysAddr, size)
    /// * PhysAddr: The starting physical address of the region.
    /// * size: The size of the region in bytes, which may be less than or equal to the page size.
    /// Returns `None` if the virtual address range is out of the address space range or not fully mapped
    pub fn translate_range(
        &self,
        vaddr: M::VirtAddr,
        size: usize,
    ) -> Option<Vec<(PhysAddr, usize)>> {
        if !self
            .va_range
            .contains_range(AddrRange::from_start_size(vaddr, size))
        {
            return None;
        }

        let mut start = vaddr;
        let end = start.add(size);

        let mut v = Vec::new();
        while start < end {
            let (start_paddr, _, page_size) = self.pt.query(start).ok()?;
            let mut end_va = start.align_down(page_size).add(page_size.into());
            end_va = end_va.min(end);

            v.push((start_paddr, (end_va.sub_addr(start)).into()));
            start = end_va;
        }
        Some(v)
    }

    /// Translate&Copy the given `VirtAddr` with LENGTH len to a mutable u8 Vec through page table.
    ///
    /// Returns `None` if the virtual address is out of range or not mapped.
    pub fn translated_byte_buffer(
        &self,
        vaddr: M::VirtAddr,
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
            let end = start.add(len);

            let mut v = Vec::new();
            while start < end {
                let (start_paddr, _, page_size) = self.page_table().query(start).unwrap();
                let mut end_va = start.align_down(page_size).add(page_size.into());
                end_va = end_va.min(end);

                v.push(unsafe {
                    core::slice::from_raw_parts_mut(
                        H::phys_to_virt(start_paddr).as_mut_ptr(),
                        (end_va.sub_addr(start)).into(),
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
    pub fn translate_and_get_limit(&self, vaddr: M::VirtAddr) -> Option<(PhysAddr, usize)> {
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

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> AddrSpace<M, PTE, H> {
    pub fn clone(&self) -> AxResult<Self> {
        let mut cloned_aspace = Self::new_empty(self.base(), self.size())?;

        for area in self.areas.iter() {
            let new_backend = area.backend().clone();
            let new_area = MemoryArea::new(area.start(), area.size(), area.flags(), new_backend);

            cloned_aspace
                .areas
                .map(new_area, &mut cloned_aspace.pt, false)
                .map_err(mapping_err_to_ax_err)?;

            match area.backend() {
                Backend::Alloc { .. } => {
                    // Alloc mappings are cloned.
                    // They are created in the new address space.
                    // The physical frames are copied to the new address space.
                    let mut addr = area.start();
                    let end = addr.add(area.size());
                    while addr < end {
                        match self.pt.query(addr) {
                            Ok((phys_addr, _, page_size)) => {
                                if !addr.is_aligned(page_size as usize) {
                                    warn!(
                                        "AddrSpace clone: addr {:#x} is not aligned to page size {:?}",
                                        addr, page_size
                                    );
                                }
                                let mut end_va = addr.align_down(page_size).add(page_size.into());
                                end_va = end_va.min(end);

                                // Copy the physical frames to the new address space.
                                let new_phys_addr = match cloned_aspace.pt.query(addr) {
                                    Ok((new_phys_addr, _, new_pgsize)) => {
                                        if page_size != new_pgsize {
                                            warn!(
                                                "AddrSpace clone: addr {:#x} page size mismatch {:?} != {:?}",
                                                addr, page_size, new_pgsize
                                            );
                                        }
                                        new_phys_addr
                                    }
                                    Err(PagingError::NotMapped) => {
                                        // The address is not mapped in the new address space.
                                        // map it!
                                        if !cloned_aspace.handle_page_fault(addr, area.flags()) {
                                            warn!(
                                                "AddrSpace clone: addr {:#x} handle page fault failed, check why?",
                                                addr
                                            );
                                        }

                                        match cloned_aspace.pt.query(addr) {
                                            Ok((new_phys_addr, _, new_pgsize)) => {
                                                if page_size != new_pgsize {
                                                    warn!(
                                                        "AddrSpace clone: addr {:#x} page size mismatch {:?} != {:?}",
                                                        addr, page_size, new_pgsize
                                                    );
                                                }
                                                new_phys_addr
                                            }
                                            Err(_) => {
                                                warn!(
                                                    "AddrSpace clone: addr {:#x} is not mapped",
                                                    addr
                                                );
                                                continue;
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        warn!("AddrSpace clone: addr {:#x} is not mapped", addr);
                                        continue;
                                    }
                                };

                                unsafe {
                                    core::ptr::copy_nonoverlapping(
                                        H::phys_to_virt(phys_addr).as_ptr(),
                                        H::phys_to_virt(new_phys_addr).as_mut_ptr(),
                                        page_size as usize,
                                    )
                                };

                                addr = end_va;
                            }
                            Err(PagingError::NotMapped) => {
                                // The address is not mapped in the original address space.
                                // Step forward to the next 4K page.
                                addr = addr.add(PAGE_SIZE_4K);
                            }
                            Err(_) => {
                                warn!("AddrSpace clone: addr {:#x} is not mapped", addr);
                            }
                        }
                    }
                }
                Backend::Linear { .. } => {
                    // Linear mappings are not cloned.
                    // They are created in the new address space.
                }
            }
        }

        Ok(cloned_aspace)
    }

    pub fn clone_cow(&mut self) -> AxResult<Self> {
        unimplemented!()
    }
}

#[allow(unused)]
impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> AddrSpace<M, PTE, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("AddrSpace")
            .field("va_range", &self.va_range)
            .field("page_table_root", &self.pt.root_paddr())
            .field("areas", &self.areas)
            .finish()
    }
}

#[allow(unused)]
impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> AddrSpace<M, PTE, H> {
    fn drop(&mut self) {
        self.clear();
    }
}
