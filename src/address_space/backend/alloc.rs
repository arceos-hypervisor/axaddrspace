use memory_addr::{PageIter4K, PhysAddr};
use page_table_multiarch::{PageTable64, MappingFlags, PageSize, PagingHandler, PagingMetaData, GenericPTE};

use super::Backend;

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> Backend<M, PTE, H> {
    /// Creates a new allocation mapping backend.
    pub const fn new_alloc(populate: bool) -> Self {
        Self::Alloc {
            populate,
            _phantom0: core::marker::PhantomData,
            _phantom1: core::marker::PhantomData,
            _phantom2: core::marker::PhantomData,
        }
    }

    pub(crate) fn map_alloc(
        &self,
        start: M::VirtAddr,
        size: usize,
        flags: MappingFlags,
        pt: &mut PageTable64<M, PTE, H>,
        populate: bool,
    ) -> bool {
        debug!(
            "map_alloc: [{:#x}, {:#x}) {:?} (populate={})",
            start,
            start.into() + size,
            flags,
            populate
        );
        if populate {
            // allocate all possible physical frames for populated mapping.
            for addr in PageIter4K::new(start, (start.into() + size).into()).unwrap() {
                if H::alloc_frame()
                    .and_then(|frame| pt.map(addr, frame, PageSize::Size4K, flags).ok())
                    .is_none()
                {
                    return false;
                }
            }
            true
        } else {
            // Map to a empty entry for on-demand mapping.
            pt.map_region(
                start,
                |_va| PhysAddr::from(0),
                size,
                MappingFlags::empty(),
                false,
                false,
            )
            .is_ok()
        }
    }

    pub(crate) fn unmap_alloc(
        &self,
        start: M::VirtAddr,
        size: usize,
        pt: &mut PageTable64<M, PTE, H>,
        _populate: bool,
    ) -> bool {
        debug!("unmap_alloc: [{:#x}, {:#x})", start, start.into() + size);
        for addr in PageIter4K::new(start, (start.into() + size).into()).unwrap() {
            if let Ok((frame, page_size, _)) = pt.unmap(addr) {
                // Deallocate the physical frame if there is a mapping in the
                // page table.
                if page_size.is_huge() {
                    return false;
                }
                H::dealloc_frame(frame);
            } else {
                // It's fine if the page is not mapped.
            }
        }
        true
    }

    pub(crate) fn handle_page_fault_alloc(
        &self,
        vaddr: M::VirtAddr,
        orig_flags: MappingFlags,
        pt: &mut PageTable64<M, PTE, H>,
        populate: bool,
    ) -> bool {
        if populate {
            false // Populated mappings should not trigger page faults.
        } else {
            // Allocate a physical frame lazily and map it to the fault address.
            // `vaddr` does not need to be aligned. It will be automatically
            // aligned during `pt.remap` regardless of the page size.
            H::alloc_frame()
                .and_then(|frame| pt.remap(vaddr, frame, orig_flags).ok())
                .is_some()
        }
    }
}
