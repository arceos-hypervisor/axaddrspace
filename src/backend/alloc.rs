use memory_addr::PageIter4K;
use page_table_multiarch::{MappingFlags, PageSize, PagingHandler};

use crate::addr::{GuestPhysAddr, HostPhysAddr};
use crate::backend::Backend;
use crate::npt::NestedPageTable as PageTable;

impl Backend {
    /// Creates a new allocation mapping backend.
    pub const fn new_alloc(populate: bool) -> Self {
        Self::Alloc { populate }
    }

    pub(crate) fn map_alloc<H: PagingHandler>(
        &self,
        start: GuestPhysAddr,
        size: usize,
        flags: MappingFlags,
        pt: &mut PageTable<H>,
        populate: bool,
    ) -> bool {
        debug!(
            "map_alloc: [{:#x}, {:#x}) {:?} (populate={})",
            start,
            start + size,
            flags,
            populate
        );
        if populate {
            // allocate all possible physical frames for populated mapping.
            for addr in PageIter4K::new(start, start + size).unwrap() {
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
                |_va| HostPhysAddr::from(0),
                size,
                MappingFlags::empty(),
                false,
                false,
            )
            .is_ok()
        }
    }

    pub(crate) fn unmap_alloc<H: PagingHandler>(
        &self,
        start: GuestPhysAddr,
        size: usize,
        pt: &mut PageTable<H>,
        _populate: bool,
    ) -> bool {
        debug!("unmap_alloc: [{:#x}, {:#x})", start, start + size);
        for addr in PageIter4K::new(start, start + size).unwrap() {
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

    pub(crate) fn handle_page_fault_alloc<H: PagingHandler>(
        &self,
        gpa: GuestPhysAddr,
        orig_flags: MappingFlags,
        pt: &mut PageTable<H>,
        populate: bool,
    ) -> bool {
        if populate {
            false // Populated mappings should not trigger page faults.
        } else {
            // Allocate a physical frame lazily and map it to the fault address.
            // `gpa` does not need to be aligned. It will be automatically
            // aligned during `pt.remap` regardless of the page size.
            H::alloc_frame()
                .and_then(|frame| pt.remap(gpa, frame, orig_flags).ok())
                .is_some()
        }
    }
}
