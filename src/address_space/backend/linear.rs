use memory_addr::{MemoryAddr, PhysAddr, PAGE_SIZE_2M};
use page_table_multiarch::{GenericPTE, MappingFlags, PageTable64, PagingHandler, PagingMetaData};

use super::Backend;

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> Backend<M, PTE, H> {
    /// Creates a new linear mapping backend.
    pub const fn new_linear(pa_va_offset: usize, allow_huge: bool) -> Self {
        Self::Linear {
            pa_va_offset,
            allow_huge,
            max_page_size_2m: false,
            flush_tlb_by_page_on_unmap: true,
        }
    }

    /// Creates a linear backend whose caller batches TLB invalidation after
    /// removing one or more areas.
    pub const fn new_linear_deferred_unmap(pa_va_offset: usize, allow_huge: bool) -> Self {
        Self::Linear {
            pa_va_offset,
            allow_huge,
            max_page_size_2m: false,
            flush_tlb_by_page_on_unmap: false,
        }
    }

    /// Creates a deferred-unmap linear backend that keeps one software area
    /// while limiting hardware leaves to at most 2 MiB.
    pub const fn new_linear_deferred_unmap_2m(pa_va_offset: usize) -> Self {
        Self::Linear {
            pa_va_offset,
            allow_huge: true,
            max_page_size_2m: true,
            flush_tlb_by_page_on_unmap: false,
        }
    }

    pub(crate) fn map_linear(
        &self,
        start: M::VirtAddr,
        size: usize,
        flags: MappingFlags,
        pt: &mut PageTable64<M, PTE, H>,
        allow_huge: bool,
        max_page_size_2m: bool,
        pa_va_offset: usize,
    ) -> bool {
        let pa_start = PhysAddr::from(start.into() - pa_va_offset);
        debug!(
            "map_linear: [{:#x}, {:#x}) -> [{:#x}, {:#x}) {:?}",
            start,
            start.add(size),
            pa_start,
            pa_start + size,
            flags
        );
        if !max_page_size_2m {
            return pt
                .map_region(
                    start,
                    |va| PhysAddr::from(va.into() - pa_va_offset),
                    size,
                    flags,
                    allow_huge,
                    false,
                )
                .is_ok();
        }

        let mut mapped = 0usize;
        while mapped < size {
            let chunk = core::cmp::min(PAGE_SIZE_2M, size - mapped);
            let chunk_start = start.add(mapped);
            if pt
                .map_region(
                    chunk_start,
                    |va| PhysAddr::from(va.into() - pa_va_offset),
                    chunk,
                    flags,
                    true,
                    false,
                )
                .is_err()
            {
                return false;
            }
            mapped += chunk;
        }
        true
    }

    pub(crate) fn unmap_linear(
        &self,
        start: M::VirtAddr,
        size: usize,
        pt: &mut PageTable64<M, PTE, H>,
        _pa_va_offset: usize,
        flush_tlb_by_page: bool,
    ) -> bool {
        debug!("unmap_linear: [{:#x}, {:#x})", start, start.add(size));
        pt.unmap_region(start, size, flush_tlb_by_page).is_ok()
    }
}
