use memory_addr::PhysAddr;
use page_table_multiarch::{PageTable64, MappingFlags, PagingHandler, PagingMetaData, GenericPTE};

use super::Backend;

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> Backend<M, PTE, H> {
    /// Creates a new linear mapping backend.
    pub const fn new_linear(pa_va_offset: usize) -> Self {
        Self::Linear { pa_va_offset }
    }

    pub(crate) fn map_linear(
        &self,
        start: M::VirtAddr,
        size: usize,
        flags: MappingFlags,
        pt: &mut PageTable64<M, PTE, H>,
        pa_va_offset: usize,
    ) -> bool {
        let pa_start = PhysAddr::from(start.into() - pa_va_offset);
        debug!(
            "map_linear: [{:#x}, {:#x}) -> [{:#x}, {:#x}) {:?}",
            start,
            start.into() + size,
            pa_start,
            pa_start + size,
            flags
        );
        pt.map_region(
            start,
            |va| PhysAddr::from(va.into() - pa_va_offset),
            size,
            flags,
            false,
            false,
        )
        .is_ok()
    }

    pub(crate) fn unmap_linear(
        &self,
        start: M::VirtAddr,
        size: usize,
        pt: &mut PageTable64<M, PTE, H>,
        _pa_va_offset: usize,
    ) -> bool {
        debug!("unmap_linear: [{:#x}, {:#x})", start, start.into() + size);
        pt.unmap_region(start, size, true).is_ok()
    }
}
