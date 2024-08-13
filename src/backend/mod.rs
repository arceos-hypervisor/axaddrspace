//! Memory mapping backends.

use memory_addr::VirtAddr;
use memory_set::MappingBackend;
use page_table_multiarch::{MappingFlags, PagingHandler};

use crate::addr::GuestPhysAddr;
use crate::npt::NestedPageTable as PageTable;

mod alloc;
mod linear;

/// A unified enum type for different memory mapping backends.
///
/// Currently, two backends are implemented:
///
/// - **Linear**: used for linear mappings. The target physical frames are
///   contiguous and their addresses should be known when creating the mapping.
/// - **Allocation**: used in general, or for lazy mappings. The target physical
///   frames are obtained from the global allocator.
#[derive(Clone)]
pub enum Backend {
    /// Linear mapping backend.
    ///
    /// The offset between the virtual address and the physical address is
    /// constant, which is specified by `pa_va_offset`. For example, the virtual
    /// address `vaddr` is mapped to the physical address `vaddr - pa_va_offset`.
    Linear {
        /// `vaddr - paddr`.
        pa_va_offset: usize,
    },
    /// Allocation mapping backend.
    ///
    /// If `populate` is `true`, all physical frames are allocated when the
    /// mapping is created, and no page faults are triggered during the memory
    /// access. Otherwise, the physical frames are allocated on demand (by
    /// handling page faults).
    Alloc {
        /// Whether to populate the physical frames when creating the mapping.
        populate: bool,
    },
}

impl<H: PagingHandler> MappingBackend<MappingFlags, PageTable<H>> for Backend {
    fn map(
        &self,
        start: VirtAddr,
        size: usize,
        flags: MappingFlags,
        pt: &mut PageTable<H>,
    ) -> bool {
        match *self {
            Self::Linear { pa_va_offset } => {
                self.map_linear(start.into(), size, flags, pt, pa_va_offset)
            }
            Self::Alloc { populate } => self.map_alloc(start.into(), size, flags, pt, populate),
        }
    }

    fn unmap(&self, start: VirtAddr, size: usize, pt: &mut PageTable<H>) -> bool {
        match *self {
            Self::Linear { pa_va_offset } => {
                self.unmap_linear(start.into(), size, pt, pa_va_offset)
            }
            Self::Alloc { populate } => self.unmap_alloc(start.into(), size, pt, populate),
        }
    }

    fn protect(
        &self,
        start: VirtAddr,
        size: usize,
        new_flags: MappingFlags,
        page_table: &mut PageTable<H>,
    ) -> bool {
        // TODO
        match page_table.protect_region(start.into(), size, new_flags, true) {
            Ok(tlb_flush_all) => {
                tlb_flush_all.ignore();
                true
            }
            Err(err) => {
                warn!("Failed to protect_region, err {err:?}");
                false
            }
        }
    }
}

impl Backend {
    pub(crate) fn handle_page_fault<H: PagingHandler>(
        &self,
        gpa: GuestPhysAddr,
        orig_flags: MappingFlags,
        page_table: &mut PageTable<H>,
    ) -> bool {
        match *self {
            Self::Linear { .. } => false, // Linear mappings should not trigger page faults.
            Self::Alloc { populate } => {
                self.handle_page_fault_alloc(gpa, orig_flags, page_table, populate)
            }
        }
    }
}
