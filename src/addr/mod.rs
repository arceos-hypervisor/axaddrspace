mod addr;
mod range;

pub use memory_addr::PAGE_SIZE_4K;

/// Guest physical address.
pub use addr::GuestPhysAddr;
/// Guest virtual address.
pub use addr::GuestVirtAddr;
/// Host virtual address.
pub type HostVirtAddr = memory_addr::VirtAddr;
/// Host physical address.
pub type HostPhysAddr = memory_addr::PhysAddr;

pub use range::*;
