use memory_addr::AddrRange;

use crate::GuestPhysAddr;

use super::{Port, SysRegAddr};


/// An address-like type that can be used to access devices.
pub trait DeviceAddr: Copy + Eq + Ord + core::fmt::Debug {}

/// A range of device addresses. It may be contiguous or not.
pub trait DeviceAddrRange {
    /// The address type of the range.
    type Addr: DeviceAddr;

    /// Returns whether the address range contains the given address.
    fn contains(&self, addr: Self::Addr) -> bool;
}


impl DeviceAddr for GuestPhysAddr {}

impl DeviceAddrRange for AddrRange<GuestPhysAddr> {
    type Addr = GuestPhysAddr;

    fn contains(&self, addr: Self::Addr) -> bool {
        Self::contains(*self, addr)
    }
}

impl DeviceAddr for SysRegAddr {}

/// A range of system register addresses.
/// 
/// Unlike [`AddrRange`], this type is inclusive on both ends.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct SysRegAddrRange {
    pub start: SysRegAddr,
    pub end: SysRegAddr,
}

impl SysRegAddrRange {
    /// Creates a new [`SysRegAddrRange`] instance.
    pub fn new(start: SysRegAddr, end: SysRegAddr) -> Self {
        Self { start, end }
    }
}

impl DeviceAddrRange for SysRegAddrRange {
    type Addr = SysRegAddr;

    fn contains(&self, addr: Self::Addr) -> bool {
        addr.0 >= self.start.0 && addr.0 <= self.end.0
    }
}

impl DeviceAddr for Port {}

/// A range of port numbers.
/// 
/// Unlike [`AddrRange`], this type is inclusive on both ends.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct PortRange {
    pub start: Port,
    pub end: Port,
}

impl PortRange {
    /// Creates a new [`PortRange`] instance.
    pub fn new(start: Port, end: Port) -> Self {
        Self { start, end }
    }
}

impl DeviceAddrRange for PortRange {
    type Addr = Port;

    fn contains(&self, addr: Self::Addr) -> bool {
        addr.0 >= self.start.0 && addr.0 <= self.end.0
    }
}