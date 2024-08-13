use core::fmt;
use core::ops::{Add, AddAssign, Sub, SubAssign};

use memory_addr::{align_down, align_offset, align_up, is_aligned, PAGE_SIZE_4K};

/// A guest physical memory address (GPA), also known as intermediate physical address (IPA),
/// this address is used in Stage 1 of a two-stage address translation mechanism, commonly employed in virtualization.
///
/// It's a wrapper type around an `usize`.
#[repr(transparent)]
#[derive(Copy, Clone, Default, Ord, PartialOrd, Eq, PartialEq)]
pub struct GuestPhysAddr(usize);

/// A guest virtual memory address (GVA),
/// which is the virtual address used by the guest operating system.
/// This address undergoes Stage 1 translation in a two-stage address translation mechanism, commonly employed in virtualization.
///
/// It's a wrapper type around an `usize`.
#[repr(transparent)]
#[derive(Copy, Clone, Default, Ord, PartialOrd, Eq, PartialEq)]
pub struct GuestVirtAddr(usize);

impl From<memory_addr::VirtAddr> for GuestPhysAddr {
    #[inline]
    fn from(addr: memory_addr::VirtAddr) -> Self {
        Self(addr.as_usize())
    }
}

impl From<GuestPhysAddr> for memory_addr::VirtAddr {
    #[inline]
    fn from(addr: GuestPhysAddr) -> memory_addr::VirtAddr {
        memory_addr::VirtAddr::from(addr.0)
    }
}

impl GuestPhysAddr {
    /// Converts an `usize` to a guest physical address.
    #[inline]
    pub const fn from(addr: usize) -> Self {
        Self(addr)
    }

    /// Converts the guest physical address to an `usize`.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Aligns the address downwards to the given alignment.
    ///
    /// See the [`align_down`] function for more information.
    #[inline]
    pub fn align_down<U>(self, align: U) -> Self
    where
        U: Into<usize>,
    {
        Self(align_down(self.0, align.into()))
    }

    /// Aligns the address upwards to the given alignment.
    ///
    /// See the [`align_up`] function for more information.
    #[inline]
    pub fn align_up<U>(self, align: U) -> Self
    where
        U: Into<usize>,
    {
        Self(align_up(self.0, align.into()))
    }

    /// Returns the offset of the address within the given alignment.
    ///
    /// See the [`align_offset`] function for more information.
    #[inline]
    pub fn align_offset<U>(self, align: U) -> usize
    where
        U: Into<usize>,
    {
        align_offset(self.0, align.into())
    }

    /// Checks whether the address has the demanded alignment.
    ///
    /// See the [`is_aligned`] function for more information.
    #[inline]
    pub fn is_aligned<U>(self, align: U) -> bool
    where
        U: Into<usize>,
    {
        is_aligned(self.0, align.into())
    }

    /// Aligns the address downwards to 4096 (bytes).
    #[inline]
    pub const fn align_down_4k(self) -> Self {
        Self(align_down(self.0, PAGE_SIZE_4K))
    }

    /// Aligns the address upwards to 4096 (bytes).
    #[inline]
    pub const fn align_up_4k(self) -> Self {
        Self(align_up(self.0, PAGE_SIZE_4K))
    }

    /// Returns the offset of the address within a 4K-sized page.
    #[inline]
    pub const fn align_offset_4k(self) -> usize {
        align_offset(self.0, PAGE_SIZE_4K)
    }

    /// Checks whether the address is 4K-aligned.
    #[inline]
    pub const fn is_aligned_4k(self) -> bool {
        is_aligned(self.0, PAGE_SIZE_4K)
    }
}

impl GuestVirtAddr {
    /// Converts an `usize` to a guest virtual address.
    #[inline]
    pub const fn from(addr: usize) -> Self {
        Self(addr)
    }

    /// Converts the address to an `usize`.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Aligns the address downwards to the given alignment.
    ///
    /// See the [`align_down`] function for more information.
    #[inline]
    pub fn align_down<U>(self, align: U) -> Self
    where
        U: Into<usize>,
    {
        Self(align_down(self.0, align.into()))
    }

    /// Aligns the address upwards to the given alignment.
    ///
    /// See the [`align_up`] function for more information.
    #[inline]
    pub fn align_up<U>(self, align: U) -> Self
    where
        U: Into<usize>,
    {
        Self(align_up(self.0, align.into()))
    }

    /// Returns the offset of the address within the given alignment.
    ///
    /// See the [`align_offset`] function for more information.
    #[inline]
    pub fn align_offset<U>(self, align: U) -> usize
    where
        U: Into<usize>,
    {
        align_offset(self.0, align.into())
    }

    /// Checks whether the address has the demanded alignment.
    ///
    /// See the [`is_aligned`] function for more information.
    #[inline]
    pub fn is_aligned<U>(self, align: U) -> bool
    where
        U: Into<usize>,
    {
        is_aligned(self.0, align.into())
    }

    /// Aligns the address downwards to 4096 (bytes).
    #[inline]
    pub const fn align_down_4k(self) -> Self {
        Self(align_down(self.0, PAGE_SIZE_4K))
    }

    /// Aligns the address upwards to 4096 (bytes).
    #[inline]
    pub fn align_up_4k(self) -> Self {
        Self(align_up(self.0, PAGE_SIZE_4K))
    }

    /// Returns the offset of the address within a 4K-sized page.
    #[inline]
    pub fn align_offset_4k(self) -> usize {
        align_offset(self.0, PAGE_SIZE_4K)
    }

    /// Checks whether the address is 4K-aligned.
    #[inline]
    pub fn is_aligned_4k(self) -> bool {
        is_aligned(self.0, PAGE_SIZE_4K)
    }
}

impl From<usize> for GuestPhysAddr {
    #[inline]
    fn from(addr: usize) -> Self {
        Self(addr)
    }
}

impl From<usize> for GuestVirtAddr {
    #[inline]
    fn from(addr: usize) -> Self {
        Self(addr)
    }
}

impl From<GuestPhysAddr> for usize {
    #[inline]
    fn from(addr: GuestPhysAddr) -> usize {
        addr.0
    }
}

impl From<GuestVirtAddr> for usize {
    #[inline]
    fn from(addr: GuestVirtAddr) -> usize {
        addr.0
    }
}

impl Add<usize> for GuestPhysAddr {
    type Output = Self;
    #[inline]
    fn add(self, rhs: usize) -> Self {
        Self(self.0 + rhs)
    }
}

impl AddAssign<usize> for GuestPhysAddr {
    #[inline]
    fn add_assign(&mut self, rhs: usize) {
        *self = *self + rhs;
    }
}

impl Sub<usize> for GuestPhysAddr {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: usize) -> Self {
        Self(self.0 - rhs)
    }
}

impl SubAssign<usize> for GuestPhysAddr {
    #[inline]
    fn sub_assign(&mut self, rhs: usize) {
        *self = *self - rhs;
    }
}

impl Add<usize> for GuestVirtAddr {
    type Output = Self;
    #[inline]
    fn add(self, rhs: usize) -> Self {
        Self(self.0 + rhs)
    }
}

impl AddAssign<usize> for GuestVirtAddr {
    #[inline]
    fn add_assign(&mut self, rhs: usize) {
        *self = *self + rhs;
    }
}

impl Sub<usize> for GuestVirtAddr {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: usize) -> Self {
        Self(self.0 - rhs)
    }
}

impl SubAssign<usize> for GuestVirtAddr {
    #[inline]
    fn sub_assign(&mut self, rhs: usize) {
        *self = *self - rhs;
    }
}

impl fmt::Debug for GuestPhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("GPA:{:#x}", self.0))
    }
}

impl fmt::Debug for GuestVirtAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("GVA:{:#x}", self.0))
    }
}

impl fmt::LowerHex for GuestPhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("GPA:{:#x}", self.0))
    }
}

impl fmt::UpperHex for GuestPhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("GPA:{:#X}", self.0))
    }
}

impl fmt::LowerHex for GuestVirtAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("GVA:{:#x}", self.0))
    }
}

impl fmt::UpperHex for GuestVirtAddr {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("GVA:{:#X}", self.0))
    }
}

/// Alias for [`GuestPhysAddr::from`].
#[macro_export]
macro_rules! gpa {
    ($addr:expr) => {
        $crate::GuestPhysAddr::from($addr)
    };
}

/// Alias for [`GuestVirtAddr::from`].
#[macro_export]
macro_rules! gva {
    ($addr:expr) => {
        $crate::GuestVirtAddr::from($addr)
    };
}
