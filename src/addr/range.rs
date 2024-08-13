use core::{fmt, ops::Range};

use crate::{GuestPhysAddr, GuestVirtAddr};

macro_rules! usize {
    ($addr:expr) => {
        ($addr).as_usize()
    };
}

macro_rules! def_range {
    ($name:ident, $addr_type:ty) => {
        #[derive(Clone, Copy, Default, PartialEq, Eq)]
        #[doc = concat!("A range of [`", stringify!($addr_type), "`].\n\n")]
        #[doc = "The range is inclusive on the start and exclusive on the end."]
        #[doc = "It is empty if `start >= end`."]
        pub struct $name {
            /// The lower bound of the range (inclusive).
            pub start: $addr_type,
            /// The upper bound of the range (exclusive).
            pub end: $addr_type,
        }

        impl $name {
            /// Creates a new address range.
            #[inline]
            pub const fn new(start: $addr_type, end: $addr_type) -> Self {
                Self { start, end }
            }

            /// Creates a new address range from the start address and the size.
            #[inline]
            pub const fn from_start_size(start: $addr_type, size: usize) -> Self {
                Self {
                    start,
                    end: <$addr_type>::from(usize!(start) + size),
                }
            }

            /// Returns `true` if the range is empty (`start >= end`).
            #[inline]
            pub const fn is_empty(self) -> bool {
                usize!(self.start) >= usize!(self.end)
            }

            /// Returns the size of the range.
            #[inline]
            pub const fn size(self) -> usize {
                usize!(self.end) - usize!(self.start)
            }

            /// Checks if the range contains the given address.
            #[inline]
            pub const fn contains(self, addr: $addr_type) -> bool {
                usize!(self.start) <= usize!(addr) && usize!(addr) < usize!(self.end)
            }

            /// Checks if the range contains the given address range.
            #[inline]
            pub const fn contains_range(self, other: Self) -> bool {
                usize!(self.start) <= usize!(other.start) && usize!(other.end) <= usize!(self.end)
            }

            /// Checks if the range is contained in the given address range.
            #[inline]
            pub const fn contained_in(self, other: Self) -> bool {
                other.contains_range(self)
            }

            /// Checks if the range overlaps with the given address range.
            #[inline]
            pub const fn overlaps(self, other: Self) -> bool {
                usize!(self.start) < usize!(other.end) && usize!(other.start) < usize!(self.end)
            }
        }

        impl<A> From<Range<A>> for $name
        where
            A: From<usize> + Into<usize>,
        {
            fn from(range: Range<A>) -> Self {
                Self::new(range.start.into().into(), range.end.into().into())
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{:#x?}..{:#x?}", usize!(self.start), usize!(self.end))
            }
        }
    };
}

def_range!(GuestVirtAddrRange, GuestVirtAddr);
def_range!(GuestPhysAddrRange, GuestPhysAddr);

/// Converts the given range expression into [`GuestVirtAddrRange`].
///
/// # Example
///
/// ```
/// use axaddrspace::gva_range;
///
/// let range = gva_range!(0x1000..0x2000);
/// assert_eq!(range.start, 0x1000.into());
/// assert_eq!(range.end, 0x2000.into());
/// ```
#[macro_export]
macro_rules! gva_range {
    ($range:expr) => {
        $crate::GuestVirtAddrRange::from($range)
    };
}

/// Converts the given range expression into [`GuestPhysAddrRange`].
///
/// # Example
///
/// ```
/// use axaddrspace::gpa_range;
///
/// let range = gpa_range!(0x1000..0x2000);
/// assert_eq!(range.start, 0x1000.into());
/// assert_eq!(range.end, 0x2000.into());
/// ```
#[macro_export]
macro_rules! gpa_range {
    ($range:expr) => {
        $crate::GuestPhysAddrRange::from($range)
    };
}
