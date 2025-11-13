use axerrno::{AxResult, ax_err};
use page_table_multiarch::{PageTable64, PagingHandler};

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        pub type NestedPageTableL4<H> = PageTable64<arch::X86_64PagingMetaDataL4, arch::X86_64PTE, H>;

    } else if #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))] {
        pub type NestedPageTableL3<H> = PageTable64<arch::Sv39MetaData, arch::Rv64PTE, H>;
        pub type NestedPageTableL4<H> = PageTable64<arch::Sv48MetaData, arch::Rv64PTE, H>;

    } else if #[cfg(target_arch = "aarch64")] {
       /// AArch64 Level 3 nested page table type alias.
        pub type NestedPageTableL3<H> = PageTable64<arch::A64HVPagingMetaDataL3, arch::A64PTEHV, H>;

        /// AArch64 Level 4 nested page table type alias.
        pub type NestedPageTableL4<H> = PageTable64<arch::A64HVPagingMetaDataL4, arch::A64PTEHV, H>;
    }
}

mod arch;

pub enum NestedPageTable<H: PagingHandler> {
    #[cfg(not(target_arch = "x86_64"))]
    L3(NestedPageTableL3<H>),
    L4(NestedPageTableL4<H>),
}

impl<H: PagingHandler> NestedPageTable<H> {
    pub fn new(level: usize) -> AxResult<Self> {
        match level {
            3 => {
                #[cfg(not(target_arch = "x86_64"))]
                {
                    Ok(NestedPageTable::L3(
                        NestedPageTableL3::try_new().map_err(|_| axerrno::AxError::NoMemory)?,
                    ))
                }
                #[cfg(target_arch = "x86_64")]
                {
                    Err(axerrno::AxError::InvalidInput)
                }
            }
            4 => Ok(NestedPageTable::L4(
                NestedPageTableL4::try_new().map_err(|_| axerrno::AxError::NoMemory)?,
            )),
            _ => Err(axerrno::AxError::InvalidInput),
        }
    }
}
