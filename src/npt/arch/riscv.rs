use page_table_entry::riscv::Rv64PTE;
use page_table_multiarch::{riscv::Sv39PageTable, PageTable64};

use crate::GuestPhysAddr;

pub type NestedPageTable<H> = PageTable64<Sv39PageTable<GuestPhysAddr>, Rv64PTE, H>;
