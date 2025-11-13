use page_table_entry::riscv::Rv64PTE;
use page_table_multiarch::{
    PageTable64,
    riscv::{Sv39MetaData, Sv48MetaData},
};

use crate::GuestPhysAddr;
