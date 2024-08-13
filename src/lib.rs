//! [ArceOS-Hypervisor](https://github.com/arceos-hypervisor/) guest VM address space management module.

#![no_std]
#![feature(const_trait_impl)]

#[macro_use]
extern crate log;
extern crate alloc;

mod addr;
mod address_space;
mod backend;
mod npt;

pub use addr::*;
pub use address_space::*;

use axerrno::AxError;
use memory_set::MappingError;

fn mapping_err_to_ax_err(err: MappingError) -> AxError {
    warn!("Mapping error: {:?}", err);
    match err {
        MappingError::InvalidParam => AxError::InvalidInput,
        MappingError::AlreadyExists => AxError::AlreadyExists,
        MappingError::BadState => AxError::BadState,
    }
}
