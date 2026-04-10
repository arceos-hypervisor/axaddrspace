// Copyright 2025 The Axvisor Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! LoongArch架构嵌套页表实现
//!
//! 本模块基于龙芯虚拟化扩展（LVZ）实现嵌套页表管理功能。
//!
//! ## 关键技术要点
//!
//! 1. **两级地址翻译**：GVA → GPA → HPA
//!    - 一级翻译（GVA→GPA）：在Guest模式下完成
//!    - 二级翻译（GPA→HPA）：在Host模式下完成，由Hypervisor维护
//!
//! 2. **TLB隔离机制**：通过GID（Guest ID）实现虚拟机隔离
//!    - GID=0：表示Host page
//!    - GID≠0：表示某个VM的Guest/VMM页
//!
//! 3. **页表格式**：复用LoongArch常规页表/TLB格式，不另起一套PTE位图
//!
//! ## 参考文档
//!
//! - 龙芯虚拟化及二进制翻译扩展手册（卷三）
//! - loongarchpagetable.md（项目文档）

use crate::{GuestPhysAddr, HostPhysAddr};
use core::arch::asm;
use core::fmt;
use page_table_entry::{GenericPTE, MappingFlags};
use page_table_multiarch::PagingMetaData;

/// 龙芯嵌套页表项结构
///
/// 基于龙芯架构TLB条目格式（TLBELO0/TLBELO1寄存器）
///
/// ## TLBELO寄存器位字段定义（基于龙芯架构参考手册）
///
/// | 位段 | 名称 | 含义 |
/// |------|------|------|
/// | 0 | V | 有效位 |
/// | 1 | D | 脏位（已修改，表示可写） |
/// | 3:4 | PLV | 特权级（0-3） |
/// | 5:6 | MAT | 内存属性类型 |
/// | 7 | G | 全局标志位 |
/// | 8:10 | PS | 页大小 |
/// | 12:47 | PPN | 物理页号 |
///
/// 注意：龙芯虚拟化扩展使用标准TLB条目格式，通过GID实现虚拟机隔离。
/// GID存储在TLB条目中但不是标准TLB格式的一部分，由硬件管理。
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct LoongArchPTE(u64);

impl LoongArchPTE {
    // 物理地址掩码：龙芯支持48位物理地址
    // bits 12..48（PPN字段）
    const PHYS_ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

    // ===== TLBELO寄存器位字段定义 =====
    // 基于龙芯架构参考手册卷一的TLBELO寄存器格式

    /// V位：条目有效（bit 0）
    const VALID: u64 = 1 << 0;

    /// D位：脏位/已修改（bit 1），表示可写
    const DIRTY: u64 = 1 << 1;

    /// NR位：非读位（bit 2），置1表示不可读
    const NR: u64 = 1 << 2;

    /// PLV位：特权级（bits 3-4），2位字段
    /// - 0：最高特权级（内核模式）
    /// - 3：最低特权级（用户模式）
    const PLV_SHIFT: u64 = 3;
    const PLV_MASK: u64 = 0b11 << 3;

    /// MAT位：内存属性类型（bits 5-6），2位字段
    /// 编码含义：
    /// - 0b00：强一致非缓存（Strongly Unordered）
    /// - 0b01：一致可缓存（Coherent Cached）
    /// - 0b10：弱一致非缓存（Weakly Unordered）
    /// - 0b11：弱一致非缓存但可执行（Weakly Unordered with Execute）
    const MAT_SHIFT: u64 = 5;
    const MAT_MASK: u64 = 0b11 << 5;

    /// G位：全局标志位（bit 7）
    /// 置1表示全局映射，不受ASID影响
    const GLOBAL: u64 = 1 << 7;

    /// PS位：页大小（bits 8-10），3位字段
    /// 编码含义：
    /// - 0b000：4KB页
    /// - 0b001：16KB页
    /// - 0b010：64KB页
    /// - 0b011：256KB页
    /// - 0b100：1MB页
    /// - 0b101：4MB页
    /// - 0b110：16MB页
    /// - 0b111：64MB页
    const PS_SHIFT: u64 = 8;
    const PS_MASK: u64 = 0b111 << 8;

    // ===== 内存属性编码常量 =====
    /// 强一致非缓存（设备内存）
    const MAT_STRONG_UNORDERED: u64 = 0b00 << Self::MAT_SHIFT;

    /// 一致可缓存（普通内存，可执行）
    const MAT_COHERENT_CACHED: u64 = 0b01 << Self::MAT_SHIFT;

    /// 弱一致非缓存
    const MAT_WEAK_UNORDERED: u64 = 0b10 << Self::MAT_SHIFT;

    /// 弱一致非缓存但可执行
    const MAT_WEAK_UNORDERED_EXEC: u64 = 0b11 << Self::MAT_SHIFT;

    // ===== 页大小编码常量 =====
    /// 4KB页（标准页）
    const PS_4K: u64 = 0b000 << Self::PS_SHIFT;

    /// 16KB页
    const PS_16K: u64 = 0b001 << Self::PS_SHIFT;

    /// 64KB页
    const PS_64K: u64 = 0b010 << Self::PS_SHIFT;

    /// 1MB页（大页）
    const PS_1M: u64 = 0b100 << Self::PS_SHIFT;

    /// 4MB页（大页）
    const PS_4M: u64 = 0b101 << Self::PS_SHIFT;

    /// 16MB页（大页）
    const PS_16M: u64 = 0b110 << Self::PS_SHIFT;
}

impl GenericPTE for LoongArchPTE {
    fn bits(self) -> usize {
        self.0 as usize
    }

    fn new_page(paddr: HostPhysAddr, flags: MappingFlags, is_huge: bool) -> Self {
        let mut pte_value = paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK;

        // 设置有效位
        pte_value |= Self::VALID;

        // 设置特权级：Hypervisor使用最高特权级（PLV=3）
        pte_value |= Self::PLV_MASK; // PLV=3

        // 设置全局标志（虚拟化映射）
        pte_value |= Self::GLOBAL;

        // 设置权限标志
        if flags.contains(MappingFlags::READ) {
            // 读权限隐含在VALID中，NR位默认为0表示可读
        }
        if flags.contains(MappingFlags::WRITE) {
            pte_value |= Self::DIRTY; // 脏位表示可写
        }

        // 设置内存属性
        if flags.contains(MappingFlags::DEVICE) {
            // 设备内存使用强一致非缓存
            pte_value |= Self::MAT_STRONG_UNORDERED;
        } else if flags.contains(MappingFlags::UNCACHED) {
            // 非缓存内存使用弱一致非缓存
            pte_value |= Self::MAT_WEAK_UNORDERED;
        } else {
            // 普通内存使用一致可缓存（可执行）
            pte_value |= Self::MAT_COHERENT_CACHED;
        }

        // 设置页大小
        if is_huge {
            // 大页使用1MB页大小（龙芯常用大页大小）
            pte_value |= Self::PS_1M;
        } else {
            pte_value |= Self::PS_4K;
        }

        // 如果请求执行权限但内存属性不是可执行，需要特殊处理
        if flags.contains(MappingFlags::EXECUTE) {
            let mat = pte_value & Self::MAT_MASK;
            // 设备内存不可执行，需要调整为可执行属性
            if mat == Self::MAT_STRONG_UNORDERED {
                pte_value = (pte_value & !Self::MAT_MASK) | Self::MAT_WEAK_UNORDERED_EXEC;
            }
        }

        Self(pte_value)
    }

    fn new_table(paddr: HostPhysAddr) -> Self {
        // 页表目录项：指向下一级页表
        let pte_value = (paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK)
            | Self::VALID              // 有效
            | Self::DIRTY              // 可写（用于更新页表）
            | Self::PLV_MASK           // PLV=3（最高特权级）
            | Self::GLOBAL             // 全局映射
            | Self::MAT_COHERENT_CACHED // 普通内存属性
            | Self::PS_4K; // 4KB页（目录表）
        Self(pte_value)
    }

    fn paddr(&self) -> HostPhysAddr {
        HostPhysAddr::from((self.0 & Self::PHYS_ADDR_MASK) as usize)
    }

    fn flags(&self) -> MappingFlags {
        let mut flags = MappingFlags::empty();

        // 有效位表示可读
        if self.0 & Self::VALID != 0 {
            flags |= MappingFlags::READ;
        }

        // 脏位表示可写
        if self.0 & Self::DIRTY != 0 {
            flags |= MappingFlags::WRITE;
        }

        // 判断内存属性
        let mat = self.0 & Self::MAT_MASK;

        // 一致可缓存和弱一致可执行属性表示可执行内存
        if mat == Self::MAT_COHERENT_CACHED || mat == Self::MAT_WEAK_UNORDERED_EXEC {
            flags |= MappingFlags::EXECUTE;
        }

        // 强一致非缓存表示设备内存
        if mat == Self::MAT_STRONG_UNORDERED {
            flags |= MappingFlags::DEVICE;
        }

        // 弱一致非缓存表示非缓存内存
        if mat == Self::MAT_WEAK_UNORDERED {
            flags |= MappingFlags::UNCACHED;
        }

        flags
    }

    fn set_paddr(&mut self, paddr: HostPhysAddr) {
        self.0 =
            (self.0 & !Self::PHYS_ADDR_MASK) | (paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK);
    }

    fn set_flags(&mut self, flags: MappingFlags, is_huge: bool) {
        // 清除原有标志位（保留物理地址）
        let paddr = self.0 & Self::PHYS_ADDR_MASK;
        *self = Self::new_page(HostPhysAddr::from(paddr as usize), flags, is_huge);
    }

    fn is_unused(&self) -> bool {
        self.0 == 0
    }

    fn is_present(&self) -> bool {
        self.0 & Self::VALID != 0
    }

    fn is_huge(&self) -> bool {
        // 检查页大小字段，大页为1MB及以上
        let ps = self.0 & Self::PS_MASK;
        ps >= Self::PS_1M // 1MB、4MB、16MB都是大页
    }

    fn clear(&mut self) {
        self.0 = 0;
    }
}

impl fmt::Debug for LoongArchPTE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("LoongArchPTE");
        f.field("raw", &self.0)
            .field("paddr", &self.paddr())
            .field("flags", &self.flags())
            .field("is_huge", &self.is_huge())
            .finish()
    }
}

/// 龙芯3级页表元数据（类似Sv39）
#[derive(Copy, Clone)]
pub struct LoongArchPagingMetaDataL3;

impl PagingMetaData for LoongArchPagingMetaDataL3 {
    const LEVELS: usize = 3;
    const VA_MAX_BITS: usize = 39; // 512GB Guest物理地址空间
    const PA_MAX_BITS: usize = 48; // 256TB主机物理地址空间

    type VirtAddr = GuestPhysAddr;

    fn flush_tlb(vaddr: Option<Self::VirtAddr>) {
        unsafe {
            // Get current GID from GSTAT CSR (0x50)
            // GID is in bits 16-23
            let gstat: usize;
            asm!("csrrd {}, 0x50", out(reg) gstat);
            let gid = (gstat >> 16) & 0xFF;

            if let Some(vaddr) = vaddr {
                // Use invtlb with op=0x7 (GID_VA) to flush specific address
                // invtlb 0x7, rj, rk - delete entries matching GID and VA
                // rj = gid, rk = virtual address
                asm!(
                    "invtlb 0x7, {0}, {1}",
                    in(reg) gid,
                    in(reg) vaddr.as_usize()
                );
            } else {
                // Use invtlb with op=0x6 (GID) to flush all entries for this GID
                // invtlb 0x6, rj, rk - delete all entries matching GID
                // rj = gid, rk = 0 (ignored)
                asm!(
                    "invtlb 0x6, {0}, $r0",
                    in(reg) gid
                );
            }
            // Memory barrier to ensure TLB operation completes
            asm!("dbar 0");
        }
    }
}

/// 龙芯4级页表元数据（类似Sv48）
#[derive(Copy, Clone)]
pub struct LoongArchPagingMetaDataL4;

impl PagingMetaData for LoongArchPagingMetaDataL4 {
    const LEVELS: usize = 4;
    const VA_MAX_BITS: usize = 48; // 256TB Guest物理地址空间
    const PA_MAX_BITS: usize = 48; // 256TB主机物理地址空间

    type VirtAddr = GuestPhysAddr;

    fn flush_tlb(vaddr: Option<Self::VirtAddr>) {
        // 与L3使用相同的TLB刷新机制
        LoongArchPagingMetaDataL3::flush_tlb(vaddr);
    }
}

// ============================================================================
// TLB刷新操作模块
// ============================================================================

/// TLB刷新操作
///
/// 龙芯虚拟化扩展提供了专门的TLB刷新指令：
/// - `invtlb`：TLB无效化指令，用于刷新TLB条目
///
/// 根据龙芯虚拟化扩展手册（卷三），TLB刷新需要考虑：
/// 1. GID（Guest ID）隔离 - 不同虚拟机的TLB条目通过GID区分
/// 2. 地址范围 - 可以刷新单个地址或整个TLB
/// 3. 操作类型 - 支持多种刷新模式
///
/// ## invtlb指令格式
///
/// `invtlb op, rj, rk`
/// - `op`：操作码，决定刷新类型
/// - `rj`：包含ASID或GID信息
/// - `rk`：包含虚拟地址（如果需要）
///
/// ## invtlb指令操作码说明（基于龙芯虚拟化扩展手册）
///
/// | 操作码 | 名称 | 功能描述 |
/// |--------|------|----------|
/// | 0x0 | ALL | 删除所有TLB条目 |
/// | 0x1 | ALL_ASID | 删除指定ASID的所有条目 |
/// | 0x2 | ALL_VA | 删除指定虚拟地址的条目（所有ASID） |
/// | 0x3 | VA_ASID | 删除指定ASID和虚拟地址的条目 |
/// | 0x4 | G0 | 删除所有G=0的条目（Host条目） |
/// | 0x5 | G1 | 删除所有G!=0的条目（Guest条目） |
/// | 0x6 | GID | 删除指定GID的所有条目 |
/// | 0x7 | GID_VA | 删除指定GID和虚拟地址的条目 |
///
/// ## 虚拟化扩展专用操作码
///
/// | 操作码 | 名称 | 功能描述 |
/// |--------|------|----------|
/// | 0x9 | GID_ALL_GUEST | 删除所有指定GID的Guest页表项 |
/// | 0xA | GID_G1_GUEST | 删除所有指定GID且G=1的Guest页表项 |
/// | 0xB | GID_G0_GUEST | 删除所有指定GID且G=0的Guest页表项 |
/// | 0xC | GID_G0_ASID | 删除指定GID、G=0、ASID的所有条目 |
/// | 0xD | GID_G0_ASID_VA | 删除指定GID、G=0、ASID、VA的条目 |
/// | 0xE | GID_G1_OR_ASID_VA | 删除指定GID且(G=1或ASID匹配)且VA匹配的条目 |
/// | 0x10 | RID_ALL_GUEST | 删除所有GID=当前RID的Guest页表项 |
/// | 0x11 | RID_ALL_SHADOW | 删除所有GID=当前RID的Shadow页表项 |
/// | 0x12 | RID_ALL_BOTH | 同时删除Guest和Shadow页表项 |
/// | 0x13 | TGID_ALL_GUEST | 删除所有GID=指定值的Guest页表项 |
/// | 0x14 | TGID_ALL_SHADOW | 删除所有GID=指定值的Shadow页表项 |
/// | 0x15 | TGID_ALL_BOTH | 同时删除指定GID的Guest和Shadow页表项 |
/// | 0x16 | TGID_GPA_SHADOW | 删除指定GID且GPA=指定VA的Shadow页表项 |
pub struct TLBFlushOps;

/// invtlb指令操作码常量
///
/// 基于龙芯虚拟化扩展手册（卷三）第11-16页的INVTLB指令定义
#[allow(dead_code)]
mod invtlb_op {
    /// 删除所有TLB条目
    pub const ALL: u32 = 0x0;
    /// 删除指定ASID的所有条目
    pub const ALL_ASID: u32 = 0x1;
    /// 删除指定虚拟地址的条目（所有ASID）
    pub const ALL_VA: u32 = 0x2;
    /// 删除指定ASID和虚拟地址的条目
    pub const VA_ASID: u32 = 0x3;
    /// 删除所有G=0的条目（Host条目）
    pub const G0: u32 = 0x4;
    /// 删除所有G!=0的条目（Guest条目）
    pub const G1: u32 = 0x5;
    /// 删除指定GID的所有条目
    pub const GID: u32 = 0x6;
    /// 删除指定GID和虚拟地址的条目
    pub const GID_VA: u32 = 0x7;

    // ===== 虚拟化扩展专用操作码 =====
    /// 删除所有指定GID的Guest页表项
    pub const GID_ALL_GUEST: u32 = 0x9;
    /// 删除所有指定GID且G=1的Guest页表项
    pub const GID_G1_GUEST: u32 = 0xA;
    /// 删除所有指定GID且G=0的Guest页表项
    pub const GID_G0_GUEST: u32 = 0xB;
    /// 删除指定GID、G=0、ASID的所有条目
    pub const GID_G0_ASID: u32 = 0xC;
    /// 删除指定GID、G=0、ASID、VA的条目
    pub const GID_G0_ASID_VA: u32 = 0xD;
    /// 删除指定GID且(G=1或ASID匹配)且VA匹配的条目
    pub const GID_G1_OR_ASID_VA: u32 = 0xE;
    /// 删除所有GID=当前RID的Guest页表项
    pub const RID_ALL_GUEST: u32 = 0x10;
    /// 删除所有GID=当前RID的Shadow页表项
    pub const RID_ALL_SHADOW: u32 = 0x11;
    /// 同时删除Guest和Shadow页表项（当前RID）
    pub const RID_ALL_BOTH: u32 = 0x12;
    /// 删除所有GID=指定值的Guest页表项
    pub const TGID_ALL_GUEST: u32 = 0x13;
    /// 删除所有GID=指定值的Shadow页表项
    pub const TGID_ALL_SHADOW: u32 = 0x14;
    /// 同时删除指定GID的Guest和Shadow页表项
    pub const TGID_ALL_BOTH: u32 = 0x15;
    /// 删除指定GID且GPA=指定VA的Shadow页表项
    pub const TGID_GPA_SHADOW: u32 = 0x16;
}

impl TLBFlushOps {
    /// 刷新指定地址的TLB条目
    ///
    /// # 参数
    ///
    /// * `vaddr` - 虚拟地址
    /// * `asid` - 地址空间ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令，需要确保：
    /// - 调用者具有足够的特权级
    /// - 地址和ASID参数有效
    #[inline]
    pub unsafe fn flush_one(vaddr: usize, asid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::VA_ASID,
            asid = in(reg) asid,
            vaddr = in(reg) vaddr,
            options(nostack, preserves_flags)
        );
        // 内存屏障，确保TLB操作完成
        asm!("dbar 0");
    }

    /// 刷新指定ASID的所有TLB条目
    ///
    /// # 参数
    ///
    /// * `asid` - 地址空间ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_asid(asid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::ALL_ASID,
            asid = in(reg) asid,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 刷新所有TLB条目
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_all() {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::ALL,
            asid = in(reg) 0,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 刷新所有Guest TLB条目（G != 0）
    ///
    /// 在虚拟化环境中，G!=0表示Guest页表项
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_all_guest() {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::G1,
            asid = in(reg) 0,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 刷新所有Host TLB条目（G = 0）
    ///
    /// 在虚拟化环境中，G=0表示Host页表项
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_all_host() {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::G0,
            asid = in(reg) 0,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 刷新指定GID的所有TLB条目
    ///
    /// # 参数
    ///
    /// * `gid` - Guest ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_gid(gid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::GID,
            asid = in(reg) gid,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 刷新指定GID和地址的TLB条目
    ///
    /// # 参数
    ///
    /// * `vaddr` - 虚拟地址
    /// * `gid` - Guest ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_gid_addr(vaddr: usize, gid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::GID_VA,
            asid = in(reg) gid,
            vaddr = in(reg) vaddr,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 刷新指定虚拟地址的所有TLB条目（跨所有ASID）
    ///
    /// # 参数
    ///
    /// * `vaddr` - 虚拟地址
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_va_all(vaddr: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::ALL_VA,
            asid = in(reg) 0,
            vaddr = in(reg) vaddr,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    // ===== 虚拟化扩展专用TLB刷新操作 =====

    /// 删除指定GID的所有Guest页表项
    ///
    /// # 参数
    ///
    /// * `gid` - Guest ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_gid_all_guest(gid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::GID_ALL_GUEST,
            asid = in(reg) gid,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 删除指定GID的所有Shadow页表项（VMM页表）
    ///
    /// Shadow页表是Hypervisor维护的GPA→HPA映射
    ///
    /// # 参数
    ///
    /// * `gid` - Guest ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_gid_all_shadow(gid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::TGID_ALL_SHADOW,
            asid = in(reg) gid,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 同时删除指定GID的Guest和Shadow页表项
    ///
    /// # 参数
    ///
    /// * `gid` - Guest ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_gid_all_both(gid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::TGID_ALL_BOTH,
            asid = in(reg) gid,
            vaddr = in(reg) 0,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }

    /// 删除指定GID且GPA=指定VA的Shadow页表项
    ///
    /// 用于刷新特定的GPA→HPA映射
    ///
    /// # 参数
    ///
    /// * `gpa` - Guest物理地址
    /// * `gid` - Guest ID
    ///
    /// # 安全性
    ///
    /// 此函数使用内联汇编执行TLB刷新指令
    #[inline]
    pub unsafe fn flush_gid_gpa_shadow(gpa: usize, gid: usize) {
        asm!(
            "invtlb {op}, {asid}, {vaddr}",
            op = const invtlb_op::TGID_GPA_SHADOW,
            asid = in(reg) gid,
            vaddr = in(reg) gpa,
            options(nostack, preserves_flags)
        );
        asm!("dbar 0");
    }
}
