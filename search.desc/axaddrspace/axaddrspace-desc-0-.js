searchState.loadedDescShard("axaddrspace", 0, "ArceOS-Hypervisor guest VM address space management module.\nThe virtual memory address space.\nAllocation mapping backend.\nA unified enum type for different memory mapping backends.\nThe memory is device memory.\nThe memory is executable.\nGuest physical address.\nGuest physical address range.\nGuest virtual address.\nGuest virtual address range.\nHost physical address.\nHost virtual address.\nLinear mapping backend.\nGeneric page table entry flags that indicate the …\nInformation about nested page faults.\nThe memory is readable.\nThe memory is uncached.\nThe memory is user accessible.\nThe memory is writable.\nAccess type that caused the nested page fault.\nGet a flags value with all known bits set.\nConverts an <code>GuestVirtAddr</code> to an <code>usize</code>.\nConverts an <code>GuestPhysAddr</code> to an <code>usize</code>.\nReturns the address space base.\nThe bitwise and (<code>&amp;</code>) of the bits in two flags values.\nThe bitwise and (<code>&amp;</code>) of the bits in two flags values.\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nGet the underlying bits value.\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nRemoves all mappings in the address space.\nThe bitwise negation (<code>!</code>) of the bits in a flags value, …\nWhether all set bits in a source flags value are also set …\nChecks if the address space contains the given address …\nThe intersection of a source flags value with the …\nGet a flags value with all bits unset.\nReturns the address space end.\nThe upper bound of the range (exclusive).\nThe upper bound of the range (exclusive).\nThe bitwise or (<code>|</code>) of the bits in each flags value.\nGuest physical address that caused the nested page fault.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nConvert from a bits value.\nConvert from a bits value exactly.\nConvert from a bits value, unsetting any unknown bits.\nThe bitwise or (<code>|</code>) of the bits in each flags value.\nGet a flags value with the bits of a flag with the given …\nConverts an <code>usize</code> to an <code>GuestVirtAddr</code>.\nConverts an <code>usize</code> to an <code>GuestPhysAddr</code>.\nHandles a page fault at the given address.\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nThe bitwise and (<code>&amp;</code>) of the bits in two flags values.\nWhether any set bits in a source flags value are also set …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nWhether all known bits in this flags value are set.\nWhether all bits in this flags value are unset.\nYield a set of contained flags values.\nYield a set of contained named flags values.\nAdd a new allocation mapping.\nAdd a new linear mapping.\nCreates a new allocation mapping backend.\nCreates a new empty address space.\nCreates a new linear mapping backend.\nThe bitwise negation (<code>!</code>) of the bits in a flags value, …\nReturns the reference to the inner page table.\nReturns the root physical address of the inner page table.\nThe intersection of a source flags value with the …\nCall <code>insert</code> when <code>value</code> is <code>true</code> or <code>remove</code> when <code>value</code> is …\nReturns the address space size.\nThe lower bound of the range (inclusive).\nThe lower bound of the range (inclusive).\nThe intersection of a source flags value with the …\nThe intersection of a source flags value with the …\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nTranslates the given <code>VirtAddr</code> into <code>PhysAddr</code>.\nTranslates the given <code>VirtAddr</code> into <code>PhysAddr</code>, and returns …\nTranslate&amp;Copy the given <code>VirtAddr</code> with LENGTH len to a …\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nRemoves mappings within the specified virtual address …\nA phantom data for the paging handler.\n<code>vaddr - paddr</code>.\nWhether to populate the physical frames when creating the …")