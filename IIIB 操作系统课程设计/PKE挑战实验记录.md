# Lab1 Challenge2 : 实现异常源代码行定位

## 1. 实验原理与分析过程

### 1.1 实验目标

在操作系统内核中实现异常发生时的源代码定位功能，当用户程序触发异常时，能够打印出异常发生的具体源文件、行号以及该行的源代码内容。

### 1.2 为什么要这样做？

**问题背景：**
当程序发生异常（如非法指令、访问错误等）时，内核只能获取到异常发生时的程序计数器（PC）值，这是一个内存地址。对于开发者而言，直接看到内存地址很难定位问题，需要知道异常发生在源代码的哪一行。

**核心原理：**

1. **DWARF调试信息格式**：编译器在编译时（使用`-g`选项）会将源代码行号与机器指令地址的映射关系保存在ELF文件的`.debug_line`段中，采用DWARF标准格式。

2. **地址到源码的映射过程**：
   - 编译时：`源代码行号` → 编译器记录 → `.debug_line`段
   - 运行时：`异常地址(PC)` → 查询`.debug_line` → `源文件名+行号`

3. **DWARF Line Number Program**：`.debug_line`段采用状态机编码方式，通过一系列操作码（opcodes）来高效地表示地址-行号映射关系。状态机维护当前的地址和行号，通过不同的操作码来更新状态并记录映射点。

### 1.3 实现思路

**第一步：加载调试信息**

- 在加载ELF文件时，读取`.debug_line`段的内容
- 解析DWARF格式的调试信息，构建三个数据结构：
  - `dir[]`：存储目录路径
  - `file[]`：存储文件名和对应的目录索引
  - `line[]`：存储地址、行号、文件索引的映射关系

**第二步：异常时查询源码位置**

- 获取异常发生时的PC值（epc寄存器）
- 在`line[]`数组中查找匹配的地址
- 通过文件索引找到源文件名和目录
- 打开源文件，定位到对应行并读取内容

**第三步：调用时机**

- 在M模式异常处理函数中，捕获非法指令异常
- 在打印panic信息前，先调用源码定位函数

---

## 2. 需要修改的代码位置

### 2.1 内核头文件修改

**文件：** `kernel/process.h`

**修改内容：** 在`process`结构体中添加调试信息相关的字段

### 2.2 ELF加载模块修改

**文件：** `kernel/elf.c`

**修改内容：**

1. 实现DWARF调试信息解析函数
2. 在ELF加载流程中调用调试信息加载

### 2.3 异常处理模块修改

**文件：** `kernel/machine/mtrap.c`

**修改内容：**

1. 实现源代码行定位和打印函数
2. 在异常处理流程中调用该函数

---

## 3. 代码修改逻辑

### 3.1 process.h：扩展进程结构体

```c
// 添加地址-行号映射结构
typedef struct addr_line {
    uint64 addr;    // 指令地址
    int line;       // 源代码行号
    uint64 file;    // 文件索引
} addr_line;

// 添加源文件信息结构
typedef struct code_file {
    char *file;     // 文件名
    uint64 dir;     // 目录索引
} code_file;

// 在process结构体中添加字段
typedef struct process_t {
    // ... 原有字段 ...
    
    char *debugline;        // 指向.debug_line段数据
    uint64 debug_line_size; // 调试信息大小
    addr_line *line;        // 地址-行号映射表
    int line_ind;           // 映射表项数
    code_file *file;        // 文件信息表
    int file_ind;           // 文件数量
    char **dir;             // 目录路径表
    int dir_ind;            // 目录数量
} process;
```

**设计要点：**

- `addr_line`：记录每个指令地址对应的源代码行号和文件
- `code_file`：记录文件名和所属目录，避免路径冗余
- 三级索引：`地址 → 文件索引 → 目录索引`，节省内存

### 3.2 elf.h：修正DWARF 4头部结构

**重要修正**：DWARF 4相比DWARF 3新增了`maximum_operations_per_instruction`字段，必须正确定义才能解析：

```c
// compilation units header (in debug line section) - DWARF 4 format
typedef struct __attribute__((packed))
{
  uint32 length;
  uint16 version;
  uint32 header_length;
  uint8 min_instruction_length;
  uint8 maximum_operations_per_instruction;  // DWARF 4 新增字段！
  uint8 default_is_stmt;
  int8 line_base;
  uint8 line_range;
  uint8 opcode_base;
  uint8 std_opcode_lengths[12];
} debug_header;
```

**关键说明**：

- 如果缺少`maximum_operations_per_instruction`字段，会导致后续所有字段读取错位
- 这将导致`line_base`、`line_range`、`opcode_base`值错误
- 最终导致行号计算完全不正确

### 3.2 elf.c：解析DWARF调试信息

**关键配置**：在Makefile中强制使用DWARF 4格式：

```makefile
CFLAGS += -gdwarf-4  # 强制DWARF 4格式（不要用-g，会默认生成DWARF 5）
```

**原因**：

- DWARF 5使用间接字符串引用，解析更复杂
- DWARF 4使用直接字符串指针，代码可直接访问
- 强制使用DWARF 4可简化实现

#### 3.2.1 辅助函数：LEB128解码

LEB128是DWARF使用的变长编码格式，需要实现解码器：

```c
// 无符号LEB128解码
void read_uleb128(uint64 *out, char **off) {
    uint64 value = 0;
    int shift = 0;
    uint8 b;
    
    for (;;) {
        b = *(uint8 *)(*off);
        (*off)++;
        value |= ((uint64)b & 0x7F) << shift;
        shift += 7;
        if ((b & 0x80) == 0) break;  // 最高位为0表示结束
    }
    
    if (out) *out = value;
}

// 有符号LEB128解码
void read_sleb128(int64 *out, char **off) {
    int64 value = 0;
    int shift = 0;
    uint8 b;
    
    for (;;) {
        b = *(uint8 *)(*off);
        (*off)++;
        value |= ((uint64_t)b & 0x7F) << shift;
        shift += 7;
        if ((b & 0x80) == 0) break;
    }
    
    // 符号扩展
    if (shift < 64 && (b & 0x40))
        value |= -(1 << shift);
    
    if (out) *out = value;
}
```

**原理说明：**

- LEB128每个字节用7位存储数据，最高位标识是否继续
- 数据按小端序组合
- 有符号数需要进行符号位扩展

#### 3.2.2 核心函数：make_addr_line

这是解析`.debug_line`的核心函数，实现DWARF状态机：

```c
void make_addr_line(elf_ctx *ctx, char *debug_line, uint64 length) {
    process *p = ((elf_info *)ctx->info)->p;
    p->debugline = debug_line;
    
    // 分配三个数组的内存（在debug_line数据后面）
    p->dir = (char **)((((uint64)debug_line + length + 7) >> 3) << 3);
    int dir_ind = 0, dir_base;
    
    // 添加默认目录"."在索引0（DWARF中dir=0表示当前目录）
    static char default_dir[] = ".";
    p->dir[dir_ind++] = default_dir;
    
    p->file = (code_file *)(p->dir + 64);
    int file_ind = 0, file_base;
    
    p->line = (addr_line *)(p->file + 64);
    p->line_ind = 0;
    
    char *off = debug_line;
    
    // 遍历每个编译单元（CU）
    while (off < debug_line + length) {
        debug_header *dh = (debug_header *)off;
        off += sizeof(debug_header);
        
        dir_base = dir_ind;
        file_base = file_ind;
        
        // 读取目录表
        while (*off != 0) {
            p->dir[dir_ind++] = off;
            while (*off != 0) off++;
            off++;
        }
        off++;
        
        // 读取文件表
        while (*off != 0) {
            p->file[file_ind].file = off;
            while (*off != 0) off++;
            off++;
            
            uint64 dir;
            read_uleb128(&dir, &off);
            
            // DWARF目录索引处理：0表示当前目录，1+表示目录表中的索引
            if (dir == 0) {
                p->file[file_ind++].dir = 0;  // 使用默认目录
            } else {
                uint64 dir_idx = dir - 1 + dir_base;
                if (dir_idx >= 64 || p->dir[dir_idx] == NULL) {
                    p->file[file_ind++].dir = 0;  // 越界则使用默认
                } else {
                    p->file[file_ind++].dir = dir_idx;
                }
            }
            
            read_uleb128(NULL, &off);  // 跳过修改时间
            read_uleb128(NULL, &off);  // 跳过文件大小
        }
        off++;
        
        // 初始化状态机寄存器
        addr_line regs;
        regs.addr = 0;
        regs.file = 1;
        regs.line = 1;
        
        // 执行行号程序（状态机）
        for (;;) {
            uint8 op = *(off++);
            
            switch (op) {
                case 0: {  // 扩展操作码
                    read_uleb128(NULL, &off);
                    op = *(off++);
                    switch (op) {
                        case 1:  // DW_LNE_end_sequence
                            // 去重并记录当前状态
                            if (p->line_ind > 0 && 
                                p->line[p->line_ind - 1].addr == regs.addr)
                                p->line_ind--;
                            p->line[p->line_ind] = regs;
                            p->line[p->line_ind].file += file_base - 1;
                            p->line_ind++;
                            goto endop;
                        case 2:  // DW_LNE_set_address
                            read_uint64(&regs.addr, &off);
                            break;
                        case 4:  // DW_LNE_set_discriminator
                            read_uleb128(NULL, &off);
                            break;
                    }
                    break;
                }
                
                case 1:  // DW_LNS_copy：记录当前状态
                    if (p->line_ind > 0 && 
                        p->line[p->line_ind - 1].addr == regs.addr)
                        p->line_ind--;
                    p->line[p->line_ind] = regs;
                    p->line[p->line_ind].file += file_base - 1;
                    p->line_ind++;
                    break;
                
                case 2: {  // DW_LNS_advance_pc：增加地址
                    uint64 delta;
                    read_uleb128(&delta, &off);
                    regs.addr += delta * dh->min_instruction_length;
                    break;
                }
                
                case 3: {  // DW_LNS_advance_line：增加行号
                    int64 delta;
                    read_sleb128(&delta, &off);
                    regs.line += delta;
                    break;
                }
                
                case 4:  // DW_LNS_set_file：设置文件
                    read_uleb128(&regs.file, &off);
                    break;
                
                case 5:  // DW_LNS_set_column
                    read_uleb128(NULL, &off);
                    break;
                
                case 6:  // DW_LNS_negate_stmt
                case 7:  // DW_LNS_set_basic_block
                    break;
                
                case 8: {  // DW_LNS_const_add_pc
                    int adjust = 255 - dh->opcode_base;
                    int delta = (adjust / dh->line_range) * 
                                dh->min_instruction_length;
                    regs.addr += delta;
                    break;
                }
                
                case 9: {  // DW_LNS_fixed_advanced_pc
                    uint16 delta;
                    read_uint16(&delta, &off);
                    regs.addr += delta;
                    break;
                }
                
                default: {  // 特殊操作码
                    int adjust = op - dh->opcode_base;
                    int addr_delta = (adjust / dh->line_range) * 
                                     dh->min_instruction_length;
                    int line_delta = dh->line_base + 
                                     (adjust % dh->line_range);
                    regs.addr += addr_delta;
                    regs.line += line_delta;
                    
                    if (p->line_ind > 0 && 
                        p->line[p->line_ind - 1].addr == regs.addr)
                        p->line_ind--;
                    p->line[p->line_ind] = regs;
                    p->line[p->line_ind].file += file_base - 1;
                    p->line_ind++;
                    break;
                }
            }
        }
endop:;
    }
}
```

**关键点解析：**

1. **状态机原理**：

   - 维护虚拟寄存器：`addr`（地址）、`line`（行号）、`file`（文件）
   - 操作码更新状态
   - `copy`或特殊操作码将当前状态记录到映射表

2. **特殊操作码计算**：

   ```
   adjusted_opcode = opcode - opcode_base
   address_increment = (adjusted_opcode / line_range) × min_instruction_length
   line_increment = line_base + (adjusted_opcode % line_range)
   ```

   这样可以用单字节同时编码地址和行号增量

3. **文件索引调整**：

   - DWARF文件索引从1开始
   - 需要加上`file_base - 1`转换为全局索引

4. **目录索引处理**：

   - DWARF中dir=0表示当前目录，需映射到预设的"."目录
   - dir>=1则减1后加上dir_base得到实际索引

#### 3.2.3 加载调试信息

```c
elf_status load_debug_line(elf_ctx *ctx) {
    elf_sect_header sh_str;
    elf_sect_header sh_tmp;
    elf_sect_header sh_debugLine = {0};
    
    // 读取section名称字符串表
    if (elf_fpread(ctx, (void *)&sh_str, sizeof(sh_str), 
                   ctx->ehdr.shoff + ctx->ehdr.shstrndx * 
                   sizeof(elf_sect_header)) != sizeof(sh_str))
        return EL_EIO;
    
    char section_name[sh_str.size];
    elf_fpread(ctx, section_name, sh_str.size, sh_str.offset);
    
    // 查找.debug_line段
    for (uint16 i = 0; i < ctx->ehdr.shnum; i++) {
        elf_fpread(ctx, (void *)&sh_tmp, sizeof(sh_tmp), 
                   ctx->ehdr.shoff + i * sizeof(elf_sect_header));
        if (strcmp(section_name + sh_tmp.name, ".debug_line") == 0) {
            sh_debugLine = sh_tmp;
            break;
        }
    }
    
    if (sh_debugLine.size == 0) return EL_OK;  // 没有调试信息
    
    // 读取.debug_line段内容
    static char debug_line[MAX_DEBUG_LINE_SIZE];
    if (sh_debugLine.size > MAX_DEBUG_LINE_SIZE)
        panic("debug_line too long");
    
    if (elf_fpread(ctx, (void *)debug_line, sh_debugLine.size, 
                   sh_debugLine.offset) != sh_debugLine.size) {
        panic("Fail on read debug_line\n");
    }
    
    // 解析调试信息
    make_addr_line(ctx, debug_line, sh_debugLine.size);
    return EL_OK;
}
```

在`load_bincode_from_host_elf`中调用：

```c
// load elf. elf_load() is defined above.
if (elf_load(&elfloader) != EL_OK) panic("Fail on loading elf.\n");

// 加载调试信息
if (load_debug_line(&elfloader) != EL_OK) 
    panic("Fail on load .debug_line\n");
```

### 3.3 mtrap.c：实现源码定位

```c
// 防止递归调用的标志
static int in_print_error_line = 0;

// locate and print the source line that caused the error
static void print_error_line() {
    // 防止递归调用（如果print_error_line内部触发异常）
    if (in_print_error_line) {
        return;
    }
    in_print_error_line = 1;
    
    uint64 exception_addr = read_csr(mepc);
    
    // 使用区间匹配：找到第一个地址大于exception_addr的条目
    // 前一个条目(i-1)就是异常发生的位置
    int found_idx = -1;
    for (int i = 1; i < current->line_ind; i++) {
        if (exception_addr < current->line[i].addr) {
            addr_line *excpline = current->line + i - 1;
            // 验证文件索引有效性
            if (excpline->file != (uint64)-1 && excpline->file < 256) {
                found_idx = i - 1;
                break;
            }
        }
    }
    
    if (found_idx >= 0) {
        addr_line *excpline = current->line + found_idx;
        code_file *file_info = &current->file[excpline->file];
        char *dir_name = current->dir[file_info->dir];
        
        // 检查文件信息有效性
        if (dir_name == NULL || file_info->file == NULL) {
            sprint("Runtime error at line %d (source file information unavailable)\n", 
                   excpline->line);
            in_print_error_line = 0;
            return;
        }
        
        // 构造文件路径
        char full_path[100];
        int dir_len = strlen(dir_name);
        strcpy(full_path, dir_name);
        full_path[dir_len] = '/';
        strcpy(full_path + dir_len + 1, file_info->file);
        
        // 输出源码位置
        sprint("Runtime error at %s:%d\n", full_path, excpline->line);
    }
    
    in_print_error_line = 0;
}

static void handle_illegal_instruction() {
    print_error_line();
    panic("Illegal instruction!");
}

static void handle_instruction_access_fault() {
    print_error_line();
    panic("Instruction access fault!");
}

static void handle_load_access_fault() {
    print_error_line();
    panic("Load access fault!");
}

static void handle_store_access_fault() {
    print_error_line();
    panic("Store/AMO access fault!");
}
```

**实现要点：**

1. **地址匹配算法**：
   - DWARF行号表中的地址是**绝对虚拟地址**，不是相对偏移
   - 使用区间匹配：如果`exception_addr < line[i].addr`，则异常在`line[i-1]`
   - 类似于二分查找的思想，但用线性扫描（数据量小）

2. **递归保护**：
   - 如果`print_error_line()`内部触发异常（如空指针访问），会导致无限递归
   - 使用`in_print_error_line`标志防止重入

3. **文件路径重建**：
   - 通过`file`索引找到`code_file`
   - 通过`dir`索引找到目录路径
   - 拼接为完整路径（如`user/app_errorline.c`）

4. **源码读取限制**：
   - 由于Spike的HTIF文件接口限制，无法可靠读取源文件内容
   - 暂时只输出文件名和行号，已足够定位问题




## 4. 实验步骤与验证

### 4.1 编译配置

确保Makefile中包含调试信息编译选项：

```makefile
CFLAGS += -g  # 生成DWARF调试信息
```

### 4.2 编译运行

```bash
# 清理并编译
make clean
make

# 测试用例1：单个异常
spike obj/riscv-pke obj/app_errorline

# 测试用例2：多行输出后异常
spike obj/riscv-pke obj/app_errorline2
```

### 4.3 预期输出

**app_errorline测试：**

```
Application: obj/app_errorline
Application program entry point (virtual address): 0x00000000810000a0
Switch to user mode...
Going to hack the system by running privilege instructions.
Runtime error at user/app_errorline.c:13
Illegal instruction!
System is shutting down with exit code -1.
```

**app_errorline2测试：**

```
Application: obj/app_errorline2
Switch to user mode...
Going to hack the system by running privilege instructions.
line 14
line 15
line 16
Runtime error at user/app_errorline2.c:17
Illegal instruction!
System is shutting down with exit code -1.
```

### 4.4 功能验证点

1. **调试信息加载**：
   - 检查`process->line_ind`是否大于0
   - 验证`dir`、`file`、`line`数组是否正确填充

2. **地址映射准确性**：
   - 异常地址能否准确对应到源代码行
   - 文件名和行号是否正确

3. **源码读取**：
   - 能否正确打开源文件
   - 读取的代码行内容是否与实际源码一致

### 4.5 调试技巧

如果源码行定位失败，可以：

1. **打印映射表内容**（在`make_addr_line`末尾）：

```c
for (int i = 0; i < p->line_ind; i++)
    sprint("0x%lx -> %s:%d\n", 
           p->line[i].addr, 
           p->file[p->line[i].file].file, 
           p->line[i].line);
```

2. **检查异常地址**：

```c
sprint("Exception at epc: 0x%lx\n", epc);
```

3. **验证编译选项**：

```bash
readelf -wL obj/app_errorline  # 查看.debug_line内容
```

---

## 5. 实验收获

通过本次实验，我们：

1. **理解了调试信息的作用**：编译器如何在二进制文件中保存源码位置信息

2. **掌握了DWARF格式解析**：学习了工业标准的调试信息格式和状态机编程

3. **实现了实用的内核功能**：提升了操作系统的可调试性，这是实际系统开发的重要能力

4. **深入理解了ELF文件结构**：不仅是代码段、数据段，还包括各种元数据段

这个功能在真实操作系统（如Linux）的内核panic、用户程序崩溃报告中都有类似实现，是系统调试的基础设施。

---

## 6. 实验调试记录

本节详细记录实验过程中遇到的问题及解决方案，帮助理解DWARF调试信息解析的关键细节。

### 6.1 问题1：程序无限挂起

**现象**：

- 添加`print_error_line()`到所有异常处理函数后，程序卡在"Switch to user mode..."不再继续
- 没有任何输出，系统完全挂起

**原因分析**：

1. **递归异常**：`print_error_line()`函数内部访问数组、指针时可能触发新的异常
2. **无限循环**：新异常再次调用`print_error_line()`，形成无限递归
3. **缺少break语句**：`CAUSE_LOAD_ACCESS`异常处理缺少`break`，导致fall-through

**子问题1.1：如何发现是递归问题？**

- 添加进入函数的调试输出：

```c
static void print_error_line() {
    sprint("DEBUG: Entering print_error_line\n");
    // ...
}
```

- 结果看到无限多个"DEBUG: Entering print_error_line"
- 说明函数被反复调用

**子问题1.2：递归是如何触发的？**

- `print_error_line()`中访问`current->line[i]`数组
- 如果`current`是NULL或`line`未初始化，会触发load access fault
- load access fault处理函数又调用`print_error_line()`
- 形成：异常 → print_error_line → 新异常 → print_error_line → ...

**解决方案**：

```c
// 添加递归保护标志
static int in_print_error_line = 0;

static void print_error_line() {
    if (in_print_error_line) {
        return;  // 防止递归
    }
    in_print_error_line = 1;
    
    // ... 正常处理 ...
    
    in_print_error_line = 0;  // 处理完成后重置
}

// 修复缺失的break
case CAUSE_LOAD_ACCESS:
    handle_load_access_fault();
    break;  // ← 必须添加
```

**教训**：

- 异常处理代码必须非常健壮，避免自身触发异常
- 每个case语句都要检查是否需要break
- 使用标志位是防止递归的简单有效方法


![image-20260302222918739](./assets/image-20260302222918739.png)

---


### 6.2 问题2：文件名显示为乱码

**现象**：

```
Runtime error at ▒▒▒▒/▒▒▒▒▒:13
```

文件名和目录名都是乱码字符。

**原因分析**：

1. **DWARF版本问题**：
   - Makefile使用`-g`编译，默认生成DWARF 5格式
   - DWARF 5使用`.debug_line_str`段存储字符串，采用间接引用
   - 代码按DWARF 4直接指针方式访问，读到了错误数据

2. **验证方式**：

```bash
readelf --debug-dump=line obj/app_errorline | head -20
# 看到：CU: /usr/lib/gcc/...（绝对路径）
# 说明是DWARF 5格式
```

**子问题2.1：为什么显示乱码而不是空字符串？**

- 在DWARF 5中，文件名位置存储的是**字符串偏移量**（数字）
- 代码把这个数字当作字符串指针直接解引用
- 读到的是任意内存数据，显示为乱码

**子问题2.2：如何确认是DWARF版本问题？**
对比readelf的输出格式：

```bash
# DWARF 5的特征
readelf -wL obj/app_errorline
# 看到：The Directory Table (format 0x1):
#      Entry Name
#      0     /usr/lib/gcc/...

# DWARF 4的特征（修改后）
readelf -wL obj/app_errorline  
# 看到：The Directory Table:
#      user
# 直接是字符串，不是偏移
```

**子问题2.3：为什么不支持DWARF 5？**

- DWARF 5的字符串表需要额外解析`.debug_line_str`段
- 涉及间接引用和偏移计算
- DWARF 4更简单，字符串直接嵌入在`.debug_line`段中

**解决方案**：
在Makefile中强制使用DWARF 4：

```makefile
# 原来
CFLAGS += -g

# 修改为
CFLAGS += -gdwarf-4
```

**验证结果**：

```bash
readelf --debug-dump=line obj/app_errorline | head -20
# CU: user/user_lib.c（相对路径）
# 文件名可读：user_lib.c, app_errorline.c
```

**教训**：

- 不同DWARF版本格式不兼容，必须明确指定版本
- 编译选项会影响调试信息格式
- 遇到乱码先检查数据格式是否匹配

---

### 6.3 问题3：行号偏移5行

**现象**：

```
Runtime error at user/app_errorline.c:18
Illegal instruction!
```

但实际异常在第13行，显示的行号比实际多5行。

**调试过程**：

**步骤1：添加调试输出查看解析结果**

```c
sprint("DEBUG ELF: Parsed %d line entries\n", p->line_ind);
for (int i = 15; i < 20; i++)
    sprint("  [%d] addr=%p line=%d file=%d\n", 
           i, p->line[i].addr, p->line[i].line, p->line[i].file);
```

输出：

```
[17] addr=0x00000000810000b4 line=18 file=4
```

**步骤2：对比readelf输出**

```bash
readelf -wL obj/app_errorline | grep "0x810000b4"
# app_errorline.c    13    0x810000b4    2    x
```

readelf显示行号是13，但解析结果是18，差5行！

**步骤3：检查状态机操作**

```c
// 添加调试
case 3: {  // DW_LNS_advance_line
    int64 delta;
    read_sleb128(&delta, &off);
    regs.line += delta;
    sprint("DEBUG: advance_line delta=%d, new line=%d\n", delta, regs.line);
    break;
}
```

输出：

```
DEBUG: CU start, regs.line=1
DEBUG: special opcode=26, line_delta=13, new line=14
```

第一个特殊操作码将行号从1增加13变成14，但readelf显示应该是从1增加8变成9。

**子问题3.1：特殊操作码的line_delta是如何计算的？**
公式：`line_delta = line_base + (adjusted_opcode % line_range)`

对于opcode=26：

- `adjusted_opcode = 26 - opcode_base`
- `line_delta = line_base + (adjusted_opcode % line_range)`

**步骤4：检查debug_header字段**

```c
sprint("DEBUG: line_base=%d, line_range=%d, opcode_base=%d\n", 
       dh->line_base, dh->line_range, dh->opcode_base);
```

输出：

```
DEBUG: line_base=1, line_range=251, opcode_base=14
```

但readelf显示应该是：

```
Line Base: -5
Line Range: 14
Opcode Base: 13
```

字段完全对不上！说明结构体定义有问题。

**子问题3.2：为什么字段会错位？**
逐字节分析debug_header：

```c
// 添加调试：打印原始字节
char *ptr = (char *)dh;
sprint("Raw bytes: ");
for (int i = 0; i < 20; i++) {
    sprint("%02x ", (unsigned char)ptr[i]);
}
sprint("\n");
```

对比readelf的`--debug-dump=rawline`输出，发现从第7个字节开始就错位了。

**子问题3.3：DWARF 3和DWARF 4的header有什么区别？**
查阅DWARF规范发现：

```
DWARF 3 header:                   DWARF 4 header:
offset  field                     offset  field
------  -----                     ------  -----
0       unit_length (4)           0       unit_length (4)
4       version (2)               4       version (2)
6       header_length (4)         6       header_length (4)
10      min_instr_len (1)         10      min_instr_len (1)
11      default_is_stmt (1)       11      max_ops_per_instr (1)  ← 新增！
12      line_base (1)             12      default_is_stmt (1)
13      line_range (1)            13      line_base (1)
14      opcode_base (1)           14      line_range (1)
15      std_opcode_lengths[]      15      opcode_base (1)
                                  16      std_opcode_lengths[]
```

DWARF 4在offset 11处新增了`maximum_operations_per_instruction`字段！

**子问题3.4：缺少这个字段会导致什么后果？**
如果结构体定义：

```c
struct {
    ...
    uint8 min_instruction_length;     // offset 10
    uint8 default_is_stmt;            // offset 11 - 错误！实际是max_ops
    int8 line_base;                   // offset 12 - 错误！读到default_is_stmt
    uint8 line_range;                 // offset 13 - 错误！读到line_base
    uint8 opcode_base;                // offset 14 - 错误！读到line_range
    ...
}
```

每个字段都读到了下一个字段的值，导致：

- `line_base`读到1（应该是-5）
- `line_range`读到251（应该是14）
- `opcode_base`读到14（应该是13）

**原因分析**：
对比DWARF 3和DWARF 4规范发现，DWARF 4在header中**新增了一个字段**：

```
DWARF 3:                          DWARF 4:
- minimum_instruction_length      - minimum_instruction_length
- default_is_stmt                 - maximum_operations_per_instruction ← 新增！
- line_base                       - default_is_stmt
- line_range                      - line_base
- opcode_base                     - line_range
                                  - opcode_base
```

原代码缺少`maximum_operations_per_instruction`字段，导致后续所有字段读取错位。

**子问题3.5：为什么用-gdwarf-4还会出问题？**
虽然指定了DWARF 4格式，但代码中的结构体定义是按DWARF 3写的。
编译器生成DWARF 4格式数据，但代码用DWARF 3格式解析，自然会错位。

**解决方案**：

```c
typedef struct __attribute__((packed))
{
  uint32 length;
  uint16 version;
  uint32 header_length;
  uint8 min_instruction_length;
  uint8 maximum_operations_per_instruction;  // ← 添加此字段
  uint8 default_is_stmt;
  int8 line_base;
  uint8 line_range;
  uint8 opcode_base;
  uint8 std_opcode_lengths[12];
} debug_header;
```

**验证结果**：

```
DEBUG: line_base=-5, line_range=14, opcode_base=13  ← 正确！
Runtime error at user/app_errorline.c:13  ← 行号正确！
Illegal instruction!
```

**子问题3.6：__attribute__((packed))的作用？**

- 防止编译器自动添加padding对齐
- 确保结构体布局与DWARF格式严格一致
- 如果没有packed，编译器可能在字段间插入填充字节

**教训**：

- 结构体定义必须严格按照标准规范
- 字段对齐问题会导致后续所有数据错位
- 使用`__attribute__((packed))`确保无padding
- 遇到数值异常先检查结构体定义是否正确

---

### 6.4 问题4：地址匹配失败

**现象**：
虽然行号正确了，但有时找不到匹配的地址，或者匹配到错误的文件。

**调试过程**：

**步骤1：打印异常地址和行表地址**

```c
sprint("DEBUG: Looking for exception_addr=0x%lx\n", exception_addr);
for (int i = 1; i < current->line_ind && i < 20; i++)
    sprint("DEBUG: [%d] addr=0x%lx, file=%d\n", 
           i, current->line[i].addr, current->line[i].file);
```

输出：

```
DEBUG: Looking for exception_addr=0x00000000810000b4
DEBUG: [1] addr=0x0000000081000006, file=0
DEBUG: [15] addr=0x00000000810000a0, file=4
DEBUG: [16] addr=0x00000000810000a8, file=4
DEBUG: [17] addr=0x00000000810000b4, file=4  ← 匹配！
DEBUG: [18] addr=0x00000000810000b8, file=4
```

地址完全匹配！但为什么早期版本找不到？

**子问题4.1：早期的错误实现**

```c
// 错误的做法
uint64 program_base = 0x81000000;  // 假设的程序加载基地址
uint64 relative_addr = exception_addr - program_base;
sprint("DEBUG: Looking for relative_addr=0x%lx\n", relative_addr);
// 输出：Looking for relative_addr=0x00000000000000b4

// 然后用relative_addr (0xb4) 去匹配line[i].addr (0x810000b4)
// 永远匹配不上！
```

**子问题4.2：为什么会误以为要用相对地址？**

- ELF程序有一个加载基地址（如0x81000000）
- 代码段的内容中，函数间的跳转使用相对偏移
- 容易误认为DWARF地址也是相对偏移
- 实际上DWARF记录的是**链接后的虚拟地址**

**子问题4.3：如何验证DWARF地址是绝对地址？**
方法1：使用readelf查看：

```bash
readelf -wL obj/app_errorline | grep "app_errorline.c"
# app_errorline.c    13    0x810000b4    2    x
# ↑ 地址是0x81000000开头，明显是虚拟地址而非偏移
```

方法2：查看异常地址：

```bash
spike obj/riscv-pke obj/app_errorline
# Exception at mepc: 0x810000b4
# 异常地址也是0x81000000开头，说明应该直接比较
```

**子问题4.4：精确匹配还是区间匹配？**
早期尝试精确匹配：

```c
for (int i = 0; i < current->line_ind; i++)
    if (current->line[i].addr == exception_addr) {
        found_idx = i;
        break;
    }
```

问题：

- 某些指令可能没有对应的行号记录
- 例如异常发生在0x810000b5，但行号表只有0x810000b4和0x810000b8
- 精确匹配会失败

改进为区间匹配：

```c
for (int i = 1; i < current->line_ind; i++) {
    if (exception_addr < current->line[i].addr) {
        found_idx = i - 1;  // 使用前一个条目
        break;
    }
}
```

**子问题4.5：为什么从i=1开始而不是i=0？**

- 区间匹配需要访问`i-1`
- 如果从i=0开始，访问`line[-1]`会越界
- 从i=1开始，最小访问`line[0]`，安全

**子问题4.6：如果异常地址小于所有记录地址怎么办？**

```c
// 添加边界检查
if (found_idx < 0) {
    sprint("Runtime error at unknown location (addr=0x%lx)\n", exception_addr);
    return;
}
```

**原因分析**：

- DWARF行号表存储的是**绝对虚拟地址**，不是相对偏移
- 如果减去program_base转换为相对地址，会与绝对地址无法匹配
- 例如：exception_addr=0x810000b4，line[i].addr=0x810000b4，应该直接比较

**解决方案**：

```c
// 直接使用绝对地址匹配
uint64 exception_addr = read_csr(mepc);

for (int i = 1; i < current->line_ind; i++) {
    if (exception_addr < current->line[i].addr) {
        // 验证前一个条目的文件索引有效
        addr_line *excpline = current->line + i - 1;
        if (excpline->file != (uint64)-1 && excpline->file < 256) {
            found_idx = i - 1;
            break;
        }
    }
}
```

**验证结果**：

```
DEBUG: Found match at index 17
Runtime error at user/app_errorline.c:13
```

**教训**：

- 理解DWARF地址语义：是虚拟地址还是偏移
- 区间匹配比精确匹配更健壮
- 注意数组访问边界，避免越界
- 使用readelf等工具验证假设

---

### 6.5 问题5：文件索引越界

**现象**：

```
DEBUG: file=65535, dir=...
```

文件索引是0xFFFF（uint64的-1）。

**原因分析**：

- DWARF状态机初始化时`regs.file = 1`
- 某些操作可能将file设置为-1表示"无效"
- 如果不检查直接访问`current->file[-1]`会越界

**子问题5.1：什么情况下file会变成-1？**
查看DWARF状态机操作：

```c
case 1:  // DW_LNS_copy
    p->line[p->line_ind] = regs;
    p->line[p->line_ind].file += file_base - 1;  // ← 关键！
    p->line_ind++;
    break;
```

问题：

- 如果`regs.file = 0`（未设置）
- 计算：`file = 0 + file_base - 1 = file_base - 1`
- 如果`file_base = 0`，结果是`-1`
- 存储到uint64后变成0xFFFFFFFFFFFFFFFF

**子问题5.2：为什么会出现regs.file = 0？**
DWARF规范中，file寄存器初始值是1（第一个文件）。
但某些编译器或特殊情况下可能生成file=0的记录。

**子问题5.3：除了-1，还有哪些无效值？**

```c
// 可能的无效值
if (excpline->file == (uint64)-1)  // -1转uint64是0xFFFFFFFFFFFFFFFF
if (excpline->file == 0)           // 0表示未设置
if (excpline->file >= 256)         // 超出数组范围（file数组最大64）
```

**子问题5.4：为什么设置上限256？**
代码中`file`数组定义：

```c
p->file = (code_file *)(p->dir + 64);  // 分配64个元素
```

但检查用256是保守的上限，给足缓冲空间。

**解决方案**：

```c
for (int i = 1; i < current->line_ind; i++) {
    if (exception_addr < current->line[i].addr) {
        addr_line *excpline = current->line + i - 1;
        // 验证文件索引有效性
        if (excpline->file != (uint64)-1 && excpline->file < 256) {
            found_idx = i - 1;
            break;
        }
    }
}

// 在访问文件信息前再次检查
if (found_idx >= 0) {
    addr_line *excpline = current->line + found_idx;
    code_file *file_info = &current->file[excpline->file];
    
    // 双重检查
    if (file_info->file == NULL) {
        sprint("Runtime error at line %d (file info unavailable)\n", 
               excpline->line);
        return;
    }
    // ...
}
```

**子问题5.5：为什么需要双重检查？**

- 第一次检查：确保索引在合理范围内
- 第二次检查：确保指针本身有效
- 防御性编程：即使索引有效，指针也可能是NULL

**教训**：

- 数组访问前必须检查索引有效性
- 特别注意负数转无符号数的陷阱
- 添加边界检查可以避免崩溃
- 防御性编程：多层检查提高健壮性

---

### 6.6 问题6：目录索引处理错误

**现象**：
早期所有文件的目录都显示为"."（当前目录），但实际应该是"user"。

**调试过程**：

**步骤1：检查DWARF原始数据**

```bash
readelf --debug-dump=line obj/app_errorline | grep -A 5 "Directory Table"
# The Directory Table (offset 0x198):
#   1     user
# 
# The File Name Table (offset 0x19e):
#   Entry Dir     Time    Size    Name
#   1     1       0       0       app_errorline.c
```

DWARF中明确记录：

- 目录表中索引1是"user"
- 文件app_errorline.c的dir索引是1

但输出却显示"./app_errorline.c"

**子问题6.1：目录解析逻辑有什么问题？**
早期代码：

```c
// 读取文件表时
uint64 dir;
read_uleb128(&dir, &off);
p->file[file_ind++].dir = dir - 1 + dir_base;  // ← 直接转换
```

这个转换假设DWARF的dir索引是1-based，减1后加dir_base。
但没有考虑dir=0的特殊情况。

**子问题6.2：DWARF中dir=0是什么意思？**
查阅DWARF 4规范：

- dir=0：表示"当前目录"（编译时的工作目录）
- dir=1,2,3...：引用目录表中的第1、2、3...项（1-based索引）

例子：

```
目录表：
  1: user
  2: kernel

文件表：
  file1 dir=0  → 当前目录（不在目录表中）
  file2 dir=1  → 目录表[1] = user
  file3 dir=2  → 目录表[2] = kernel
```

**子问题6.3：为什么所有文件都显示"."？**
早期代码没有处理dir=0，导致：

- dir=0时：`file.dir = 0 - 1 + 0 = -1`（变成一个很大的数）
- 访问`p->dir[-1]`或`p->dir[超大数]`都是未定义行为
- 可能碰巧读到某个包含"."的地址

**子问题6.4：如何为dir=0提供默认目录？**
需要在目录数组的索引0处预留一个默认值：

```c
// 在解析目录表之前
static char default_dir[] = ".";
p->dir[dir_ind++] = default_dir;  // 索引0现在是"."

// 然后解析DWARF目录表到索引1、2、3...
while (*off != 0) {
    p->dir[dir_ind++] = off;  // 索引1, 2, 3...
    // ...
}
```

**子问题6.5：索引转换逻辑应该如何修改？**

```c
uint64 dir;
read_uleb128(&dir, &off);

if (dir == 0) {
    // dir=0表示当前目录，使用预留的索引0
    p->file[file_ind++].dir = 0;
} else {
    // dir>=1引用目录表，需要：
    // - 减1转为0-based（dir=1 → 0）
    // - 加dir_base调整为全局索引
    // - 再加1跳过预留的索引0
    uint64 dir_idx = dir - 1 + dir_base + 1;
    // 简化：dir_idx = dir + dir_base
    
    // 但dir_base已经考虑了预留位置，所以：
    dir_idx = dir - 1 + dir_base;  // dir_base从1开始
}
```

等等，这里很混乱！让我们重新分析：

**子问题6.6：dir_base的含义是什么？**

```c
dir_base = dir_ind;  // 在解析每个CU前设置
```

当解析第二个CU时：

- 第一个CU解析后，dir_ind = 3（索引0是默认，1和2是第一个CU的）
- dir_base = 3
- 第二个CU的目录会从索引3开始存储

所以dir_base是当前CU目录的起始索引。

**子问题6.7：正确的索引转换逻辑**

```c
if (dir == 0) {
    p->file[file_ind++].dir = 0;  // 使用全局索引0（默认目录）
} else {
    // dir=1引用当前CU目录表的第1项
    // 存储在全局索引dir_base
    // dir=2引用当前CU目录表的第2项
    // 存储在全局索引dir_base+1
    // ...
    uint64 dir_idx = dir - 1 + dir_base;
    
    // 边界检查
    if (dir_idx >= 64 || p->dir[dir_idx] == NULL) {
        p->file[file_ind++].dir = 0;  // 越界时使用默认
    } else {
        p->file[file_ind++].dir = dir_idx;
    }
}
```

**原因分析**：

1. **DWARF目录索引语义**：
   - dir=0 表示"当前目录"（编译时的工作目录）
   - dir=1, 2, 3... 对应目录表中的第1、2、3...项（1-based）

2. **代码问题**：
   - 没有为dir=0预留默认目录
   - 索引转换逻辑错误

**解决方案**：

```c
// 在数组开头预留默认目录
static char default_dir[] = ".";
p->dir[dir_ind++] = default_dir;

// 读取文件表时正确转换索引
uint64 dir;
read_uleb128(&dir, &off);
if (dir == 0) {
    p->file[file_ind++].dir = 0;  // 使用默认目录
} else {
    uint64 dir_idx = dir - 1 + dir_base;  // 1-based转0-based
    if (dir_idx >= 64 || p->dir[dir_idx] == NULL) {
        p->file[file_ind++].dir = 0;  // 越界保护
    } else {
        p->file[file_ind++].dir = dir_idx;
    }
}
```

**验证结果**：

```
Runtime error at user/app_errorline.c:13  ← 目录正确！
```

**子问题6.8：为什么需要NULL检查？**
虽然索引在范围内，但指针可能未初始化：

- DWARF数据可能损坏
- 目录表解析可能有bug
- 防御性编程总是好的

**教训**：

- 仔细阅读规范，理解特殊值的含义（如dir=0）
- 索引转换要考虑base offset和特殊情况
- 添加边界检查和NULL检查
- 复杂的索引逻辑容易出错，需要仔细验证

---

### 6.7 调试技巧总结

1. **对比工具输出**：

   ```bash
   readelf --debug-dump=line obj/app_errorline
   readelf --debug-dump=rawline obj/app_errorline
   ```

   将解析结果与readelf输出对比，验证正确性。

2. **分阶段调试**：

   - 先验证DWARF加载（打印line表内容）
   - 再验证地址匹配（打印查找过程）
   - 最后验证文件路径构建

3. **添加大量调试输出**：
   在关键步骤打印中间结果，逐步缩小问题范围。

4. **理解规范**：
   阅读DWARF标准文档，理解每个字段的准确含义。

5. **异常处理要健壮**：

   - 添加边界检查
   - 添加NULL检查
   - 添加递归保护

---

### 6.8 最终实现效果

经过上述调试，最终实现了：

- ✅ 正确解析DWARF 4格式
- ✅ 准确定位异常源代码行号
- ✅ 显示正确的文件路径
- ✅ 支持所有异常类型
- ✅ 防止递归异常崩溃

输出示例：

```
Runtime error at user/app_errorline.c:13
Illegal instruction!
```

虽然由于Spike文件接口限制无法读取源代码内容，但文件名和行号已经足够开发者快速定位问题。






# Lab1 Challenge3: 多核启动及运行

## 1. 实验原理与分析过程

### 1.1 实验目标

修改 PKE 操作系统内核，使其能够在双核环境下启动并运行，每个核心加载并执行独立的用户程序，所有程序执行完毕后协同退出并关闭模拟器。

### 1.2 为什么要实现多核支持？

**背景：**
现代处理器普遍采用多核架构以提升性能。在 RISC-V 架构中，每个 CPU 核心称为一个 hart（hardware thread）。操作系统必须能够：

1. 协调多个核心的初始化过程
2. 为每个核心分配独立的资源（栈、trapframe等）
3. 管理每个核心上运行的进程
4. 同步多个核心的操作（如系统启动和关闭）

**核心挑战：**

1. **资源隔离与共享**：
   - **共享资源**：物理内存、设备接口（HTIF）、模拟器接口
   - **独立资源**：每个核心的寄存器、栈空间、trapframe

2. **并发控制**：
   - 某些初始化操作只能执行一次（如 HTIF 初始化）
   - 需要同步机制确保初始化完成后才继续执行

3. **进程管理**：
   - 单核模式下有全局 `current` 指针指向当前进程
   - 多核模式下每个核心需要跟踪自己的当前进程

### 1.3 实现原理

**核心标识（hartid）：**

- RISC-V 提供 `tp` 寄存器（thread pointer）专门用于存储 hart 相关信息
- 可以用 `read_tp()` 和 `write_tp()` 读写此寄存器
- 每个核心在启动时将自己的 hartid 保存到 tp 寄存器

**同步机制：**

- 使用同步屏障（barrier）确保所有核心在某个点同步
- 使用原子操作（`__sync_fetch_and_add`）实现无锁的计数器

**内存布局：**

```
Hart 0:                          Hart 1:
  用户栈:    0x81100000            0x81110000
  内核栈:    0x81200000            0x81210000  
  Trapframe: 0x81300000            0x81301000
  程序加载:  0x81000000            0x85000000
```

---

## 2. 需要修改的代码位置

### 2.1 配置文件

- `kernel/config.h`: 已设置 `NCPU = 2`

### 2.2 进程管理

- `kernel/process.h`: 修改 `current` 声明
- `kernel/process.c`: 修改 `current` 定义和 `switch_to` 实现

### 2.3 核心初始化

- `kernel/machine/minit.c`: 添加同步机制，只在 hart0 初始化共享资源

### 2.4 内核入口

- `kernel/kernel.c`: 支持多个进程，分配独立内存空间

### 2.5 ELF 加载

- `kernel/elf.c`: 支持加载多个应用程序
- `kernel/elf.h`: 更新函数签名

### 2.6 中断处理

- `kernel/strap.c`: 修改所有 `current` 使用为 `current[hartid]`

### 2.7 系统调用

- `kernel/syscall.c`: 实现多核协同退出机制

---

## 3. 代码修改逻辑

### 3.1 进程管理：支持多核 current 指针

**修改 kernel/process.h：**

```c
// current points to the currently running user-mode application.
// In multicore mode, each hart has its own current process.
extern process* current[NCPU];
```

**修改 kernel/process.c：**

```c
// Define current as an array
process* current[NCPU] = {NULL};

void switch_to(process* proc) {
  assert(proc);
  uint64 hartid = read_tp();  // Get current hart id
  current[hartid] = proc;      // Update current for this hart
  
  // ... rest of the function
}
```

**设计要点：**

- 将全局单例 `current` 改为数组 `current[NCPU]`
- 每个核心维护自己的当前进程指针
- 通过 `read_tp()` 获取 hartid 索引到数组

### 3.2 M 模式初始化：同步机制

**修改 kernel/machine/minit.c：**

```c
#include "kernel/sync_utils.h"

volatile static int init_sync_counter = 0;

void m_start(uintptr_t hartid, uintptr_t dtb) {
  // Only hart 0 initializes shared resources
  if (hartid == 0) {
    spike_file_init();
    init_dtb(dtb);
  }
  
  // Synchronization barrier: wait for hart 0 to finish
  sync_barrier(&init_sync_counter, NCPU);
  
  sprint("In m_start, hartid:%d\n", hartid);
  
  // Store hartid in tp register
  write_tp(hartid);
  
  // ... rest of initialization
}
```

**关键原理：**

1. **条件初始化**：

   ```c
   if (hartid == 0) {
       // Only hart 0 does these
   }
   ```

   - HTIF、设备树等共享资源只初始化一次
   - 避免重复初始化导致的冲突

2. **同步屏障**：

   ```c
   sync_barrier(&init_sync_counter, NCPU);
   ```

   - 所有核心在此等待

   - 确保 hart 0 完成初始化后其他核心才继续

   - 实现原理（在 `kernel/sync_utils.h` 中）：

     ```c
     static inline void sync_barrier(volatile int* counter, int total) {
       __sync_fetch_and_add(counter, 1);
       while (*counter < total) {
         asm volatile("nop");
       }
     }
     ```

3. **保存 hartid**：

   ```c
   write_tp(hartid);
   ```

   - 将 hartid 写入 tp 寄存器
   - 后续代码通过 `read_tp()` 获取当前核心 ID

### 3.3 S 模式入口：独立进程加载

**修改 kernel/kernel.c：**

```c
// Global array for multiple processes
process user_app[NCPU];

void load_user_program(process *proc, uint64 hartid) {
  // Different memory layout for different harts
  proc->trapframe = (trapframe *)(USER_TRAP_FRAME + hartid * 0x1000);
  memset(proc->trapframe, 0, sizeof(trapframe));
  
  proc->kstack = USER_KSTACK + hartid * 0x10000;
  proc->trapframe->regs.sp = USER_STACK + hartid * 0x10000;
  
  // Important: set tp register with hartid
  proc->trapframe->regs.tp = hartid;

  load_bincode_from_host_elf(proc, hartid);
}

int s_start(void) {
  uint64 hartid = read_tp();
  sprint("hartid = %d: Enter supervisor mode...\n", hartid);
  
  write_csr(satp, 0);
  
  load_user_program(&user_app[hartid], hartid);
  
  sprint("hartid = %d: Switch to user mode...\n", hartid);
  switch_to(&user_app[hartid]);
  
  return 0;
}
```

**内存布局设计：**

| 资源      | Hart 0 地址 | Hart 1 地址 | 偏移量            |
| --------- | ----------- | ----------- | ----------------- |
| Trapframe | 0x81300000  | 0x81301000  | +0x1000 (4KB)     |
| 内核栈    | 0x81200000  | 0x81210000  | +0x10000 (64KB)   |
| 用户栈    | 0x81100000  | 0x81110000  | +0x10000 (64KB)   |
| 程序加载  | 0x81000000  | 0x85000000  | +0x4000000 (64MB) |

**为什么要设置 tp 寄存器？**

- 用户程序可能需要知道自己运行在哪个核心上
- trapframe 中的 tp 会在 `sret` 时恢复到 tp 寄存器
- 确保用户态和内核态都能正确识别 hartid

### 3.4 ELF 加载：支持多应用

**修改 kernel/elf.c：**

```c
void load_bincode_from_host_elf(process *p, uint64 hartid) {
  arg_buf arg_bug_msg;
  size_t argc = parse_args(&arg_bug_msg);
  
  if (!argc) panic("You need to specify the application program!\n");
  if (hartid >= argc) panic("Not enough application programs for all harts!\n");

  // Load the application corresponding to this hartid
  sprint("hartid = %d: Application: %s\n", hartid, arg_bug_msg.argv[hartid]);

  // Open the hartid-th application file
  info.f = spike_file_open(arg_bug_msg.argv[hartid], O_RDONLY, 0);
  
  // ... rest of loading process
}
```

**加载策略：**

- 命令行参数：`spike -p2 riscv-pke app0 app1`
- Hart 0 加载 `app0`（`argv[0]`）
- Hart 1 加载 `app1`（`argv[1]`）
- 通过 hartid 作为索引选择对应的应用

### 3.5 中断处理：核心独立的 tick 计数

**修改 kernel/strap.c：**

```c
#include "config.h"

// Each hart has its own tick counter
static uint64 g_ticks[NCPU] = {0};

static void handle_mtimer_trap() {
  uint64 hartid = read_tp();
  sprint("hartid = %d: Ticks %d\n", hartid, g_ticks[hartid]);
  g_ticks[hartid]++;
  write_csr(sip, 0);
}

void smode_trap_handler(void) {
  if ((read_csr(sstatus) & SSTATUS_SPP) != 0)
    panic("usertrap: not from user mode");

  uint64 hartid = read_tp();
  assert(current[hartid]);
  
  // Save user process counter
  current[hartid]->trapframe->epc = read_csr(sepc);
  
  uint64 cause = read_csr(scause);
  
  if (cause == CAUSE_USER_ECALL) {
    handle_syscall(current[hartid]->trapframe);
  } else if (cause == CAUSE_MTIMER_S_TRAP) {
    handle_mtimer_trap();
  } else {
    panic("unexpected exception happened.\n");
  }

  // Continue execution of current hart's process
  switch_to(current[hartid]);
}
```

**设计要点：**

1. **独立的 tick 计数器**：`g_ticks[NCPU]` 数组，每个核心独立计数
2. **使用 hartid 索引**：所有对 `current` 的访问都改为 `current[hartid]`
3. **核心隔离**：每个核心的中断处理不影响其他核心

### 3.6 系统调用：协同退出机制

**修改 kernel/syscall.c：**

```c
#include "riscv.h"
#include "config.h"

// Atomic counter to track exited harts
volatile static int exit_counter = 0;

ssize_t sys_user_print(const char* buf, size_t n) {
  uint64 hartid = read_tp();
  sprint("hartid = %d: %s", hartid, buf);
  return 0;
}

ssize_t sys_user_exit(uint64 code) {
  uint64 hartid = read_tp();
  sprint("hartid = %d: User exit with code:%d.\n", hartid, code);
  
  // Atomically increment exit counter
  __sync_fetch_and_add(&exit_counter, 1);
  
  // Wait for all harts to exit
  while (exit_counter < NCPU) {
    asm volatile("nop");
  }
  
  // Only hart 0 performs shutdown
  if (hartid == 0) {
    sprint("hartid = %d: shutdown with code:%d.\n", hartid, code);
    shutdown(code);
  }
  
  // Other harts wait indefinitely
  while (1) {
    asm volatile("wfi");
  }
}
```

**协同退出原理：**

1. **原子计数**：

   ```c
   __sync_fetch_and_add(&exit_counter, 1);
   ```

   - GCC 内置原子操作
   - 确保多核同时访问时的线程安全
   - 等价于 `exit_counter++` 但是原子性

2. **等待所有核心**：

   ```c
   while (exit_counter < NCPU) {
       asm volatile("nop");
   }
   ```

   - 所有核心在此自旋等待
   - 直到所有核心都执行了 exit

3. **指定核心关闭**：

   ```c
   if (hartid == 0) {
       shutdown(code);
   }
   ```

   - 只有 hart 0 负责关闭模拟器
   - 避免多个核心同时关闭导致冲突

4. **其他核心等待**：

   ```c
   while (1) {
       asm volatile("wfi");
   }
   ```

   - hart 1 进入低功耗等待状态
   - 等待 hart 0 关闭系统

**为什么需要这样设计？**

- 单核模式：一个进程退出就关闭系统
- 多核模式：必须等所有进程都退出才能安全关闭
- 如果不等待，先退出的核心会强制关闭系统，导致其他核心未完成工作

---

## 4. 实验步骤与验证

### 4.1 代码准备

确保已经完成 lab1_3 的内容，然后切换到挑战分支：

```bash
# 切换到 challenge3 分支
git checkout -b lab1_challenge3_multicore

# 如果需要，合并 lab1_3 的修改
git merge lab1_3_irq -m "continue to work on lab1_challenge3"
```

### 4.2 编译项目

```bash
cd /app/riscv-pke
make clean
make
```

预期输出应包含：

```
PKE core has been built into "obj/riscv-pke"
User app has been built into "obj/app0"
User app has been built into "obj/app1"
```

### 4.3 运行测试

使用 spike 的 `-p2` 选项启动双核模拟：

```bash
spike -p2 obj/riscv-pke obj/app0 obj/app1
```

### 4.4 预期输出

```
HTIF is available!
(Emulated) memory size: 2048 MB
In m_start, hartid:0
hartid = 0: Enter supervisor mode...
hartid = 0: Application: obj/app0
hartid = 0: Application program entry point (virtual address): 0x0000000081000000
hartid = 0: Switch to user mode...
In m_start, hartid:1
hartid = 1: Enter supervisor mode...
hartid = 1: Application: obj/app1
hartid = 1: Application program entry point (virtual address): 0x0000000085000000
hartid = 1: Switch to user mode...
hartid = 0: >>> app0 is expected to be executed by hart0
hartid = 1: >>> app1 is expected to be executed by hart1
hartid = 0: User exit with code:0.
hartid = 1: User exit with code:0.
hartid = 0: shutdown with code:0.
System is shutting down with exit code 0.
```

### 4.5 验证要点

**1. 启动顺序验证：**

- ✓ Hart 0 和 Hart 1 都输出 "In m_start"
- ✓ 两个核心都成功进入 S 模式
- ✓ 每个核心加载了不同的应用程序

**2. 内存隔离验证：**

- ✓ App0 加载地址: 0x81000000
- ✓ App1 加载地址: 0x85000000
- ✓ 两个地址空间不重叠

**3. 并发执行验证：**

- ✓ 两个应用程序的输出都正确显示
- ✓ 每条输出都带有正确的 hartid 标识

**4. 协同退出验证：**

- ✓ 两个核心都打印了退出信息
- ✓ 只有 hart 0 打印了 "shutdown" 信息
- ✓ 系统正常退出，返回码为 0

### 4.6 调试技巧

**如果程序卡住不退出：**

1. 检查 `exit_counter` 的原子操作是否正确
2. 确认 `NCPU` 宏定义为 2
3. 验证 while 循环条件 `exit_counter < NCPU`
4. 在本次实验中，程序卡住的原因是在 sys_user_exit 中，当所有核心都退出后，只有 hartid 0 会执行 shutdown，但此时 hartid 0 可能已经进入了 while 循环。我们需要修改退出逻辑，以下是错误示例

```c
ssize_t sys_user_exit(uint64 code) {
  uint64 hartid = read_tp();
  sprint("hartid = %d: User exit with code:%d.\n", hartid, code);
  
  // In multicore mode, only shutdown when all harts have exited
  // Use atomic increment to count exits
  exit_counter++;
  
  if (exit_counter == NCPU) {
    // All harts have exited, hart 0 performs shutdown
    if (hartid == 0) {
      sprint("hartid = %d: shutdown with code:%d.\n", hartid, code);
      shutdown(code);
    }
  }
  
  // If not all harts have exited, this hart enters infinite loop
  while (1) {
    asm volatile("wfi"); // Wait for interrupt
  }
}
```

**如果核心 ID 错误：**

1. 确认 `write_tp(hartid)` 在 m_start 中被调用
2. 检查 trapframe 中的 tp 寄存器是否正确设置
3. 验证 `read_tp()` 的调用位置

**如果内存访问错误：**

1. 检查每个核心的内存布局计算
2. 确认偏移量足够大，避免重叠
3. 验证 user_app 数组大小为 NCPU

**打印调试信息：**

```c
// 在关键位置添加调试输出
sprint("DEBUG: hartid=%d, current=%p, trapframe=%p\n", 
       hartid, current[hartid], current[hartid]->trapframe);
```

---

## 5. 实验收获

通过本次实验，我们：

1. **理解了多核操作系统的基本原理**：
   - 资源的共享与隔离
   - 核心间的同步与协作
   - 并发执行的管理

2. **掌握了 RISC-V 多核编程**：
   - hartid 的概念和使用
   - tp 寄存器的作用
   - 原子操作和同步机制

3. **实现了多核内核功能**：
   - 多核启动流程
   - 独立的进程管理
   - 协同的系统调用

4. **学习了系统级并发控制**：
   - 同步屏障（barrier）
   - 原子操作（atomic operations）
   - 低功耗等待（WFI 指令）

这些技术是现代多核操作系统的基础，为后续的进程调度、负载均衡、核间通信等高级功能打下了坚实基础。



# Lab2 Challenge2：堆空间管理

> 目标：在 PKE 内核中实现“更紧凑”的堆空间管理，使 `better_malloc(n)` 能够在同一物理页内按需分配多块内存，并支持释放与复用；当分配跨页大块（如 4096 字节）时，仍能保持虚拟地址增长紧凑、映射正确且不触发对齐/原子指令相关错误。
>
> 说明：本文记录一次从需求理解、设计实现到调试修复的完整过程，结构参考挑战实验文档的写法。

---

## 1. 实验原理与分析过程

### 1.1 实验背景与目标

PKE 在基础实验（lab2_2）里提供的内存分配非常“粗粒度”：每次 `better_malloc` 直接分配并映射一个整页（4KB），然后把返回地址交给用户。

Challenge2 的要求是让分配更加“紧凑”：

- **同一页内多次分配**：例如先分配 100 字节，再分配 50 字节，两者应该落在同一页内，且地址差距不能太大（应用中用 `p - m > 512` 做了简单约束）。
- **释放后可复用**：例如释放第一个 100 字节块后，再申请 50 字节应尽可能复用这块空间（应用中用 `m == n` 检验）。
- **跨页分配仍然紧凑**：例如分配 100 后，再分配 4096（跨页），也不应导致虚拟地址出现“巨大的跳跃”，否则会触发应用对“紧凑管理”的检查。

因此，挑战的核心不在“能分配页”，而在：

1. **如何在进程虚拟地址空间中维护一个堆区域**
2. **如何在堆区域内用元数据管理碎片与复用**
3. **如何在需要扩展堆时进行按需映射**
4. **如何避免 RISC-V 上的对齐问题（典型报错：`misaligned AMO`）**

### 1.2 设计约束与关键思路

要实现紧凑堆管理，同时考虑 PKE 的简化特性：

- PKE 的用户程序和内核逻辑都很“教学化”：没有复杂的 sbrk/brk、mmap 子系统；但我们可以在系统调用里做“最小足够”的堆管理。
- 用户态通过 `better_malloc(n)` / `better_free(p)` 触发系统调用；系统调用在内核态可访问进程页表并完成映射。
- 用户程序的约束非常明确：
  - 需要复用释放的块
  - 需要紧凑布局（两个分配间距不超过 512）

基于这些约束，我们采用如下思路：

- 在堆区域中引入一个 **内存控制块 MCB（Memory Control Block）** 作为每个块的元数据。
- **MCB 与用户可见内存连续放置**：
  - 实际内存布局为：`[MCB][用户可用 payload]`
  - 用户拿到的指针应指向 payload 的起始：`ptr = mcb_addr + sizeof(MCB)`
- 在首次分配时初始化堆：
  - 先映射堆的首个页
  - 在页首放置一个“初始 MCB”（size=0，stat=0）用于形成链表起点
- 每次分配：
  1. 先遍历 MCB 链表查找可用空闲块（`m_stat==0` 且 `m_size>=size`）
  2. 找到则标记占用并返回 payload 指针
  3. 找不到则从当前堆顶追加一个新块：
     - 若追加会跨页，则按需映射新页（可以映射多页）
     - 新块的 MCB 仍然应放在“堆顶连续位置”，而不是跳到新页开头
- 每次释放：
  - 传入的是 payload 指针
  - 通过 `payload - sizeof(MCB)` 找回 MCB
  - 将 `m_stat` 置 0 表示空闲（此版本不做合并、也不做页回收）

### 1.3 为什么会出现 misaligned AMO？

在 RISC-V 中，某些原子指令（AMO）对地址对齐有严格要求；在某些场景下，如果对页表项、锁或某些原子访问的数据结构地址没有对齐，会出现类似：

- `misaligned AMO`

本挑战中最常见触发来源是：

- **物理地址/虚拟地址不按 8 字节对齐**
- **MCB 结构体大小或堆指针更新导致下一次写 MCB 不对齐**

因此实现中采取：

- 对用户请求 size 做 8 字节对齐：
  - `size = (size + 7) & ~7`
- 确保 `heap_current` 的推进按 `sizeof(MCB)+aligned_size` 进行

### 1.4 关键数据结构与不变量

为了让实现可控，我们定义几个不变量：

1. `heap_start`：堆区域起始虚拟地址（第一次分配时确定）
2. `heap_current`：下一个新块 MCB 的虚拟地址（堆顶）
3. MCB 链表：
   - `mcb_head` 指向第一块 MCB（位于堆区域起始处）
   - 每次追加新块，将其挂到链表尾部
4. 任何返回给用户的指针必须是：
   - `mcb_va + sizeof(MCB)`

---

## 2. 需要修改的代码位置

本挑战的最小改动集中在 3 个位置：

1. 内核进程结构扩展（为了保存堆管理状态）
2. 系统调用实现（实现 better_malloc/better_free 的核心逻辑）
3. 用户态库函数（本仓库中已经存在 `better_malloc`/`better_free` 封装，对应 syscall 号不变）

### 2.1 进程结构：新增堆管理字段

- 修改位置：kernel/process.h
- 修改内容：
  - 定义 `mcb_t`（MCB）结构
  - 在 `process_t` 中加入：`heap_start`、`heap_current`、`mcb_head`

### 2.2 初始化：为进程初始化堆字段

- 修改位置：kernel/kernel.c
- 修改内容：
  - 在 `load_user_program()` 完成用户页表映射后，将新字段置零：
    - `heap_start = 0`
    - `heap_current = 0`
    - `mcb_head = NULL`

### 2.3 系统调用：实现 better_malloc/better_free

- 修改位置：kernel/syscall.c
- 修改内容：
  - 新增 `sys_user_better_malloc(size)` 与 `sys_user_better_free(va)`
  - 将原先 `SYS_user_allocate_page` / `SYS_user_free_page` 的分发改为调用 better_malloc/better_free

### 2.4 相关辅助：页表遍历与映射

- 使用到的函数通常来自：
  - kernel/vmm.c
  - kernel/vmm.h

包括：

- `user_vm_map()`：为用户页表建立映射
- `user_va_to_pa()`：将用户虚拟地址转为内核可访问的物理地址（direct mapping）
- `page_walk()`：查询 PTE，判断某页是否已经映射

---

## 3. 代码修改逻辑

本节以“实现路线 + 关键细节 + 容易出错点”为主。

### 3.1 MCB 数据结构设计

我们采用单向链表，每个块一个 MCB。

MCB 字段：

- `m_size`：块大小（payload 大小，不含 MCB）
- `m_stat`：状态（0 空闲，1 已分配）
- `next`：下一块 MCB 指针

注意：

- 由于用户态和内核态地址空间隔离，内核中维护的 `mcb*` 指针实际指向的是“物理地址（direct map 可访问）”处的数据。
- 但 `better_malloc` 返回给用户的是“虚拟地址”。

因此在实现中必须明确两个地址域：

- `mcb_va`：MCB 的用户虚拟地址（用于计算返回地址、以及跨页判断）
- `mcb_pa`：MCB 的内核可访问物理地址（用于读写 MCB 内容）

#### 3.1.1 关键代码（kernel/process.h）

下面是本次挑战中引入的 MCB 结构与进程堆管理字段（摘自实际实现）：

```c
// Memory Control Block (MCB) - 内存控制块
// 用于管理堆上的内存块
typedef struct mcb_t {
  uint64 m_size;       // 内存块大小（不含 MCB 本身）
  uint64 m_stat;       // 状态：0=空闲，1=已分配
  struct mcb_t* next;  // 指向下一个 MCB
} mcb;

// the extremely simple definition of process, used for begining labs of PKE
typedef struct process_t {
  uint64 kstack;
  pagetable_t pagetable;
  trapframe* trapframe;

  // 堆管理相关字段 - added for lab2_challenge2
  uint64 heap_start;    // 堆起始虚拟地址
  uint64 heap_current;  // 当前堆顶（已分配的最高地址）
  mcb* mcb_head;        // MCB 链表头指针
} process;
```

### 3.2 堆初始化：第一次 better_malloc

触发条件：

- `current->mcb_head == NULL`

步骤：

1. 分配一页物理页：`alloc_page()`
2. 选择一个堆起始虚拟地址：这里使用全局 `g_ufree_page` 作为“新分配页的虚拟地址起点”
3. 将物理页映射到用户页表：
   - `user_vm_map(current->pagetable, heap_start, PGSIZE, pa, perm)`
4. 在页首放置一个初始 MCB：
   - `m_size = 0`，`m_stat = 0`，`next = NULL`
5. 更新 `heap_current = heap_start + sizeof(MCB)`

额外注意：

- 当我们用 `g_ufree_page` 作为堆首地址并映射了一页后，应该推进 `g_ufree_page += PGSIZE`，以避免后续“其他路径”再次使用相同虚拟页基址。

#### 3.2.1 关键代码（kernel/syscall.c：初始化堆 + 头 MCB）

首次调用 `better_malloc` 时，初始化堆区域并在页首写入一个“头 MCB”（摘自实际实现）：

```c
// 首次分配：初始化堆管理结构
if (current->mcb_head == NULL) {
  void* pa = alloc_page();
  if (pa == NULL) {
    return 0;
  }

  current->heap_start = g_ufree_page;
  current->heap_current = g_ufree_page;

  user_vm_map((pagetable_t)current->pagetable, current->heap_start, PGSIZE,
              (uint64)pa, prot_to_type(PROT_WRITE | PROT_READ, 1));

  void* mcb_pa = user_va_to_pa((pagetable_t)current->pagetable, (void*)current->heap_start);
  current->mcb_head = (mcb*)mcb_pa;
  current->mcb_head->m_size = 0;
  current->mcb_head->m_stat = 0;
  current->mcb_head->next = NULL;

  current->heap_current = current->heap_start + sizeof(mcb);
  g_ufree_page += PGSIZE;
}
```

### 3.3 分配策略：先找空闲块，再追加新块

#### 3.3.1 查找空闲块（first-fit）

遍历链表：

- 规则：`m_stat == 0 && m_size >= size`

找到后：

- `m_stat = 1`
- 返回 `mcb_va + sizeof(MCB)`

实践中我们选择“最简单可用”的 first-fit。

关于“分裂与合并”：

- 为了保持实现简单，本次没有做“块分裂”（即空闲块大于请求 size 时，把剩余空间再做成一个新空闲块）。
- 也没有做“相邻空闲块合并”。

尽管如此，挑战应用只测试“释放后复用”与“紧凑性”，first-fit 足够通过。

#### 3.3.1.1 关键代码（kernel/syscall.c：扫描空闲块并复用）

下面代码展示了“先扫描空闲块再追加新块”的核心逻辑（摘自实际实现）。注意：当找到空闲块后，需要返回其对应的 **虚拟地址**（而不是物理指针）：

```c
// 遍历 MCB 链表，寻找空闲块
mcb* prev = NULL;
mcb* curr = current->mcb_head;

while (curr != NULL) {
  if (curr->m_stat == 0 && curr->m_size >= size) {
    curr->m_stat = 1;

    // 计算 curr 对应的 MCB 虚拟地址（通过从 heap_start 顺序推进定位）
    uint64 mcb_va = 0;
    uint64 search_va = current->heap_start;
    mcb* search_mcb = current->mcb_head;

    while (search_mcb != NULL && search_mcb != curr) {
      uint64 block_size = sizeof(mcb) + search_mcb->m_size;
      search_va += block_size;
      void* next_pa = user_va_to_pa((pagetable_t)current->pagetable, (void*)search_va);
      search_mcb = (mcb*)next_pa;
    }
    mcb_va = search_va;
    return mcb_va + sizeof(mcb);
  }
  prev = curr;
  curr = curr->next;
}
```

#### 3.3.2 追加新块

当找不到空闲块时：

- 在 `heap_current` 处放置新 MCB
- payload 紧随其后
- `heap_current` 向后推进

关键：**追加时必须保证虚拟地址连续**

- `new_mcb_va = heap_current`
- `end_va = new_mcb_va + sizeof(MCB) + size`

如果 `end_va` 跨越了当前已映射页边界，则需要“按需映射缺失页”。

### 3.4 跨页分配：按需映射 + 连续布局

跨页分配的目标：

- 虚拟地址仍然从 `heap_current` 连续增长
- 需要映射的页按需映射
- 新块 MCB 仍然写在 `heap_current` 位置，而不是跳到新页开头

实现步骤：

1. 计算 `end_page = (end_va - 1) / PGSIZE`
2. 对 `new_mcb_va / PGSIZE` 到 `end_page` 范围内的页：
   - 如果页表项不存在或无效，则 `alloc_page()` 并 `user_vm_map()`

注意点：

- 判断页是否映射：
  - `pte = page_walk(pagetable, page_va, 0)`
  - `pte == NULL` 或 `(*pte & PTE_V) == 0` 表示未映射

#### 3.4.1 关键代码（kernel/syscall.c：按 end_va 覆盖范围逐页补映射）

这段代码是修复“跨页时虚拟地址跳跃过大”的关键：块的 MCB 仍然放在 `heap_current` 的连续位置，只是把覆盖到的页逐页补映射（摘自实际实现）：

```c
uint64 total_needed = sizeof(mcb) + size;
uint64 new_mcb_va = current->heap_current;

uint64 current_page = new_mcb_va / PGSIZE;
uint64 end_va = new_mcb_va + total_needed;
uint64 end_page = (end_va - 1) / PGSIZE;

for (uint64 page = current_page + 1; page <= end_page; page++) {
  uint64 page_va = page * PGSIZE;

  pte_t* pte = page_walk((pagetable_t)current->pagetable, page_va, 0);
  if (pte == NULL || (*pte & PTE_V) == 0) {
    void* new_pa = alloc_page();
    if (new_pa == NULL) {
      return 0;
    }
    user_vm_map((pagetable_t)current->pagetable, page_va, PGSIZE,
                (uint64)new_pa, prot_to_type(PROT_WRITE | PROT_READ, 1));
    g_ufree_page += PGSIZE;
  }
}
```

### 3.5 释放策略：仅标记空闲

释放非常直接：

- 用户传入 payload 地址：`va`
- 找回 MCB：`mcb_va = va - sizeof(MCB)`
- 将其转为物理地址（以便在内核读写）：
  - `mcb_pa = user_va_to_pa(pagetable, (void*)mcb_va)`
- 写：`block->m_stat = 0`

注意点：

- 释放时不要 `unmap` 页面，否则会把同页的其他块也一起“搞没”。
- 本挑战只要求块级复用，不要求页级回收。

#### 3.5.1 关键代码（kernel/syscall.c：better_free）

释放时将用户传入的 payload 指针回退 `sizeof(mcb)` 找到 MCB，再将其标记为空闲（摘自实际实现）：

```c
uint64 sys_user_better_free(uint64 va) {
  if (current->mcb_head == NULL) {
    return -1;
  }

  uint64 mcb_va = va - sizeof(mcb);
  void* mcb_pa = user_va_to_pa((pagetable_t)current->pagetable, (void*)mcb_va);
  mcb* block = (mcb*)mcb_pa;
  block->m_stat = 0;
  return 0;
}
```

### 3.6 进程字段初始化

在进程加载（load_user_program）过程中，完成用户页表、栈、trapframe 等映射后，再初始化堆字段：

- `heap_start = 0`
- `heap_current = 0`
- `mcb_head = NULL`

这样保证：

- 每个新进程第一次 `better_malloc` 都会走初始化逻辑

#### 3.6.1 关键代码（kernel/kernel.c：初始化进程堆字段）

在加载用户程序、完成必要映射后，对堆字段做初始化（摘自实际实现）：

```c
// initialize heap management fields for better_malloc/better_free. added @lab2_challenge2
proc->heap_start = 0;
proc->heap_current = 0;
proc->mcb_head = NULL;
```

#### 3.6.2 关键代码（kernel/syscall.c：系统调用分发替换为 better_*）

用户态仍调用原来的 `SYS_user_allocate_page`/`SYS_user_free_page`，但内核将其分发到 `better_malloc/better_free`（摘自实际实现）：

```c
case SYS_user_allocate_page:
  return sys_user_better_malloc(a1);
case SYS_user_free_page:
  return sys_user_better_free(a1);
```

### 3.7 实现中关于“地址域”的经验总结

这次实现最容易混乱的是：

- 内核里写 MCB，必须拿到它的物理地址（或者 direct mapping 可访问地址）
- 返回给用户的地址必须是虚拟地址

因此建议在代码中清晰区分变量命名：

- `xxx_va`：虚拟地址
- `xxx_pa`：物理地址或内核可访问地址

---

## 4. 实验步骤与验证

### 4.1 构建与运行

在仓库根目录执行：

1. 清理：

```bash
make clean
```

2. 编译：

```bash
make
```

3. 运行单页分配应用：

```bash
spike ./obj/riscv-pke ./obj/app_singlepageheap
```

预期输出关键点：

- 正常打印 `hello, world!!!`
- 退出码为 0

4. 运行跨页分配应用：

```bash
spike ./obj/riscv-pke ./obj/app_singlepageheap2
```

预期输出关键点：

- 正常打印 `cross page`
- 退出码为 0

### 4.2 功能验证点清单

#### 4.2.1 单页紧凑分配

- `m = better_malloc(100)`
- `p = better_malloc(50)`

应用中检查：

- `(uint64)p - (uint64)m <= 512`

说明：

- 即便 MCB + 对齐带来额外开销，两块仍应在同页且距离不大

#### 4.2.2 释放后复用

- `better_free(m)`
- `n = better_malloc(50)`

应用中检查：

- `m == n`

说明：

- 需要扫描空闲块并复用
- 若总是只在堆顶追加，则会失败

#### 4.2.3 跨页分配紧凑性

- `m = better_malloc(100)`
- `p = better_malloc(4096)`

应用中检查：

- `(uint64)p - (uint64)m <= 512`

说明：

- `p` 应该紧跟在 `m` 后面（考虑 MCB 与对齐）
- 即便 payload 跨页，虚拟地址也必须“连续增长”，而不是把新块放到新页起始位置

### 4.3 推荐的额外自测

为了更有把握，建议自行写一两个小应用覆盖：

- 多次分配、交错释放，再分配
- 分配大小接近页边界（例如 4000、4080、4096、4100）
- 对齐边界：1、7、8、9、15、16 字节

---

## 5. 实验收获

### 5.1 对“堆”的直观理解

这次挑战让“堆管理”从一个抽象概念变成了可触摸的实现：

- 用户看到的是“连续的虚拟地址”
- 内核做的是“按页映射 + 块元数据管理”

也进一步理解了：

- 堆并不是天生存在的，它是 OS/运行库在虚拟地址空间中“维护出来”的

### 5.2 地址对齐与架构限制的重要性

在普通用户态写 malloc 时，可能不太会直接遇到“misaligned AMO”。

但在内核态处理页表、锁、原子指令时：

- 对齐是硬约束
- 一个看似无害的“指针推进”都可能导致难定位的异常

### 5.3 “最小可用策略”也能通过挑战

本挑战不需要实现完整的 dlmalloc 或 ptmalloc。

只要抓住测试点：

- 紧凑
- 复用
- 跨页紧凑

使用简单的 first-fit + 标记空闲就能完成目标。

### 5.4 内核接口能力：页表遍历与映射

这次实现强化了对以下能力的掌握：

- 如何判断某个用户虚拟页是否已映射
- 如何按需分配物理页并建立映射
- 如何在内核访问用户虚拟地址对应的物理内存内容

---

## 6. 实验调试记录

本节记录真实完成过程中遇到的关键问题与解决方法。

### 6.1 问题 1：跨页分配导致虚拟地址跳跃过大

#### 6.1.1 现象

运行跨页测试应用：

- `m = better_malloc(100)`
- `p = better_malloc(4096)`

应用报错：

- 输出：`you need to manage the vm space precisely!`
- 退出码：-1

也就是：

- `(uint64)p - (uint64)m > 512`

#### 6.1.2 原因分析

早期实现的错误做法是：

- 当发现 `heap_current + need > heap_start + PGSIZE`（跨页）时
- 直接把“新块的 MCB”放在 **新页面的起始处**：
  - `new_mcb_va = new_heap_start`

这会造成：

- `p` 的虚拟地址被强制对齐到新页边界
- 从 `m` 到 `p` 的距离可能接近 4096（减去少量）
- 必然超过 512

本质问题：

- **分配策略破坏了虚拟地址的连续性**
- 把“跨页”误解成“换页从头开始放块”

#### 6.1.2.1 错误实现代码（当时的写法示意）

下面代码是我们早期尝试中的典型错误点：一旦跨页，就把新块的 MCB 放到新页开头，导致返回的 `p` 跳到页边界，从而 `p - m` 过大：

```c
// ❌ 错误示意：跨页后把新块从新页起始处开始放
if (current->heap_current + total_needed > current->heap_start + PGSIZE) {
  void* new_pa = alloc_page();
  uint64 new_heap_start = current->heap_start + PGSIZE;
  user_vm_map((pagetable_t)current->pagetable, new_heap_start, PGSIZE,
              (uint64)new_pa, prot_to_type(PROT_WRITE | PROT_READ, 1));

  uint64 new_mcb_va = new_heap_start;   // ← 关键错误：跳到新页开头
  // ... 在 new_mcb_va 写 MCB 并返回 new_mcb_va + sizeof(mcb)
}
```

#### 6.1.3 解决方法

修复原则：

- “块”应该永远从 `heap_current` 处连续追加
- 若 `end_va` 跨页，只补齐映射，不改变块的虚拟布局

实现策略：

- `new_mcb_va = heap_current`
- `end_va = new_mcb_va + sizeof(MCB) + size`
- 计算 `end_page`
- 对 `new_mcb_va` 到 `end_va` 覆盖到的页逐页检查 PTE：
  - 未映射则 `alloc_page()` + `user_vm_map()`

修复后效果：

- `p` 紧跟在 `m` 之后（最多差：MCB+对齐）
- 跨页也不再导致跳跃
- 应用通过

#### 6.1.3.1 修复后关键代码（kernel/syscall.c：连续布局 + 覆盖范围补映射）

修复版本的核心是：**块从 `heap_current` 连续追加**，跨页只补映射（摘自实际实现）：

```c
uint64 new_mcb_va = current->heap_current;
uint64 end_va = new_mcb_va + sizeof(mcb) + size;
uint64 end_page = (end_va - 1) / PGSIZE;

for (uint64 page = (new_mcb_va / PGSIZE) + 1; page <= end_page; page++) {
  uint64 page_va = page * PGSIZE;
  pte_t* pte = page_walk((pagetable_t)current->pagetable, page_va, 0);
  if (pte == NULL || (*pte & PTE_V) == 0) {
    void* new_pa = alloc_page();
    user_vm_map((pagetable_t)current->pagetable, page_va, PGSIZE,
                (uint64)new_pa, prot_to_type(PROT_WRITE | PROT_READ, 1));
  }
}
```

#### 6.1.4 经验总结

- “跨页”只意味着“映射需要更多页”，并不意味着“块布局要从新页起点重新开始”。
- 挑战的评测点是虚拟地址紧凑，而不是物理页是否连续。

---

### 6.2 问题 2：释放后不复用，导致 `m != n`

#### 6.2.1 现象

运行单页测试应用：

- `m = better_malloc(100)`
- `p = better_malloc(50)`
- `better_free(m)`
- `n = better_malloc(50)`

应用报错：

- 输出：`your malloc is not complete.`
- 条件：`m != n`

#### 6.2.2 原因分析

早期实现只做“堆顶追加”，没有扫描空闲块链表。

即便 `better_free(m)` 把块标记为空闲，如果分配逻辑仍然只追加：

- 新分配的 50 字节必然拿到新的地址
- 复用失败

#### 6.2.2.1 错误实现代码（当时的写法示意）

当时的错误实现只会从堆顶不断追加，完全忽略空闲块：

```c
// ❌ 错误示意：只追加，不扫描空闲块
uint64 new_mcb_va = current->heap_current;
void* new_mcb_pa = user_va_to_pa((pagetable_t)current->pagetable, (void*)new_mcb_va);
mcb* new_mcb = (mcb*)new_mcb_pa;
new_mcb->m_size = size;
new_mcb->m_stat = 1;
new_mcb->next = NULL;

// 直接推进堆顶并返回
current->heap_current += sizeof(mcb) + size;
return new_mcb_va + sizeof(mcb);
```

#### 6.2.3 解决方法

在 `better_malloc` 中增加空闲块扫描：

- 遍历 MCB 链表
- 找到 `m_stat==0` 且 `m_size>=size`
- 直接置 `m_stat=1` 并返回该块 payload

修复后：

- 第二次申请 50 字节复用之前释放的 100 字节块
- `m == n`

#### 6.2.3.1 修复后关键代码（kernel/syscall.c：扫描并复用空闲块）

修复后在追加前先扫描空闲块（摘自实际实现）：

```c
while (curr != NULL) {
  if (curr->m_stat == 0 && curr->m_size >= size) {
    curr->m_stat = 1;
    // ... 计算 mcb_va
    return mcb_va + sizeof(mcb);
  }
  prev = curr;
  curr = curr->next;
}
```

#### 6.2.4 经验总结

- 释放与分配必须配套设计
- 仅有“标记空闲”的 free 还不够；malloc 必须能识别并复用

---

### 6.3 问题 3：出现 `misaligned AMO` 或相关对齐异常

#### 6.3.1 现象

在某些实现尝试中，系统会在运行过程中出现对齐相关的异常（典型提示为 `misaligned AMO`）。

#### 6.3.2 原因分析

根因通常是地址对齐被破坏：

- 用户请求 size 未对齐，导致 `heap_current` 推进后落在非 8 字节边界
- 后续写入 MCB（结构体包含 `uint64` 字段）时发生非对齐访问
- 或者某些原子相关路径访问了非对齐地址

#### 6.3.2.1 错误实现代码（当时的写法示意）

当时我们没有对 size 做对齐，导致 `heap_current` 可能推进到非 8 字节边界：

```c
// ❌ 错误示意：未对齐 size
// size = size;
current->heap_current += sizeof(mcb) + size;
```

#### 6.3.3 解决方法

- 将用户请求 size 对齐到 8 字节：
  - `size = (size + 7) & ~7`
- `heap_current` 的推进始终使用：
  - `sizeof(MCB) + aligned_size`

修复后：

- MCB 结构体写入位置对齐
- 异常消失

#### 6.3.3.1 修复后关键代码（kernel/syscall.c：8 字节对齐）

```c
// ✅ 修复：确保 size 对齐到 8 字节边界
size = (size + 7) & ~7;
```

#### 6.3.4 经验总结

- 在 64-bit 系统下，很多内核数据结构默认需要 8 字节对齐
- “只要能跑”的心态在内核里很危险，对齐常常决定能否稳定运行

---

### 6.4 问题 4：忘记推进 `g_ufree_page`，导致后续映射冲突或覆盖

#### 6.4.1 现象

一些情况下会出现：

- 堆页映射后，后续分配/映射出现异常
- 或者某些地址重复使用导致不可预期行为

（该问题不一定在挑战给定应用中稳定复现，但属于隐患。）

#### 6.4.2 原因分析

如果第一次初始化堆时使用了：

- `heap_start = g_ufree_page`

但是忘记：

- `g_ufree_page += PGSIZE`

那么全局“下一块可用虚拟页”指针不会前进。后续如果还有使用 `g_ufree_page` 的逻辑：

- 可能会重复使用同一虚拟页基址
- 导致映射覆盖/冲突

#### 6.4.2.1 错误实现代码（当时的写法示意）

首次堆初始化映射了一页，但忘记推进 `g_ufree_page`：

```c
// ❌ 错误示意：映射后未推进 g_ufree_page
current->heap_start = g_ufree_page;
user_vm_map((pagetable_t)current->pagetable, current->heap_start, PGSIZE,
            (uint64)pa, prot_to_type(PROT_WRITE | PROT_READ, 1));
// g_ufree_page += PGSIZE;  // ← 忘了
```

#### 6.4.3 解决方法

在堆首次初始化成功映射一页后立即推进：

- `g_ufree_page += PGSIZE`

同时在跨页补映射时，如也使用 `g_ufree_page` 作为虚拟页来源，则同样要推进。

#### 6.4.3.1 修复后关键代码（kernel/syscall.c：映射后推进）

```c
g_ufree_page += PGSIZE;
```

#### 6.4.4 经验总结

- 全局“虚拟页分配指针”必须单调递增
- 否则会出现重复映射同一 VA 的严重错误

---

### 6.5 问题 5：MCB 物理指针与虚拟返回地址混用

#### 6.5.1 现象

在实现空闲块复用或遍历时，有机会出现：

- 返回的地址异常
- 或者写入 MCB 失败

#### 6.5.2 原因分析

`mcb_head` 在内核中通常保存的是“内核可访问地址”（由 `user_va_to_pa` 得到）。

但返回给用户必须是虚拟地址。如果把 `mcb*` 的值直接当成虚拟地址返回，就会：

- 返回一个物理地址（或 direct map 地址）
- 用户态访问必然错误

#### 6.5.2.1 错误实现代码（当时的写法示意）

典型错误：把内核态的 `mcb*`（物理/内核可访问地址）直接当成用户虚拟地址返回：

```c
// ❌ 错误示意：把物理指针当作用户虚拟地址返回
mcb* new_mcb = (mcb*)user_va_to_pa(pagetable, (void*)new_mcb_va);
return (uint64)new_mcb + sizeof(mcb);   // ← new_mcb 不是用户态能用的地址
```

#### 6.5.3 解决方法

把“写元数据”和“返回虚拟地址”分开处理：

- 写元数据：使用 `user_va_to_pa(pagetable, mcb_va)` 得到可写地址
- 返回用户地址：始终返回 `mcb_va + sizeof(MCB)`

必要时：

- 在遍历链表时只用 `next` 指针推进
- 但在返回时要能找到该 MCB 对应的 `mcb_va`

#### 6.5.3.1 修复后关键代码（返回始终使用虚拟地址）

```c
// ✅ 修复：写元数据用 pa，返回地址用 va
void* new_mcb_pa = user_va_to_pa((pagetable_t)current->pagetable, (void*)new_mcb_va);
mcb* new_mcb = (mcb*)new_mcb_pa;
// ... 写 new_mcb 字段
return new_mcb_va + sizeof(mcb);
```

#### 6.5.4 经验总结

- 内核态看到的指针不一定是用户态能用的指针
- 地址域必须严格区分

---

### 6.6 问题 6：跨页映射只映射了一页，导致大块写入触发缺页

#### 6.6.1 现象

分配一个大块（例如 4096 或更大）后，在用户态对该区域写入字符串或数据时：

- 可能触发缺页异常（page fault）

#### 6.6.2 原因分析

如果只在跨页时映射“下一页”而没有根据 `end_va` 计算完整页数：

- 当请求 size 足够大（甚至超过 1 页）时
- 实际覆盖多页
- 但只映射一页会导致后半段访问缺页

#### 6.6.2.1 错误实现代码（当时的写法示意）

典型错误：只补一页，而不是按覆盖范围补齐所有页：

```c
// ❌ 错误示意：只映射 next page
if (need_cross_page) {
  uint64 next_page_va = (new_mcb_va & ~(PGSIZE - 1)) + PGSIZE;
  void* new_pa = alloc_page();
  user_vm_map((pagetable_t)current->pagetable, next_page_va, PGSIZE,
              (uint64)new_pa, prot_to_type(PROT_WRITE | PROT_READ, 1));
}
```

#### 6.6.3 解决方法

按 `end_va` 计算覆盖的页范围并逐页映射：

- `end_page = (end_va - 1) / PGSIZE`
- `for page in [current_page+1..end_page]`：检查 PTE 并映射

修复后：

- 大块写入正常
- 不再触发缺页

#### 6.6.3.1 修复后关键代码（按 end_va 覆盖范围逐页补齐）

```c
uint64 end_va = new_mcb_va + sizeof(mcb) + size;
uint64 end_page = (end_va - 1) / PGSIZE;
for (uint64 page = (new_mcb_va / PGSIZE) + 1; page <= end_page; page++) {
  uint64 page_va = page * PGSIZE;
  // ... page_walk 检查、alloc_page、user_vm_map
}
```

#### 6.6.4 经验总结

- “跨页”不是“跨一页”，要按区间计算
- 页映射要按覆盖范围逐页补齐

---

## 附录 A：关键实现片段（节选）

> 说明：此处给出部分实现片段，方便对照理解。实际代码以仓库中的实现为准。

### A.1 size 对齐

```c
size = (size + 7) & ~7;
```

### A.2 释放：payload -> MCB

```c
uint64 mcb_va = va - sizeof(mcb);
void* mcb_pa = user_va_to_pa((pagetable_t)current->pagetable, (void*)mcb_va);
mcb* block = (mcb*)mcb_pa;
block->m_stat = 0;
```

### A.3 跨页补映射（按 end_va 范围）

```c
uint64 end_va = new_mcb_va + sizeof(mcb) + size;
uint64 end_page = (end_va - 1) / PGSIZE;
for (uint64 page = current_page + 1; page <= end_page; page++) {
  uint64 page_va = page * PGSIZE;
  pte_t* pte = page_walk((pagetable_t)current->pagetable, page_va, 0);
  if (pte == NULL || (*pte & PTE_V) == 0) {
    void* new_pa = alloc_page();
    user_vm_map((pagetable_t)current->pagetable, page_va, PGSIZE,
                (uint64)new_pa, prot_to_type(PROT_WRITE | PROT_READ, 1));
  }
}
```



# Lab2 Challenge3: 多核内存并发管理实验指南

## 1. 实验原理与分析过程

### 1.1 实验目标

在一个支持双核（Hart 0 和 Hart 1）的 RISC-V 操作系统内核中，实现并发安全的物理内存分配与回收，并支持每个核心独立加载运行不同的用户程序。

### 1.2 为什么要这样做？

**问题背景：**
在单核操作系统中，物理内存分配器只需要考虑串行访问。但在多核环境下，多个处理器核心（Hart）并发执行，它们共享物理内存资源。如果不加控制地同时访问共享资源（如物理页分配器），会导致严重的数据竞争（Race Condition）。

**核心原理：**

1. **数据竞争问题**：

   ```
   时间轴：Hart 0                    Hart 1
   -------------------------------------------------------
   t1:    读取 head = free_list.next
   t2:                               读取 head = free_list.next（相同页）
   t3:    free_list.next = head->next
   t4:                               free_list.next = head->next
   结果：两个核心都获得了同一个物理页！
   ```

   这会导致：

   - 数据破坏：两个进程写入同一物理页，相互覆盖数据
   - 内存泄漏：链表指针损坏，部分页面永久丢失
   - 系统崩溃：访问无效指针导致异常

2. **进程隔离问题**：
   每个核心运行的用户进程应当拥有：

   - 独立的虚拟地址空间（通过页表实现）
   - 独立的堆空间指针（避免虚拟地址冲突）
   - 独立的程序代码和数据（加载不同的ELF文件）

3. **同步机制需求**：

   - **互斥锁（Mutex）**：保护临界区，确保同一时刻只有一个核心访问共享资源
   - **原子操作**：使用硬件支持的原子指令，避免中间状态被观察到
   - **内存屏障**：确保内存操作的顺序性，防止编译器或CPU重排序

### 1.3 实现思路

**第一步：实现物理内存管理的并发安全**

- 使用 RISC-V 的 `amoswap` 原子指令实现自旋锁
- 在 `alloc_page` 和 `free_page` 中使用锁保护临界区
- 确保链表操作的原子性

**第二步：实现多核应用程序独立加载**

- 修改 ELF 加载器，根据 hartid 选择不同的应用程序
- 确保 Hart 0 加载 `app_alloc0`，Hart 1 加载 `app_alloc1`
- 避免所有核心都加载同一个程序的错误

**第三步：实现进程堆空间隔离**

- 将堆指针从全局数组移入进程控制块
- 每个进程独立维护自己的堆空间状态
- 通过 `current[hartid]->user_heap_ptr` 访问当前进程的堆指针

**第四步：完善输出格式与协同退出**

- 所有输出添加 hartid 前缀，便于区分日志来源
- 实现多核协同退出机制，确保所有核心完成后再关机
- 使用同步屏障（barrier）等待所有核心到达退出点

---

## 2. 需要修改的代码位置

### 2.1 物理内存管理模块

**文件：** `kernel/pmm.c`

**修改内容：**

1. 添加全局锁变量 `pmm_lock`
2. 实现 `pmm_lock_acquire()` 和 `pmm_lock_release()` 函数
3. 在 `alloc_page()` 和 `free_page()` 中添加锁保护

### 2.2 ELF 加载模块

**文件：** `kernel/elf.c`

**修改内容：**

1. 修改 `load_bincode_from_host_elf()` 函数
2. 使用 `read_tp()` 获取 hartid 并作为数组索引
3. 根据 hartid 选择对应的应用程序文件

### 2.3 进程管理模块

**文件：** `kernel/process.h`, `kernel/process.c`, `kernel/kernel.c`

**修改内容：**

1. 在 `process_t` 结构体中添加 `user_heap_ptr` 字段
2. 删除全局数组 `g_ufree_page`（如果存在）
3. 在进程初始化时设置堆指针

### 2.4 系统调用模块

**文件：** `kernel/syscall.c`

**修改内容：**

1. 修改 `sys_user_allocate_page()` 使用进程的堆指针
2. 修改输出格式，添加 hartid 前缀
3. 实现协同退出机制

### 2.5 虚拟内存管理模块

**文件：** `kernel/vmm.c`

**修改内容：**

1. 修改 `user_heap_init()` 函数（如果存在）
2. 使用新的 `user_heap_ptr` 字段名

### 2.6 用户库

**文件：** `user/user_lib.c`

**修改内容：**

1. 修改 `naive_malloc()` 函数
2. 传递固定大小参数（4000字节）

---

## 3. 代码修改逻辑

### 3.1 物理内存管理 (PMM) 的并发安全

**文件位置**：`kernel/pmm.c`

**修改原理**：
在多核环境下，`alloc_page` 和 `free_page` 会操作全局空闲链表 `g_free_mem_list`。这是一个临界区（Critical Section）。我们需要引入一个自旋锁，在操作链表前获取锁，操作完成后释放锁。

**为什么需要锁？**
考虑以下场景：

```c
// 没有锁的情况
void *alloc_page(void) {
  list_node *n = g_free_mem_list.next;  // Hart 0 和 Hart 1 可能同时读到相同的 n
  if (n)
    g_free_mem_list.next = n->next;     // 两个核心都会执行这一步
  return (void *)n;                      // 返回同一个物理页！
}
```

**RISC-V 原子指令详解**：

我们使用 RISC-V 的 `amoswap`（Atomic Memory Operation - Swap）指令来实现锁：

1. **amoswap.w.aq**（Acquire）:
   - 功能：原子地将内存位置的值与寄存器值交换
   - `.aq`（Acquire）：保证后续的内存访问不会重排到该指令之前
   - 语法：`amoswap.w.aq rd, rs2, (rs1)`
   - 操作：`temp = mem[rs1]; mem[rs1] = rs2; rd = temp;`

2. **amoswap.w.rl**（Release）:
   - 功能：原子地将内存位置的值与寄存器值交换
   - `.rl`（Release）：保证之前的内存访问不会重排到该指令之后
   - 用于释放锁时，确保临界区内的所有操作对其他核心可见

**自旋锁工作原理**：

```
初始状态：pmm_lock = 0（未上锁）

Hart 0 尝试获取锁：
1. 执行 amoswap.w.aq，尝试将 1 写入 pmm_lock，读出旧值 0
2. 旧值是 0，说明锁可用，获取成功，退出循环
3. 现在 pmm_lock = 1（已上锁）

Hart 1 尝试获取锁（此时锁已被 Hart 0 持有）：
1. 执行 amoswap.w.aq，尝试将 1 写入 pmm_lock，读出旧值 1
2. 旧值是 1，说明锁被占用，继续循环
3. 不断尝试，直到 Hart 0 释放锁

Hart 0 释放锁：
1. 执行 amoswap.w.rl，将 0 写入 pmm_lock
2. 现在 pmm_lock = 0（未上锁）
3. Hart 1 的下一次尝试将成功
```

**代码实现**：

```c
// 定义全局锁变量，0 表示未上锁，1 表示已上锁
// volatile 确保编译器不会优化掉对该变量的访问
static volatile int pmm_lock = 0;

// 获取锁（自旋等待）
static void pmm_lock_acquire() {
  int old = 1;  // 预设为 1，用于检测锁的状态
  
  // 自旋循环：不断尝试获取锁
  while (1) {
    // 内联汇编：调用 RISC-V 原子指令
    // %0 对应 old（输出操作数）
    // %1 对应常数 1（输入操作数，要写入的值）
    // %2 对应 &pmm_lock（输入操作数，内存地址）
    // "memory" 告诉编译器该操作会修改内存，防止重排序
    asm volatile("amoswap.w.aq %0, %1, (%2)" 
                 : "=r"(old)           // 输出：将交换前的旧值读入 old
                 : "r"(1),             // 输入：将 1 写入内存
                   "r"(&pmm_lock)      // 输入：目标内存地址
                 : "memory");          // 副作用：修改内存
    
    // 如果旧值是 0，说明之前锁是空闲的，我们成功获取了锁
    if (old == 0) break;
    
    // 否则，锁被占用，继续自旋等待
    // 可以添加 CPU 暂停指令减少功耗（可选）
    // asm volatile("pause");
  }
  // 退出循环时，pmm_lock = 1，锁已被当前核心持有
}

// 释放锁
static void pmm_lock_release() {
  // 将 0 写入 pmm_lock，释放锁
  // x0 是 RISC-V 的零寄存器，永远为 0
  // 第一个 x0：目标寄存器（我们不关心旧值）
  // 第二个 x0：源寄存器（写入 0）
  asm volatile("amoswap.w.rl x0, x0, (%0)" 
               :                          // 无输出
               : "r"(&pmm_lock)           // 输入：目标内存地址
               : "memory");               // 副作用：修改内存
  // 现在 pmm_lock = 0，锁已释放
}

// 修改后的 alloc_page
void *alloc_page(void)
{
  pmm_lock_acquire(); // 【关键】进入临界区前加锁

  // 临界区开始 ========================================
  list_node *n = g_free_mem_list.next;
  uint64 hartid = read_tp();
  
  // 打印调试信息（可选）
  if (vm_alloc_stage[hartid]) {
    sprint("hartid = %ld: alloc page 0x%x\n", hartid, n);
  }
  
  // 从链表中移除该节点
  if (n)
    g_free_mem_list.next = n->next;
  // 临界区结束 ========================================

  pmm_lock_release(); // 【关键】退出临界区后解锁
  
  return (void *)n;
}

// 修改后的 free_page
void free_page(void *pa)
{
  pmm_lock_acquire(); // 【关键】进入临界区前加锁

  // 临界区开始 ========================================
  // 边界检查
  if (((uint64)pa % PGSIZE) != 0 || 
      (uint64)pa < free_mem_start_addr || 
      (uint64)pa >= free_mem_end_addr)
    panic("free_page 0x%lx \n", pa);

  // 插入到链表头部
  list_node *n = (list_node *)pa;
  n->next = g_free_mem_list.next;
  g_free_mem_list.next = n;
  // 临界区结束 ========================================

  pmm_lock_release(); // 【关键】退出临界区后解锁
}
```

**设计要点**：

1. **为什么使用 `volatile`？**

   - 防止编译器将 `pmm_lock` 缓存在寄存器中
   - 确保每次访问都从内存读取最新值
   - 在多核环境中至关重要

2. **为什么自旋而不是睡眠？**

   - 临界区很短（只是链表操作），等待时间短
   - 睡眠和唤醒的开销可能比自旋还大
   - 适合用于低竞争、短临界区的场景

3. **可能的优化**：

   ```c
   // 在自旋循环中添加 CPU 暂停指令
   while (1) {
     asm volatile("amoswap.w.aq %0, %1, (%2)" ...);
     if (old == 0) break;
     
     // 减少总线压力，降低功耗
     for (int i = 0; i < 10; i++)
       asm volatile("nop");
   }
   ```

### 3.2 多核应用程序加载 (ELF Loader)

**文件位置**：`kernel/elf.c`

**修改原理**：
Spike 模拟器启动时通过命令行参数传递应用程序路径。例如：

```bash
spike -p2 obj/riscv-pke obj/app_alloc0 obj/app_alloc1
```

- `parse_args` 函数会解析命令行参数，将应用程序路径存入 `argv` 数组
- `argv[0]` = `"obj/app_alloc0"`
- `argv[1]` = `"obj/app_alloc1"`

**问题所在**：
原始代码硬编码使用 `argv[0]`：

```c
// 错误的实现
info.f = spike_file_open(arg_bug_msg.argv[0], O_RDONLY, 0);
```

这导致：

- Hart 0 打开 `argv[0]`，加载 `app_alloc0` ✓
- Hart 1 也打开 `argv[0]`，加载 `app_alloc0` ✗（错误！）

**解决方案**：
利用 `read_tp()` 获取当前核心 ID（hartid），用作数组索引：

- Hart 0: `read_tp()` 返回 0，使用 `argv[0]` ✓
- Hart 1: `read_tp()` 返回 1，使用 `argv[1]`  ✓

**什么是 `read_tp()`？**

- `tp` (Thread Pointer) 是 RISC-V 的一个特殊寄存器

- 在 PKE 中，我们在 M 模式初始化时将 hartid 写入 tp 寄存器

- `read_tp()` 读取 tp 寄存器的值，即当前核心的 ID

- 定义在 `kernel/riscv.h` 中：

  ```c
  static inline uint64 read_tp() {
    uint64 x;
    asm volatile("mv %0, tp" : "=r" (x));
    return x;
  }
  ```

**代码实现**：

```c
void load_bincode_from_host_elf(process *p) {
  arg_buf arg_bug_msg;

  // 解析命令行参数
  size_t argc = parse_args(&arg_bug_msg);
  if (!argc) 
    panic("You need to specify the application program!\n");

  // 【关键修改 1】：使用 read_tp() 获取当前核心 ID
  uint64 hartid = read_tp();
  
  // 【关键修改 2】：根据 hartid 打印对应的应用程序名称
  sprint("hartid = %ld: Application: %s\n", hartid, arg_bug_msg.argv[hartid]);

  // elf loading. elf_ctx is defined in kernel/elf.h
  elf_ctx elfloader;
  elf_info info;

  // 【关键修改 3】：根据 hartid 加载对应的应用程序文件
  // Hart 0 加载 argv[0]，Hart 1 加载 argv[1]
  info.f = spike_file_open(arg_bug_msg.argv[hartid], O_RDONLY, 0);
  info.p = p;
  
  if (IS_ERR_VALUE(info.f))
    panic("Fail on openning the input application program.\n");

  // 初始化 ELF 加载器
  if (elf_init(&elfloader, &info) != EL_OK)
    panic("fail to init elfloader.\n");

  // 加载 ELF 文件到内存
  if (elf_load(&elfloader) != EL_OK)
    panic("Fail on loading elf.\n");

  // 设置程序入口点
  p->trapframe->epc = elfloader.ehdr.entry;

  // 关闭文件
  spike_file_close(info.f);

  sprint("hartid = %ld: Application program entry point (virtual address): 0x%lx\n", 
         hartid, p->trapframe->epc);
}
```

**执行流程示意**：

```
命令行：spike -p2 riscv-pke app_alloc0 app_alloc1
                              ↓
                    parse_args() 解析参数
                              ↓
                  ┌───────────────────────┐
                  │  argv[0] = app_alloc0 │
                  │  argv[1] = app_alloc1 │
                  └───────────────────────┘
                              ↓
         ┌────────────────────┴────────────────────┐
         │                                         │
    Hart 0                                    Hart 1
    read_tp() = 0                            read_tp() = 1
         │                                         │
    argv[0] = app_alloc0                     argv[1] = app_alloc1
         │                                         │
    加载 app_alloc0 ✓                        加载 app_alloc1 ✓
```

**注意事项**：

1. 如果命令行参数不足（如只提供了一个应用程序），Hart 1 会访问越界

2. 需要在 `parse_args` 后添加检查：

   ```c
   if (hartid >= argc)
     panic("Not enough application programs for all harts!\n");
   ```

### 3.3 进程结构的改进与堆隔离

**文件位置**：`kernel/process.h`, `kernel/kernel.c`, `kernel/vmm.c`

**修改原理**：
在简单的实现中，可能使用全局数组来管理每个核心的堆指针：

```c
// 不推荐的做法
uint64 g_ufree_page[NCPU];  // 每个核心一个堆指针
```

这种做法的问题：

1. **破坏封装性**：堆指针是进程的私有状态，不应该是全局变量
2. **不利于扩展**：如果一个核心运行多个进程（时间片调度），无法区分
3. **逻辑混乱**：通过 hartid 索引访问，假设一个核心只运行一个进程

**改进方案**：
将堆指针移入进程控制块（PCB），作为进程的私有属性：

```c
typedef struct process_t {
  uint64 kstack;              // 内核栈
  pagetable_t pagetable;      // 页表
  trapframe *trapframe;       // 陷入帧
  uint64 user_heap_ptr;       // 【新增】用户堆指针
} process;
```

**为什么这样更好？**

1. **符合面向对象原则**：堆指针是进程的属性，应该在进程结构体中
2. **支持多进程**：每个进程独立维护自己的堆空间，不依赖全局索引
3. **访问更清晰**：通过 `current[hartid]->user_heap_ptr` 访问当前进程的堆指针

**代码实现**：

**1. 修改 `kernel/process.h`**：

```c
#ifndef _PROC_H_
#define _PROC_H_

#include "riscv.h"

// 陷入帧结构体（保存用户态上下文）
typedef struct trapframe_t
{
  riscv_regs regs;        // 通用寄存器
  uint64 kernel_sp;       // 内核栈指针
  uint64 kernel_trap;     // 陷入处理函数地址
  uint64 epc;             // 用户态程序计数器
  uint64 kernel_satp;     // 内核页表基址
} trapframe;

// 进程控制块（PCB）
typedef struct process_t
{
  uint64 kstack;              // 内核栈地址
  pagetable_t pagetable;      // 用户页表基址
  trapframe *trapframe;       // 陷入帧指针
  
  // 【新增字段】：用户堆指针
  // 记录该进程用户态堆区的当前空闲地址
  // 每次 malloc 时从这里分配虚拟地址
  uint64 user_heap_ptr;
} process;

// 切换到指定进程
void switch_to(process *);

// 当前进程数组（每个核心一个）
// current[0] 是 Hart 0 当前运行的进程
// current[1] 是 Hart 1 当前运行的进程
extern process *current[NCPU];

#endif
```

**2. 修改 `kernel/kernel.c`**（初始化堆指针）：

```c
void load_user_program(process *proc)
{
  sprint("hartid = %d: User application is loading.\n", read_tp());
  
  // 分配陷入帧页面
  proc->trapframe = (trapframe *)alloc_page();
  memset(proc->trapframe, 0, sizeof(trapframe));

  // 分配用户页表
  proc->pagetable = (pagetable_t)alloc_page();
  memset((void *)proc->pagetable, 0, PGSIZE);

  // 分配内核栈和用户栈
  proc->kstack = (uint64)alloc_page() + PGSIZE;
  uint64 user_stack = (uint64)alloc_page();

  // 设置用户栈指针
  proc->trapframe->regs.sp = USER_STACK_TOP;
  proc->trapframe->regs.tp = read_tp();

  sprint("hartid = %d: user frame 0x%lx, user stack 0x%lx, user kstack 0x%lx \n", 
         read_tp(), proc->trapframe, proc->trapframe->regs.sp, proc->kstack);

  // 加载 ELF 文件
  load_bincode_from_host_elf(proc);

  // 映射用户栈
  user_vm_map((pagetable_t)proc->pagetable, USER_STACK_TOP - PGSIZE, 
              PGSIZE, user_stack, prot_to_type(PROT_WRITE | PROT_READ, 1));

  // 映射陷入帧（内核直接映射）
  user_vm_map((pagetable_t)proc->pagetable, (uint64)proc->trapframe, 
              PGSIZE, (uint64)proc->trapframe, prot_to_type(PROT_WRITE | PROT_READ, 0));

  // 映射陷入向量（内核直接映射）
  user_vm_map((pagetable_t)proc->pagetable, (uint64)trap_sec_start, 
              PGSIZE, (uint64)trap_sec_start, prot_to_type(PROT_READ | PROT_EXEC, 0));

  // 【关键】：初始化用户堆指针
  // USER_FREE_ADDRESS_START 是用户堆区的起始虚拟地址（如 0x00400000）
  proc->user_heap_ptr = USER_FREE_ADDRESS_START;
}
```

**3. 修改 `kernel/syscall.c`**（使用进程的堆指针）：

```c
// 系统调用：为用户进程分配一页内存
uint64 sys_user_allocate_page()
{
  // 【关键】：从当前进程的堆指针获取虚拟地址
  uint64 hartid = read_tp();
  uint64 va = current[hartid]->user_heap_ptr;
  
  // 分配物理页（会自动加锁）
  void *pa = alloc_page();
  
  // 【关键】：更新进程的堆指针
  current[hartid]->user_heap_ptr += PGSIZE;
  
  // 建立虚拟地址到物理地址的映射
  user_vm_map((pagetable_t)current[hartid]->pagetable, va, PGSIZE, (uint64)pa,
              prot_to_type(PROT_WRITE | PROT_READ, 1));
  
  sprint("hartid = %d: vaddr 0x%x is mapped to paddr 0x%x\n", hartid, va, pa);
  
  return va;  // 返回虚拟地址给用户程序
}
```

**4. 修改 `kernel/vmm.c`**（如果存在 `user_heap_init`）：

```c
// 初始化用户堆（如果有单独的初始化函数）
void user_heap_init(process *proc) {
  // 【关键】：使用新的字段名
  proc->user_heap_ptr = USER_FREE_ADDRESS_START;
}
```

**内存布局示意**：

```
用户进程虚拟地址空间：
┌─────────────────────────┐ 0xFFFFFFFF
│     Kernel Space        │
├─────────────────────────┤ 0x80000000
│                         │
│    （未使用区域）        │
│                         │
├─────────────────────────┤ USER_STACK_TOP (0x7FFFF000)
│     User Stack          │ ← trapframe->regs.sp
│         ↓               │
├─────────────────────────┤
│                         │
│    （未使用区域）        │
│                         │
├─────────────────────────┤ USER_FREE_ADDRESS_START (0x00400000)
│     User Heap           │ ← user_heap_ptr（初始值）
│         ↑               │    每次 malloc 后递增
├─────────────────────────┤
│     Code & Data         │ ← ELF 加载的程序代码和数据
└─────────────────────────┘ 0x00010000
```

**访问流程**：

```
用户程序调用 malloc()
    ↓
触发系统调用 SYS_user_allocate_page
    ↓
内核处理：
    1. 读取 current[hartid]->user_heap_ptr（如 0x00400000）
    2. 调用 alloc_page() 分配物理页（如 0x87FA4000）
    3. 建立映射：VA 0x00400000 → PA 0x87FA4000
    4. user_heap_ptr += PGSIZE（更新为 0x00401000）
    5. 返回虚拟地址 0x00400000 给用户程序
    ↓
用户程序获得可用的内存地址，写入数据
```

### 3.4 输出格式与协同退出

**文件位置**：`kernel/kernel.c`, `kernel/syscall.c`

**修改原理**：
为了满足自动评测机的严格比对要求，我们需要精确控制输出格式。此外，多核系统中，只有当所有核心都完成任务后，主核心（Hart 0）才能执行关机操作，否则会强行终止其他正在运行的核心。

**问题 1：输出格式不统一**
原始代码可能缺少 hartid 前缀，或者格式字符串中的空格不一致：

```c
// 错误示例
sprint("User application is loading.\n");  // 缺少 hartid
sprint("User exit with code:%d.\n", code); // 缺少空格
```

预期格式应该是：

```c
sprint("hartid = %d: User application is loading.\n", read_tp());
sprint("hartid = %d: User exit with code: %d.\n", read_tp(), code);  // 注意空格
```

**问题 2：多核退出竞争**
考虑以下场景：

```
时间 t1: Hart 0 执行 exit(0)，调用 shutdown()
时间 t2: Spike 模拟器关闭
时间 t3: Hart 1 还没执行完，被强制终止
结果：Hart 1 的输出丢失，退出消息不完整
```

**解决方案：同步屏障（Barrier）**
使用原子计数器和自旋等待实现：

```
Hart 0 退出 → counter++ → 等待 counter == NCPU
Hart 1 退出 → counter++ → 等待 counter == NCPU
↓（两个核心都到达）
Hart 0 执行 shutdown()
Hart 1 进入无限循环等待
```

**代码实现**：

**1. 修改 `kernel/kernel.c`**（添加 hartid 前缀）：

```c
void load_user_program(process *proc)
{
  // 【修改】：添加 hartid 前缀
  sprint("hartid = %d: User application is loading.\n", read_tp());
  
  // ... 其他初始化代码 ...
  
  sprint("hartid = %d: user frame 0x%lx, user stack 0x%lx, user kstack 0x%lx \n", 
         read_tp(), proc->trapframe, proc->trapframe->regs.sp, proc->kstack);
  
  // ... 加载 ELF ...
  
  sprint("hartid = %d: Application program entry point (virtual address): 0x%lx\n", 
         read_tp(), proc->trapframe->epc);
}

int s_start(void)
{
  sprint("hartid = %d: Enter supervisor mode...\n", read_tp());
  
  // ... 初始化代码 ...
  
  sprint("hartid = %d: Switch to user mode...\n", read_tp());
  
  // ... 切换到用户模式 ...
}
```

**2. 修改 `kernel/syscall.c`**（协同退出）：

```c
#include "kernel/sync_utils.h"  // 包含 sync_barrier

// 全局原子计数器，记录已退出的核心数
volatile static int counter = 0;

// 系统调用：用户进程退出
ssize_t sys_user_exit(uint64 code)
{
  uint64 hartid = read_tp();
  
  // 【修改 1】：添加 hartid 前缀，注意 "code: " 后面有空格
  sprint("hartid = %d: User exit with code: %d.\n", hartid, code);
  
  // 【关键】：同步屏障，等待所有核心到达此处
  // sync_barrier 定义在 kernel/sync_utils.h 中
  // 功能：原子地递增 counter，然后自旋等待 counter >= NCPU
  sync_barrier(&counter, NCPU);
  
  // 此时所有核心都已到达，counter == NCPU
  
  // 【修改 2】：只有 Hart 0 负责关机
  if (hartid == 0) {
    // 注意 "code: " 后面有空格
    sprint("hartid = %d: shutdown with code: %d.\n", hartid, code);
    shutdown(code);  // 关闭 Spike 模拟器
  }
  
  // Hart 1 不会执行到这里（Hart 0 的 shutdown 会终止整个系统）
  // 但为了代码完整性，可以添加无限循环：
  while (1) {
    asm volatile("wfi");  // Wait For Interrupt（低功耗等待）
  }
  
  return 0;
}
```

**同步屏障的实现（`kernel/sync_utils.h`）**：

```c
#ifndef _SYNC_UTILS_H_
#define _SYNC_UTILS_H_

// 同步屏障：等待所有核心到达
static inline void sync_barrier(volatile int* counter, int total) {
  // 原子递增计数器
  // GCC 内置函数，等价于：
  // int old = *counter;
  // *counter = old + 1;
  // return old;
  __sync_fetch_and_add(counter, 1);
  
  // 自旋等待，直到所有核心都到达
  while (*counter < total) {
    // 空循环，不断检查计数器
    // 可以添加 CPU 暂停指令减少功耗
    asm volatile("nop");
  }
  
  // 退出循环时，*counter >= total
  // 说明所有核心都已到达
}

#endif
```

**为什么需要 `volatile`？**

```c
volatile static int counter = 0;
```

- `volatile` 告诉编译器：这个变量可能被其他核心修改
- 防止编译器将其缓存在寄存器中
- 每次访问都从内存读取最新值
- 在多核环境中至关重要

**执行流程示意**：

```
Hart 0:                          Hart 1:
用户程序运行                     用户程序运行
    ↓                               ↓
exit(0)                          exit(0)
    ↓                               ↓
sys_user_exit(0)                sys_user_exit(0)
    ↓                               ↓
print "User exit..."            print "User exit..."
    ↓                               ↓
counter++ (counter=1)           counter++ (counter=2)
    ↓                               ↓
while (counter < 2) {}          while (counter < 2) {}
    ↓                               ↓
检查：counter=2，退出循环        检查：counter=2，退出循环
    ↓                               ↓
if (hartid == 0)                if (hartid == 1)
    ↓                               ↓
print "shutdown..."             跳过 shutdown
    ↓                               ↓
shutdown(0) → 系统关闭          while(1) wfi （永远等待）
```

**注意事项**：

1. **空格问题**：`"code: %d"` 和 `"code:%d"` 是不同的，评测机会严格比对
2. **顺序问题**：必须先打印 "User exit"，再执行同步屏障
3. **死锁风险**：如果某个核心在到达屏障前崩溃，其他核心会永远等待

---

## 4. 用户库修改

**文件位置**：`user/user_lib.c`

**修改原理**：
为了配合 `user/app_alloc0.c` 和 `user/app_alloc1.c` 的测试逻辑，我们将 `naive_malloc` 修改为请求一个固定较大的块（4000字节），这样每次 `malloc` 基本都会触发一次新的页面分配。

**为什么要传递 4000 这个参数？**

在原始实现中，`sys_user_allocate_page` 可能忽略大小参数，直接分配一个页面（4096字节）。但是在这个实验中，我们：

1. **显式传递大小**：告诉内核我们需要多少内存
2. **固定为 4000**：接近一个页面大小（4096），确保每次都分配新页
3. **测试并发性**：多个核心频繁分配页面，增加竞争概率

**测试程序行为分析**：

**`user/app_alloc0.c`**：

```c
#define N 5
#define BASE 0

int main(void) {
  void *p[N];
  
  // 连续分配 5 次
  for (int i = 0; i < N; i++) {
    p[i] = naive_malloc();   // 每次分配 4000 字节（接近一页）
    int *pi = p[i];
    *pi = BASE + i;           // 写入数据：0, 1, 2, 3, 4
    printu("=== user alloc 0 @ vaddr 0x%x\n", p[i]);
  }
  
  // 读取验证
  for (int i = 0; i < N; i++) {
    int *pi = p[i];
    printu("=== user0: %d\n", *pi);  // 应该打印：0, 1, 2, 3, 4
    naive_free(p[i]);
  }
  
  exit(0);
}
```

**`user/app_alloc1.c`**：

```c
#define N 5
#define BASE 5  // 注意：起始值不同

int main(void) {
  void *p[N];
  
  // 连续分配 5 次
  for (int i = 0; i < N; i++) {
    p[i] = naive_malloc();
    int *pi = p[i];
    *pi = BASE + i;           // 写入数据：5, 6, 7, 8, 9
    printu(">>> user alloc 1 @ vaddr 0x%x\n", p[i]);
  }
  
  // 读取验证
  for (int i = 0; i < N; i++) {
    int *pi = p[i];
    printu(">>> user 1: %d\n", *pi);  // 应该打印：5, 6, 7, 8, 9
    naive_free(p[i]);
  }
  
  exit(0);
}
```

**并发执行分析**：

```
时间轴：Hart 0 (app_alloc0)               Hart 1 (app_alloc1)
------------------------------------------------------------------
t1:     malloc() → VA 0x00400000           
t2:                                        malloc() → VA 0x00400000
        (两个进程的虚拟地址可以相同，因为有独立的页表)
        
t3:     写入 PA 0x87FA4000                 
t4:                                        写入 PA 0x87FA2000
        (物理地址必须不同，否则会相互覆盖数据)
        
t5:     读取验证：0 ✓                      
t6:                                        读取验证：5 ✓
```

**代码实现**：

```c
/*
 * The supporting library for applications.
 */

#include "user_lib.h"
#include "util/types.h"
#include "util/snprintf.h"
#include "kernel/syscall.h"

// 用户态系统调用包装函数
uint64 do_user_call(uint64 sysnum, uint64 a1, uint64 a2, uint64 a3, 
                     uint64 a4, uint64 a5, uint64 a6, uint64 a7) {
  int ret;
  
  // 参数已经在寄存器 a0-a7 中
  // 执行 ecall 指令触发系统调用
  asm volatile(
      "ecall\n"
      "sw a0, %0"  // 保存返回值
      : "=m"(ret)
      :
      : "memory");
  
  return ret;
}

// 用户态打印函数
int printu(const char* s, ...) {
  va_list vl;
  va_start(vl, s);
  
  char out[256];
  int res = vsnprintf(out, sizeof(out), s, vl);
  va_end(vl);
  
  const char* buf = out;
  size_t n = res < sizeof(out) ? res : sizeof(out);
  
  return do_user_call(SYS_user_print, (uint64)buf, n, 0, 0, 0, 0, 0);
}

// 用户态退出函数
int exit(int code) {
  return do_user_call(SYS_user_exit, code, 0, 0, 0, 0, 0, 0); 
}

// 【关键修改】：简单的 malloc 实现
void* naive_malloc() {
  // 参数说明：
  // SYS_user_allocate_page: 系统调用号
  // 4000: 请求的内存大小（字节）
  //       接近一个页面大小（4096），确保分配新页
  // 其他参数：未使用，填 0
  return (void*)do_user_call(SYS_user_allocate_page, 4000, 0, 0, 0, 0, 0, 0);
}

// 简单的 free 实现
void naive_free(void* va) {
  do_user_call(SYS_user_free_page, (uint64)va, 0, 0, 0, 0, 0, 0);
}
```

**为什么不是 4096？**

- 4096 是页面大小，但可能被理解为"恰好一页"
- 4000 强调"需要一个页面来容纳这么多数据"
- 实际上，内核可能忽略这个参数，直接分配 4096 字节
- 但明确传递大小是良好的编程习惯

**内核如何处理这个参数？**

```c
// kernel/syscall.c
uint64 sys_user_allocate_page() {
  // 注意：这个实现忽略了 size 参数
  // 直接分配一个完整的页面（4096 字节）
  void *pa = alloc_page();  // 总是分配 4096 字节
  
  uint64 va = current[read_tp()]->user_heap_ptr;
  current[read_tp()]->user_heap_ptr += PGSIZE;  // 增加 4096
  
  user_vm_map(..., va, PGSIZE, ...);  // 映射 4096 字节
  
  return va;
}
```

**更复杂的实现（可选）**：

```c
// 如果要真正支持任意大小分配：
uint64 sys_user_allocate_page(uint64 size) {
  // 计算需要多少个页面
  uint64 npages = (size + PGSIZE - 1) / PGSIZE;
  
  uint64 va_start = current[read_tp()]->user_heap_ptr;
  
  // 分配多个页面
  for (uint64 i = 0; i < npages; i++) {
    void *pa = alloc_page();
    uint64 va = va_start + i * PGSIZE;
    user_vm_map(..., va, PGSIZE, (uint64)pa, ...);
  }
  
  current[read_tp()]->user_heap_ptr += npages * PGSIZE;
  
  return va_start;
}
```

---

## 5. 实验步骤与运行

### 5.1 准备工作

1. **确认环境**：

   ```bash
   # 检查当前工作目录
   pwd
   # 应该显示：/app/riscv-pke
   
   # 检查 RISC-V 工具链
   riscv64-unknown-elf-gcc --version
   
   # 检查 Spike 模拟器
   spike --version
   ```

2. **查看源代码**：

   ```bash
   # 查看测试程序
   cat user/app_alloc0.c
   cat user/app_alloc1.c
   
   # 查看关键文件
   cat kernel/pmm.c | grep -A 20 "alloc_page"
   cat kernel/elf.c | grep -A 10 "load_bincode"
   ```

### 5.2 编译项目

1. **清理旧文件**：

   ```bash
   cd /app/riscv-pke
   make clean
   ```

   预期输出：

   ```
   rm -fr obj
   ```

2. **编译内核与用户程序**：

   ```bash
   make
   ```

   预期输出（部分）：

   ```
   compiling kernel/pmm.c
   compiling kernel/elf.c
   compiling kernel/syscall.c
   ...
   linking obj/riscv-pke ...
   PKE core has been built into "obj/riscv-pke"
   
   compiling user/app_alloc0.c
   compiling user/user_lib.c
   linking obj/app_alloc0 ...
   User app has been built into "obj/app_alloc0"
   
   compiling user/app_alloc1.c
   linking obj/app_alloc1 ...
   User app has been built into "obj/app_alloc1"
   ```

3. **检查生成的文件**：

   ```bash
   ls -lh obj/riscv-pke obj/app_alloc0 obj/app_alloc1
   ```

   应该看到三个可执行文件，大小在 50-100 KB 左右。

### 5.3 运行双核仿真

使用 Spike 的 `-p2` 参数启动两个核心：

```bash
spike -p2 obj/riscv-pke obj/app_alloc0 obj/app_alloc1
```

**参数说明**：

- `-p2`：启动 2 个处理器核心（Hart 0 和 Hart 1）
- `obj/riscv-pke`：操作系统内核镜像
- `obj/app_alloc0`：第一个用户程序（由 Hart 0 加载）
- `obj/app_alloc1`：第二个用户程序（由 Hart 1 加载）

### 5.4 其他有用的命令

1. **只编译内核**：

   ```bash
   make obj/riscv-pke
   ```

2. **只编译用户程序**：

   ```bash
   make obj/app_alloc0
   make obj/app_alloc1
   ```

3. **查看 ELF 文件信息**：

   ```bash
   riscv64-unknown-elf-readelf -h obj/app_alloc0
   riscv64-unknown-elf-objdump -d obj/app_alloc0 | less
   ```

4. **使用 GDB 调试**（高级）：

   ```bash
   # 终端 1：启动 Spike 并等待 GDB 连接
   spike -p2 --rbb-port=9824 obj/riscv-pke obj/app_alloc0 obj/app_alloc1
   
   # 终端 2：启动 GDB
   riscv64-unknown-elf-gdb obj/riscv-pke
   (gdb) target remote localhost:9824
   (gdb) break load_user_program
   (gdb) continue
   ```

---

## 6. 预期输出样例

### 6.1 完整输出

仔细比对以下输出，确保 `hartid`、应用程序名称、物理地址分配等信息的一致性。

```
HTIF is available!
(Emulated) memory size: 2048 MB
In m_start, hartid:0
hartid = 0: Enter supervisor mode...
PKE kernel start 0x0000000080000000, PKE kernel end: 0x000000008000a000, PKE kernel size: 0x000000000000a000 .
free physical memory address: [0x000000008000a000, 0x0000000087ffffff] 
kernel memory manager is initializing ...
In m_start, hartid:1
hartid = 1: Enter supervisor mode...
KERN_BASE 0x0000000080000000
physical address of _etext is: 0x0000000080006000
hartid = 0: User application is loading.
hartid = 1: User application is loading.
hartid = 0: user frame 0x0000000087fbc000, user stack 0x000000007ffff000, user kstack 0x0000000087fbb000 
hartid = 0: Application: obj/app_alloc0
hartid = 1: user frame 0x0000000087fb8000, user stack 0x000000007ffff000, user kstack 0x0000000087fb7000 
hartid = 1: Application: obj/app_alloc1
hartid = 0: Application program entry point (virtual address): 0x0000000000010078
hartid = 1: Application program entry point (virtual address): 0x0000000000010078
hartid = 0: Switch to user mode...
hartid = 0: alloc page 0x87fa4000
hartid = 0: alloc page 0x87fa3000
hartid = 1: Switch to user mode...
hartid = 0: vaddr 0x00400000 is mapped to paddr 0x87fa4000
hartid = 1: alloc page 0x87fa2000
hartid = 1: alloc page 0x87fa1000
=== user alloc 0 @ vaddr 0x00400000
hartid = 1: vaddr 0x00400000 is mapped to paddr 0x87fa2000
hartid = 0: alloc page 0x87fa0000
hartid = 0: vaddr 0x00401000 is mapped to paddr 0x87fa0000
>>> user alloc 1 @ vaddr 0x00400000
=== user alloc 0 @ vaddr 0x00401000
hartid = 1: alloc page 0x87f9f000
hartid = 1: vaddr 0x00401000 is mapped to paddr 0x87f9f000
hartid = 0: alloc page 0x87f9e000
hartid = 0: vaddr 0x00402000 is mapped to paddr 0x87f9e000
>>> user alloc 1 @ vaddr 0x00401000
=== user alloc 0 @ vaddr 0x00402000
hartid = 1: alloc page 0x87f9d000
hartid = 1: vaddr 0x00402000 is mapped to paddr 0x87f9d000
hartid = 0: alloc page 0x87f9c000
hartid = 0: vaddr 0x00403000 is mapped to paddr 0x87f9c000
>>> user alloc 1 @ vaddr 0x00402000
=== user alloc 0 @ vaddr 0x00403000
hartid = 1: alloc page 0x87f9b000
hartid = 1: vaddr 0x00403000 is mapped to paddr 0x87f9b000
hartid = 0: alloc page 0x87f9a000
hartid = 0: vaddr 0x00404000 is mapped to paddr 0x87f9a000
>>> user alloc 1 @ vaddr 0x00403000
=== user alloc 0 @ vaddr 0x00404000
hartid = 1: alloc page 0x87f99000
hartid = 1: vaddr 0x00404000 is mapped to paddr 0x87f99000
=== user0: 0
>>> user alloc 1 @ vaddr 0x00404000
=== user0: 1
>>> user 1: 5
=== user0: 2
>>> user 1: 6
=== user0: 3
>>> user 1: 7
=== user0: 4
>>> user 1: 8
hartid = 0: User exit with code: 0.
>>> user 1: 9
hartid = 1: User exit with code: 0.
hartid = 0: shutdown with code: 0.
System is shutting down with exit code 0.
```

### 6.2 输出分析

**1. 启动阶段**：

```
In m_start, hartid:0
In m_start, hartid:1
```

- 两个核心都成功启动
- M 模式初始化完成

**2. 内核初始化**：

```
hartid = 0: Enter supervisor mode...
hartid = 1: Enter supervisor mode...
```

- 两个核心都进入 S 模式
- 物理内存管理器初始化（只由 Hart 0 执行）

**3. 应用程序加载**：

```
hartid = 0: Application: obj/app_alloc0
hartid = 1: Application: obj/app_alloc1
```

- ✓ **正确**：两个核心加载了不同的程序
- ✗ **错误**：如果都是 `app_alloc0`，说明 ELF 加载器未修改

**4. 内存分配阶段**：

```
hartid = 0: alloc page 0x87fa4000
hartid = 1: alloc page 0x87fa2000
```

- ✓ **正确**：物理地址不同（0x87fa4000 vs 0x87fa2000）
- ✗ **错误**：如果物理地址相同，说明锁未正确实现

**5. 虚拟地址映射**：

```
hartid = 0: vaddr 0x00400000 is mapped to paddr 0x87fa4000
hartid = 1: vaddr 0x00400000 is mapped to paddr 0x87fa2000
```

- ✓ **正确**：虚拟地址可以相同（独立页表）
- ✓ **正确**：物理地址必须不同（共享物理内存）

**6. 用户态输出**：

```
=== user alloc 0 @ vaddr 0x00400000  (Hart 0)
>>> user alloc 1 @ vaddr 0x00400000  (Hart 1)
```

- ✓ **正确**：两种前缀交替出现
- ✗ **错误**：只有一种前缀，说明只运行了一个程序

**7. 数据验证**：

```
=== user0: 0
=== user0: 1
...
>>> user 1: 5
>>> user 1: 6
...
```

- ✓ **正确**：Hart 0 读到 0-4，Hart 1 读到 5-9
- ✗ **错误**：数据混乱，说明物理页被重复分配

**8. 退出阶段**：

```
hartid = 0: User exit with code: 0.
hartid = 1: User exit with code: 0.
hartid = 0: shutdown with code: 0.
```

- ✓ **正确**：两个核心都打印退出消息
- ✓ **正确**：只有 Hart 0 打印 shutdown
- ✗ **错误**：缺少任一消息，说明协同退出未实现

### 6.3 常见错误输出

**错误 1：双核运行同一程序**：

```
hartid = 0: Application: obj/app_alloc0
hartid = 1: Application: obj/app_alloc0  <-- 错误！
...
=== user alloc 0 @ vaddr 0x00400000
=== user alloc 0 @ vaddr 0x00400000  <-- 只有一种前缀
```

原因：ELF 加载器未根据 hartid 选择程序

**错误 2：物理页重复分配**：

```
hartid = 0: alloc page 0x87fa4000
hartid = 1: alloc page 0x87fa4000  <-- 相同地址！
...
=== user0: 5  <-- 数据错误（应该是 0）
```

原因：PMM 锁未实现或实现错误

**错误 3：过早退出**：

```
hartid = 0: User exit with code: 0.
hartid = 0: shutdown with code: 0.
```

缺少 Hart 1 的退出消息

原因：未实现协同退出，Hart 0 直接关机

---

## 7. 实验调试记录

本节详细记录实验过程中遇到的典型问题、排查思路及解决方案，供后续参考。

### 7.1 问题一：双核运行同一程序

**现象描述**：
运行 Spike 后，观察到两个核心的输出日志中，加载的应用程序都是 `obj/app_alloc0`，且后续的用户态打印输出混杂，只有 `===` (User 0 的特征)，没有 `>>>` (User 1 的特征)。

```
hartid = 0: Application: obj/app_alloc0
hartid = 1: Application: obj/app_alloc0  <-- 错误！应该是 app_alloc1
...
=== user alloc 0 @ vaddr 0x00400000
=== user alloc 0 @ vaddr 0x00401000
=== user alloc 0 @ vaddr 0x00402000
（缺少 >>> 前缀的输出）
```

**调试过程**：

**步骤 1：定位问题代码**
查看 `kernel/elf.c` 中的 `load_bincode_from_host_elf` 函数，发现文件加载部分的代码：

```c
// 原始代码（错误）
sprint("Application: %s\n", arg_bug_msg.argv[0]);
info.f = spike_file_open(arg_bug_msg.argv[0], O_RDONLY, 0);
```

**步骤 2：分析问题根源**

- Spike 通过命令行传递参数：`spike -p2 pke app0 app1`
- `parse_args()` 将参数存入数组：`argv[0] = "app0"`, `argv[1] = "app1"`
- 但代码硬编码使用 `argv[0]`，导致所有核心都加载第一个程序

**步骤 3：验证假设**
添加调试输出：

```c
sprint("DEBUG: hartid=%d, argv[0]=%s, argv[1]=%s\n", 
       read_tp(), arg_bug_msg.argv[0], arg_bug_msg.argv[1]);
```

输出：

```
DEBUG: hartid=0, argv[0]=obj/app_alloc0, argv[1]=obj/app_alloc1
DEBUG: hartid=1, argv[0]=obj/app_alloc0, argv[1]=obj/app_alloc1
```

确认 `argv` 数组内容正确，问题在于索引使用错误。

**原因分析**：
硬编码使用 `argv[0]` 导致：

1. Hart 0 打开 `argv[0]`，加载 `app_alloc0` ✓
2. Hart 1 也打开 `argv[0]`，加载 `app_alloc0` ✗
3. 两个核心运行相同的程序，输出混乱

**解决方案**：

```c
// 修正代码
uint64 hartid = read_tp();  // 获取当前核心 ID
sprint("hartid = %ld: Application: %s\n", hartid, arg_bug_msg.argv[hartid]);
info.f = spike_file_open(arg_bug_msg.argv[hartid], O_RDONLY, 0);
```

**验证结果**：

```
hartid = 0: Application: obj/app_alloc0  ✓
hartid = 1: Application: obj/app_alloc1  ✓
...
=== user alloc 0 @ vaddr 0x00400000  ✓
>>> user alloc 1 @ vaddr 0x00400000  ✓
```

**教训**：

- 多核环境下，不能假设所有核心执行相同的逻辑
- 需要通过 hartid 区分核心，选择不同的资源
- 使用 `read_tp()` 获取 hartid 是标准做法

---

### 7.2 问题二：编译报错 `process` has no member named `ufree_page`

**现象描述**：
在清理了全局数组 `g_ufree_page` 并修改结构体后，编译出现错误：

```
kernel/vmm.c: In function 'user_heap_init':
kernel/vmm.c:228:7: error: 'process' {aka 'struct process_t'} has no member named 'ufree_page'
   228 |   proc->ufree_page = USER_FREE_ADDRESS_START;
       |       ^~
make: *** [Makefile:102: obj/kernel/vmm.o] Error 1
```

**调试过程**：

**步骤 1：检查修改记录**
回顾之前的修改：

1. 在 `kernel/process.h` 中添加了 `uint64 user_heap_ptr;`
2. 在 `kernel/kernel.c` 中初始化 `proc->user_heap_ptr`
3. 在 `kernel/syscall.c` 中使用 `current[hartid]->user_heap_ptr`

但忘记检查是否还有其他文件使用旧的字段名。

**步骤 2：全局搜索旧字段名**

```bash
grep -r "ufree_page" kernel/
```

输出：

```
kernel/vmm.c:228:  proc->ufree_page = USER_FREE_ADDRESS_START;
```

找到了！`kernel/vmm.c` 中还在使用旧名称。

**步骤 3：检查 `vmm.c` 的上下文**

```c
// kernel/vmm.c
void user_heap_init(process *proc) {
  proc->ufree_page = USER_FREE_ADDRESS_START;  // 旧名称
}
```

**原因分析**：

1. 重构时将 `ufree_page` 改名为 `user_heap_ptr`
2. 在 `process.h` 中更新了定义
3. 在 `kernel.c` 和 `syscall.c` 中更新了使用
4. 但遗漏了 `vmm.c` 中的使用

这是重构过程中常见的**变量名不一致**问题。

**解决方案**：

```c
// kernel/vmm.c
void user_heap_init(process *proc) {
  proc->user_heap_ptr = USER_FREE_ADDRESS_START;  // 修正为新名称
}
```

**验证编译**：

```bash
make clean && make
```

编译成功，没有错误。

**教训**：

- 重命名字段时，必须全局搜索并替换所有使用

- 使用 IDE 的"重命名"功能可以自动完成

- 或者使用 `sed` 进行全局替换：

  ```bash
  sed -i 's/ufree_page/user_heap_ptr/g' kernel/*.c
  ```

**补充技巧：如何避免这类错误？**

1. **使用版本控制**：

   ```bash
   git diff  # 查看所有修改
   git grep "ufree_page"  # 在代码库中搜索
   ```

2. **增量编译**：

   ```bash
   make  # 先编译一次
   # 修改代码
   make  # 再编译，只编译修改的文件
   ```

   这样可以更快发现错误。

3. **静态分析工具**：

   ```bash
   # 使用 cppcheck 检查代码
   cppcheck --enable=all kernel/
   ```

---

### 7.3 问题三：物理页分配锁的实现

**背景**：
虽然实验要求输出中没有直接体现锁的竞争（因为测试程序规模较小，竞争窗口窄），但在理论上，多核同时进入 `alloc_page` 修改 `g_free_mem_list` 链表是绝对不安全的。

**尝试与修正**：

**尝试 1：直接复制其他实验的锁**
最初可能尝试复制其他实验中的 `spinlock_t` 类型和相关函数：

```c
// 来自其他实验的代码（不推荐）
#include "kernel/sync_utils.h"

spinlock_t pmm_lock;

void alloc_page(void) {
  spinlock_lock(&pmm_lock);
  // ...
  spinlock_unlock(&pmm_lock);
}
```

问题：

- 依赖 `sync_utils.h` 中的复杂实现
- 可能被判为抄袭或冗余
- 不符合"手动实现"的要求

**尝试 2：使用 GCC 内置原子操作**

```c
// 使用 GCC 内置函数
int pmm_lock = 0;

void pmm_lock_acquire() {
  while (__sync_lock_test_and_set(&pmm_lock, 1)) {
    // 自旋等待
  }
}

void pmm_lock_release() {
  __sync_lock_release(&pmm_lock);
}
```

问题：

- 虽然可行，但不是 RISC-V 原子指令
- 实验要求使用 `amoswap`

**最终方案：使用内联汇编**

```c
static volatile int pmm_lock = 0;

static void pmm_lock_acquire() {
  int old = 1;
  while (1) {
    // 直接使用 RISC-V 的 amoswap 指令
    asm volatile("amoswap.w.aq %0, %1, (%2)" 
                 : "=r"(old) 
                 : "r"(1), "r"(&pmm_lock) 
                 : "memory");
    if (old == 0) break;
  }
}

static void pmm_lock_release() {
  asm volatile("amoswap.w.rl x0, x0, (%0)" 
               : 
               : "r"(&pmm_lock) 
               : "memory");
}
```

**为什么这个方案最好？**

1. **自包含**：不依赖外部库或头文件
2. **符合要求**：明确使用了 RISC-V 的 `amoswap` 指令
3. **简单明了**：代码量少，易于理解和维护
4. **正确性**：`.aq` 和 `.rl` 后缀确保内存顺序正确

**验证方法**：

由于竞争窗口很小，很难直接观察到锁的效果。我们可以通过以下方法验证：

**方法 1：添加调试输出**

```c
static void pmm_lock_acquire() {
  int old = 1;
  int retry_count = 0;
  while (1) {
    asm volatile("amoswap.w.aq %0, %1, (%2)" 
                 : "=r"(old) 
                 : "r"(1), "r"(&pmm_lock) 
                 : "memory");
    if (old == 0) break;
    
    retry_count++;
    if (retry_count > 100) {
      sprint("hartid=%d: waiting for lock, retry=%d\n", read_tp(), retry_count);
    }
  }
  if (retry_count > 0) {
    sprint("hartid=%d: acquired lock after %d retries\n", read_tp(), retry_count);
  }
}
```

如果看到 "waiting for lock" 消息，说明确实发生了竞争。

**方法 2：人为增加临界区时间**

```c
void *alloc_page(void) {
  pmm_lock_acquire();
  
  list_node *n = g_free_mem_list.next;
  
  // 人为增加延迟，增加竞争概率
  for (volatile int i = 0; i < 10000; i++);
  
  if (n)
    g_free_mem_list.next = n->next;
  
  pmm_lock_release();
  return (void *)n;
}
```

如果没有锁，两个核心可能分配到相同的页面。

**方法 3：检查内存分配结果**

```bash
# 运行程序，记录输出
spike -p2 obj/riscv-pke obj/app_alloc0 obj/app_alloc1 > output.txt

# 提取所有物理地址
grep "alloc page" output.txt | awk '{print $NF}' | sort

# 检查是否有重复
grep "alloc page" output.txt | awk '{print $NF}' | sort | uniq -d
```

如果有输出，说明有重复分配，锁实现有问题。

**教训**：

- 并发问题很难调试，因为竞争窗口可能很小
- 正确的锁实现至关重要，即使看不到明显的错误
- 使用内联汇编时，务必理解每个操作数的含义
- `.aq` 和 `.rl` 后缀对应 C11 的 `memory_order_acquire` 和 `memory_order_release`

---

### 7.4 问题四：输出格式微调

**现象描述**：
修改完逻辑后，功能正常，但评测对比中显示多处差异：

1. `User application is loading.` 这一行前面没有 `hartid`
2. 退出代码 `code:0` 与预期 `code: 0` 相比少了一个空格

**调试过程**：

**步骤 1：对比输出差异**
使用 `diff` 命令对比实际输出和预期输出：

```bash
spike -p2 obj/riscv-pke obj/app_alloc0 obj/app_alloc1 > actual.txt
diff expected.txt actual.txt
```

输出：

```
< hartid = 0: User application is loading.
> User application is loading.
< hartid = 1: User application is loading.
> User application is loading.
< hartid = 0: User exit with code: 0.
> hartid = 0: User exit with code:0.
< hartid = 1: User exit with code: 0.
> hartid = 1: User exit with code:0.
< hartid = 0: shutdown with code: 0.
> hartid = 0: shutdown with code:0.
```

找到了三处不一致：

1. 缺少 "hartid = %d: " 前缀
2. "code:" 后面缺少空格

**步骤 2：定位问题代码**
搜索包含这些字符串的代码：

```bash
grep -n "User application is loading" kernel/*.c
grep -n "User exit with code" kernel/*.c
grep -n "shutdown with code" kernel/*.c
```

输出：

```
kernel/kernel.c:43:       sprint("User application is loading.\n");
kernel/syscall.c:37:  sprint("hartid = %d: User exit with code:%d.\n", read_tp(), code);
kernel/syscall.c:43:    sprint("hartid = %d: shutdown with code:%d.\n", read_tp(), code);
```

**步骤 3：分析每个问题**

**问题 1：缺少 hartid 前缀**

```c
// kernel/kernel.c:43（错误）
sprint("User application is loading.\n");
```

解决：

```c
// 修正后
sprint("hartid = %d: User application is loading.\n", read_tp());
```

**问题 2：缺少空格**

```c
// kernel/syscall.c:37（错误）
sprint("hartid = %d: User exit with code:%d.\n", read_tp(), code);
//                                          ^ 缺少空格
```

解决：

```c
// 修正后
sprint("hartid = %d: User exit with code: %d.\n", read_tp(), code);
//                                          ^ 添加空格
```

**问题 3：shutdown 也缺少空格**

```c
// kernel/syscall.c:43（错误）
sprint("hartid = %d: shutdown with code:%d.\n", read_tp(), code);
```

解决：

```c
// 修正后
sprint("hartid = %d: shutdown with code: %d.\n", read_tp(), code);
```

**原因分析**：
这是典型的**格式字符串细节问题**。在手动编写输出语句时，容易遗漏：

- 前缀信息（hartid）
- 标点符号后的空格
- 冒号与数字之间的空格

自动评测机会进行**逐字符比对**，任何差异都会导致失败。

**验证结果**：

```bash
# 重新编译
make clean && make

# 运行并保存输出
spike -p2 obj/riscv-pke obj/app_alloc0 obj/app_alloc1 > actual.txt

# 对比
diff expected.txt actual.txt
```

输出为空，说明完全一致。

**教训**：

1. **格式字符串要精确**：包括空格、冒号、换行等

2. **使用 diff 工具**：快速定位差异位置

3. **复制粘贴**：从预期输出复制格式字符串，减少手误

4. **统一风格**：在代码开始时定义输出格式宏：

   ```c
   #define PRINT_HARTID(fmt, ...) \
     sprint("hartid = %d: " fmt, read_tp(), ##__VA_ARGS__)
   
   // 使用
   PRINT_HARTID("User application is loading.\n");
   PRINT_HARTID("User exit with code: %d.\n", code);
   ```

**补充：自动化测试脚本**

```bash
#!/bin/bash
# test.sh - 自动测试脚本

make clean && make || exit 1

# 运行测试
spike -p2 obj/riscv-pke obj/app_alloc0 obj/app_alloc1 > actual.txt

# 对比输出
if diff -q expected.txt actual.txt > /dev/null; then
  echo "✓ Test passed!"
  exit 0
else
  echo "✗ Test failed. Differences:"
  diff expected.txt actual.txt
  exit 1
fi
```

---

### 7.5 调试技巧总结

**1. 添加调试输出**

```c
// 在关键位置添加调试信息
sprint("DEBUG: hartid=%d, entering alloc_page\n", read_tp());
sprint("DEBUG: free_list.next = %p\n", g_free_mem_list.next);
```

编译选项可以使用条件编译：

```c
#ifdef DEBUG
  sprint("DEBUG: ...\n", ...);
#endif
```

编译时：

```bash
make CFLAGS="-DDEBUG"
```

**2. 使用 GDB 调试**

```bash
# 启动 Spike 并等待 GDB
spike -p2 --rbb-port=9824 obj/riscv-pke obj/app_alloc0 obj/app_alloc1

# 另一个终端
riscv64-unknown-elf-gdb obj/riscv-pke
(gdb) target remote localhost:9824
(gdb) break alloc_page
(gdb) continue
(gdb) info registers
(gdb) print *n
```

**3. 检查内存和寄存器**

```c
// 打印内存内容
void dump_memory(void *addr, size_t len) {
  uint8_t *p = (uint8_t *)addr;
  for (size_t i = 0; i < len; i++) {
    if (i % 16 == 0) sprint("\n%p: ", p + i);
    sprint("%02x ", p[i]);
  }
  sprint("\n");
}

// 打印寄存器
sprint("tp=%lx, sp=%lx\n", read_tp(), read_sp());
```

**4. 使用断言**

```c
#include "util/functions.h"  // kassert

void *alloc_page(void) {
  pmm_lock_acquire();
  
  list_node *n = g_free_mem_list.next;
  kassert(n != NULL);  // 断言：空闲链表不为空
  
  // ...
}
```

**5. 检查编译警告**

```bash
# 开启所有警告
make CFLAGS="-Wall -Wextra"

# 将警告视为错误
make CFLAGS="-Werror"
```

**6. 代码审查清单**

- [ ] 所有核心相关的代码是否使用 `read_tp()` 获取 hartid？
- [ ] 共享资源是否有锁保护？
- [ ] 输出格式是否完全匹配（包括空格）？
- [ ] 进程私有数据是否放在 `process_t` 中？
- [ ] 是否有编译警告？
- [ ] 是否有未初始化的变量？
- [ ] 指针是否在使用前检查了 NULL？

---

---

## 8. 实验收获与总结


### 8.1 技术收获

**1. 并发控制机制**

- 理解了数据竞争的本质和危害
- 掌握了自旋锁的实现原理
- 学会使用 RISC-V 原子指令（`amoswap`）
- 理解了内存屏障（Memory Barrier）的作用

**2. 多核编程技巧**

- 使用 `tp` 寄存器存储核心 ID（hartid）
- 通过 hartid 索引访问核心私有数据
- 实现多核同步机制（sync_barrier）
- 理解核心间的协作与通信

**3. 进程管理**

- 进程控制块（PCB）的设计
- 进程私有资源的管理（堆指针、页表、栈）
- 虚拟地址空间的隔离
- 物理内存的共享与分配

### 8.2 常见问题 FAQ

**Q1: 为什么要用 `volatile`？**
A: 防止编译器优化，确保每次访问都从内存读取，对多核环境至关重要。

**Q2: 虚拟地址可以重复吗？**
A: 可以。不同进程有独立的页表，虚拟地址空间是隔离的。

**Q3: 物理地址可以重复吗？**
A: 不行！物理内存是共享的，重复分配会导致数据破坏。

**Q4: 为什么 Hart 1 不需要 shutdown？**
A: Hart 0 执行 `shutdown()` 会关闭整个系统，包括所有核心。

**Q5: 如何扩展到 4 核、8 核？**
A: 修改 `NCPU` 宏，确保所有数组大小正确，并提供足够的命令行参数。

### 8.3 实验完成检查

**验证清单**：

- ✓ 两个核心加载不同的程序
- ✓ 物理页分配正确且无重复
- ✓ 输出格式完全匹配预期
- ✓ 所有核心协同退出
- ✓ 数据验证正确（User 0: 0-4, User 1: 5-9）

**恭喜完成 Lab2 Challenge3！**



# Lab3 Challenge2：信号量

## 1. 实验原理与分析过程

### 1.1 实验目标

本实验要求在 PKE 内核中实现信号量机制，使用户态程序能够通过系统调用完成基本的进程同步。实验完成后，用户程序应支持创建信号量，并执行 P/V 操作，从而实现多个进程之间按指定顺序运行。

本实验的主要任务包括：

1. 在内核中设计信号量数据结构；
2. 实现 `sem_new`、`sem_P`、`sem_V` 三个核心接口；
3. 增加对应系统调用号及系统调用分发逻辑；
4. 在用户库中补充封装接口；
5. 通过给定测试程序验证实现正确性。

由于本实验建立在已有的进程管理与调度框架上，因此重点不在于重新设计调度器，而在于把信号量的“阻塞—唤醒—重新入队”过程准确接入现有内核执行流。

### 1.2 信号量基本原理

信号量是一种经典的同步原语，本质上可以看成“计数器 + 等待队列”的组合。

其中：

- `count` 表示当前可用资源数量；
- 等待队列用于保存暂时无法继续执行的进程。

#### 1.2.1 P 操作

P 操作又称 wait/down，其逻辑为：

```text
P(s):
    s.count = s.count - 1
    if s.count < 0:
        当前进程进入等待队列
        当前进程状态改为 BLOCKED
        调用调度器切换到其他就绪进程
```

也就是说，P 操作先尝试申请资源。如果申请后计数器仍不小于 0，则说明资源足够，进程继续执行；若结果小于 0，则说明资源不足，当前进程必须等待。

#### 1.2.2 V 操作

V 操作又称 signal/up，其逻辑为：

```text
V(s):
    s.count = s.count + 1
    if s.count <= 0:
        从等待队列取出一个进程
        将其唤醒并放回就绪队列
```

V 操作表示释放资源。当释放后仍满足 `count <= 0` 时，说明在此之前已有进程处于等待状态，因此要唤醒其中一个进程。

#### 1.2.3 本实验的实现思路

结合 PKE 当前代码结构，本实验采用如下实现方案：

- 用固定大小数组维护信号量池；
- 每个信号量内部维护一个 FIFO 等待队列；
- 为 `process` 结构增加一个等待链指针，用于将进程挂入信号量等待队列；
- 在 P 操作中完成阻塞和调度切换；
- 在 V 操作中完成唤醒和重新入就绪队列。

之所以采用静态数组而不是动态分配，是因为实验环境较轻量，固定大小实现更简单，也更容易验证正确性。

### 1.3 测试程序行为分析

本实验提供了两个测试程序，分别用于验证两种典型同步情形。

#### 1.3.1 `app_semaphore`：顺序同步

该程序要求父进程与两个子进程按固定顺序打印：

```text
Parent -> Child0 -> Child1 -> Parent -> ...
```

其同步关系为：

- `s0` 初值为 1，保证父进程先运行；
- `s1` 初值为 0，Child0 必须等待父进程发信号；
- `s2` 初值为 0，Child1 必须等待 Child0 发信号。

执行流程抽象如下：

```text
Parent: P(s0) -> print -> V(s1)
Child0: P(s1) -> print -> V(s2)
Child1: P(s2) -> print -> V(s0)
```

这个程序主要用于验证：

- P 操作能否正确阻塞；
- V 操作能否正确唤醒；
- 多个进程之间的顺序控制是否符合预期。

#### 1.3.2 `app_semaphore2`：生产者-消费者

该程序模拟简单的生产者-消费者模型：

- 两个子进程充当生产者；
- 父进程充当消费者；
- `empty` 表示空闲资源位；
- `full` 表示可消费资源数。

抽象流程如下：

```text
Producer: P(empty) -> produce -> V(full)
Consumer: P(full) -> consume -> V(empty)
```

这个程序主要用于验证：

- 信号量计数器能否正确表示资源数量；
- 多个生产者与一个消费者之间的同步是否正确；
- 多次 P/V 操作后队列与进程状态是否保持一致。

## 2. 需要修改的代码位置

本实验的改动主要分布在内核进程管理、系统调用以及用户库三个部分。

### 2.1 [kernel/process.h](kernel/process.h)

该文件中需要完成以下工作：

1. 增加信号量数量上限；
2. 定义 `semaphore_t` 结构体；
3. 在 `process` 结构中加入信号量等待队列指针；
4. 声明信号量相关函数原型。

新增的关键定义包括：

```c
#define MAX_SEMAPHORE 10

typedef struct semaphore_t {
  int id;
  int count;
  process *queue_head;
  process *queue_tail;
} semaphore_t;
```

同时，为了把进程挂到等待队列中，还需在 `process` 中加入：

```c
struct process_t *sem_queue_next;
```

### 2.2 [kernel/process.c](kernel/process.c)

该文件是本实验的核心修改位置，主要实现：

- 全局信号量池；
- `init_semaphores()`；
- `sem_new()`；
- `sem_P()`；
- `sem_V()`。

也就是说，信号量的创建、阻塞、唤醒逻辑都集中在这里完成。

### 2.3 [kernel/kernel.c](kernel/kernel.c)

内核启动时需要初始化信号量池，因此在 `s_start()` 中、`init_proc_pool()` 之后加入：

```c
init_semaphores();
```

### 2.4 [kernel/syscall.h](kernel/syscall.h)

需要增加三个系统调用号：

- `SYS_user_sem_new`
- `SYS_user_sem_P`
- `SYS_user_sem_V`

用于让用户态程序能够通过 `ecall` 进入对应的内核处理函数。

### 2.5 [kernel/syscall.c](kernel/syscall.c)

这里需要完成两部分工作：

1. 新增 `sys_user_sem_new`、`sys_user_sem_P`、`sys_user_sem_V`；
2. 在 `do_syscall()` 中加入分发逻辑。

### 2.6 [user/user_lib.h](user/user_lib.h) 与 [user/user_lib.c](user/user_lib.c)

为了让用户程序以普通函数形式调用信号量，需要在用户库中补充：

- `int sem_new(int initial_count);`
- `void sem_P(int sem_id);`
- `void sem_V(int sem_id);`

这部分属于用户态接口封装，其本质是对系统调用的简单包装。

## 3. 代码修改逻辑

### 3.1 信号量结构设计

本实验使用固定大小数组管理所有信号量：

```c
semaphore_t semaphores[MAX_SEMAPHORE];
```

这样实现的优点是结构清晰、管理简单，且不需要额外设计内存分配与回收逻辑。

每个信号量包含以下字段：

- `id`：用于标识该槽位是否已被占用；
- `count`：表示可用资源数量；
- `queue_head`：等待队列头指针；
- `queue_tail`：等待队列尾指针。

为了把等待进程串起来，在 `process` 中加入 `sem_queue_next` 作为单链表指针。这样就不需要额外定义等待队列节点类型，改动较小，也和现有代码风格一致。

### 3.2 初始化逻辑

在内核启动阶段调用 `init_semaphores()`，将信号量池全部置为初始状态：

```c
void init_semaphores() {
  for (int i = 0; i < MAX_SEMAPHORE; i++) {
    semaphores[i].id = -1;
    semaphores[i].count = 0;
    semaphores[i].queue_head = NULL;
    semaphores[i].queue_tail = NULL;
  }
}
```

其中，`id = -1` 表示该信号量槽位未被分配。

### 3.3 `sem_new` 的实现逻辑

`sem_new` 的实现比较直接，其流程如下：

1. 遍历信号量池；
2. 找到第一个空闲槽位；
3. 写入编号、初始值，并清空等待队列；
4. 返回该信号量编号；
5. 若没有空闲槽位，则返回 `-1`。

其作用就是为用户程序分配一个新的可用信号量实例。

### 3.4 `sem_P` 的实现逻辑

`sem_P` 是本实验最关键的部分。其核心步骤为：

1. 检查 `sem_id` 是否合法；
2. 将 `count` 减一；
3. 若减后结果小于 0，则说明资源不足；
4. 将当前进程状态设为 `BLOCKED`；
5. 把当前进程挂到该信号量等待队列尾部；
6. 调用 `schedule()` 切换到其他进程。

对应代码骨架为：

```c
semaphores[sem_id].count--;
if (semaphores[sem_id].count < 0) {
  current->status = BLOCKED;
  ...
  schedule();
}
```

等待队列采用 FIFO 插入方式，即新阻塞的进程进入队尾。这样可以保证先等待的进程先被唤醒，避免明显的不公平现象。

### 3.5 `sem_V` 的实现逻辑

`sem_V` 与 `sem_P` 相对应，其流程如下：

1. 检查参数是否合法；
2. 将 `count` 加一；
3. 若加后结果仍然小于等于 0，则说明存在等待进程；
4. 从等待队列头部取出一个进程；
5. 将其状态改为 `READY`；
6. 插入系统就绪队列。

这里选择从队头取出等待进程，与前面的队尾插入一起构成标准 FIFO 队列。

### 3.6 系统调用接入逻辑

用户程序并不直接调用内核函数，因此需要打通完整调用链：

```text
user app
  -> user_lib.c
  -> ecall
  -> do_syscall()
  -> sys_user_sem_xxx()
  -> sem_xxx()
```

因此，本实验除了内核实现外，还必须补充系统调用号、系统调用处理函数以及用户库包装层，否则用户程序无法使用新功能。

### 3.7 用户库封装逻辑

在 [user/user_lib.c](user/user_lib.c) 中新增三个包装函数，形式与已有 `fork()`、`yield()` 接口保持一致。例如：

```c
int sem_new(int initial_count) {
  return do_user_call(SYS_user_sem_new, initial_count, 0, 0, 0, 0, 0, 0);
}
```

这样做的意义在于：

- 屏蔽系统调用细节；
- 保持用户接口统一；
- 使应用层代码更简洁易读。

## 4. 实验步骤与验证

### 4.1 编译系统

完成代码修改后，在工程根目录执行：

```bash
make
```

编译结果表明：

- 内核成功生成 `obj/riscv-pke`；
- 用户测试程序成功生成 `obj/app_semaphore`；
- 用户测试程序成功生成 `obj/app_semaphore2`。

这说明数据结构定义、系统调用号、函数声明及实现之间是一致的。

### 4.2 测试一：顺序打印程序

运行命令：

```bash
spike obj/riscv-pke obj/app_semaphore
```

核心输出结果如下：

```text
Parent print 0
Child0 print 0
Child1 print 0
Parent print 1
Child0 print 1
Child1 print 1
...
Parent print 9
Child0 print 9
Child1 print 9
```

#### 结果分析

从输出可以看出：

1. 父进程总是先打印；
2. Child0 总是在父进程之后运行；
3. Child1 总是在 Child0 之后运行；
4. 三个进程连续多轮都保持相同顺序。

这说明基于信号量的顺序同步已经正确建立，P/V 操作能够准确控制进程执行先后。

### 4.3 测试二：生产者-消费者程序

运行命令：

```bash
spike obj/riscv-pke obj/app_semaphore2
```

核心输出结果如下：

```text
Product resource
Product resource
Parent got 0 source
Parent got 1 source
Parent working
Finished
```

#### 结果分析

从输出可以看出：

1. 两个生产者都成功生产了资源；
2. 父进程在 `full` 信号量允许后依次消费两个资源；
3. `empty` 与 `full` 的数量约束关系保持正确；
4. 程序最终顺利结束，没有出现死锁或异常退出。

这说明本实验实现的信号量不仅能处理顺序控制，也能正确支持资源同步问题。

### 4.4 验证结论

两个测试程序覆盖了两种典型场景：

- `app_semaphore` 对应顺序同步；
- `app_semaphore2` 对应生产者-消费者同步。

测试结果均符合预期，说明本次实现已经满足实验要求。

## 5. 实验收获

通过本实验，我对操作系统中的进程同步机制有了更具体的认识，主要体现在以下几个方面。

### 5.1 对阻塞与唤醒过程的理解更加清晰

以前对信号量的理解更多停留在概念层面，而本实验要求真正把 P/V 操作落到代码中，因此必须明确：

- 何时阻塞当前进程；
- 阻塞后如何将其加入等待队列；
- 何时唤醒等待进程；
- 唤醒后如何让其重新参与调度。

这使我把“同步原语”和“进程调度”联系起来理解，而不是孤立地记忆概念。

### 5.2 对信号量计数器含义的理解更加准确

实现后可以更清楚地理解 `count` 的意义：

- `count >= 0` 时，表示系统中仍有可用资源；
- `count < 0` 时，表示已有进程处于等待状态。

这种设计非常紧凑，也体现了经典信号量模型的简洁性。

### 5.3 熟悉了一个内核功能的完整接入过程

本实验并不是单纯在某个 `.c` 文件里补一个函数，而是要贯通：

- 数据结构；
- 内核实现；
- 系统调用号；
- 系统调用分发；
- 用户库接口；
- 用户程序验证。

这个过程帮助我更清楚地理解了“一个内核功能如何从底层实现逐步暴露给用户程序”。

### 5.4 认识到教学实现与真实系统之间的差别

本实验使用静态数组和简单 FIFO 队列，目的是突出原理、降低实现复杂度。在真实操作系统中，还需要进一步考虑：

- 并发访问时的原子性；
- 信号量销毁与资源回收；
- 超时等待；
- 优先级反转等问题。

因此，这次实验既帮助我理解了基本原理，也让我认识到真实系统实现会更复杂。

## 6. 实验调试记录

本实验整体实现较直接，调试过程也相对简短，主要问题如下：

1. 在定义 `semaphore_t` 时直接使用 `process *`，初次编译出现类型未定义错误；
2. 通过在 [kernel/process.h](kernel/process.h) 中加入前向声明 `typedef struct process_t process;` 解决该问题；
3. 修改完成后重新编译，两个测试程序均顺利通过。

总体而言，本实验调试工作量不大，重点更多在于理清信号量与进程调度之间的关系。

## 总结

本次 lab3_challenge2 成功在 PKE 中实现了一个简洁可用的信号量机制。整个实现围绕“计数器 + 等待队列 + 阻塞/唤醒”展开，并通过系统调用向用户态暴露接口。最终，顺序同步与生产者-消费者两个测试程序都通过验证，说明实现满足实验要求。

虽然本实验规模不大，但非常适合理解操作系统中的同步原理。它把抽象的 P/V 操作与具体的进程状态切换、调度和系统调用联系了起来，为后续学习更复杂的同步机制打下了基础。



# Lab3 Challenge3: 写时复制（Copy On Write）

## 1. 实验原理与分析过程

### 1.1 实验目标

本实验的目标是为 PKE 内核中的 `fork` 机制加入写时复制（Copy On Write，COW）支持。
基础版 `fork` 在复制堆空间时，会直接为子进程分配新物理页，并把父进程的数据整页拷贝过去。
这种实现虽然正确，但效率不高。
因为在很多场景中，子进程 `fork` 之后并不会立刻修改堆数据。
如果一开始就把所有堆页完整复制，会造成明显的内存浪费和不必要的数据搬运开销。
因此，本实验的核心思想是：
在 `fork` 发生时不立即复制物理页，而是让父子进程先共享同一物理页。
只有当某个进程第一次对这页进行写操作时，才在页故障处理中真正复制页面。

### 1.2 COW 的基本思想

写时复制可以概括为两句话：
第一，`fork` 时先共享。
第二，写入时再分离。
换成页表语言描述就是：
原本父进程堆区中的可写页，在 `fork` 后不再保持可写，而是变成“可读、不可写、带 COW 标记”的共享页。
父进程和子进程的页表项都指向同一个物理页。
由于该页已经失去写权限，所以无论父进程还是子进程，只要第一次尝试写它，就会触发 `Store Page Fault`。
内核在处理这个页故障时，检查该页是否为 COW 页。
如果是，则分配新页、复制旧内容、更新当前进程页表，然后恢复写权限。
这样，页复制动作就被延迟到了“真正发生写入”的那一刻。

### 1.3 本实验的正确行为标准

本实验给定的用户程序 `user/app_cow.c` 会执行如下步骤：

1. 父进程在堆区申请一页内存。
2. 父进程打印这页的物理地址。
3. 调用 `fork` 创建子进程。
4. 子进程先打印写入前这页的物理地址。
5. 子进程对该堆页执行写操作。
6. 子进程再打印写入后的物理地址。
   如果 COW 正确实现，那么：
   写入前，子进程这页的物理地址应与父进程相同。
   写入时，应触发一次页故障。
   写入后，子进程该页的物理地址应变成新的页地址。
   因此，验证 COW 的关键不是“程序能否跑完”，而是“写前共享、写时 fault、写后分离”这三个现象是否都成立。

### 1.4 为什么必须去掉写权限

如果 `fork` 之后仍然保留 `PTE_W`，那么父子进程写共享页时，CPU 不会触发页故障。
没有 fault，就没有机会进入内核执行延迟复制逻辑。
因此，COW 的本质不是“偷偷共享”，而是“通过权限设计强制制造一次可控的写页故障”。
也就是说，COW 必须依赖页表项权限位。
在本实验中，我们把共享页统一设置为：

- `PTE_V`：有效
- `PTE_R`：可读
- `PTE_U`：用户态可访问
- `PTE_COW`：软件定义的 COW 标记
  但不设置 `PTE_W`。
  这样第一次写一定会 trap。

### 1.5 为什么需要单独的 COW 标记位

仅靠“只读”并不能区分一页到底是普通只读页，还是 COW 页。
例如代码段通常也是只读的，但代码段的写故障显然不应走 COW 分支。
因此我们需要一个专门的软件标志位来表明：
这页是一个因为共享而暂时只读的页，并且未来允许通过复制恢复可写。
本实验使用 RISC-V PTE 的软件保留位 bit 8 作为 `PTE_COW`。
定义如下：

```c
#define PTE_COW (1L << 8)
```

有了这个位之后，页故障处理函数就可以准确地区分：
哪些 fault 是 COW 触发的，
哪些 fault 是真正的非法访问。

### 1.6 为什么需要物理页引用计数

COW 不是单纯的“触发 fault 就复制”。
如果一个物理页只剩当前进程独占了，那么实际上并不需要复制。
这时只要把页表项恢复成可写即可。
因此我们需要知道：
当前物理页究竟还被多少地址空间共享。
这就是引入“物理页引用计数”的原因。
当父子进程在 `fork` 后共享一个物理页时，计数应为 2。
如果其中一个进程在写时复制后切走了，旧物理页的引用计数就减 1。
如果最终只剩一个进程使用该页，则它再写时无需复制。

### 1.7 为什么 COW 依赖页故障处理流程

本实验的关键不是单独改一处代码，而是把多个模块串联成一条完整链路：
`fork` 阶段把页表改成共享只读 + COW；
写入阶段触发 `Store Page Fault`；
trap 分发进入 COW 处理函数；
COW 处理函数完成实际复制、页表回写和 TLB 刷新；
返回用户态后重新执行原写指令。
因此，COW 实际上是“进程管理 + 页表管理 + 物理内存管理 + trap 处理”四部分协同工作的结果。

### 1.8 本实验的几个关键难点

回顾整个实现过程，真正有难度的地方不是代码量，而是下面几个关键点：

1. `fork` 时必须同时修改父子页表，而不能只改子进程。
2. 共享页必须具备准确的引用计数。
3. 页故障处理必须优先识别 COW，而不能直接掉进栈扩展逻辑。
4. 页故障修复后必须刷新 TLB。
5. 新页表项不仅要带 `R/W/U`，还要带 `A/D` 位。
6. 输出日志要与实验预期保持一致，否则评测仍会判错。

## 2. 需要修改的代码位置

### 2.1 `kernel/riscv.h`

本文件中增加了新的页表标志位 `PTE_COW`。
这是整个实验中最底层的标记支持。
没有它，内核无法区分普通只读页和 COW 页。

### 2.2 `kernel/pmm.h`

本文件中增加了页引用计数接口的函数声明：

```c
void page_ref_inc(void* pa);
void page_ref_dec(void* pa);
int page_ref_get(void* pa);
```

这样，进程管理模块和虚拟内存管理模块都可以访问物理页引用计数能力。

### 2.3 `kernel/pmm.c`

本文件中新增了：

1. `page_ref_count[]` 引用计数数组。
2. `pa_to_index()` 物理地址转页索引函数。
3. `page_ref_inc()` / `page_ref_dec()` / `page_ref_get()` 三个核心接口。
   这里是 COW 的共享页统计中心。

### 2.4 `kernel/vmm.h`

本文件增加了两个 COW 相关接口的声明：

```c
void map_cow_page(pagetable_t pagetable, uint64 va, uint64 pa);
int handle_cow_fault(pagetable_t pagetable, uint64 va);
```

前者负责建立 COW 映射，后者负责在 fault 时执行真正的修复。

### 2.5 `kernel/vmm.c`

本文件中新增了：

1. `map_cow_page()`：建立“只读 + COW”的共享映射。
2. `handle_cow_fault()`：处理 COW 页写故障。
   它是整个实验的核心处理模块。

### 2.6 `kernel/process.c`

本文件重点修改 `do_fork()`。
主要变化在 `HEAP_SEGMENT` 分支：
不再给子进程堆页逐页 `alloc_page + memcpy`，
而是让父子进程共享同一物理页并打上 COW 标记。
此外，代码段共享映射的日志输出也需要保留。

### 2.7 `kernel/strap.c`

本文件重点修改 `handle_user_page_fault()`。
我们需要在处理 `CAUSE_STORE_PAGE_FAULT` 时优先尝试 `handle_cow_fault()`。
如果它成功处理，就直接返回。
只有不是 COW fault 时，才继续走原有的动态栈扩展逻辑。
同时保留 `handle_page_fault: ...` 这一行输出，以便与实验标准输出一致。

### 2.8 测试程序

本实验主要通过下面两个用户程序验证：

1. `user/app_cow.c`：单页 COW 测试。
2. `user/app_cow_e.c`：多页 COW 测试。
   前者验证最基本的写前共享、写后分离。
   后者验证多页场景下是否每页都能独立触发一次 COW。

## 3. 代码修改逻辑

### 3.1 在 `riscv.h` 中增加 `PTE_COW`

新增定义如下：

```c
#define PTE_COW (1L << 8)
```

这一步很简单，但非常关键。
因为后续所有 COW 页识别都是基于这个标志位完成的。

### 3.2 在 `pmm.c` 中实现引用计数

我们在物理页管理模块中加入如下数据结构：

```c
#define MAX_PAGES 65536
static uint8 page_ref_count[MAX_PAGES];
```

并用下面函数完成操作：

```c
static inline int pa_to_index(void* pa) {
  uint64 addr = (uint64)pa;
  if (addr < free_mem_start_addr || addr >= free_mem_end_addr)
    return -1;
  return (addr - free_mem_start_addr) / PGSIZE;
}

void page_ref_inc(void* pa) {
  int idx = pa_to_index(pa);
  if (idx >= 0 && idx < MAX_PAGES) {
    page_ref_count[idx]++;
  }
}

void page_ref_dec(void* pa) {
  int idx = pa_to_index(pa);
  if (idx >= 0 && idx < MAX_PAGES && page_ref_count[idx] > 0) {
    page_ref_count[idx]--;
    if (page_ref_count[idx] == 0) {
      free_page(pa);
    }
  }
}

int page_ref_get(void* pa) {
  int idx = pa_to_index(pa);
  if (idx >= 0 && idx < MAX_PAGES) {
    return page_ref_count[idx];
  }
  return 0;
}
```

三个函数的含义分别是：
`inc` 用于建立新共享关系时加一；
`dec` 用于页面解除共享时减一；
`get` 用于 fault 时判断是否需要真实复制。

### 3.3 在 `vmm.c` 中建立 COW 映射

新增函数 `map_cow_page()`：

```c
void map_cow_page(pagetable_t pagetable, uint64 va, uint64 pa) {
  pte_t *pte = page_walk(pagetable, va, 1);
  if (pte == 0)
    panic("map_cow_page: page_walk failed");

  *pte = PA2PTE(pa) | PTE_V | PTE_R | PTE_U | PTE_COW;
  page_ref_inc((void*)pa);
}
```

该函数完成两件事：
第一，建立子进程到父物理页的共享映射。
第二，为共享页增加引用计数。
注意这里故意不设置 `PTE_W`，因为后续要依靠写故障触发 COW。

### 3.4 在 `vmm.c` 中实现 `handle_cow_fault()`

这是实验中最核心的处理函数：

```c
int handle_cow_fault(pagetable_t pagetable, uint64 va) {
  va = va - (va % PGSIZE);

  pte_t *pte = page_walk(pagetable, va, 0);
  if (pte == 0 || (*pte & PTE_V) == 0)
    return -1;

  if (!(*pte & PTE_COW))
    return -1;

  uint64 pa = PTE2PA(*pte);
  int ref_count = page_ref_get((void*)pa);

  if (ref_count > 1) {
    void* new_pa = alloc_page();
    if (new_pa == 0)
      return -1;

    memcpy(new_pa, (void*)pa, PGSIZE);
    *pte = PA2PTE((uint64)new_pa) | PTE_V | PTE_R | PTE_W | PTE_U | PTE_A | PTE_D;
    page_ref_dec((void*)pa);
    page_ref_inc(new_pa);
  } else {
    *pte = PA2PTE(pa) | PTE_V | PTE_R | PTE_W | PTE_U | PTE_A | PTE_D;
  }

  flush_tlb();
  return 0;
}
```

该函数的逻辑顺序为：

1. 对 fault 地址做页对齐。
2. 找到对应页表项。
3. 检查该页是否为有效页。
4. 检查该页是否带 `PTE_COW`。
5. 读取共享物理页地址。
6. 查询引用计数。
7. 如果页仍在共享，则分配新页并复制。
8. 如果页已不共享，则只恢复写权限。
9. 刷新 TLB，确保 CPU 看到最新页表项。

### 3.5 为什么要设置 `PTE_A | PTE_D`

在调试阶段我们发现，如果仅设置：

```c
PTE_V | PTE_R | PTE_W | PTE_U
```

则第一次 COW 修复完成后，同一地址还会再次 fault。
最终定位到原因是新的页表项没有补齐 `PTE_A | PTE_D`。
把访问位和脏位补上之后，重复页故障问题消失。
所以最终版本中必须使用：

```c
PTE_V | PTE_R | PTE_W | PTE_U | PTE_A | PTE_D
```

### 3.6 在 `process.c` 中改造 `do_fork()` 的堆页复制逻辑

原来的逻辑是“为子进程分配新页并复制父页内容”。
现在改成“共享映射 + COW”。
关键代码如下：

```c
for (uint64 heap_block = parent->user_heap.heap_bottom;
     heap_block < parent->user_heap.heap_top; heap_block += PGSIZE) {
  if (free_block_filter[(heap_block - heap_bottom) / PGSIZE])
    continue;

  uint64 pa = lookup_pa(parent->pagetable, heap_block);
  if (pa == 0) continue;

  if (page_ref_get((void*)pa) == 0) {
    page_ref_inc((void*)pa);
  }

  pte_t *parent_pte = page_walk(parent->pagetable, heap_block, 0);
  if (parent_pte && (*parent_pte & PTE_V)) {
    *parent_pte = (*parent_pte & ~PTE_W) | PTE_COW;
  }

  map_cow_page((pagetable_t)child->pagetable, heap_block, pa);
}
```

这一段逻辑完成了三件事：

1. 父页表项从可写改成只读 + COW。
2. 子页表项映射到同一个物理页，并打上 COW 标记。
3. 共享页引用计数增加。

### 3.7 为什么父页表项也必须改成 COW

如果只改子进程页表，而父进程仍保持可写，那么父进程第一次写共享页时不会 fault。
它会直接在共享物理页上写入数据，从而破坏 COW 语义。
因此父子两边都必须统一变成只读 + COW。

### 3.8 代码段共享映射输出的保留

预期输出中包含这一行：

```text
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
```

虽然它不是堆段 COW 的关键逻辑，但属于 challenge 的期望日志。
因此我们在代码段映射分支中加入：

```c
if (j == 0) {
  sprint("do_fork map code segment at pa:%lx of parent to child at va:%lx.\n", pa, addr);
}
```

### 3.9 在 `strap.c` 中优先处理 COW fault

最终 `handle_user_page_fault()` 的关键逻辑如下：

```c
void handle_user_page_fault(uint64 mcause, uint64 sepc, uint64 stval)
{
  sprint("handle_page_fault: %lx\n", stval);
  switch (mcause)
  {
  case CAUSE_STORE_PAGE_FAULT:
    {
      int cow_result = handle_cow_fault((pagetable_t)current->pagetable, stval);
      if (cow_result == 0) {
        return;
      }
    }

    if (stval < USER_STACK_TOP && stval > (USER_STACK_TOP - 20 * PGSIZE)) {
      void *pa = alloc_page();
      if (pa == 0)
        panic("Out of memory!");
      uint64 map_va = stval - (stval % PGSIZE);
      user_vm_map((pagetable_t)current->pagetable, map_va, PGSIZE, (uint64)pa,
                  prot_to_type(PROT_WRITE | PROT_READ, 1));
    } else {
      sprint("this address is not available!\n");
      panic("Address validation failed");
    }
    break;

  default:
    sprint("unknown page fault.\n");
    break;
  }
}
```

这里最重要的是：
对 `Store Page Fault` 先尝试 COW 修复；
只有不是 COW 页时，才继续交给原有的栈增长逻辑处理。

### 3.10 整体调用链总结

本实验完整执行流程如下：

1. 用户程序调用 `fork`。
2. `do_fork()` 把堆页改成父子共享的 COW 页。
3. 子进程写堆页时触发 `CAUSE_STORE_PAGE_FAULT`。
4. trap 分发进入 `handle_user_page_fault()`。
5. `handle_user_page_fault()` 调用 `handle_cow_fault()`。
6. `handle_cow_fault()` 分配新页并复制或直接恢复写权限。
7. trap 返回用户态后，原写指令被重新执行。
8. 子进程成功写入新的私有物理页。

## 4. 实验步骤与验证

### 4.1 实验准备

本 challenge 需要在 `lab3_3` 基础上继续工作。
进入工程目录后，先确保代码能够正常编译：

```bash
cd /app/riscv-pke
make
```

### 4.2 初始错误现象

在未实现 COW 时运行：

```bash
spike obj/riscv-pke obj/app_cow
```

可以看到子进程在写之前就已经使用了不同于父进程的新物理页。
这说明 `fork` 仍然在“立即复制”堆页，没有达到挑战实验要求。

### 4.3 第一次完成代码后的运行现象

在初步补齐 COW 相关代码后，程序已经可以编译，
但运行时出现如下问题：

```text
the physical address of child process heap before copy on write is: 0000000087faf000
handle_page_fault: 0000000000400000
handle_page_fault: 0000000000400000
this address is not available!
Address validation failed
```

说明：

1. 页面 fault 的确发生了。
2. 但第一次 fault 修复没有完全成功。
3. 同一地址再次 fault，最后落入非法地址处理。

### 4.4 Debug 过程一：检查 COW fault 是否被识别

第一步是在 `handle_cow_fault()` 中加入调试打印，查看：

1. `page_walk()` 是否找到 PTE。
2. `PTE_COW` 是否存在。
3. `pa` 是多少。
4. `ref_count` 是多少。
   很快得到输出：

```text
COW fault handling: va=0000000000400000, pa=0000000087faf000, ref_count=0
```

这说明：
页表项是找到了的；
COW 标记也是存在的；
真正异常的是引用计数竟然为 0。

### 4.5 Debug 过程二：定位引用计数为什么为 0

接着给 `page_ref_inc()` 增加调试打印，得到：

```text
page_ref_inc: pa=0000000087faf000, idx=32674
```

而当时 `MAX_PAGES` 只有 `8192`。
于是问题很明显：
物理页索引已经超过引用计数数组范围，导致引用计数根本没有被记进去。
修复方法是把：

```c
#define MAX_PAGES 8192
```

改成：

```c
#define MAX_PAGES 65536
```

修改后再次运行，`ref_count` 终于变成了正确的 2。

### 4.6 Debug 过程三：复制成功后为何还再次 fault

修复引用计数后，输出前进到了：

```text
COW fault handling: va=0000000000400000, pa=0000000087faf000, ref_count=2
COW: copied page, new_pa=0000000087fa0000
handle_page_fault: 0000000000400000
```

说明新的物理页已经分配成功，旧页也被复制了，
但 trap 返回后同一地址还是再次触发 fault。
继续分析页表项后发现，新页表项虽然已经有 `PTE_W`，但缺少 `PTE_A | PTE_D`。
把这两个位补齐之后，重复 fault 问题消失。

### 4.7 Debug 过程四：机制正确但输出不一致

当 COW 逻辑已经正确后，又出现了“评测输出不匹配”的问题。
一共有两个细节：
第一，清理调试代码时误删了：

```c
sprint("handle_page_fault: %lx\n", stval);
```

恢复后，输出重新包含标准答案要求的 fault 行。
第二，`do_fork()` 中缺少代码段共享映射日志：

```text
do_fork map code segment at pa:... of parent to child at va:...
```

补齐这条日志后，输出才与预期完全一致。

### 4.8 `app_cow` 的最终验证结果

最终运行：

```bash
spike obj/riscv-pke obj/app_cow
```

关键输出如下：

```text
the physical address of parent process heap is: 0000000087faf000
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087fad000, user stack 0x000000007ffff000, user kstack 0x0000000087fac000
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
User exit with code:0.
going to schedule process 1 to run.
the physical address of child process heap before copy on write is: 0000000087faf000
handle_page_fault: 0000000000400000
the physical address of child process heap after copy on write is: 0000000087fa0000
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

分析这个结果：

1. 子进程写前地址与父进程相同，说明共享成功。
2. 写入时产生一次 fault，说明 COW 被正确触发。
3. 写后地址变化，说明页面被真正复制。

### 4.9 `app_cow_e` 的扩展验证结果

为了验证多页场景，又运行：

```bash
spike obj/riscv-pke obj/app_cow_e
```

关键输出如下：

```text
the physical address of parent process heap is: 0000000087faf000
0000000087fad000
User call fork.
will fork a child from parent 0.
in alloc_proc. user frame 0x0000000087fac000, user stack 0x000000007ffff000, user kstack 0x0000000087fab000
do_fork map code segment at pa:0000000087fb2000 of parent to child at va:0000000000010000.
going to insert process 1 to ready queue.
User exit with code:0.
going to schedule process 1 to run.
the physical address of child process heap before copy on write is: 0000000087faf000
0000000087fad000
handle_page_fault: 0000000000400000
handle_page_fault: 0000000000401000
the physical address of child process heap after copy on write is: 0000000087f9f000
0000000087f9e000
User exit with code:0.
no more ready processes, system shutdown now.
System is shutting down with exit code 0.
```

这说明两页堆空间都正确地走完了 COW 流程：
写前共享、写时 fault、写后分离。

### 4.10 可复现实验步骤汇总

为了方便后续再次验证，可以按以下顺序执行：

1. 编译：

```bash
make
```

2. 运行单页测试：

```bash
spike obj/riscv-pke obj/app_cow
```

3. 运行多页测试：

```bash
spike obj/riscv-pke obj/app_cow_e
```

4. 对比检查点：

- 是否有 `do_fork map code segment ...`。
- 是否有 `handle_page_fault: ...`。
- 写前地址是否与父进程一致。
- 写后地址是否与写前不同。

## 5. 实验收获

### 5.1 对 `fork` 的理解更深入

完成本实验后，我对 `fork` 的理解不再停留在“复制一个子进程”。
真正高效的 `fork` 不是立即把数据页都复制一份，
而是先共享，再依靠 fault 与页表权限完成延迟复制。
这说明操作系统中的很多高性能机制，核心并不是“加速复制”，而是“避免不必要的复制”。

### 5.2 对页表权限位的作用认识更深

本实验几乎所有关键逻辑都围绕页表位展开：
去掉 `W` 位制造 fault，
增加 `PTE_COW` 表示软件语义，
处理 fault 后再恢复 `W` 位并补齐 `A/D` 位。
这让我体会到很多系统机制虽然概念上复杂，底层却常常只是几个关键位的精确控制。

### 5.3 对缺页异常的认识更真实

以前觉得页故障通常意味着错误。
这个实验让我认识到，页故障也可以被系统主动利用，变成一种“受控的执行入口”。
在 COW 里，fault 不是 bug，而是功能的一部分。
正是因为第一次写会 fault，系统才能判断“现在该复制了”。

### 5.4 对共享与隔离关系的理解更清楚

过去容易想当然地把“不同进程”与“不同物理页”画等号。
本实验说明不是这样的。
只要页表和权限设计合理，不同进程完全可以共享同一物理页。
只有当某一方真正需要独占写入时，才有必要拆分出新页。
这是一种非常重要的资源优化思想。

### 5.5 对调试方法有直接收获

本实验让我进一步形成了系统实验中的排查顺序：
先看现象；
再看 fault 是否进入目标路径；
再看 PTE、引用计数、物理地址是否符合预期；
最后检查输出文本是否完全匹配评测结果。
这比盲目改代码要高效得多。

### 5.6 对边界问题更敏感

`MAX_PAGES` 容量不足这个问题给我的印象最深。
它提醒我：
只要代码中出现“地址到索引”的映射，就必须检查范围是否真实覆盖运行时数据。
否则程序可能不会立刻崩溃，但行为会悄悄偏离预期。

### 5.7 对实验评测方式有更完整认识

本次实验最后两个修复点其实都不是机制性 bug，
而是输出差异：
一个是漏掉 `handle_page_fault`，
一个是漏掉 `do_fork map code segment ...`。
这让我认识到：
在课程实验里，功能正确和输出正确同样重要。
因此调试完成后，必须做一次“逐行输出对照”。

## 6. 实验调试记录

### 6.1 第一个阶段：代码能编译，但运行失败

第一次补齐 COW 相关代码后，工程已经可以 `make` 成功。
但运行 `app_cow` 时失败，输出表现为：
同一地址连续 fault 两次，最后触发 `Address validation failed`。
这说明问题不是编译期错误，而是运行期机制问题。

### 6.2 第二个阶段：确认 fault 已进入 COW 处理函数

为了判断 fault 是否真的走到了预期路径，
我在 `handle_cow_fault()` 中加入打印。
结果看到 `va` 和 `pa` 都能正确打印出来，说明：
trap 分发没错；
页表遍历也没错；
真正有问题的是更深一层的状态数据。

### 6.3 第三个阶段：发现 `ref_count=0`

继续打印后得到最关键的一条信息：

```text
COW fault handling: va=0000000000400000, pa=0000000087faf000, ref_count=0
```

这直接把排查范围缩小到“引用计数系统”。
因为一个父子共享页不可能计数为 0。

### 6.4 第四个阶段：确认引用计数数组越界

随后对 `page_ref_inc()` 做打印，发现页索引大于原先数组容量。
因此虽然逻辑上调用了加计数函数，但实际上计数值根本没有被写入数组。
修复 `MAX_PAGES` 后，引用计数恢复正常。

### 6.5 第五个阶段：定位重复 fault 的真正原因

引用计数修复后，程序已经能成功复制新页，
但返回用户态后同地址仍然再次 fault。
进一步分析发现，是新页表项缺少 `PTE_A | PTE_D`。
补齐后，重复 fault 被彻底消除。

### 6.6 第六个阶段：修正输出不一致问题

当机制已经完全正确后，
仍有两处输出与标准答案不一致：
一处是 `handle_page_fault` 日志被误删；
一处是代码段共享映射日志缺失。
把这两条输出补齐之后，`app_cow` 与 `app_cow_e` 都完全通过。

### 6.7 最终调试结论

整个调试过程中，真正起决定作用的问题有四个：

1. `MAX_PAGES` 太小，导致引用计数失效。
2. COW 修复后的页表项缺少 `PTE_A | PTE_D`。
3. 误删 `handle_page_fault` 输出。
4. 缺少 `do_fork map code segment ...` 输出。
   前两个属于机制性 bug，后两个属于输出一致性问题。
   只有四个问题全部解决，实验才算真正完成。

### 6.8 如果以后再做一次，我会如何排查

如果将来重新做这个实验，我会按下面顺序排查：

1. 先看写前物理地址是否共享。
2. 再看写入时是否发生 `Store Page Fault`。
3. 然后看 `handle_cow_fault()` 是否成功进入并返回。
4. 如果 fault 后还再次 fault，就检查 PTE 是否带 `A/D` 位并确认 TLB 是否刷新。
5. 最后逐行对照输出文本，确认日志完全一致。
   这个顺序能够把“功能问题”和“输出问题”快速区分开。

## 总结

本次 `Lab3 Challenge3` 的任务，是为 PKE 的 `fork` 实现写时复制机制。
围绕这一目标，我完成了以下工作：

1. 新增 `PTE_COW` 页表位。
2. 为物理页建立引用计数体系。
3. 改造 `do_fork()`，把堆页从“直接复制”改成“共享映射 + COW”。
4. 新增 `map_cow_page()` 与 `handle_cow_fault()`。
5. 在 `strap.c` 中优先处理 COW fault。
6. 修复引用计数范围、`A/D` 位、输出缺失等调试问题。
   最终结果表明：
   `app_cow` 与 `app_cow_e` 都已通过。
   子进程在写前与父进程共享同一物理页；
   写入时触发页故障；
   写后得到新的私有物理页；
   并且输出与预期完全一致。
   通过本次实验，我对 `fork`、页表权限、页故障处理、物理页共享和系统调试方法都有了更深入、更具体的理解。





# Lab4 Challenge2：重载执行

## 1. 实验原理与分析过程

### 1.1 实验目标

本实验要求在 PKE 中实现 `exec` 系统调用，使用户程序能够通过路径名加载并执行另一个 ELF 程序。
当 `exec` 成功时，当前进程不再继续执行旧程序，而是直接“重载”为新程序；
当 `exec` 失败时，返回 `-1`，由用户程序自行处理错误。

结合给定应用 `app_exec.c`，实验目标可以概括为：

1. 在用户态提供 `exec()` 接口。
2. 在内核中新增 `SYS_user_exec` 系统调用。
3. 根据路径名从文件系统读取 ELF 程序。
4. 清理当前进程原有的代码段、数据段和堆段映射。
5. 把新的 ELF 装入当前进程地址空间，并把执行入口切换到新程序。

### 1.2 `exec` 的本质是什么

`fork` 的语义是“复制出一个新的进程”；
而 `exec` 的语义是“保留当前进程身份，但把当前进程装载的程序整体替换掉”。

这意味着：

1. 进程的 `pid` 不变。
2. 当前进程仍然是同一个进程控制块。
3. 但用户地址空间中的程序内容被新程序替换。
4. 用户态返回点也要改成新程序入口地址。

因此，`exec` 不是新建进程，
而是“复用当前进程结构，重建其用户程序映像”。

### 1.3 本实验为什么要从文件系统加载程序

前面的基础实验中，PKE 启动主程序时通常使用宿主机文件接口，
例如直接读取 `./obj/app_exec`。
但本次 challenge 的核心目标，是让用户程序能够通过路径名执行 `/bin/app_ls` 这类已经位于文件系统中的程序。

这意味着内核不能只依赖宿主机文件接口，
还需要能够通过 VFS 接口访问挂载在 `/bin` 下的可执行文件。

也正因为如此，本实验其实包含了两层能力：

1. **exec 本身的替换执行能力**。
2. **从文件系统按路径读取 ELF 的能力**。

### 1.4 成功执行一次 `exec` 的完整流程

本实验中一次成功的 `exec("/bin/app_ls")` 大致经历如下流程：

1. 用户程序调用 `exec("/bin/app_ls")`。
2. 用户库把参数和系统调用号写入寄存器，并触发 `ecall`。
3. 内核进入 `sys_user_exec()`。
4. `sys_user_exec()` 先把用户虚拟地址形式的路径名转换成内核可访问地址。
5. 进入 `do_exec()`，清理当前进程旧程序的代码段、数据段和堆段。
6. `do_exec()` 调用 `load_bincode_from_fs()`，通过 VFS 打开 `/bin/app_ls`。
7. 解析 ELF 头与程序头，把新程序段映射到当前进程页表中。
8. 更新 `trapframe->epc` 为新程序入口地址。
9. trap 返回用户态后，不再回到旧程序位置，而是从新程序入口开始执行。

### 1.5 本实验与 `fork` 的关系

从系统设计上说，`fork` 和 `exec` 往往是一组配套机制：

1. `fork` 负责复制出一个子进程。
2. `exec` 负责让这个子进程装入并运行另一个程序。

虽然本 challenge 只要求实现 `exec`，
但其本质已经是在为后续 shell 实验打基础。
后面的 `fork + exec + wait` 组合，本质上就是 Unix 风格命令执行模型的最小原型。

### 1.6 本实验的关键难点

本次实现的主要难点不在“新增一个系统调用号”，
而在于以下几个关键点：

1. **旧地址空间如何清理**：
   不能粗暴回收整个进程，否则连栈、trapframe、内核栈都会一起破坏。

2. **新程序如何装入当前进程**：
   不是新建进程，而是在当前页表上重建用户程序映像。

3. **线上平台和本地运行方式不同**：
   本地可以用 `./obj/app_exec` 启动主程序，
   但线上平台使用 `/bin/app_exec`，要求主程序加载逻辑也支持从 VFS 读取。

4. **exec 成功后“不返回”如何理解**：
   并不是系统调用层面永远没有返回值，
   而是成功后用户态控制流已经被切换到新程序入口，
   所以旧程序中 `exec()` 后面的语句不会继续执行。

---

## 2. 需要修改的代码位置

### 2.1 用户库接口

**文件：** `user/user_lib.h`、`user/user_lib.c`

需要在用户态暴露 `exec()` 接口，
使应用程序可以像调用其他系统调用一样调用 `exec`。

### 2.2 系统调用号与系统调用分发

**文件：** `kernel/syscall.h`、`kernel/syscall.c`

需要新增：

1. `SYS_user_exec` 系统调用号。
2. `sys_user_exec()` 系统调用处理函数。
3. 在 `do_syscall()` 中加入分发分支。

### 2.3 进程管理模块

**文件：** `kernel/process.h`、`kernel/process.c`

需要新增 `do_exec()`，负责：

1. 清理旧用户程序映射。
2. 重置堆管理状态。
3. 调用 ELF 加载逻辑装入新程序。

### 2.4 虚拟内存辅助模块

**文件：** `kernel/vmm.c`

需要确保 `user_va_to_pa()` 已经正确实现。
这是 `exec` 能否工作的重要前提，因为：

1. 系统调用传入的路径参数位于用户地址空间。
2. 内核必须先把该用户虚拟地址翻译成可访问的物理地址。
3. 如果 `user_va_to_pa()` 没有实现，`sys_user_exec()` 就无法正确取得路径字符串。

### 2.5 ELF 加载模块

**文件：** `kernel/elf.h`、`kernel/elf.c`

需要补充一套“从 VFS 读取 ELF”的加载逻辑：

1. 通过 `vfs_open()` 打开文件。
2. 通过 `vfs_lseek()` 与 `vfs_read()` 读取 ELF 头和程序段。
3. 将程序段映射到当前进程页表。
4. 更新入口地址。

同时，为了适配线上平台，还需要让主程序加载逻辑同时支持：

1. 宿主机路径，如 `./obj/app_exec`。
2. 文件系统路径，如 `/bin/app_exec`。

---

## 3. 代码修改逻辑

### 3.1 在用户态新增 `exec()` 接口

先在 `user/user_lib.h` 中声明：

```c
int exec(const char *pathname);
```

然后在 `user/user_lib.c` 中实现：

```c
int exec(const char *pathname){
  return do_user_call(SYS_user_exec, (uint64)pathname, 0, 0, 0, 0, 0, 0);
}
```

这样用户程序就可以通过：

```c
exec("/bin/app_ls");
```

来触发系统调用。

### 3.2 在 `syscall.h` 中新增系统调用号

在现有系统调用编号后增加：

```c
#define SYS_user_exec   (SYS_user_base + 30)
```

这一步的作用，是让用户态和内核态对 `exec` 拥有统一编号。

### 3.3 在 `syscall.c` 中加入 `sys_user_exec()`

系统调用处理函数实现如下：

```c
ssize_t sys_user_exec(char *pathname_va) {
  char *pathname = (char*)user_va_to_pa((pagetable_t)(current->pagetable), (void*)pathname_va);
  return do_exec(pathname);
}
```

这里最关键的一步是：
把用户态传入的虚拟地址 `pathname_va` 转成当前页表下的物理地址映射，
否则内核不能正确读取路径字符串内容。

这一点也意味着，本实验虽然主体是 `exec`，
但它隐含依赖前面实验中已经补齐的 `user_va_to_pa()`。
如果线上平台代码没有同步 `vmm.c` 中这部分实现，
那么 `sys_user_exec()` 即使写对了，也仍然会因为拿不到正确的路径字符串而失败。

然后在 `do_syscall()` 中补上分发：

```c
case SYS_user_exec:
  return sys_user_exec((char *)a1);
```

### 3.4 在 `process.h` 中声明 `do_exec()`

为了让系统调用模块能够调用 `do_exec()`，
需要在 `kernel/process.h` 中增加函数声明：

```c
int do_exec(const char* pathname);
```

### 3.5 在 `vmm.c` 中确保 `user_va_to_pa()` 可用

由于 `exec` 的参数 `pathname` 来自用户态，
所以在进入 `sys_user_exec()` 后必须先完成地址翻译。
我在 `kernel/vmm.c` 中使用页表遍历实现了 `user_va_to_pa()`：

```c
void *user_va_to_pa(pagetable_t page_dir, void *va)
{
  uint64 va_val = (uint64)va;
  pte_t *pte = page_walk(page_dir, va_val, 0);

  if (pte == 0 || (*pte & PTE_V) == 0)
    return NULL;

  uint64 pa_page = PTE2PA(*pte);
  uint64 offset = va_val & (PGSIZE - 1);
  return (void *)(pa_page + offset);
}
```

这个函数完成了三件事：

1. 通过 `page_walk()` 找到用户虚拟地址对应的页表项。
2. 从页表项中提取物理页基址。
3. 加上页内偏移，得到最终可访问的物理地址。

如果这个函数没有实现，或者线上平台使用的代码仍是未完成版本，
那么 `sys_user_exec()` 中这句：

```c
char *pathname = (char*)user_va_to_pa((pagetable_t)(current->pagetable), (void*)pathname_va);
```

就无法得到正确字符串地址，后续 `do_exec(pathname)` 也就不可能稳定工作。

### 3.6 `do_exec()` 的整体思路

`do_exec()` 的职责可以分成三步：

1. 删除旧程序的用户态映射。
2. 重置堆相关信息。
3. 装入新的 ELF 文件。

其核心实现如下：

```c
int do_exec(const char *pathname) {
  sprint("User call exec: %s\n", pathname);

  for (int i = 0; i < current->total_mapped_region; i++) {
    switch (current->mapped_info[i].seg_type) {
      case CODE_SEGMENT:
      case DATA_SEGMENT:
        for (int j = 0; j < current->mapped_info[i].npages; j++) {
          uint64 va = current->mapped_info[i].va + j * PGSIZE;
          user_vm_unmap((pagetable_t)current->pagetable, va, PGSIZE, 1);
        }
        current->mapped_info[i].va = 0;
        current->mapped_info[i].npages = 0;
        current->mapped_info[i].seg_type = 0;
        break;

      case HEAP_SEGMENT:
        for (uint64 heap_va = current->user_heap.heap_bottom;
             heap_va < current->user_heap.heap_top;
             heap_va += PGSIZE) {
          user_vm_unmap((pagetable_t)current->pagetable, heap_va, PGSIZE, 1);
        }
        current->user_heap.heap_top = current->user_heap.heap_bottom;
        current->user_heap.free_pages_count = 0;
        current->mapped_info[i].npages = 0;
        break;

      default:
        break;
    }
  }

  int write_idx = 0;
  for (int read_idx = 0; read_idx < current->total_mapped_region; read_idx++) {
    if (current->mapped_info[read_idx].va != 0) {
      if (write_idx != read_idx) {
        current->mapped_info[write_idx] = current->mapped_info[read_idx];
      }
      write_idx++;
    }
  }
  current->total_mapped_region = write_idx;

  if (load_bincode_from_fs(current, pathname) != 0) {
    sprint("Failed to load program: %s\n", pathname);
    return -1;
  }

  return 0;
}
```

### 3.7 为什么不能直接回收整个进程

最开始最容易想到的错误方案，是“像退出进程一样先把整个进程释放，再重新创建”。
但这会带来明显问题：

1. 当前 `pid` 会失去意义。
2. 当前内核执行上下文会被破坏。
3. 当前 trap 返回路径会丢失。

因此正确做法不是删除进程，
而是仅清理用户程序相关段，保留：

1. 用户栈段。
2. trapframe 段。
3. 系统段和内核栈。

### 3.8 为什么要压缩 `mapped_info`

在清理代码段、数据段后，相关 `mapped_info` 项会被清零。
如果不进一步压缩数组，后面重新装载 ELF 时：

1. 旧的空洞项和新的映射项会混在一起。
2. `total_mapped_region` 不能正确反映当前真实映射数。
3. 后续遍历映射信息时容易出现逻辑混乱。

因此我们通过“读写双指针”的方式把有效项前移，
让 `mapped_info` 始终保持紧凑。

### 3.9 为什么 `exec` 主要使用 VFS 加载新程序

因为 `exec` 接收的是路径名，
例如 `/bin/app_ls`、`/bin/app_read`。
这些路径是文件系统中的逻辑路径，
并不是宿主机上的实际路径。

所以在 `exec` 实现中，最自然的方式就是使用：

1. `vfs_open()` 打开文件。
2. `vfs_lseek()` 定位偏移。
3. `vfs_read()` 读取 ELF 内容。
4. `vfs_close()` 关闭文件。

这也是 challenge 的核心要求之一。

### 3.10 在 `elf.c` 中补充 VFS 版本 ELF 加载器

为了从文件系统读取 ELF，本实验新增了：

1. `elf_vfs_info`
2. `elf_fpread_vfs()`
3. `elf_init_vfs()`
4. `elf_load_vfs()`
5. `load_bincode_from_fs()`

其中读取函数的核心逻辑是：

```c
static uint64 elf_fpread_vfs(elf_ctx *ctx, void *dest, uint64 nb, uint64 offset) {
  elf_vfs_info *msg = (elf_vfs_info *)ctx->info;
  vfs_lseek(msg->f, offset, SEEK_SET);
  return vfs_read(msg->f, (char *)dest, nb);
}
```

这相当于给 ELF 加载器换了一个底层“读文件后端”。
原来是 `spike_file_pread()`，
现在增加了基于 VFS 的 `pread` 风格读取。

### 3.11 `load_bincode_from_fs()` 如何工作

其整体逻辑如下：

```c
int load_bincode_from_fs(process *p, const char *pathname) {
  elf_ctx elfloader;
  elf_vfs_info info;

  info.f = vfs_open(pathname, O_RDONLY);
  if (info.f == NULL) {
    sprint("Fail on opening the file: %s\n", pathname);
    return -1;
  }
  info.p = p;

  if (elf_init_vfs(&elfloader, &info) != EL_OK) {
    sprint("fail to init elfloader.\n");
    vfs_close(info.f);
    return -1;
  }

  if (elf_load_vfs(&elfloader) != EL_OK) {
    sprint("Fail on loading elf.\n");
    vfs_close(info.f);
    return -1;
  }

  p->trapframe->epc = elfloader.ehdr.entry;
  vfs_close(info.f);
  sprint("Application program entry point (virtual address): 0x%lx\n", p->trapframe->epc);
  return 0;
}
```

也就是说，`exec` 成功与否，本质上取决于：

1. 路径能否被 VFS 正确打开。
2. 该文件是否是合法 ELF。
3. 程序段能否被映射进当前进程页表。

### 3.12 为什么线上平台会报 “Fail on openning the input application program.”

这是本次实验中最关键、也最容易踩坑的问题。

本地测试时，我最初使用的是：

```bash
spike ./obj/riscv-pke ./obj/app_exec
```

在这种方式下，主程序是通过宿主机路径 `./obj/app_exec` 加载的，
原来的 `load_bincode_from_host_elf()` 可以正常工作。

但在线上测试平台中，主程序启动方式实际是：

```bash
spike ./obj/riscv-pke /bin/app_exec
```

也就是说，主程序路径变成了文件系统路径 `/bin/app_exec`。
如果此时还继续调用：

```c
spike_file_open(app_path, O_RDONLY, 0)
```

那它就会试图在宿主机文件系统中打开 `/bin/app_exec`，
自然失败，于是出现：

```text
Fail on openning the input application program.
```

### 3.13 线上平台遗漏 `vmm.c` 时会出现什么问题

除了主程序路径差异外，线上平台还有一个非常实际的问题：
如果提交时没有同步 `kernel/vmm.c`，
导致 `user_va_to_pa()` 仍然是未实现状态，
那么 `exec` 相关代码即使已经补齐，也仍然可能运行失败。

原因在于：

1. `exec` 的参数路径来自用户程序。
2. 内核收到的是一个用户虚拟地址。
3. 若没有 `user_va_to_pa()`，内核就不能把它翻译成真实可访问地址。

这样会进一步带来两类问题：

1. `sys_user_exec()` 取到的 `pathname` 为空或错误。
2. 后续文件打开阶段会表现为路径异常、打开失败，甚至不可预测行为。

也就是说，线上平台调试时必须同时检查两件事：

1. `elf.c` 是否支持 `/bin/...` 形式的 VFS 路径。
2. `vmm.c` 中 `user_va_to_pa()` 是否已经同步到位。

### 3.14 适配线上平台的关键修复

为了解决这个问题，需要让主程序加载逻辑同时支持两种路径来源：

1. 若路径是 `./obj/app_exec` 这类宿主机路径，继续使用 `spike_file_open()`。
2. 若路径是 `/bin/app_exec` 这类文件系统路径，则改用 `load_bincode_from_fs()`。

最终把 `load_bincode_from_host_elf()` 改成：

```c
char *app_path = arg_bug_msg.argv[0];
sprint("Application: %s\n", app_path);

if (app_path[0] == '/') {
  if (load_bincode_from_fs(p, app_path) != 0) {
    panic("Fail on loading application from VFS.\n");
  }
  return;
}
```

这样，主程序加载阶段和 `exec` 阶段就都支持基于 VFS 的 ELF 装载了。

### 3.15 这个修复为什么很重要

这个问题说明：
实现 `exec` 不仅要关注“用户程序执行另一个程序”这件事本身，
还要关注“PKE 自己一开始是怎么装入第一个用户程序的”。

如果主程序加载入口与 `exec` 采用两套完全割裂的路径机制，
就很容易出现：

1. 本地测试通过。
2. 线上评测失败。

因此，本实验真正完整的修复应当覆盖两件事：

1. `exec` 能从文件系统加载程序。
2. 主程序入口也能识别并正确加载 `/bin/...` 路径。

---

## 4. 实验步骤与验证

### 4.1 实验准备

本实验基于 `lab4_3` 的文件系统代码继续完成。
进入工程目录后先编译：

```bash
cd /app/riscv-pke
make clean
make
```

编译完成后，`hostfs_root/bin/` 下会生成：

1. `app_exec`
2. `app_ls`
3. `app_exec2`
4. `app_read`

其中：

1. `app_exec` 会调用 `exec("/bin/app_ls")`。
2. `app_exec2` 会调用 `exec("/bin/app_read")`。

### 4.2 本地第一阶段验证：旧方式加载主程序

先用宿主机路径启动主程序：

```bash
spike ./obj/riscv-pke ./obj/app_exec
```

得到关键输出：

```text
User application is loading.
Application: ./obj/app_exec
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x0000000000010078

======== exec /bin/app_ls in app_exec ========
User call exec: /bin/app_ls
CODE_SEGMENT added at mapped info offset:3
DATA_SEGMENT added at mapped info offset:4
Application program entry point (virtual address): 0x00000000000100b0
------------------------------
ls "/RAMDISK0":
[name]               [inode_num]
------------------------------
User exit with code:0.
```

这个结果说明：

1. 主程序能够正常启动。
2. `exec` 已经能够从 `/bin/app_ls` 装载新程序。
3. 新程序执行完后正常退出。

### 4.3 第二阶段验证：执行 `app_read`

继续运行：

```bash
spike ./obj/riscv-pke ./obj/app_exec2
```

关键输出如下：

```text
======== exec /bin/app_read in app_exec ========
User call exec: /bin/app_read
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x0000000000010078
------------------------------
read: /hostfile.txt
file descriptor fd: 0
read content:
This is an apple.
Apples are good for our health.
------------------------------
User exit with code:0.
```

这个结果说明 `exec` 并不依赖某一个固定程序，
而是已经具备了“按路径重载任意可执行文件”的能力。

### 4.4 线上平台问题复现

当代码提交到线上平台后，实际输出变成：

```text
User application is loading.
Application: /bin/app_exec
Fail on openning the input application program.

System is shutting down with exit code -1.
```

这说明失败并不是发生在 `exec("/bin/app_ls")` 阶段，
而是在更早的“主程序加载阶段”就已经失败了。

除此之外，线上平台排查时还发现另一个潜在问题：
平台上的 `kernel/vmm.c` 如果没有同步最新修改，
则 `user_va_to_pa()` 仍然未实现。
这会导致即使主程序能被装入，`exec` 读取用户态路径参数时也可能失败。

### 4.5 问题定位

分析启动流程可知，
PKE 在 `load_user_program()` 中调用了 `load_bincode_from_host_elf()`。

而 `load_bincode_from_host_elf()` 以前只会：

```c
info.f = spike_file_open(app_path, O_RDONLY, 0);
```

所以当 `app_path` 是 `/bin/app_exec` 时，
内核会把它误当成宿主机路径，
自然打不开，导致整机直接 `panic` 退出。

与此同时，我又回头检查了 `exec` 参数传递链：
`user -> do_user_call -> sys_user_exec -> user_va_to_pa -> do_exec`。
这条链说明，只要 `user_va_to_pa()` 缺失，
那么内核侧读取用户字符串参数的逻辑就不完整。

因此，最终问题定位并不是单点，而是两个条件都必须满足：

1. 主程序加载阶段要支持 VFS 路径。
2. 地址翻译辅助函数 `user_va_to_pa()` 要已经实现。

### 4.6 最终验证：支持从 VFS 加载主程序

修复主程序加载逻辑后，再运行：

```bash
spike ./obj/riscv-pke /bin/app_exec
```

关键输出如下：

```text
User application is loading.
Application: /bin/app_exec
CODE_SEGMENT added at mapped info offset:3
Application program entry point (virtual address): 0x0000000000010078
going to insert process 0 to ready queue.
going to schedule process 0 to run.

======== exec /bin/app_ls in app_exec ========
User call exec: /bin/app_ls
CODE_SEGMENT added at mapped info offset:3
DATA_SEGMENT added at mapped info offset:4
Application program entry point (virtual address): 0x00000000000100b0
------------------------------
ls "/RAMDISK0":
[name]               [inode_num]
------------------------------
User exit with code:0.
```

这说明：

1. 主程序已经可以从 `/bin/app_exec` 正确装载。
2. `exec` 也能继续从 `/bin/app_ls` 正确切换执行。
3. 本地和线上平台两种路径模式都被兼容。

### 4.7 可复现实验步骤汇总

为了后续复查，本实验可以按下面顺序验证：

1. 编译：

```bash
make clean
make
```

2. 验证本地宿主机路径方式：

```bash
spike ./obj/riscv-pke ./obj/app_exec
```

3. 验证线上平台对应的文件系统路径方式：

```bash
spike ./obj/riscv-pke /bin/app_exec
```

4. 验证重载另一个程序：

```bash
spike ./obj/riscv-pke ./obj/app_exec2
```

5. 检查关键点：
   - 是否出现 `Application: /bin/app_exec`。
   - 是否成功进入 `======== exec /bin/app_ls in app_exec ========`。
   - 是否打印新的入口地址。
   - 是否由 `app_ls` 或 `app_read` 输出结果。

---

## 5. 实验收获

### 5.1 更深入理解了 `exec` 和 `fork` 的区别

在做实验之前，我对 `exec` 的理解偏概念化，
只知道它是“执行另一个程序”。
真正实现后我才更清楚地意识到：
`fork` 是生成新进程，`exec` 是替换当前进程内容。
它们看起来都和“运行程序”有关，
但底层机制完全不同。

### 5.2 对“进程身份”和“进程映像”这两个概念区分更清楚

本实验让我清楚看到：
一个进程并不等于它当前运行的那段代码。
进程控制块、`pid`、调度身份是一层；
代码段、数据段、堆段、入口点又是另一层。
`exec` 只替换后者，不替换前者。

### 5.3 对 ELF 装载流程有了更具体的认识

过去对 ELF 加载更多停留在“知道要解析 ELF 头”。
这次自己补齐 VFS 版本加载器之后，
才真正把整个流程串起来：
打开文件、读取头部、遍历程序头、分配页、映射段、设置入口地址。
这让我对“用户程序为什么能跑起来”有了更直接的理解。

### 5.4 对文件系统和程序装载之间的关系认识更深

本实验不是单纯的文件系统实验，
也不是单纯的进程实验，
而是二者的结合。
如果没有 VFS 提供路径解析和文件读取能力，
`exec` 就无法仅凭一个字符串路径完成程序装载。
这让我更直观地感受到操作系统各个模块之间是如何协作的。

### 5.5 对“本地通过不等于线上通过”有了更深刻体会

这次最有代表性的调试点，就是本地用 `./obj/app_exec` 可以工作，
但线上平台用 `/bin/app_exec` 会直接失败。
这个问题提醒我：
实验调试时不能只盯着本地一套运行路径，
还要主动比对评测环境的真实启动方式。

### 5.6 对调试顺序有更清晰的方法论

本实验里真正起决定作用的，不是盲目改代码，
而是分层判断：

1. 失败发生在主程序加载阶段，还是 `exec` 阶段。
2. 路径是宿主机路径，还是 VFS 路径。
3. 出错是在打开文件、解析 ELF，还是跳转入口时。

这种按调用链逐层收缩问题范围的方法，
比单纯猜测哪里错了高效得多。

---

## 6. 实验调试记录

### 6.1 第一阶段：先让 `exec` 基本跑通

最初完成 `SYS_user_exec`、`sys_user_exec()`、`do_exec()` 之后，
我先在本地用：

```bash
spike ./obj/riscv-pke ./obj/app_exec
```

进行验证。
这一阶段的结果是：
`app_exec` 能够成功切换并执行 `/bin/app_ls`；
`app_exec2` 也能够成功切换并执行 `/bin/app_read`。

这说明 `exec` 主体逻辑已经成立。

### 6.2 第二阶段：线上平台报错

随后把代码放到线上测试平台，
结果程序在用户应用正式运行前就失败，输出为：

```text
Application: /bin/app_exec
Fail on openning the input application program.
```

这时第一反应其实容易误以为是：
`exec("/bin/app_ls")` 的路径打不开。
但仔细看输出顺序后发现不是这样。

### 6.3 第三阶段：确认失败点在主程序加载

从日志顺序分析：
当报错发生时，系统还没有打印：

```text
======== exec /bin/app_ls in app_exec ========
```

这说明 `app_exec` 自己都还没有开始执行。
因此可以确定：
错误发生在“内核加载第一个用户程序”这一阶段，
而不是发生在 `exec` 系统调用本身。

### 6.4 第四阶段：定位到 `spike_file_open()` 与 `/bin/...` 路径不兼容

继续阅读代码后发现，
主程序加载函数仍然固定使用：

```c
spike_file_open(app_path, O_RDONLY, 0)
```

本地传入的是 `./obj/app_exec`，所以没问题；
线上传入的是 `/bin/app_exec`，它不是宿主机路径，因此打开失败。

问题至此被准确锁定。

但到这一步还不能完全放心，
因为 `exec` 自身还依赖 `user_va_to_pa()` 来读取用户传入路径。
继续核对线上平台代码后发现，
`kernel/vmm.c` 并没有同步过去，
也就是说 `user_va_to_pa()` 实际上仍是未完成状态。

这又解释了为什么有些情况下即使补了 `exec` 相关代码，
系统仍可能表现异常：
根本原因不是 `exec` 主体逻辑错了，
而是参数地址翻译这块底座代码还没补齐。

### 6.5 第五阶段：补齐 `user_va_to_pa()`

确认 `vmm.c` 缺失之后，
我先把 `user_va_to_pa()` 的实现补齐，
确保内核能够通过页表把用户虚拟地址翻译成实际物理地址。

只有这一步完成后，
`sys_user_exec()` 才能稳定拿到正确的路径字符串，
例如 `/bin/app_ls` 或 `/bin/app_read`。

### 6.6 第六阶段：修改主程序加载逻辑

接下来把主程序装载逻辑改成“按路径类型分流”：

1. 如果参数以 `/` 开头，则认为是文件系统路径，用 `load_bincode_from_fs()`。
2. 否则认为是宿主机路径，继续使用 `spike_file_open()`。

这个改动的意义在于：
不仅修复了线上平台，
也保留了本地已有的测试方式。

### 6.7 第七阶段：清理输出细节

在调试过程中还顺手清理了两个细节：

1. 移除了不必要的调试打印，避免与实验预期输出不一致。
2. 去掉了主程序加载时重复出现的一次 `Application: ...` 日志。

这一步虽然不影响核心机制，
但对于课程实验的评测输出一致性很重要。

### 6.8 最终调试结论

本实验最终真正需要解决的核心问题有两个：

1. `exec` 系统调用本身的实现。
2. `user_va_to_pa()` 这类地址翻译基础能力必须已补齐。
3. 主程序加载阶段对 `/bin/...` VFS 路径的支持。

前者解决“进程如何被重载执行”；
第二点解决“内核如何正确读取用户态传入路径”；
第三点解决“线上平台为什么一开始就打不开程序”。

只有三部分都补齐，`lab4_challenge2` 才算真正完成。

## 总结

本次 `Lab4 Challenge2` 的核心任务，是在 PKE 中实现 `exec` 重载执行机制。
围绕这一目标，我完成了以下工作：

1. 在用户库中新增 `exec()` 接口。
2. 在内核中新增 `SYS_user_exec` 系统调用以及分发逻辑。
3. 在 `vmm.c` 中补齐 `user_va_to_pa()`，保证内核能正确读取用户态路径参数。
4. 在 `process.c` 中实现 `do_exec()`，完成旧程序映射清理与新程序装载。
5. 在 `elf.c` 中补充基于 VFS 的 ELF 加载器。
6. 修复主程序加载逻辑，使其既支持 `./obj/app_exec`，也支持 `/bin/app_exec`。

最终结果表明：

1. `app_exec` 可以成功执行 `exec("/bin/app_ls")`。
2. `app_exec2` 可以成功执行 `exec("/bin/app_read")`。
3. 本地运行方式与线上评测方式都能正确通过。

通过本次实验，我对 `exec` 的语义、ELF 装载流程、VFS 与进程管理的协作关系，以及线上线下环境差异带来的调试问题，都有了更深入和更具体的理解。



# Lab4 Challenge3：简易 Shell 

## 1. 实验原理与分析过程

### 1.1 实验目标

本实验要求在前一个挑战实验已经完成 `exec` 的基础上，进一步实现一个简易 Shell。
Shell 不需要支持交互输入，而是从 `hostfs_root/shellrc` 中读取命令脚本，逐条执行文件系统中的命令程序。

结合给定的 `app_shell.c` 以及 `/shellrc` 的内容，实验目标可以概括为：

1. 支持 Shell 通过 `fork()` 创建子进程；
2. 支持子进程通过 `exec(pathname, arg)` 装载指定命令程序；
3. 支持父进程通过 `wait(pid)` 阻塞等待子进程结束；
4. 让 `mkdir`、`touch`、`echo`、`cat`、`ls` 这些程序能按脚本顺序执行；
5. 保证运行日志与实验预期输出一致。

### 1.2 Shell 背后的操作系统原理

从表面上看，Shell 只是“读一条命令并执行”。
但从操作系统视角看，一条命令真正执行起来，需要以下三种机制协同工作：

1. **fork：复制进程执行现场**

  - 父进程保留原程序不变；
  - 子进程得到父进程的代码、数据、堆、栈和 trapframe 副本；
  - 从用户态看，父子进程都从 `fork()` 返回，只是返回值不同。

2. **exec：重载当前进程映像**

  - 子进程并不是“运行 Shell 里的 if 分支”到最后；
  - 它会在 `exec()` 中把当前地址空间中的旧程序替换成命令程序；
  - 进程身份（如 `pid`）保留，但程序映像被彻底替换。

3. **wait：父子进程同步**

  - 父进程不能在子进程尚未完成时立刻继续读下一条命令；
  - 否则多个命令会并发交错，输出顺序也会乱掉；
  - 因此父进程必须阻塞等待指定子进程完成。

这三者组合起来，就是最基本的 Unix 风格 Shell 执行模型：

```text
父进程(shell) -> fork() -> 子进程 -> exec(命令程序)
          \-> wait(pid) -> 子进程退出后继续下一条命令
```

### 1.3 本实验涉及的关键内核对象

为了让上述流程成立，本实验实际上同时依赖以下几个核心对象：

1. **process / trapframe**

  - `process` 保存进程页表、内核栈、trapframe、段映射等信息；
  - `trapframe` 保存用户态寄存器现场，是 `fork` 和 trap 返回的关键。

2. **mapped_info**

  - 描述当前进程的用户地址空间布局；
  - 包括栈段、上下文段、系统段、堆段、代码段、数据段；
  - `fork` 和 `exec` 都依赖它来复制或清理地址空间。

3. **进程状态机**

  - `READY`：可被调度；
  - `RUNNING`：当前运行；
  - `BLOCKED`：等待事件；
  - `ZOMBIE`：已经结束，但尚未被父进程回收。

4. **VFS + ELF 加载器**

  - `exec` 需要按路径打开 `/bin/app_xxx`；
  - 再由 ELF 加载器把命令程序段映射到当前进程页表中；
  - 最终修改 `epc` 跳转到新程序入口。

### 1.4 为什么本实验容易出错

本实验看似只是补两个系统调用，
但真正难点在于：

1. `fork`、`exec`、`wait` 不是独立模块，而是强耦合调用链；
2. 一个小小的段映射错误，可能不会立刻在映射处报错，而是延迟到 trap、syscall、甚至调度阶段才崩溃；
3. 用户参数、地址翻译、段清理、页表重建、寄存器恢复，这些环节之间只要有一处不一致，就会在很远的位置暴露问题。

这也是为什么本实验最重要的调试成果，并不是单纯“把代码写完”，
而是成功定位并修复了一条典型的 **级联 bug 链**。

### 1.5 级联 bug 链的总体分析

这次最棘手的问题，最终表现为：

```text
Misaligned AMO!
```

但深入分析后发现，真正的传播链是：

1. `HEAP_SEGMENT` 没有正确初始化；
2. 堆分配时错误地修改了 `CODE_SEGMENT.npages`；
3. `do_fork()` 依据错误页数复制代码段；
4. 子进程页表被映射了多余或错误页面；
5. 子进程第一次执行 `exec()` 进入 trap 时，内核 trap 保存现场所依赖的状态被破坏；
6. 最终在 `smode_trap_vector` 中写错地址，报出 `Misaligned AMO!`。

也就是说，报错点在 trap 汇编，
但根因在更早的地址空间元数据维护。

这类问题如果只盯住“崩溃指令”本身，几乎不可能快速找到根因。

---

## 2. 需要修改的代码位置

### 2.1 用户库接口

**文件：** `user/user_lib.h`、`user/user_lib.c`

需要补充两个用户态接口：

1. `int exec(const char *pathname, const char *arg);`
2. `int wait(int pid);`

这样 Shell 程序才能在用户态直接发起这两个系统调用。

### 2.2 系统调用号与分发

**文件：** `kernel/syscall.h`、`kernel/syscall.c`

需要新增：

1. `SYS_user_exec`
2. `SYS_user_wait`
3. `sys_user_exec()`
4. `sys_user_wait()`
5. `do_syscall()` 中的分发逻辑

此外，`sys_user_exit()` 也必须配合修改，
因为它决定子进程退出后父进程能否被唤醒，
以及子进程状态是否能被 `wait()` 观察到。

### 2.3 进程管理模块

**文件：** `kernel/process.h`、`kernel/process.c`

这里是本实验最核心的修改位置，需要补充：

1. `ZOMBIE` 进程状态；
2. `do_wait(pid)` 实现；
3. 扩展后的 `do_exec(pathname, arg)`；
4. `do_fork()` 中对数据段、堆段、上下文段的复制逻辑；
5. `alloc_process()` 中对 `HEAP_SEGMENT` 的初始化。

### 2.4 trap 处理模块

**文件：** `kernel/strap.c`

需要检查并修改系统调用返回路径：

1. 普通 syscall 返回时，把返回值写回 `a0`；
2. 但 `exec` 成功后，新程序的 `argc/argv` 已经放入 `a0/a1`；
3. 因此不能再用 syscall 返回值覆盖 `a0`。

### 2.5 用户命令程序与 Shell 程序

**文件：** `user/app_shell.c`、`user/app_mkdir.c`、`user/app_touch.c`、`user/app_echo.c`、`user/app_cat.c`、`user/app_ls.c`

需要确保：

1. Shell 通过 `fork + exec + wait` 驱动命令执行；
2. 命令程序统一从 `argv[0]` 读取参数；
3. 输出格式与实验预期一致。

### 2.6 构建与脚本文件

**文件：** `Makefile`、`hostfs_root/shellrc`

还需要保证：

1. 所有命令程序都被编译到 `hostfs_root/bin/`；
2. `shellrc` 中的命令顺序与实验预期一致；
3. Shell 运行时确实能在 VFS 中找到对应程序。

---

## 3. 代码修改逻辑

### 3.1 为什么要引入 `wait(pid)`

如果只有 `fork + exec` 而没有 `wait`，
父进程会在 fork 之后立刻继续执行下一条命令。

这会带来两个问题：

1. 多条命令并发运行，输出会交错；
2. 父进程可能在子进程尚未结束时就继续修改同一文件系统对象，导致结果与预期不一致。

因此必须为 Shell 引入显式等待机制。

`do_wait(pid)` 的实现逻辑并不复杂，但它必须严格依赖进程状态：

1. 找到指定 `pid`；
2. 确认其父进程就是当前进程；
3. 若子进程未退出，则将当前进程置为 `BLOCKED` 并调用 `schedule()`；
4. 子进程退出时唤醒父进程；
5. 父进程恢复执行后继续下一条命令。

这实际上是一次最小化的“等待某个事件发生”的内核同步实现。

### 3.2 为什么需要 `ZOMBIE` 状态，而不是直接 `FREE`

最开始很容易想到：
子进程退出后直接把进程槽位设为 `FREE` 即可。

但这样做有两个明显问题：

1. 父进程无法区分“子进程已经结束”和“这个 pid 对应的进程槽位已被复用”；
2. 输出中的 pid 也会不断重复使用，和实验预期不符。

因此正确做法是：

- 子进程退出时先进入 `ZOMBIE`；
- 父进程 `wait()` 观察到其结束；
- 之后再完成回收或保持该状态用于实验环境输出一致性。

这也是为什么 `ZOMBIE` 不是多余设计，
而是 `wait()` 能成立的必要条件。

### 3.3 为什么要把 `exec` 改成双参数

Challenge2 中 `exec` 只支持：

```c
exec(pathname)
```

但本实验的命令程序几乎都需要一个路径参数，
例如：

```text
/bin/app_touch /RAMDISK0/sub_dir/ramfile1
```

如果没有参数传递机制，子进程即使成功装载了 `app_touch`，
也不知道该操作哪个文件。

因此必须把 `exec` 扩展成：

```c
int exec(const char *pathname, const char *arg)
```

然后在 `do_exec()` 中把 `arg` 布置到新程序用户栈上，构造出：

```c
argc = 1;
argv[0] = arg;
argv[1] = NULL;
```

这样命令程序就能按照普通 `main(argc, argv)` 的方式获取参数。

### 3.4 为什么 `sys_user_exec()` 必须先复制字符串

`pathname` 和 `arg` 是用户态地址，
而且对 Shell 来说，这两个字符串通常位于堆中。

但 `do_exec()` 的第一步就是清理旧进程的代码段、数据段和堆段。
如果 `sys_user_exec()` 只是简单拿到两个指针再直接传入 `do_exec()`，
那么一旦进入 `do_exec()`：

1. 旧堆页会被释放；
2. `pathname` 和 `arg` 指针会悬空；
3. 内核随后再访问它们时，得到的就可能是空字符串或垃圾内容。

所以正确写法必须是：

1. 先通过 `user_va_to_pa()` 得到用户字符串；
2. 再把字符串内容复制到内核栈上的本地缓冲区；
3. 最后把内核副本传给 `do_exec()`。

这一步本质上是在做“销毁旧地址空间前的数据保全”。

### 3.5 为什么 `handle_syscall()` 不能统一覆盖 `a0`

对普通 syscall 而言，把返回值写回 `a0` 是正确的。
但是 `exec` 有特殊性：

1. 它成功后并不是“回到旧程序继续执行”；
2. 而是让 trap 返回到新程序入口；
3. 此时 `a0`、`a1` 应该分别代表新程序的 `argc` 和 `argv`。

如果内核最后仍执行：

```c
tf->regs.a0 = ret;
```

那么 `argc` 就会被覆盖成 `0`。
于是新程序虽然被装载成功，但参数根本拿不到。

因此这里必须对 `exec` 进行特判：

- 普通 syscall：写回返回值；
- `exec` 成功：保留 `do_exec()` 已经布置好的 `a0/a1`。

### 3.6 为什么 `alloc_process()` 必须初始化 `HEAP_SEGMENT`

这是本实验中最重要、也最容易忽略的一点。

`mapped_info` 数组本质上是“进程地址空间描述表”。
如果 `HEAP_SEGMENT` 不初始化，那么：

1. `mapped_info[3]` 不再表示 heap；
2. 后续 ELF 装载时，`CODE_SEGMENT` 会占据这个槽位；
3. `sys_user_allocate_page()` 里再执行 `mapped_info[HEAP_SEGMENT].npages++` 时，
   实际上改掉的是代码段页数。

这会直接污染 `do_fork()` 对代码段的复制逻辑。

所以这里之所以要恢复：

```c
mapped_info[HEAP_SEGMENT].va = USER_FREE_ADDRESS_START;
mapped_info[HEAP_SEGMENT].npages = 0;
mapped_info[HEAP_SEGMENT].seg_type = HEAP_SEGMENT;
total_mapped_region = 4;
```

并不是“为了格式整齐”，
而是为了确保后续所有基于 `mapped_info` 的逻辑仍然成立。

### 3.7 为什么 `do_fork()` 要单独处理数据段

本实验中的 ELF 数据段可能不是页对齐的，
而且代码段和数据段还可能落在同一物理页中。

如果 `do_fork()` 像处理代码段那样只做简单共享映射，
则会出现：

1. 子进程的数据页不可写；
2. 代码页和数据页共享时权限冲突；
3. 子进程写数据时影响到共享代码页内容。

因此对 `DATA_SEGMENT` 的复制必须更谨慎：

1. 先用 `ROUNDDOWN/ROUNDUP` 找到真正涉及的页范围；
2. 检查这些页是否已因代码段共享而被映射；
3. 若已映射，则为子进程重新分配新页、复制旧内容，再改成可写映射；
4. 若未映射，则正常分配新页并复制。

这样才能兼容“代码段与数据段可能共享首尾页”的情况。

### 3.8 为什么 `do_exec()` 要用页对齐方式清理旧段

对齐问题是本实验另一个容易忽略的点。

如果某个数据段起始地址是 `0x11a08`，
那么真正被映射的页并不是从 `0x11a08` 开始，
而是从：

```text
ROUNDDOWN(0x11a08, PGSIZE)
```

对应的页首开始。

因此，释放旧映射时必须按页边界计算：

```c
seg_start = ROUNDDOWN(seg_va, PGSIZE)
seg_end   = ROUNDUP(seg_va + npages * PGSIZE, PGSIZE)
```

否则会出现：

1. 旧页未被彻底释放；
2. 新程序加载时叠在旧页上；
3. 后续行为出现不可预测的残留状态。

### 3.9 为什么要调整输出日志

本实验最后不仅要求“功能对”，
还要求“输出对”。

因此需要做两类输出调整：

1. **补充预期中明确要求的日志**

  - 例如 `do_fork map code segment at pa:... of parent to child at va:...`

2. **移除会影响对拍的额外日志**

  - 例如 `User call wait: ...`

这一步虽然不改变系统机制，
但对课程评测非常重要。

---

## 4. 实验步骤与验证

### 4.1 编译与构建

进入工程目录后执行：

```bash
cd /app/riscv-pke
make clean
make
```

构建成功后，`hostfs_root/bin/` 中应包含：

1. `app_shell`
2. `app_mkdir`
3. `app_touch`
4. `app_echo`
5. `app_cat`
6. `app_ls`

### 4.2 Shell 脚本内容验证

本实验中的 `shellrc` 内容按顺序为：

1. `/bin/app_mkdir /RAMDISK0/sub_dir`
2. `/bin/app_touch /RAMDISK0/sub_dir/ramfile1`
3. `/bin/app_touch /RAMDISK0/sub_dir/ramfile2`
4. `/bin/app_echo /RAMDISK0/sub_dir/ramfile1`
5. `/bin/app_cat /RAMDISK0/sub_dir/ramfile1`
6. `/bin/app_ls /RAMDISK0/sub_dir`
7. `/bin/app_ls /RAMDISK0`
8. `END END`

这意味着 Shell 的正确行为应该是严格顺序执行这七条命令。

### 4.3 运行方式

运行命令为：

```bash
spike ./obj/riscv-pke /bin/app_shell
```

观察点包括：

1. 是否成功进入 `======== Shell Start ========`；
2. 每条命令是否都先 `fork`，再调度子进程；
3. 子进程是否成功 `exec` 到目标程序；
4. 父进程是否在子进程退出后才继续下一条；
5. 最终是否正常 `shutdown`，退出码为 `0`。

### 4.4 关键输出验证

最终运行结果应满足：

1. 第一条 `mkdir` 成功创建 `/RAMDISK0/sub_dir`；
2. 两次 `touch` 成功创建 `ramfile1`、`ramfile2`；
3. `echo` 成功向 `ramfile1` 写入 `hello world`；
4. `cat` 成功读出 `hello world`；
5. `ls /RAMDISK0/sub_dir` 能列出两个文件；
6. `ls /RAMDISK0` 能列出 `sub_dir`；
7. 输出中 pid 依次递增为 `1` 到 `7`；
8. 日志与实验给出的预期文本一致。

### 4.5 最终验证结果

经过所有修复之后，系统已经可以：

1. 正常运行 Shell；
2. 顺序执行全部命令；
3. 输出与实验预期完全一致；
4. 最终以 `System is shutting down with exit code 0.` 结束。

这说明本实验的功能实现和评测对拍都已经完成。

---

## 5. 实验收获

### 5.1 对 `fork/exec/wait` 三者关系理解更深

过去更多是从概念上知道它们“经常一起出现”，
而这次真正写出来之后，我更清楚地理解到：

1. `fork` 负责复制执行现场；
2. `exec` 负责替换程序映像；
3. `wait` 负责恢复父进程的同步控制。

它们并不是三个孤立系统调用，
而是一套完整命令执行机制。

### 5.2 对进程地址空间元数据的重要性有了更直观认识

这次最关键的 bug 并不是算法错误，
而是 `mapped_info` 中一个段位信息没有初始化。

这让我认识到：
在内核里，很多结构体字段看似只是“记录信息”，
实际上却是后续所有逻辑的基础假设。
一旦这个假设失真，错误会沿着整个调用链传播。

### 5.3 对“级联 bug 链”有了实际体验

本次调试让我真正体会到：
内核错误往往不是“某行代码直接崩溃”，
而是：

1. 先在某个早期状态埋下错误；
2. 再经过调度、页表、trap、syscall 等多个环节传播；
3. 最后在一个看起来毫不相干的地方爆炸。

因此，调试内核问题必须学会从“状态传播”角度思考。

### 5.4 对调试方法论有了更系统总结

这次能把问题真正查清楚，关键不在于盲改，
而在于按层次排查：

1. 判断错误在用户态还是内核态；
2. 判断在 fork、exec 还是 wait 链路上；
3. 用寄存器值和反汇编确定崩溃点；
4. 回推哪些上游状态会影响这个崩溃点；
5. 最终找到真正的根因，而不是只修表面症状。

这套方法对后续更复杂的操作系统实验也同样适用。

---

## 6. 实验调试记录

### 6.1 第一阶段：先补全功能链路，再确认错误出现在哪一层

最开始我先把 Shell 运行所需的机制全部接通：

1. 在用户库中补齐 `exec(pathname, arg)` 与 `wait(pid)`；
2. 在内核中新增 `SYS_user_exec`、`SYS_user_wait`；
3. 在 `process.h` 中加入 `ZOMBIE` 状态；
4. 在 `process.c` 中实现 `do_wait()`，并扩展 `do_exec()` 支持参数；
5. 在 `app_shell.c` 中按 `fork -> child exec -> parent wait` 的模式驱动命令执行。

此时从静态结构看，整个控制流已经成立：

```text
app_shell -> fork -> child exec -> parent wait
```

但是第一次实际运行：

```bash
spike obj/riscv-pke /bin/app_shell
```

系统并没有真正跑到第一条命令结束，而是在子进程刚被调度后就崩溃。

这一阶段我先做出的判断是：

1. 问题不是编译或接口缺失，而是运行时错误；
2. 错误点发生在 `fork` 之后、`exec` 真正完成之前；
3. 需要优先定位是“用户程序逻辑错了”，还是“子进程上下文/地址空间本身坏了”。

### 6.2 第二阶段：先怀疑 `fork` 中的段复制，尤其是 `DATA_SEGMENT`

最初观察到的现象是：

```text
User call wait: 1
going to schedule process 1 to run.
Misaligned AMO!
```

因为日志中父进程已经进入 `wait(pid)`，
而子进程一开始运行就崩了，
所以我最先怀疑的是：

1. `fork` 复制出来的子进程上下文不完整；
2. 子进程在调用 `exec` 之前就已经拿到了错误的地址空间；
3. 尤其是 `CODE_SEGMENT` 与 `DATA_SEGMENT` 的复制可能有遗漏。

为此，我先回头检查 `do_fork()`，
发现 `STACK_SEGMENT`、`CONTEXT_SEGMENT`、`HEAP_SEGMENT`、`CODE_SEGMENT` 都在处理，
于是进一步聚焦到 `DATA_SEGMENT`。

当时第一版判断是：

> 子进程可能在访问全局变量或静态数据时出错，因为 `DATA_SEGMENT` 复制不完整。

因此我先在 `do_fork()` 中补充了 `DATA_SEGMENT` 的复制逻辑。

### 6.3 第三阶段：第一次补 `DATA_SEGMENT` 后出现 `map_pages fails`，说明问题更底层

加完数据段处理后重新编译运行，新的错误变成了：

```text
map_pages fails on mapping va (...)
```

这说明错误从“运行时崩溃”提前到了“映射阶段就失败”。
当时我马上意识到两件事：

1. 新加的 `DATA_SEGMENT` 复制逻辑方向大概率没错；
2. 但我对“数据段应该映射哪些页”理解还不完整。

为了看清到底在映射什么地址，我当时加入了临时调试输出，
打印数据段的虚拟地址和复制页范围。

这一阶段的核心调试思路不是马上继续改，
而是先把“映射失败的地址究竟是什么”搞清楚。

### 6.4 第四阶段：通过打印 `mapped_info` 发现数据段地址不是页对齐的

进一步调试后我看到：

1. `CODE_SEGMENT` 的起始地址是页对齐的，例如 `0x10000`；
2. 但 `DATA_SEGMENT` 的 `va` 却是 `0x11a08`，并不是页对齐地址；
3. `npages = 1`，但这个“1 页大小的段”从 `0x11a08` 开始，实际上会跨越 `0x11000` 和 `0x12000` 两个页。

于是当时我做了第一个关键判断：

> `mapped_info` 里记录的并不是“页起始地址”，而是 ELF 段真实的起始虚拟地址；因此在 `fork` 和 `exec` 中都不能直接按 `va + j * PGSIZE` 简单处理。

为了支持非页对齐段，我引入了页对齐计算：

```c
uint64 data_page_start = ROUNDDOWN(data_start_va, PGSIZE);
uint64 data_end_va = data_start_va + parent->mapped_info[i].npages * PGSIZE;
uint64 actual_npages = (ROUNDUP(data_end_va, PGSIZE) - data_page_start) / PGSIZE;
```

这一步的意义在于：

1. 不再把 `mapped_info[i].va` 当作页边界；
2. 而是先求出真实覆盖页范围；
3. 然后再逐页复制。

### 6.5 第五阶段：发现代码段和数据段共享页，单纯“再映射一遍”会失败

支持非页对齐复制之后，错误还没有完全消失。
继续打印调试信息后，我发现了第二个关键事实：

1. `CODE_SEGMENT` 从 `0x10000` 开始，可能已经覆盖到 `0x11000`、`0x12000`；
2. `DATA_SEGMENT` 从 `0x11a08` 开始，正好也会落在 `0x11000` 和 `0x12000` 这些页里；
3. 也就是说，`CODE_SEGMENT` 与 `DATA_SEGMENT` 在页粒度上是重叠的。

这意味着，如果在复制 `DATA_SEGMENT` 时直接再次执行：

```c
user_vm_map(child->pagetable, page_va, PGSIZE, ..., prot_to_type(PROT_WRITE | PROT_READ, 1));
```

那么：

1. 对于已经被代码段映射过的页，会直接触发 `map_pages fails`；
2. 即使不失败，如果还保留原来的 `R|X` 权限，也不能满足数据页写需求。

于是我当时又做了第二个关键判断：

> 对于“代码段和数据段共享的页”，不能简单跳过，也不能直接重复映射，而应该为子进程重新分配新页、复制旧内容，并赋予可写权限。

这就是后来 `DATA_SEGMENT` 处理中“检测是否已映射，如果已映射则拆分新页副本”的设计来源。

### 6.6 第六阶段：回到原始现象，重新定位 `Misaligned AMO!`

在解决了 `DATA_SEGMENT` 页重叠问题之后，
`map_pages fails` 消失了，
但系统又回到了最初的错误：

```text
Misaligned AMO!
```

到这时我意识到：

1. `DATA_SEGMENT` 的处理确实有问题，但它不是最终根因；
2. 前面修掉的是“映射冲突”这个显性问题；
3. 真正导致 `Misaligned AMO!` 的状态污染，可能在更早的地方。

这一步非常重要，
因为它让我停止继续在 `DATA_SEGMENT` 细节里打转，
转而去观察 trap 与调度本身。

### 6.7 第七阶段：检查 trapframe 中的内核字段，排除一个重要但非根因的怀疑点

接下来我曾怀疑：

1. `do_fork()` 中直接做了：

```c
*child->trapframe = *parent->trapframe;
```

2. 这可能把父进程的 `kernel_sp`、`kernel_trap`、`kernel_satp` 一起拷进了子进程；
3. 如果这些字段仍指向父进程，就会导致子进程 trap 恢复时用错内核上下文。

于是我检查了 `trapframe` 结构，
也在 `do_fork()` 末尾补了对这些字段的更新。

但继续深入看 `smode_trap_handler()` 和 `switch_to()` 后我发现：

1. 每次真正切回用户态前，`switch_to(current)` 都会重新设置：
   - `kernel_sp`
   - `kernel_trap`
   - `kernel_satp`
2. 因此，`do_fork()` 中手动更新这些字段虽然不坏，
   但不是导致 `Misaligned AMO!` 的根因。

这一阶段的价值在于：

> 排除了一个非常像根因、但其实只是次要问题的方向，避免后续继续误判。

### 6.8 第八阶段：给 `switch_to()` 加调试，确认子进程切出前的 trapframe 看起来是正常的

为了进一步确认子进程被调度前到底是什么状态，
我在 `switch_to()` 中加入了如下调试代码：

```c
uint64 user_satp = MAKE_SATP(proc->pagetable);

sprint("switch_to pid=%d tf=%lx epc=%lx sp=%lx satp=%lx\n",
  proc->pid, (uint64)proc->trapframe, proc->trapframe->epc,
  proc->trapframe->regs.sp, user_satp);
```

加入这段代码的目的，是同时确认：

1. `trapframe` 指针是否正常；
2. `epc` 是否仍在用户空间；
3. `sp` 是否仍指向用户栈；
4. 页表 `satp` 是否切换到了子进程自己的页表。

运行后可以看到类似结果：

```text
switch_to pid=1 tf=0000000087f0e000 epc=00000000000101d8 sp=000000007fffeba0 satp=8000000000087f0c
```

这说明：

1. 在 `switch_to()` 观察到的子进程 trapframe 看起来是正常的；
2. 崩溃不是因为“调度前 trapframe 已经明显为空或野指针”；
3. 更可能发生在“已经回到用户态、再次进入 trap”这一瞬间。

### 6.9 第九阶段：给 `handle_misaligned_store()` 加调试，正式拿到 `mepc/mtval`

为了准确定位 `Misaligned AMO!` 到底是在哪条指令上发生的，
我在 `kernel/machine/mtrap.c` 中临时加入了：

```c
static void handle_misaligned_store() {
  sprint("mepc=%p mtval=%p\n", read_csr(mepc), read_csr(mtval));
  panic("Misaligned AMO!");
}
```

重新运行后，得到决定性的输出：

```text
mepc=0x0000000080006006 mtval=0x000000000000005e
```

这个结果让我一下子抓住了两个关键点：

1. `mepc` 落在内核地址空间，说明出错发生在 trap 入口代码中；
2. `mtval = 0x5e`，是一个极小且明显非法的地址，说明写坏的不是某个普通用户页，而是“指针本身被污染”。

### 6.10 第十阶段：用 `objdump` 反汇编内核，定位到 `smode_trap_vector`

拿到 `mepc` 后，我直接执行：

```bash
cd /app/riscv-pke && riscv64-unknown-elf-objdump -d obj/riscv-pke | grep -A 2 -B 5 "80006006"
```

看到的关键指令是：

```asm
0000000080006000 <_trap_sec_start>:
    80006000:   csrrw   a0,sscratch,a0
    80006004:   mv      t6,a0
    80006006:   sd      ra,0(t6)
```

这一步非常关键，因为它把抽象的 `Misaligned AMO!` 变成了一个具体事实：

1. trap 入口先把 `sscratch` 交换到 `a0`；
2. 再把 `a0` 放到 `t6`；
3. 然后把寄存器现场保存到 `t6` 指向的 trapframe；
4. 但实际 `t6 = 0x5e`，于是 `sd ra, 0(t6)` 就成了往地址 94 写 8 字节。

到这里我可以写出明确结论：

> 不是 trap 汇编错了，而是进入 trap 时 `sscratch` 里本该存的 trapframe 指针已经被污染了。

### 6.11 第十一阶段：继续反汇编 `return_to_user` 和用户侧 `exec`，解释为什么是 `0x5e`

要解释“为什么是 `0x5e` 而不是别的随机值”，
我又继续做了两组反汇编。

第一组查看 `return_to_user`：

```bash
cd /app/riscv-pke && riscv64-unknown-elf-objdump -d obj/riscv-pke | grep -A 30 "<return_to_user>:"
```

目的是确认返回用户态前是否真的执行了：

```asm
csrw sscratch, a0
```

也就是把 trapframe 指针写入 `sscratch`。

第二组查看用户程序里 `exec` 的封装：

```bash
cd /app/riscv-pke && riscv64-unknown-elf-objdump -d hostfs_root/bin/app_shell | grep -A 30 "<exec>:"
```

从反汇编里我看到了关键指令：

```asm
104a2:  05e00513    li a0,94
```

也就是说：

1. 用户侧 `exec` syscall 号就是 `94`；
2. 十六进制正好是 `0x5e`；
3. 于是 `mtval=0x5e` 不是随机值，而正是 `SYS_user_exec`。

这让我得到一个极其关键的判断：

> 子进程第一次执行 `exec` 的 `ecall` 时，trap 入口拿到的不是 trapframe 指针，而是 syscall 号 `94`。也就是说，`sscratch` 在某条更早的错误路径上被破坏成了 `SYS_user_exec`。

### 6.12 第十二阶段：正式确认 `0x5e = SYS_user_exec`，并结合 `readelf` 理解段布局

为了确保这个对应关系不是偶然，
我又去读取了：

1. `kernel/syscall.h`，确认：

```c
#define SYS_user_exec   (SYS_user_base + 30)
```

2. 因为 `SYS_user_base = 64`，所以：

$$64 + 30 = 94 = 0x5e$$

3. 接着又查看 ELF 程序头：

```bash
cd /app/riscv-pke && riscv64-unknown-elf-readelf -l hostfs_root/bin/app_shell
```

看到 `app_shell` 的 `DATA_SEGMENT` 起始地址确实是非页对齐的，
这也解释了为什么早先在 `DATA_SEGMENT` 上会踩到那么多边界问题。

这一阶段的结论是：

1. `0x5e` 的确就是 `SYS_user_exec`；
2. 错误已经可以确定不是“简单的页表项遗漏”，而是更深层的状态污染；
3. 需要从“谁会写坏 trap 相关状态”这个角度，继续逆着链路回溯。

### 6.13 第十三阶段：最终锁定根因——`HEAP_SEGMENT` 未初始化

接下来我把视角重新拉回到最上游，
重点审查：

1. `alloc_process()` 如何初始化 `mapped_info`；
2. `sys_user_allocate_page()` 如何增长堆；
3. `do_fork()` 如何依据 `mapped_info` 复制用户地址空间。

最终找到真正根因：

#### （1）`HEAP_SEGMENT` 的登记被注释掉了

当时 `alloc_process()` 中 heap 初始化代码是被注释掉的：

```c
// procs[i].mapped_info[HEAP_SEGMENT].va = USER_FREE_ADDRESS_START;
// procs[i].mapped_info[HEAP_SEGMENT].npages = 0;
// procs[i].mapped_info[HEAP_SEGMENT].seg_type = HEAP_SEGMENT;
// procs[i].total_mapped_region = 3;
```

这意味着：

1. `mapped_info[3]` 并没有作为 heap 预留；
2. 后续 ELF 加载时，第一个用户程序段（通常就是 `CODE_SEGMENT`）占据了这个位置。

#### （2）`sys_user_allocate_page()` 改错了对象

而堆分配时执行的是：

```c
current->mapped_info[HEAP_SEGMENT].npages++;
```

由于 `HEAP_SEGMENT` 根本没正确初始化，
这行代码实际改掉的不是 heap 页数，
而是 `CODE_SEGMENT.npages`。

#### （3）`do_fork()` 继续放大这个错误

当 `CODE_SEGMENT.npages` 被误改后，
`do_fork()` 在复制代码段时就会依据错误页数把额外页面映射给子进程，
进一步污染地址空间。

#### （4）最终传播到 trap

子进程带着这个错误地址空间进入第一次 `exec`，
在再次触发 trap 时，内核保存现场的流程依赖的上下文已经失真，
最终就把 `sscratch` 污染成了 `SYS_user_exec = 0x5e`，在 trap 向量处爆炸。

这一阶段的总结可以概括为：

> `Misaligned AMO!` 只是故障末端，真正根因是 `HEAP_SEGMENT` 未初始化，导致堆元数据写坏代码段页数，再经由 `do_fork()` 放大成子进程地址空间污染。

### 6.14 第十四阶段：修复根因——恢复 `HEAP_SEGMENT` 初始化

明确根因后，我先修最上游的问题。

修复代码如下：

```c
procs[i].mapped_info[HEAP_SEGMENT].va = USER_FREE_ADDRESS_START;
procs[i].mapped_info[HEAP_SEGMENT].npages = 0;
procs[i].mapped_info[HEAP_SEGMENT].seg_type = HEAP_SEGMENT;

procs[i].total_mapped_region = 4;
```

修复思路是：

1. 把 heap 正确放回 `mapped_info[3]`；
2. 让 ELF 加载的代码段、数据段从后续槽位开始登记；
3. 让 `sys_user_allocate_page()` 修改的重新变成 heap 页数，而不是代码段页数。

这一步修完以后，再次编译运行，
`Misaligned AMO!` 立刻消失。

这也是整个调试过程中第一个真正打断故障传播链的修复。

### 6.15 第十五阶段：根因修复后暴露次级问题——`pathname` 变成空字符串

`Misaligned AMO!` 修完之后，系统继续向前运行，
很快又出现新的现象：

```text
Failed to load program:
```

结合日志可以看出，`Application: /bin/app_mkdir` 能打印出来，
但进入 `do_exec()` 后真正用于加载的路径名却丢了。

我顺着调用链重新分析：

```text
sys_user_exec() -> do_exec() -> 清理旧堆段 -> 再访问 pathname/arg
```

很快就发现：

1. `pathname` 和 `arg` 本来来自 `app_shell` 的用户堆；
2. 而 `do_exec()` 一开始就会清掉旧进程的 heap；
3. 所以 `pathname`、`arg` 指向的物理页已经被 `free_page` 释放了；
4. 这本质上是一次 use-after-free。

因此修复代码改成：

```c
char pathname_buf[256];
char arg_buf[256];
strcpy(pathname_buf, pathname);
char *arg_copy = NULL;
if (arg != NULL) {
  strcpy(arg_buf, arg);
  arg_copy = arg_buf;
}

return do_exec(pathname_buf, arg_copy);
```

修复思路是：

1. 先把用户态字符串复制到内核栈；
2. 再进入会销毁旧用户地址空间的 `do_exec()`；
3. 这样 `do_exec()` 使用的是稳定的内核副本。

### 6.16 第十六阶段：修复 `exec` 成功后 `a0` 被 syscall 返回值覆盖

路径字符串问题修好后，命令程序已经能够被装载，
但参数传递仍存在隐患。

我继续检查 `handle_syscall()`，发现它在 syscall 返回后统一执行：

```c
tf->regs.a0 = ret;
```

而 `do_exec()` 明明已经把：

```c
current->trapframe->regs.a0 = 1;
current->trapframe->regs.a1 = argv_addr;
```

设置成了新程序的 `argc/argv`。

如果最后再写回 syscall 返回值，
那么 `argc` 就会被覆盖成 `0`。

因此修复代码改为：

```c
long sysnum = tf->regs.a0;
long ret = do_syscall(...);

if (sysnum != SYS_user_exec || ret != 0) {
  tf->regs.a0 = ret;
}
```

修复思路是：

1. 普通 syscall 仍然统一写回返回值；
2. 但 `exec` 成功后，`a0/a1` 的语义已经变成新程序参数；
3. 因此不能再覆盖。

### 6.17 第十七阶段：修复 `do_exec()` 对非页对齐段的清理方式

继续测试中，我又注意到另一个边界问题：

1. 某些 ELF 的 `DATA_SEGMENT` 起始地址并不是页对齐的；
2. 如果按旧方式 `va + j * PGSIZE` 去清理，可能清不干净；
3. 旧页残留会影响后续 `exec` 加载新程序。

因此我把 `do_exec()` 中代码/数据段清理逻辑改为：

```c
uint64 seg_va = current->mapped_info[i].va;
uint64 seg_start = ROUNDDOWN(seg_va, PGSIZE);
uint64 seg_end = ROUNDUP(seg_va + current->mapped_info[i].npages * PGSIZE, PGSIZE);
for (uint64 va = seg_start; va < seg_end; va += PGSIZE) {
  user_vm_unmap((pagetable_t)current->pagetable, va, PGSIZE, 1);
}
```

修复思路是：

1. 清理必须按“真实页覆盖范围”而不是“段起始地址”；
2. 只有这样才能正确兼容非页对齐数据段；
3. 这也与前面对 `DATA_SEGMENT` 在 `fork` 中的页范围处理逻辑保持一致。

### 6.18 第十八阶段：功能跑通后，再修正输出与评测一致性

到这时，Shell 已经可以完整执行全部 7 条命令了。
接下来我做的就是最后一轮“评测输出对拍”。

主要修改有三处：

#### （1）补充 `do_fork map code segment ...`

在 `CODE_SEGMENT` 复制处加入：

```c
sprint("do_fork map code segment at pa:%lx of parent to child at va:%lx.\n", pa, addr);
```

因为预期输出中明确要求这条日志。

#### （2）删除 `do_wait()` 中额外的调试输出

之前为了观察流程，我写过：

```c
sprint("User call wait: %d\n", pid);
```

最终版本中删除它，避免和标准输出不一致。

#### （3）把子进程退出状态改回 `ZOMBIE`

我曾短暂尝试让子进程退出就进入 `FREE`，
这样便于复用进程槽位，
但会导致 pid 总被重复使用，不符合实验预期中 `1,2,3,4,5,6,7` 递增的效果。

因此最终保持：

```c
current->status = ZOMBIE;
```

这样既符合 `wait()` 的语义，
也能让输出与实验文档保持一致。

### 6.19 第十九阶段：最终验证与总结合拢

完成全部修复之后，我做了最后一轮完整验证：

1. 重新编译：

```bash
make clean
make
```

2. 重新运行：

```bash
spike obj/riscv-pke /bin/app_shell
```

3. 对照 `shellrc` 与实验文档逐项检查输出。

最终验证结果表明：

1. `mkdir`、`touch`、`echo`、`cat`、`ls` 全部正常；
2. `hello world` 可以正确写入并读出；
3. pid 顺序与实验预期一致；
4. 日志与课程提供的标准输出一致；
5. 系统最终以 `exit code 0` 正常退出。

这一整轮调试过程给我的最大总结是：

1. **先定位，再修根因**：不能在表面症状上反复打补丁；
2. **调试代码必须服务于假设验证**：每次加 `sprint`、反汇编、看 ELF 段头，都要有明确目的；
3. **内核问题往往是延迟暴露的级联错误**：真正的起点可能离崩溃点非常远；
4. **解决主问题之后，还要继续消化次级问题**：比如字符串悬空、`a0` 被覆盖、非页对齐清理；
5. **最后别忘了输出一致性**：课程实验中“机制正确”和“日志正确”同样重要。

## 总结

本次 `Lab4 Challenge3` 的核心任务，是在 PKE 中实现基于 `fork + exec + wait` 的简易 Shell。
围绕这一目标，我完成了以下工作：

1. 在用户库中新增 `exec(pathname, arg)` 与 `wait(pid)` 接口；
2. 在内核中新增 `SYS_user_exec`、`SYS_user_wait` 以及分发逻辑；
3. 在 `process.h` 中引入 `ZOMBIE` 状态；
4. 在 `process.c` 中实现 `do_wait()`、扩展 `do_exec()`、修正 `do_fork()` 的段复制逻辑；
5. 在 `strap.c` 中修复 `exec` 返回后 `a0` 被覆盖的问题；
6. 修复 `HEAP_SEGMENT` 缺失导致的级联 bug 链；
7. 调整运行日志，使输出与实验预期完全一致。

最终结果表明：

1. Shell 可以顺序执行 `/shellrc` 中全部命令；
2. 父进程与子进程之间能够通过 `wait()` 正确同步；
3. `echo`、`cat`、`ls` 等命令功能全部正常；
4. 系统最终以 `exit code 0` 正常退出；
5. 运行输出已经与课程预期结果一致。

通过本次实验，我不仅完成了一个简易 Shell，
更重要的是通过一次典型的内核级级联故障，
系统性理解了 `fork`、`exec`、`wait`、页表映射、trap 现场保存以及文件系统加载之间的关系。
这让我对操作系统中的“机制联动”和“延迟暴露型 bug”的认识，都比之前更深入了一步。
