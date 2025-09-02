#AUIPC≤‚ ‘    revise date:2021/1/24 tiger   compact mode£∫ Text at  Address 0
#“¿¥Œ ‰≥ˆ 0x00430004 0x00430014 0x00430024 0x00430034 0x00430044 0x00430054 0x00430064 0x00430074

.text
nop
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
auipc s1,0x430
add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print
addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   

