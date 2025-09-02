#sra移位测试    revise date:2021/1/24 tiger
#依次输出  0x87600000 0xf8760000 0xff876000 0xfff87600 0xffff8760 0xfffff876 0xffffff87 0xfffffff8 0xffffffff

.text

addi t0,zero,1     #sll 移位次数
addi t1,zero,3     #sll 移位次数
addi s1,zero,  0x8
slli s1,s1,8
addi s1,s1,0x76     

slli s1,s1,20     #

add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print

addi t3,zero,8

sra_branch:
sra s1,s1,t0     #先移1位
sra s1,s1,t1     #再移3位
add a0,zero,s1          
addi a7,zero,34         # system call for print
ecall                  # print
addi t3,t3, -1    
bne t3,zero,sra_branch   #循环8次

addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   
