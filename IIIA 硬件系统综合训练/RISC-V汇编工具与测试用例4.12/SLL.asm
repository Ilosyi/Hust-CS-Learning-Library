#sll   revise date:2018/3/12 tiger
#ÒÀ´ÎÊä³ö  0x00000876 0x00008760 0x00087600 0x00876000 0x08760000 0x87600000 0x76000000 0x60000000 0x00000000

.text

addi t0,zero,1     
addi t1,zero,3     
addi s1,zero,  0x8
slli s1,s1,8
addi s1,s1,0x76     

add a0,zero,s1           
addi a7,zero,34         # system call for print
ecall                  # print

addi t3,zero,8

sll_branch:
sll s1,s1,t0       #1
sll s1,s1,t1       #æµ‹è¯•
add a0,zero,s1          
addi a7,zero,34         # system call for print
ecall                  # print
addi t3,t3, -1    
bne t3,zero,sll_branch

addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   





