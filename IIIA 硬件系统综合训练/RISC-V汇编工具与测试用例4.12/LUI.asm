#LUI≤‚ ‘    revise date:2021/1/24 tiger
#“¿¥Œ ‰≥ˆ  0xfedcffff 0x0ba98000 0x07654000 0x03210000 0xfedcffff 0x0ba98000 0x07654000 0x03210000 0xfedcffff 0x0ba98000 0x07654000 0x03210000 0xfedcffff 0x0ba98000 0x07654000 0x03210000 0xfedcffff 0x0ba98000 0x07654000 0x03210000 0xfedcffff 0x0ba98000 0x07654000 0x03210000 0xfedcffff 0x0ba98000 0x07654000 0x03210000 0xfedcffff 0x0ba98000 0x07654000 0x03210000

.text

addi t3,zero,   0x8

lui_branch:
lui s1,   0xFEDC0 
addi s0,zero, -1
srli  s0,s0,16
or s1,s1,s0

add a0,zero,s1          
addi a7,zero,34         # system call for print
ecall    
lui s1,   0xBA98
add a0,zero,s1          
ecall    
lui s1,   0x7654     
add a0,zero,s1          
ecall    
lui s1,   0x3210     
add a0,zero,s1          
ecall    
                           # print
addi t3,t3, -1    
bne t3,zero,lui_branch

addi   a7,zero,10         # system call for exit
ecall                     # we are out of here.   
