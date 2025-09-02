#REMU≤‚ ‘    revise date:2021/1/24 tiger
# ‰≥ˆ 0x87540110

.text
addi s1,x0,0x88
slli s1,s1,8
addi s1,s1,0x48
addi s2,x0,0x87
slli s2,s2,8
addi s2,s2,0x54
remu s2,s2,s1
slli s2,s2,16
ori s2,s2,0x110


add a0,zero,s2
addi a7,zero,34         # system call for print
ecall         
          
                                    # print
addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   

