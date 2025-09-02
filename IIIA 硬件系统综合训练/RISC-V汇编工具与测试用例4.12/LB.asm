#LB测试    revise date:2021/1/24 tiger
#依次输出 0xffffff81 0xffffff82 0xffffff83 0xffffff84 0xffffff85 0xffffff86 0xffffff87 0xffffff88 0xffffff89 0xffffff8a 0xffffff8b 0xffffff8c 0xffffff8d 0xffffff8e 0xffffff8f 0xffffff90 0xffffff91 0xffffff92 0xffffff93 0xffffff94 0xffffff95 0xffffff96 0xffffff97 0xffffff98 0xffffff99 0xffffff9a 0xffffff9b 0xffffff9c 0xffffff9d 0xffffff9e 0xffffff9f 0xffffffa0
.text
addi t1,zero,0     #init_addr 
addi t3,zero,16     #counter
nop

#预先写入数据，实际是按字节顺序存入 0x81,82,84,86,87,88,89.......等差数列
addi s1,zero,  0x84
slli s1,s1,8
addi s1,s1,0x83 
addi s2,zero,  0x04
slli s2,s2,8
addi s2,s2,0x04
slli s1,s1,8
addi s1,s1,0x82 
slli s1,s1,8
addi s1,s1,0x81
slli s2,s2,8
addi s2,s2,0x04
slli s2,s2,8
addi s2,s2,0x04    #    init_data= 0x84838281 next_data=init_data+ 0x04040404

lb_store:
sw s1,(t1)
add s1,s1,s2   #data +1
addi t1,t1,4    # addr +4  
addi t3,t3,-1   #counter
bne t3,zero,lb_store

addi t3,zero,32   #循环次数
addi t1,zero,0    # addr 
lb_branch:
lb s1,(t1)         #测试指令
add a0,zero,s1          
addi a7,zero,34         #输出
ecall                  
addi t1,t1, 1    
addi t3,t3, -1    
bne t3,zero,lb_branch



addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   
