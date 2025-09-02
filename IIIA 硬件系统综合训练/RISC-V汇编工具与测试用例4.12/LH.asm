#LH 测试    revise date:2021/1/24 tiger
#依次输出  0xffff8281 0xffff8483 0xffff8685 0xffff8887 0xffff8a89 0xffff8c8b 0xffff8e8d 0xffff908f 0xffff9291 0xffff9493 0xffff9695 0xffff9897 0xffff9a99 0xffff9c9b 0xffff9e9d 0xffffa09f 0xffffa2a1 0xffffa4a3 0xffffa6a5 0xffffa8a7 0xffffaaa9 0xffffacab 0xffffaead 0xffffb0af 0xffffb2b1 0xffffb4b3 0xffffb6b5 0xffffb8b7 0xffffbab9 0xffffbcbb 0xffffbebd 0xffffc0bf
.text

addi t1,zero,0     #init_addr 
addi t3,zero,16     #counter

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


lh_store:
sw s1,(t1)
add s1,s1,s2   #data +1
addi t1,t1,4    # addr +4  
addi t3,t3,-1   #counter
bne t3,zero,lh_store

addi t3,zero,32
addi t1,zero,0    # addr  
lh_branch:
lh s1,(t1)     #测试指令
add a0,zero,s1          
addi a7,zero,34         
ecall                  # print
addi t1,t1, 2    
addi t3,t3, -1    
bne t3,zero,lh_branch

addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   
