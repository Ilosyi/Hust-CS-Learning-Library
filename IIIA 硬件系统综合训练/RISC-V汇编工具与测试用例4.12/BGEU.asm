
#bgeu 测试    大于跳转  递减运算 ，从正数开始向零运算  revise date:2022/2/24 tiger
#依次输出0x0000000f 0x0000000e 0x0000000d 0x0000000c 0x0000000b 0x0000000a 0x00000009 0x00000008 0x00000007 0x00000006 0x00000005 0x00000004 0x00000003 0x00000002 0x00000001 
.text


addi s1,zero,15  
addi s2,zero,1

bgeu_branch:
sub s3,s2,s1
add a0,zero,s1        
addi a7,zero,34         
ecall                  # 输出当前值
addi s1,s1,-1  
bgeu s3,s2,bgeu_branch    #当前测试指令


addi   a7,zero,10         
ecall                  # 程序暂停或退出
