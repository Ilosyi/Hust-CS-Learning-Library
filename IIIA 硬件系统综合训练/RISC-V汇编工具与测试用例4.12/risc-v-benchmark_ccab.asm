#############################################################
#测试jal,jalr指令，
#############################################################
.text
 addi s1,zero, 1   #测试jal,jalrr指令
 j jmp_next1       #jal x0,8
 addi s1,zero, 1
 addi s2,zero, 2
 addi s3,zero, 3
jmp_next1:
 j jmp_next2
 addi s1,zero, 1
 addi s2,zero, 2
 addi s3,zero, 3
jmp_next2:
 j jmp_next3
 addi s1,zero, 1
 addi s2,zero, 2
 addi s3,zero, 3
jmp_next3:
 j jmp_next4
 addi s1,zero, 1
 addi s2,zero, 2
 addi s3,zero, 3
jmp_next4:jal jmp_count

#移位测试  需要支持超addi,sll,add,ecall,srl,sll,sra,beq,j,ecall    revise date:2015/12/16 tiger

.text
addi s0,zero,1     #简单移位，循环测试，0号区域显示的是初始值1左移1位重复15次的值，1号区域是累加值
addi s1,zero,1  
slli s1, s1, 31   #逻辑左移31位 s1=0x80000000
 

###################################################################
#                逻辑左移测试 
# 显示区域依次显示0x80000000 0x20000000 0x08000000 0x02000000 0x00800000 0x00200000 0x00080000 0x00020000 0x00008000 0x00002000 0x00000800 0x00000200 0x00000080 0x00000020 0x00000008 0x00000002 0x00000000  
###################################################################
LogicalRightShift:            #逻辑右移测试，将最高位1逐位向右右移直至结果为零

add    a0,zero,s1       #display s1
addi   a7,zero,34        # display hex
ecall                 # we are out of here.  
     
srli s1, s1, 2   
beq s1, zero, shift_next1
j LogicalRightShift

shift_next1:

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  


###################################################################
#                逻辑右移测试 
# 显示区域依次显示0x00000004 0x00000010 0x00000040 0x00000100 0x00000400 0x00001000 0x00004000 0x00010000 0x00040000 0x00100000 0x00400000 0x01000000 0x04000000 0x10000000 0x40000000 0x00000000 
###################################################################

addi s1,zero, 1   
LogicalLeftShift:         #逻辑左移测试，将最低位1逐位向左移直至结果为零
slli s1, s1, 2  

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  
      
beq s1, zero, ArithRightShift
j LogicalLeftShift


###################################################################
#                算术右移测试 
# 显示区域依次显示0x80000000 0xf0000000 0xff000000 0xfff00000 0xffff0000 0xfffff000 0xffffff00 0xfffffff0 0xffffffff 
###################################################################
ArithRightShift:          #算术右移测试，#算术移位测试，80000000算术右移，依次显示为F0000000,FF000000,FFF00000,FFFF0000直至FFFFFFFF

addi s1,zero,1  
slli s1, s1, 31   #逻辑左移31位 s1=0x80000000

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  

srai s1, s1, 3    #s1=0X80000000-->0XF0000000

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  


srai s1, s1, 4    #0XF0000000-->0XFF000000

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  


srai s1, s1, 4    #0XFF000000-->0XFFF00000

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  

srai s1, s1, 4    

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  

srai s1, s1, 4    

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  


srai s1, s1, 4    

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  


srai s1, s1, 4    

add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.  


srai s1, s1, 4    


add    a0,zero,s1       #display s1
addi   a7,zero,34         # display hex
ecall                 # we are out of here.    

#############################################################
#走马灯测试,测试addi,andi,sll,srl,sra,or,ori,nor,ecall  LED按走马灯方式来回显示数据
#############################################################

.text
addi s0,zero,1 
slli s3, s0, 31      # s3=0x80000000
srai s3, s3, 31      # s3=0xFFFFFFFF   
add s0,zero,zero   # s0=0         
addi s2,zero,12 

addi s6,zero,3  #走马灯计数
zmd_loop:

addi s0, s0, 1    #计算下一个走马灯的数据
andi s0, s0, 15  

#######################################
addi t0,zero,8    
addi t1,zero,1
left:

slli s3, s3, 4   #走马灯左移
or s3, s3, s0

add    a0,zero,s3       # display s3
addi   a7,zero,34         # system call for LED display 
ecall                 # display 

sub t0,t0,t1
bne t0,zero,left
#######################################

addi s0, s0, 1   #计算下一个走马灯的数据
addi t6,zero,15
and s0, s0, t6
slli s0, s0, 28     

addi t0,zero,8
addi t1,zero,1

zmd_right:

srli s3, s3, 4  #走马灯右移
or s3, s3, s0

add    a0,zero,s3       # display s3
addi   a7,zero,34         # system call for LED display 
ecall                 # display 

sub t0,t0,t1
bne t0,zero,zmd_right
srli s0, s0, 28  
#######################################

sub s6,s6,t1
beq s6,zero, exit
j zmd_loop

exit:

add t0,zero,zero
xori t0,t0,-1      #test r  xori
slli t0,t0,8
ori t0,t0,255

add   a0,zero,t0       # display t0
addi   a7,zero,34         # system call for LED display 
ecall                 # display 

#################################################################################
#本程序实现0-15号字单元的降序排序,此程序可在rars 仿真器中运行
#运行时请将rars Setting中的Memory Configuration设置为Compact，data at address 0
#################################################################################
 .text
sort_init:
 addi s0,zero,-1
 addi s1,zero,0
 
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
  sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
  sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
 sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
  sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
  sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
  sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
  sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
    sw s0,0(s1)
 addi s0,s0,1
 addi s1,s1,4
   
 addi s0,s0,1
 
 add s0,zero,zero   
 addi s1,zero,60   #排序区间
sort_loop:
 lw s3,0(s0)     
 lw s4,0(s1)
 slt t0,s3,s4
 beq t0,zero,sort_next   #降序排序
 sw s3, 0(s1)
 sw s4, 0(s0)   
sort_next:
 addi s1, s1, -4   
 bne s0, s1, sort_loop  
 
 add    a0,zero,s0       #display s0
 addi   a7,zero,34         # display hex
 ecall                 # we are out of here.  DISP: disp r0, 0
 
 addi s0,s0,4
 addi s1,zero,60
 bne s0, s1, sort_loop
                  
 addi   a7,zero,10         # system call for exit
 ecall                  # we are out of here.   
 
#############################################
# insert your ccmb benchmark program here!!!
#############################################

#j benchmark_start       #delete this instruction for ccmb bencmark
#C1 instruction benchmark

#AUIPC测试    revise date:2021/1/24 tiger   compact mode： Text at  Address 0
#依次输出 0x00430004 0x00430014 0x00430024 0x00430034 0x00430044 0x00430054 0x00430064 0x00430074

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



#C2 instruction benchmark

#sltiu 测试    revise date:2021/1/24 tiger
#依次输出  0x00001997 0x00001996 0x00001995 0x00001994 0x00001993 0x00001992 0x00001991 0x00001990 0x0000198f 0x0000198e 0x0000198d 0x0000198c 0x0000198b 0x0000198a 0x00001989 0x00001988 0x00001987 0x00001986 0x00001985 0x00001984 0x00001983 0x00001982 0x00001981 0x00001980 0x0000197f 0x0000197e 0x0000197d 0x0000197c 0x0000197b 0x0000197a 0x00001979 0x00001978 0x00001977 0x00001976 0x00001975 0x00001974 0x00001973 0x00001972 0x00001971 0x00001970 0x0000196f 0x0000196e 0x0000196d 0x0000196c 0x0000196b 0x0000196a 0x00001969 0x00001968 0x00001967 0x00001966 0x00001965 0x00001964 0x00001963 0x00001962 0x00001961 0x00001960 0x0000195f 0x0000195e 0x0000195d 0x0000195c 0x0000195b 0x0000195a 0x00001959 0x00001958 0x00001957 0x00001956 0x00001955 0x00001954 0x00001953 0x00001952 0x00001951 0x00001950 0x0000194f 0x0000194e 0x0000194d 0x0000194c 0x0000194b 0x0000194a 0x00001949

.text
addi t0,zero,-1    
addi t1,zero,0     

addi s0,zero,0x19
slli s0,s0,8
addi s1,zero,0x97


sltiu_branch:
add a0,s0,s1          
addi a7,zero,34        # system call for print
ecall                  # print
add s1,s1,t0     
sltiu t1,s1, 0x49
beq t1,zero,sltiu_branch

addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   


#Mem instruction benchmark


#LBU 测试   revise date:2021/1/24 tiger
#依次输出   0x00000081 0x00000082 0x00000083 0x00000084 0x00000085 0x00000086 0x00000087 0x00000088 0x00000089 0x0000008a 0x0000008b 0x0000008c 0x0000008d 0x0000008e 0x0000008f 0x00000090 0x00000091 0x00000092 0x00000093 0x00000094 0x00000095 0x00000096 0x00000097 0x00000098 0x00000099 0x0000009a 0x0000009b 0x0000009c 0x0000009d 0x0000009e 0x0000009f 0x000000a0
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



lbu_store:
sw s1,(t1)
add s1,s1,s2   #data +1
addi t1,t1,4    # addr +4  
addi t3,t3,-1   #counter
bne t3,zero,lbu_store

addi t3,zero,32
addi t1,zero,0    # addr +4  
lbu_branch:
lbu s1,(t1)     #测试指令
add a0,zero,s1          
addi a7,zero,34         
ecall                  # 输出
addi t1,t1, 1    
addi t3,t3, -1    
bne t3,zero,lbu_branch

addi   a7,zero,10         # system call for exit
ecall                  # we are out of here.   


#Branch instruction benchmark



#bge 测试    大于等于零跳转   递减运算 ，从正数开始向零运算revise date:2022/1/24 tiger  
#依次输出0x0000000f 0x0000000e 0x0000000d 0x0000000c 0x0000000b 0x0000000a 0x00000009 0x00000008 0x00000007 0x00000006 0x00000005 0x00000004 0x00000003 0x000000020 x000000010 x00000000
.text
addi s1,zero,15  #初始值
bge_branch:
add a0,zero,s1          
addi a7,zero,34         
ecall                  # 输出当前值
addi s1,s1,-1
bge s1,zero,bge_branch   #测试指令

addi   a7,zero,10         #停机指令
ecall                  # 系统调用

 
 #处理器实现中请用停机指令实现ecall

jmp_count: addi s0,zero, 0
       addi s0,s0, 1
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       ecall                 # we are out of here.  
       
       addi s0,s0, 2
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       ecall                 # we are out of here.  
       
       addi s0,s0, 3
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       ecall                 # we are out of here.  
       
       addi s0,s0, 4       
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       ecall                 # we are out of here.  
       
       addi s0,s0, 5              
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       ecall                 # we are out of here.  

       addi s0,s0, 6              
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       ecall                 # we are out of here.  

       addi s0,s0, 7              
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       ecall                 # we are out of here.  

       addi s0,s0, 8              
       add    a0,zero,s0      
       addi   a7,zero,34         # display hex
       addi   a7,zero,34         # display hex       
       ecall                 # we are out of here.  
       
       ret  #persudo instruction  jalr x0,x1,0
