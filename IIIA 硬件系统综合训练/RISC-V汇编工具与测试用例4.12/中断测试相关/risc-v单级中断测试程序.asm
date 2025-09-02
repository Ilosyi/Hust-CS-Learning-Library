.text

addi sp, zero, 2000   #内存堆栈初始化

#############################################################
#走马灯测试,测试addi,andi,slli,srli,srai,or,ori,nor,ecall  LED按走马灯方式来回显示数据
#############################################################

.text
START:

addi s0,zero,1 
slli s3, s0, 31      # s3=0x80000000
srai s3, s3, 31      # s3=0xFFFFFFFF   
add s0,zero,zero   # s0=0         
addi s2,zero,12 

addi s6,zero,8  #走马灯计数
zmd_loop:

addi s0, s0, 1    #计算下一个走马灯的数
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

addi s0, s0, 1   #计算下一个走马灯的数
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
xori t0,t0,-1      #test nor  ori
slli t0,t0,16
ori t0,t0,-1

add   a0,zero,t0       # display t0
addi   a7,zero,34         # system call for LED display 
ecall                 # display 

j START   # loop forever



InteruptProgram1:
###########################################################
#  exceptoin 1
#  使用?s6? s5?s4?s3?s0?a0?a7
###########################################################

sw a7, 0(sp)         #保护现场
addi sp, sp, 4
sw a0, 0(sp)
addi sp, sp, 4
sw s0, 0(sp)
addi sp, sp, 4
sw s3, 0(sp)
addi sp, sp, 4
sw s4, 0(sp)
addi sp, sp, 4
sw s5, 0(sp)
addi sp, sp, 4
sw s6, 0(sp)
addi sp, sp, 4

#################  中断服务

addi s6,zero,1      #走马灯演示数字， 不同中断号显示不同数值，分别设置为1,2,3
addi s4,zero,3      #循环次数初始值   
addi s5,zero,1      #计数器累加值


IntLoop1:
add s0,zero,s6   

IntLeftShift1:       


slli s0, s0, 4  
or s3,s0,s4
add    a0,zero,s3       #display s0
addi   a7,zero,34       #display hex
ecall               

bne s0, zero, IntLeftShift1
sub s4,s4,s5            #循环次数递减
bne s4, zero, IntLoop1

################# 恢复现场

addi sp, sp, -4        
lw s6, 0(sp)
addi sp, sp, -4
lw s5, 0(sp)
addi sp, sp, -4
lw s4, 0(sp)
addi sp, sp, -4
lw s3, 0(sp)
addi sp, sp, -4
lw s0, 0(sp)
addi sp, sp, -4
lw a0, 0(sp)
addi sp, sp, -4
lw a7, 0(sp)

################  中断返回

uret              #同步开中断，mepc-->pc



InteruptProgram2:
#############################################################################################
#  exceptoin 2
#  使用?s6? s5?s4?s3?s0?a0?a7
#############################################################################################


################# 保护现场

sw a7, 0(sp)        
addi sp, sp, 4
sw a0, 0(sp)
addi sp, sp, 4
sw s0, 0(sp)
addi sp, sp, 4
sw s3, 0(sp)
addi sp, sp, 4
sw s4, 0(sp)
addi sp, sp, 4
sw s5, 0(sp)
addi sp, sp, 4
sw s6, 0(sp)
addi sp, sp, 4

################# 中断服务

addi s6,zero,2       #走马灯演示数字， 不同中断号显示不同数值，分别设置为1,2,3
addi s4,zero,3       #循环次数初始值   
addi s5,zero,1       #计数器累加值


IntLoop2:
add s0,zero,s6   

IntLeftShift2:       


slli s0, s0, 4  
or s3,s0,s4
add    a0,zero,s3       #display s0
addi   a7,zero,34       #display hex
ecall                  

bne s0, zero, IntLeftShift2
sub s4,s4,s5           #循环次数递减
bne s4, zero, IntLoop2

################# 恢复现场

addi sp, sp, -4
lw s6, 0(sp)
addi sp, sp, -4
lw s5, 0(sp)
addi sp, sp, -4
lw s4, 0(sp)
addi sp, sp, -4
lw s3, 0(sp)
addi sp, sp, -4
lw s0, 0(sp)
addi sp, sp, -4
lw a0, 0(sp)
addi sp, sp, -4
lw a7, 0(sp)


################  中断返回
uret              #同步开中断，mepc-->pc


InteruptProgram3:
#############################################################################################
#  exceptoin 3
#  使用?s6? s5?s4?s3?s0?a0?a7
#############################################################################################

################# 保护现场

sw a7, 0(sp)
addi sp, sp, 4
sw a0, 0(sp)
addi sp, sp, 4
sw s0, 0(sp)
addi sp, sp, 4
sw s3, 0(sp)
addi sp, sp, 4
sw s4, 0(sp)
addi sp, sp, 4
sw s5, 0(sp)
addi sp, sp, 4
sw s6, 0(sp)
addi sp, sp, 4

################# 中断服务

addi s6,zero,3       #走马灯演示数字， 不同中断号显示不同数值，分别设置为1,2,3
addi s4,zero,3       #循环次数初始值   
addi s5,zero,1       #计数器累加值

IntLoop:
add s0,zero,s6   

IntLeftShift:       

slli s0, s0, 4  
or s3,s0,s4
add    a0,zero,s3       #display s0
addi   a7,zero,34       # display hex
ecall                 

bne s0, zero, IntLeftShift
sub s4,s4,s5           #循环次数递减
bne s4, zero, IntLoop


################# 恢复现场

addi sp, sp, -4
lw s6, 0(sp)
addi sp, sp, -4
lw s5, 0(sp)
addi sp, sp, -4
lw s4, 0(sp)
addi sp, sp, -4
lw s3, 0(sp)
addi sp, sp, -4
lw s0, 0(sp)
addi sp, sp, -4
lw a0, 0(sp)
addi sp, sp, -4
lw a7, 0(sp)


################  中断返回
uret              #同步开中断，mepc-->pc
