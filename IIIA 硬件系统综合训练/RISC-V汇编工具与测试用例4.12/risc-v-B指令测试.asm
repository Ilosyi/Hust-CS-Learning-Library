# benchmark for beq and bne inst instructions
.text
addi s0,x0,1
addi s2,x0,255
addi s1,x0,1
addi s3,x0,3
beq s0, s2, Next1
beq s0, s0, Next1
addi s1,x0,1    # can not execute
addi s2,x0,2
addi s3,x0,3

Next1:
 add    a0,x0,s0       # display s0
 addi   a7,x0,34         # system call for display
 ecall                 # syscall include 2 hidden operand: v0,a0£¬may cause data hazzard

 bne s1, s1, Next2
 bne s1, s2, Next2

 addi s1,x0,1    # can not execute
 addi s2,x0,2
 addi s3,x0,3

Next2:
 add    a0,x0,s3       # display s3
 addi   a7,x0,34        # system call for display
 ecall                 # syscall include 2 hidden operand: v0,a0£¬may cause data hazzard

 addi   a7,x0,10      # system call for exit
 ecall                  # we are out of here.   

