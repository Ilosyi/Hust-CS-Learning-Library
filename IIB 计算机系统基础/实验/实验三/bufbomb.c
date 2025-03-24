/* 
 * bufbomb.c - Bomb program that is solved using a buffer overflow attack 
 * 
 * Copyright (c) 2002-2011, R. Bryant and D. O'Hallaron, All rights reserved.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include "support.h"
#include "gencookie.h"
#include "stack.h"

/* 
 * This version of bufbomb uses mmap() to shift the stack to a stable
 * location, regardless of any stack randomization by the runtime system.
 */
#ifndef USE_MMAP
#define USE_MMAP
#endif

/* The "bottom" of the shifted stack will be at this address. This
 * location works on every Linux system we've tried it on over a
 * period of years. */
#ifdef USE_MMAP
#include <sys/mman.h>
#endif

/* Included for historical reasons. Better to run a VirtualBox Ubuntu
   Linux VM on your PC or Mac than to use Cygwin */
#ifdef __CYGWIN__
#include "getopt.h"
#endif

/* Binary code for HLT (halt) instruction */
#define HALT_INSTR 0xF4

/* Levels 0-3 are called once */
#define NORMAL_CNT 1
/* $begin getbuf-c */
/* Buffer size for getbuf */
#define NORMAL_BUFFER_SIZE 32

/* $end getbuf-c */
/* Level 4 (nitro mode) is called multiple times */
#define KABOOM_CNT 5
/* $begin kaboom-c */
/* Buffer size for getbufn */
#define KABOOM_BUFFER_SIZE 512

/* $end kaboom-c */
/* Global variables */
char *userid = NULL; /* user id [set by -u] */
int notify = 0;      /* if true, send exploits to grading server [set by -s] */
int autograde = 0;   /* if true, run in autograde mode with timeout [set by -g]*/

FILE *infile = NULL; /* always stdin */
unsigned cookie = 0; /* unique cookie computed from userid */
int success = 0;     /* set by validate() to indicate successful exploit */

/* Function prototypes */
void validate(int);
char *Gets(char *);
int getbuf();
int getbufn();
int uniqueval();

/*
 * The following is the part of the code students will actually look
 * at.  They are put at the beginning of the file to make them easier
 * to find in the diassembly, and to make their locations more stable
 * when other parts of the code get modified.
 */

/* 
 * smoke - On return from getbuf(), the level 0 exploit executes
 * the code for smoke() instead of returning to test().
 */
/* $begin smoke-c */
void smoke()
{
    printf("Smoke!: You called smoke()\n");
    validate(0);
    exit(0);
}
/* $end smoke-c */

/* 
 * fizz - On return from getbuf(), the level 1 exploit executes the
 * code for fizz() instead of returning to test(), and makes it appear
 * that fizz() was passed the users's unique cookie as the argument.
 */
/* $begin fizz-c */
void fizz(int val)
{
    if (val == cookie) {
		printf("Fizz!: You called fizz(0x%x)\n", val);
		validate(1);

    } else
		printf("Misfire: You called fizz(0x%x)\n", val);
    exit(0);
}
/* $end fizz-c */

/* 
 * bang - On return from getbuf(), the level 2 exploit executes the
 * code for bang() instead of returning to test(). Before transferring
 * control, it must execute code on the stack that sets a global
 * variable to the user's cookie.
 */
/* $begin bang-c */
int global_value = 0;

void bang(int val)
{
    if (global_value == cookie) {
        printf("Bang!: You set global_value to 0x%x\n", global_value);
        validate(2);
    } else
        printf("Misfire: global_value = 0x%x\n", global_value);
    exit(0);
}
/* $end bang-c */

/* 
 * test - This function calls the function with the buffer overflow
 * bug. The exploits for levels 0-2 return from the getbuf() call to a
 * different function, which then immediately exits. The level 3
 * exploit must return to test() with local variable val set to the
 * user's cookie. This is tricky because, unlike the previous levels,
 * which simply transferred control, the exploit code must restore the
 * stack to support a proper return.
 */
/* $begin boom-c */
void test()
{
    int val;
    /* Put canary on stack to detect possible corruption */
    volatile int local = uniqueval(); 

    val = getbuf(); 

    /* Check for corrupted stack */
    if (local != uniqueval()) {
		printf("Sabotaged!: the stack has been corrupted\n");
    }
    else if (val == cookie) {
		printf("Boom!: getbuf returned 0x%x\n", val);
		validate(3);
    } else {
        printf("Dud: getbuf returned 0x%x\n", val);
    }
}
/* $end boom-c */


/*
 * testn - Calls the function with the buffer overflow bug exploited
 * by the level 4 exploit.
 */
void testn()
{
    int val;
    volatile int local = uniqueval();

    val = getbufn();

    /* Check for corrupted stack */
    if (local != uniqueval()) {
		printf("Sabotaged!: the stack has been corrupted\n");
    }
    else if (val == cookie) {
		printf("KABOOM!: getbufn returned 0x%x\n", val);
		validate(4);
    }
    else {
		printf("Dud: getbufn returned 0x%x\n", val);
    }
}

/******************
 * Helper functions
 ******************/

/* 
 * Gets - Like gets(), except that can optionally (when hexformat
 * nonzero) accept format where characters are typed as pairs of hex
 * digits.  Nondigit characters are ignored.  Stops when encounters
 * newline.  In addition, it stores the string in global buffer
 * gets_buf.
 */
#define GETLEN 1024

int  gets_cnt = 0;
char gets_buf[3*GETLEN+1];

static char trans_char[16] = 
	{'0', '1', '2', '3', '4', '5', '6', '7', 
	 '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

static void save_char(char c) {
    if (gets_cnt < GETLEN) {
		gets_buf[3*gets_cnt] = trans_char[(c>>4)&0xF];
		gets_buf[3*gets_cnt+1] = trans_char[c&0xF];
		gets_buf[3*gets_cnt+2] = ' ';
		gets_cnt++;
    }
}

static void save_term()
{
    gets_buf[3*gets_cnt] = '\0';
}

char *Gets(char *dest)
{
    int c;
    char *sp = dest;

    gets_cnt = 0;

    while ((c = getc(infile)) != EOF && c != '\n') {
		*sp++ = c;
		save_char(c);
    }

    *sp++ = '\0';
    save_term();
    return dest;
}


/*
 * usage - prints usage information
 */
static void usage(char *name)
{
    printf("Usage: %s -u <userid> [-nsh]\n",  name);
    printf("  -u <userid> User ID\n");
    printf("  -n          Nitro mode\n");
    printf("  -s          Submit your solution to the grading server\n");
    printf("  -h          Print help information\n");
    exit(0);
}

/* 
 * Signal handlers for bus errors, seg faults, and illegal instruction
 * faults
 */
void bushandler(int sig)
{
    printf("Crash!: You caused a bus error!\n");
    printf("Better luck next time\n");
    exit(0);
}

void seghandler(int sig)
{
    printf("Ouch!: You caused a segmentation fault!\n");
    printf("Better luck next time\n");
    exit(0);
}

void illegalhandler(int sig)
{
    printf("Oops!: You executed an illegal instruction\n");
    printf("Better luck next time\n");
    exit(0);
}

/* 
 * launch - Calls either test (normal mode) or testn (nitro mode)
 */
static void launch(int nitro, int offset)
{
    int localbuf[16];
    size_t stable_tweak = 0;
    int *space;
    /*
     * This little hack adjusts the stack. Without it, the stack
     * offset is different when the program is executed in the shell
     * and when it is executed in gdb.  For normal mode, it tries to
     * put it into a stable position from one run to the next.  In
     * nitro mode, it makes it even less stable than it would normally
     * be.  You don't need to understand it to do the assignment.
     */
    stable_tweak = (((size_t) localbuf) & 0x3FF0); 
    space = (int *) alloca(stable_tweak + offset);

    /* Fill full of halt instructions, so that will get seg fault */
    memset(space, HALT_INSTR, stable_tweak);

    /* Call the appropriate function */
    printf("Type string:");
    if (nitro)
		testn();
    else
		test();
    if (!success) {
		printf("Better luck next time\n");
		success = 0;
    }
}


/* 
 * launcher - New version of the launching code that uses mmap() to 
 * generate a stable stack position, regardless of any stack randomization
 * used by the runtime system. 
 */

/* Must put context information in global vars, since stack will get
   messed up */
int global_nitro = 0;
int global_offset = 0;
volatile void *stack_top;
volatile void *global_save_stack = NULL;


void launcher(int nitro, int offset)
{
#ifdef USE_MMAP
    void *new_stack;
#endif

    /* Assign from stack to globals before we move the stack location */
    global_nitro = nitro;
    global_offset = offset;

#ifdef USE_MMAP
    new_stack = mmap(START_ADDR, STACK_SIZE, PROT_EXEC|PROT_READ|PROT_WRITE,
					 MAP_PRIVATE | MAP_GROWSDOWN | MAP_ANONYMOUS | MAP_FIXED,
					 0, 0);
    if (new_stack != START_ADDR) {
		fprintf(stderr, "Internal error.  Couldn't use mmap. Try different value for START_ADDR\n");
		exit(1);
    }
    stack_top = new_stack + STACK_SIZE - 8;
    asm("movl %%esp,%%eax ; movl %1,%%esp ; movl %%eax,%0"
		: "=r" (global_save_stack)
		: "r"  (stack_top)
		: "%eax"
		);
#endif

    launch(global_nitro, global_offset);


#ifdef USE_MMAP
    asm("movl %0,%%esp"
		: 
		: "r" (global_save_stack)
		);
    munmap(new_stack, STACK_SIZE);
#endif
}

/*
 * uniqueval - Compute random value that will change from one execution to the next
 */
int uniqueval(){
    srandom(getpid());
    return random();
}


/* 
 * main - The main routine
 */
int main(int argc, char *argv[])
{
    int cookie_tweak = 0;
    int nitro = 0; /* Run in unstable mode? */
    int i;
    int *offsets;
    int cnt = NORMAL_CNT; /* By default, call launcher once */
    char c;

    /* Install handlers for the inevitable faults */
    signal(SIGSEGV, seghandler);
    signal(SIGBUS,  bushandler);
    signal(SIGILL,  illegalhandler);

    /* Parse command line arguments */
    infile = stdin;
    while ((c = getopt(argc, argv, "gsnhu:")) != -1)
		switch(c) {
		case 'h': /* print help info */
			usage(argv[0]);
			break;
		case 'n': /* run in nitro mode */
			nitro = 1; 
			cnt = KABOOM_CNT; /* Call launcher multiple times */
			break;
		case 'u': /* userid */
			userid = strdup(optarg);
			cookie = gencookie(userid);
			break;
		case 's': /* submit exploit string to the grading server */
			if (!NOTIFY)
				printf("This is a quiet bomb. Ignoring -s flag.\n");
			notify = NOTIFY; 
			break;
		case 'g': /* autograding mode, sets timeout */
			autograde = 1;
			break;
		default:
			usage(argv[0]);
		}

    /* Userid is a required argument */
    if (!userid) {
        printf("%s: Missing required argument (-u <userid)\n", argv[0]);
        usage(argv[0]);
    }

    /* Initialize the bomb. If this is a notifying bomb, make sure we're running
       on one of the legal hosts defined in config.h */
    initialize_bomb();

    /* Print some basic info */
    printf("Userid: %s\n", userid);
    printf("Cookie: 0x%x\n", cookie);

    /* Set up random stack offsets for nitro mode */
    srandom(cookie);
    cookie_tweak =  (random() & 0x0FF0) + 0x100;

    offsets = (int *) calloc(cnt, sizeof(int));
    offsets[0] = 0;
    for (i = 1; i < cnt; i++)
		/* Random number x s.t. |x| <= 128 & x is multiple of 16 */
		offsets[i] = 128 - (random() & 0xF0);
    /*  
     * Now call the launcher, once by default, multiple times for
     * nitro mode
     */
    for (i = 0; i < cnt; i++)
		launcher(nitro, offsets[i]+cookie_tweak);

    return 0;
}

