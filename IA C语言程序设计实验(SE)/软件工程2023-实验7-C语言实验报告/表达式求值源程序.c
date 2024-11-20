#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
int main() {
char u[] = "UVWXYZ";
char v[] = "xyz";
struct T {
int x;
char c;
char* t;
}a[] = { {11,'A',u},{100, 'B',v}}, * p = a;

printf("%d\n", (++p)->x); p--;
printf("%c\n", (p++, p->c)); p--;
printf("%c\n", (*p++->t, *p->t)); p--;
printf("%c\n", *(++p)->t); p--;
printf("%c\n", *++p->t), --p->t;
printf("%c\n", ++ * p->t);
}
