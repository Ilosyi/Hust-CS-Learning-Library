#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
char* strcopy(char*, const char*);
int main(void)
{
	char s1[100],  s2[100],  s3[100];
	printf("Input a string:\n");
	scanf("%s", s2);
	strcopy(s1, s2);
	printf("%s\n", s1);

	printf("Input a string again:\n");
	scanf("%s", s2);
	strcopy(s3, s2);
	printf("%s\n", s3);
	return 0;
}

/*将字符串s复制给字符串t，并且返回串t的首地址*/
char* strcopy(char* t, const char* s)
{
	char* p = t;
	while (*t++ = *s++);
	return p;
}
