#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define N 4
/*对指针数组s指向的size个字符串进行升序排序*/
void strsort(char* s[], int size)
{
	char *temp;
	int i, j;
	for (i = 0; i < size - 1; i++)
		for (j = 0; j < size - i - 1; j++)
			if (strcmp(s[j],s[j+1])>0)
			{
				temp = s[j];
				s[j]=s[j+1];
				s[j + 1] = temp;
			}
}
//修改后的函数
void Strsort(char**s, int size)
{
	char* temp;
	int i, j;
	
	for (i = 0; i < size - 1; i++)
		for (char **p=s,j=0; j < size - i - 1; j++,p++)
			if (strcmp(*p, *(p+1))>0)
			{
				temp = *p;
				*p = *(p+1);
				*(p+1)= temp;
			}
}

int main()
{
	int i;
	char* s[N], t[50];
	for (i = 0; i < N; i++)
	{
		gets_s(t,50);
		s[i] = (char*)malloc(strlen(t) + 1);
		strcpy(s[i],t);
	}
	strsort(s,N);
	for (i = 0; i < N; i++) { puts(s[i]); free(s[i]); }
	return 0;
}
