#define  _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
void strcate(char[], char[]);
void strdelc(char[], char);
char a[] = "Language", b[] = "Programming";
int main(void)
{
	
	printf("%s %s\n", b, a);
	strcate(b, a);	printf("%s %s\n", b, a);
	strdelc(b, 'a');	printf("%s\n", b);
	return 0;
}
void strcate(char t[], char s[])//programming  Language
{
	int i = 0, j = 0;
	while (t[i++]);
	i--;
	while ((t[i++] = s[j++]) != '\0')
	{
		;
	}
}
void strdelc(char s[], char c)
{
	int j, k;
	for (j = k = 0; s[j] != '\0'; j++)
	{ 
		if (s[j] != c)	s[k++] = s[j]; 
	}
	s[k] = '\0';

}
