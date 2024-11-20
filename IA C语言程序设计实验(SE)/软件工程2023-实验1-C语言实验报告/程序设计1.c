#include<stdio.h> 
int main()
{
    char c;
    while((c=getchar())!=EOF)
    {
        if(c>='A'&&c<='Z')//检验范围
        putchar(c+32);//根据ASCII表，大小写字母差值为32
        else
        putchar(c);
    } 
        return 0;
}
