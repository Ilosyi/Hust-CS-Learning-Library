#include<stdio.h>
#include<stdlib.h>
int main(int argc , char* argv[])
{
	char ch;
	FILE *fp;
	if(argc!=2)
    {
		printf("Arguments error!\n");
		exit(-1);
	}
	if((freopen(argv[1],"r",stdin))==NULL)// // 使用 freopen 将标准输出重定向到文本文件
    {       /* fp 指向 filename */
		printf("Can't open %s file!\n",argv[1]);
		exit(-1);
	}

	while((ch=getchar())!=EOF)          /* 从filename中读字符 */
	   putchar(ch);                  /* 向显示器中写字符 */
    return 0;	
}
