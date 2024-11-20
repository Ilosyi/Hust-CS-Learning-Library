#include <string.h>
#include <stdio.h>

// 自定义函数，用于替换字符串
void str_replace(char* position, int len_target, char* replacement)
{
	int len_replace;
	char* tmp;
	len_replace = strlen(replacement);
	// 替换字符串比目标字符串短，往前移动
	if (len_replace < len_target)
	{
		tmp = position + len_target;
		while (*tmp)
		{
			*(tmp - (len_target - len_replace)) = *tmp; // len_target - len_replace是移动的距离 
			tmp++;                                        // 不断左移
		}
		*(tmp - (len_target - len_replace)) = *tmp; // 移动终止符号	
	}
	else
		// 替换比目标长，往后移动
		if (len_replace > len_target)
		{
			tmp = position;
			while (*tmp) tmp++;
			while (tmp >= position + len_target)
			{
				*(tmp + (len_replace - len_target)) = *tmp;
				tmp--;
			}
		}
	strncpy(position, replacement, len_replace);
}

int main()
{
	char str1[1000] = { 0 };
	char str2[10], str3[10];
	int count = 0;
	char* p;

	// 从用户输入获取目标字符串和替换字符串
	scanf("%s", str2);
	scanf("%s", str3);

	// 读取文本文件
	FILE* fp = fopen("experiment/src/step8/source.txt", "r");
	fread(str1, 1, 1000, fp);

	// 开始查找字符串str2 
	p = strstr(str1, str2);
	while (p)
	{
		count++;
		// 每找到一个str2，就用str3来替换 
		str_replace(p, strlen(str2), str3);
		p = p + strlen(str3);
		p = strstr(p, str2);
	}

	// 输出替换的总次数和修改后的文本内容
	printf("%d\n", count);
	printf("%s", str1);

	// 关闭文件
	fclose(fp);

	return 0;
}
