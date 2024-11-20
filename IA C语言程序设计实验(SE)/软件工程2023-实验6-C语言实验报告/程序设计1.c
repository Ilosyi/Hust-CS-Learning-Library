#define _CRT_SECURE_NO_WARNINGS
# include<stdio.h>
# include<string.h>
# include<stdlib.h>

int main(int argc, char* argv[]) {
    // argc 表示命令行参数的数量
    // argv 是一个指向参数字符串数组的指针，其中 argv[0] 存储程序的名称，后续元素存储传递给程序的参数
    // 注意：IDE中配置的参数通常不包括程序名称，因此从 argv[1] 开始获取实际的参数

    // 输出参数数量
    printf("Number of command-line arguments: %d\n", argc - 1);

    // 输出每个参数
    for (int i = 1; i < argc; i++) {
        printf("Argument %d: %s\n", i, argv[i]);
    }

    return 0;
}
//Visual studio 调试-最后一行工程属性-配置属性-调试-命令行参数
