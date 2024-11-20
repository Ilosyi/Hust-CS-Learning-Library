 #include <stdio.h>

int main() 
{
    int x;
    while (scanf("%d", &x) != EOF) {

        int a = (x >> 24) & 0xFF;  // 第一个八位
        int b = (x >> 16) & 0xFF;  // 第二个八位
        int c = (x >> 8) & 0xFF;   // 第三个八位
        int d = x & 0xFF;          // 第四个八位
        printf("%d.%d.%d.%d\n", a, b, c, d);
    }
    return 0;
}
