#include <stdio.h>

int main() {
    int x;
    char a[33]; 
    scanf("%d", &x);

    for (int i = 31; i >= 0; i--)
     {
        a[i] = (x & 1) + '0'; 
        x >>= 1; 
    }
    a[32] = '\0';
    printf("%s\n", a);
    return 0;
}
