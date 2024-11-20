#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#define M 10
#define N 3

int main() {
    int a[M + 1] = { 0 };
    int count = 0;
    int x = 0;

    while (count < M - 1) 
    {
        int step = 0;      
        while (step < N) {
            x = x % M+1;  

            if (!a[x]) {
                step++;
            }
        }    
        a[x] = 1;
        printf("%6d", x);
        count++;
    }
    for (int i = 1; i <= M; i++) {
        if (!a[i]) {
            printf("%6d", i);
            break;
        }
    }
    return 0;
}
