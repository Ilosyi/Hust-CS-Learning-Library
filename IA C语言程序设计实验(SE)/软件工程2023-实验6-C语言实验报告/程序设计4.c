#include <stdio.h>
#include <string.h>
int panduan(char c) {
    if (c == '\t') {
        return 0;
    }
    else {
        return 1;
    }
}

int main() {
    char s[80];
    char c;
    int i = 0;
    int LineStart = 1;  

    while ((c = getchar()) != EOF) {
        if (panduan(c)) {
            if (LineStart && c == ' ') {
                continue;  
            }
            s[i++] = c;
            if (c == '\n') {
                LineStart = 1;  
            }
            else {
                LineStart = 0;  
            }
        }
    }

    for (int j = 0; j < i; j++) {
        printf("%c", s[j]);
    }

    return 0;
}
