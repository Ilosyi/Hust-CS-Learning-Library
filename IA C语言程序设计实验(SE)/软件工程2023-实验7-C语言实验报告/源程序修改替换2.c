#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

struct s_list {
    int data;             /* 数据域 */
    struct s_list* next;  /* 指针域 */
};

void create_list(struct s_list** headp, int* p);  /* 将 headp 作为指向指针的指针传递 */
/* 打印链表的辅助函数 */

int main(void) {
    struct s_list* head = NULL, * p;
    int s[] = { 1, 2, 3, 4, 5, 6, 7, 8, 0 }; /* 0为结束标记 */

    create_list(&head, s);  /* 将头指针的地址传递给 create_list */
     p = head;
    while (p) {
        printf("%d\t", p->data);
        p = p->next;
    }
    printf("\n");     

    return 0;
}

void create_list(struct s_list** headp, int* p) {
    struct s_list* loc_head = NULL;

    while (*p) {
        struct s_list* newx = (struct s_list*)malloc(sizeof(struct s_list));
        newx->data = *p++;
        newx->next = loc_head; /* 在链表开头插入新节点 */
        loc_head = newx;
    }

    *headp = loc_head; /* 更新 headp 指针，使其指向最后创建的节点 */
}

