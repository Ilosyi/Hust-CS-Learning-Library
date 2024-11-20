#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
void task0() { printf("task0 is called!\n"); }
void task1() { printf("task1 is called!\n"); }
void task2() { printf("task2 is called!\n"); }
void task3() { printf("task3 is called!\n"); }
void task4() { printf("task4 is called!\n"); }
void task5() { printf("task5 is called!\n"); }
void task6() { printf("task6 is called!\n"); }
void task7() { printf("task7 is called!\n"); }
void execute(void (*tasks[])(), char* Task, int num) {
    for (int i = 0; i < num; i++) {
        tasks[Task[i]]();
    }
}
void scheduler(char* input)
{
    int num = 0;
    void (*function[8])() = { task0, task1, task2, task3, task4, task5, task6,task7 };
    char Task[20] = { 0 };

    for (int i = 0; input[i] != '\0'; i++)
    {
        int task_number = input[i] - '0';
        Task[i] = task_number;
        if (task_number >= 0 && task_number < 8)
        {
            num++;
        }

    }
    execute(function, Task, num);
}

int main() {
    char input[20];
    scanf("%s", input, 20);
    scheduler(input);
    return 0;
}
