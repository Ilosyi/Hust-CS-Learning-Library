#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

struct Student                  //定义学生结构体
{
	char ID[10];
	char name[20];
	int English, math, physics, Cprogramming;
	double ave;
	struct Student* next;
};
typedef struct Student stu;  //自定义名

stu* head = NULL;   //头指针

//输入学生信息
void addStudent()
{
	printf("请输入要输入的学生人数:\n");
	int n;
	scanf("%d", &n);
	printf("请输入学生的ID,姓名，英语，数学，物理，C语言成绩:\n");
	for (int i = 0; i < n; i++) {
		stu* student = (stu*)malloc(sizeof(stu)); //分配空间
		scanf("%s", student->ID, 10);
		scanf("%s", student->name, 20);
		scanf("%d %d %d %d",
			&student->English, &student->math,
			&student->physics, &student->Cprogramming);   //输入成绩
		
		student->next = NULL;      //设置为空
		if (head == NULL)
		{
			head = student;        //head指向第一位
		}
		else
		{
			stu* current = head;
			while (current->next != NULL)
			{
				current = current->next;      //向下查找到结尾
			}
			current->next = student;   //让上一个结构体指向本个
		}

	}
	printf("完成了%d位同学的成绩输入。\n", n);
}
//展示学生信息
void display1() {

	struct Student* current = head;
	printf("学生信息如下:\n");
	while (current != NULL) {
		printf("%-10s\t", current->ID);
		printf("%-10s\t", current->name);
		printf("%-3d%-3d%-3d%-3d\n", current->English,current->math,current->physics
			,current->Cprogramming);
		current = current->next;
	}
}
void display2() 
{
	struct Student* current = head;

	while (current != NULL) {
		printf("%s\t", current->ID);
		printf("%s\t", current->name);
		printf("%d\t", current->Cprogramming + current->English + current->physics
			+ current->math);
		printf("%.2f\n", current->ave);
		current = current->next;
	}
}
void sta()
{		
		struct Student* current = head;
		while (current != NULL) {
			current->ave = (current->English + current->math +
				current->physics + current->Cprogramming) / 4.0;
			current = current->next;
		}
	
}
//排序算法
void sortScores() {
	// 如果链表为空或只有一个节点，无需排序
	if (head == NULL || head->next == NULL) {
		return;
	}

	int swap = 1;      // 用于检测是否发生交换
	stu* temp = NULL;  // 用于交换节点
	stu* end = NULL;   // 指向已排序部分的末尾

	while (swap) {//若没有发生交换，即swap为0，则退出循环
		swap = 0;          // 在每次外部循环开始时重置交换标志
		stu* current = head;  // 从链表头开始遍历

		while (current->next != end) {
			stu* next = current->next;
			if (current->ave > next->ave)
			{
				// 交换节点
				if (current == head) {
					head = next;  // 重置头指针
				}
				else {
					temp->next = next;//temp为prev，更改指向
				}

				current->next = next->next; //交换
				next->next = current;
				temp = next;//temp向右移动
				swap = 1;  // 标记发生了交换
			}
			else {
				temp = current;
				current = current->next;
			}
		}
		end = current;  // 每轮大循环结束时，已排序部分的末尾为当前节点
	}
}
//修改学生信息
void modify() {
	char targetID[10];
	int Type;
	char Content[20];
	printf("修改项目:\n");
	printf("0.姓名\n1.英语成绩\n2.数学成绩\n3.物理成绩\n4.C语言成绩\n");
	printf("请输入要修改的学生的学号，修改项目，修改后的内容：");
	scanf("%s%d%s", targetID,&Type, Content);

	stu* current = head;
	while (current != NULL) {
		if (strcmp(current->ID, targetID) == 0) //比较是否相等
		{
			switch (Type) {
			case 0:
				strcpy(current->name, Content);
				break;
			case 1:
				current->English = atoi(Content);
				break;
			case 2:
				current->math = atoi(Content);//atoi将字符串转化为整型值
				break;
			case 3:
				current->physics = atoi(Content);
				break;
			case 4:
				current->Cprogramming = atoi(Content);
				break;
			default:
				printf("无效的修改类型\n");
				return;
			}
			current->ave = (double)(current->English + current->math
				+ current->physics + current->Cprogramming) / 4.0;
			printf("%s ", current->ID);
			printf("%s ", current->name);
			printf("%d %d %d %d\n", current->English, current->math,
				current->physics, current->Cprogramming);
			return;
		}
		current = current->next;//current下移

	}
}
void menu()
{
	printf("请输入选项\t\n");
	printf("1.输入每个学生的各项信息\n");
	printf("2.输出每个学生的各项信息\n");
	printf("3.修改指定学生的指定数据项的内容\n");
	printf("4.统计每个同学的平均成绩\n");
	printf("5.输出各位同学的学号、姓名、四门课程的总成绩和平均成绩\n");
	printf("6.按照平均成绩进行升序排序\n");
	printf("0.退出程序\n");
}
int main()
{
	menu();
	int choice = -1;
	while (choice != 0)
	{
		scanf("%d", &choice);
		switch (choice)
		{
		case 1:addStudent();
			break;
		case 2:display1();
			break;
		case 3:modify();
			break;
		case 4:sta();
			break;
		case 5:display2();
			break;
		case 6:sortScores();
			break;
		case 0:exit(0);

		default:printf("无效输入\n");

		}
	}
	//释放内存
	stu* current = head;
	while (current != NULL) {
		stu* next = current->next;
		free(current);
		current = next;
	}

	return 0;
}
