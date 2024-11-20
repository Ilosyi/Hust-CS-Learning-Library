#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#define N 10
#define M 3

int circle[N] = { 0 };
void Solve(int n, int m, int pos)
{
	int cnt;
	if (n == 1)
	{
		return;
	}
	for (cnt = 1; cnt <= m; pos++)
	{
		pos %= N;
		if (circle[pos] == 0) cnt++;
	}
	circle[pos - 1] = 1;
	printf("%6d", pos);
	Solve(n - 1, m, pos);
}
int main()
{
	int i;
	Solve(N, M, 0);
	for (i = 0; circle[i]; i++)
		;
	printf("%6d", i + 1);
	return 0;
}
