#include <iostream>
#include<vector>
#include<queue>
using namespace std;

//有一个n×m 的棋盘，在某个点(x,y)上有一个马，要求你计算出马到达棋盘上任意一个点最少要走几步
//到达不了的点用-1表示
/*
算法步骤：
1.马走日字，共有8个方向
2.使用BFS，从马的位置开始，每次向8个方向走一步，直到所有点都被访问
3.使用队列存储每次的位置，使用pair存储坐标
4.使用一个二维数组存储每个点到马的位置的最短距离
5.将无法到达的点标记为-1
6.注意初始点位置可能被重新访问，再次标记为0
7.输出棋盘
*/
void BFS(vector<vector<int>>& chess, int x, int y,int n,int m);
int main() {
	int n, m, x, y;//n行m列，马的位置(x,y)
	cin >> n >> m >> x >> y;
	//棋盘
	vector<vector<int>> chess(n + 1, vector<int>(m + 1,0));
	BFS(chess, x, y,n,m);//计算马到达棋盘上任意一个点最少要走几步
	for (int i = 1; i <= n; i++) //输出棋盘
    {
		for (int j = 1; j <= m; j++)
        {
			cout << chess[i][j] << " ";
		}
		cout << endl;
	}
	return 0;
}

//BFS
void BFS(vector<vector<int>>& chess, int x, int y,int n,int m) {
    //马走日字，共有8个方向
    // 定义方向数组，表示马的八个可能移动方向
    int dx[8] = { 1, 1, 2, 2, -1, -1, -2, -2 };
    int dy[8] = { 2, -2, 1, -1, 2, -2, 1, -1 };

	queue<pair<int, int>> q;//队列,用pair存储坐标
    q.push({ x, y });
    chess[x][y] = 0; // 起始点到自身的距离为0

    while (!q.empty()) 
    {
		// 取出队首元素
        int nowX = q.front().first;
		int nowY = q.front().second;
        q.pop();

        for (int i = 0; i < 8; i++) 
        {
            int newX = nowX + dx[i];
            int newY = nowY + dy[i];
            // 判断范围，且未被访问
            if (newX >= 1 && newX <= n && newY >= 1 && newY <= m && chess[newX][newY] == 0) 
            {
                chess[newX][newY] = chess[nowX][nowY] + 1; // 更新步数
				q.push({ newX, newY });//将新位置加入队列
            }
        }
    }

    // 将无法到达的点标记为-1
    for (int i = 1; i <= n; i++) 
    {
        for (int j = 1; j <= m; j++) 
        {
            if (chess[i][j] == 0 && !(i == x && j == y)) 
            {
                chess[i][j] = -1;
            }
        }
    }
	chess[x][y] = 0; //注意上面if判断中，初始点位置可能被重新访问，再次标记为0
}
