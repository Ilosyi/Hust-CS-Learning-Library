#include <iostream>
#include<vector>
#include<queue>
#include<map>
using namespace std;


struct Point
{
	int x;
	int y;
	int dist;
};

int StartX, StartY, EndX, EndY;//起点坐标，终点坐标
map<pair<int, int>, pair<int, int>> Portals;//传送门
void findPortals(vector<vector<char>>& maze, int n, int m);
void BFS(vector<vector<char>>& maze, vector<vector<int>>& mark, int n, int m);

int main()
{
	int n, m;
	cin >> n >> m;
	//迷宫
	vector<vector<char>> maze(n, vector<char>(m, 0));
	//标记数组
	vector<vector<int>> mark(n, vector<int>(m, INT_MAX));//记录到达每个点的最短时间
	//读取地图	
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++) 
		{
			cin >> maze[i][j];
			if (maze[i][j] == '@')//获取起点坐标
			{
				StartX = i;
				StartY = j;
			}
			else if (maze[i][j] == '=')//获取终点坐标
			{
				EndX = i;
				EndY = j;
			}

		}
	}
	findPortals(maze, n, m);//找到传送门
	//调试：遍历传送门
	/*for (auto it = Portals.begin(); it != Portals.end(); it++)
	{
		cout << it->first.first << " " << it->first.second << " " << it->second.first << " " << it->second.second << endl;
	}*/	
	BFS(maze, mark, n, m);//BFS
}
void findPortals(vector<vector<char>>& maze, int n, int m)
{
	//遍历地图，找到传送门
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (isupper(maze[i][j]))
			{
				//如果该传送门已经有记录
				if (Portals.find({ i, j }) != Portals.end())
				{
					continue;
				}
				int objectX = -1, objectY = -1;
				//找到传送门对应的另一个传送门
				for (int k = 0; k < n; k++)
				{
					for (int l = 0; l < m; l++)
					{
						if (maze[k][l] == maze[i][j] && (k != i || l != j))
						{
							objectX = k;
							objectY = l;
							break;
						}
					}
				}
				Portals[{i, j}] = { objectX, objectY };
				Portals[{objectX, objectY}] = { i, j };
			}
		}
	}
}
void BFS(vector<vector<char>>& maze, vector<vector<int>>& mark, int n, int m)
{
	//方向数组
	int dx[4] = { 0, 0, 1, -1 };
	int dy[4] = { 1, -1, 0, 0 };
	
	//队列
	queue<Point> q;
	
	q.push({ StartX, StartY, 0 });//起点入队
	mark[StartX][StartY] = 0;//标记到达起点的时间为0
	//BFS
	while (!q.empty())
	{
		Point now = q.front();
		q.pop();
		//调试：当前访问的点
		//cout <<"当前点:"<< now.x << " " << now.y << " " << now.dist << endl;
		//如果到达终点，输出时间
		if (now.x==EndX&&now.y==EndY)
		{
			cout << now.dist << endl;
			return;
		}
		//如果是传送门
		if (isupper(maze[now.x][now.y]))
		{
			auto portal = Portals[{now.x, now.y}];
			now.x = portal.first;
			now.y = portal.second;
			//cout << "传送到:"<<now.x << " " << now.y << " " << now.dist << endl;
		}
		//遍历四个方向
		for (int i = 0; i < 4; i++)
		{
			int newX = now.x + dx[i];
			int newY = now.y + dy[i];
			if (newX >= 0 && newX < n && newY >= 0 && newY < m && maze[newX][newY] != '#' && mark[newX][newY] > now.dist + 1) {
				mark[newX][newY] = now.dist + 1;
				q.push({ newX, newY, now.dist + 1 });
				
			}
			
		}
	}
	
}
