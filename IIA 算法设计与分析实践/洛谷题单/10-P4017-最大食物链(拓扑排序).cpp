#include <iostream>
#include <vector>
#include <queue>
using namespace std;

const int MOD = 80112002;

int main()
{
    int n, m;//生物种类和被吃的关系数
    cin >> n >> m;

    // 动态分配内存
    vector<int> in_degree(n + 1, 0);//入度
    vector<int> out_degree(n + 1, 0);//出度
	vector<vector<int>> adj(n + 1);//邻接表
	vector<int> dp(n + 1, 0);//路径数量
	queue<int> topoQueue;//拓扑排序队列
	int result = 0;//最终结果

    // 读取边的信息并构建图
    for (int i = 0; i < m; ++i)
    {
		int A, B;//B吃A
        cin >> A >> B;
        ++in_degree[B]; // 目标节点入度+1
        ++out_degree[A]; // 起始节点出度+1
        adj[A].push_back(B); // 建立一条单向边
    }

    // 初次寻找入度为0的点(最佳生产者)
    for (int i = 1; i <= n; ++i)
    {
        if (in_degree[i] == 0)
        {
            dp[i] = 1; // 初始化路径数量
            topoQueue.push(i); // 压入队列
        }
    }

    // 拓扑排序并计算路径数量
    while (!topoQueue.empty())
    {
        int u = topoQueue.front(); // 取出队首
        topoQueue.pop(); // 弹出队首

        // 遍历当前节点的所有邻接节点
		for (int v : adj[u])//等同于for(int j=0;j<adj[u].size();j++) v=adj[u][j]
        {
            in_degree[v]--; // 将邻接节点的入度-1
            dp[v] = (dp[v] + dp[u]) % MOD; // 更新到邻接节点的路径数量
            if (in_degree[v] == 0)
            {
                topoQueue.push(v); // 如果邻接节点的入度为0，压入队列
            }
        }
    }

    // 寻找出度为0的点(最佳消费者)并累加路径数量
    for (int i = 1; i <= n; ++i)
    {
        if (out_degree[i] == 0)
        {
            result = (result + dp[i]) % MOD; // 累加答案
        }
    }

    cout << result << endl; // 输出最终结果
    return 0; // 程序结束
}

