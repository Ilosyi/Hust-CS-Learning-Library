#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

// 定义结构体用于存储边
struct Edge
{
    int from, to, weight;
};

bool solveConstraints(int n, int m, vector<Edge> &edges)
{
    vector<int> dist(n + 1, INT_MAX); // 距离数组，初始为无穷大
    dist[0] = 0;                      // 虚拟源点的距离设为0

    // 使用Bellman-Ford算法解决差分约束问题
    for (int i = 0; i <= n; i++)
    {
        for (auto &edge : edges)
        {
            if (dist[edge.from] != INT_MAX && dist[edge.to] > dist[edge.from] + edge.weight)
            {
                dist[edge.to] = dist[edge.from] + edge.weight;
            }
        }
    }

    // 检查是否存在负环
    for (auto &edge : edges)
    {
        if (dist[edge.from] != INT_MAX && dist[edge.to] > dist[edge.from] + edge.weight)
        {
            return false; // 存在负环，表示无解
        }
    }

    return true; // 如果没有负环，返回可行
}

int main()
{
    int n, m;
    cin >> n >> m;

    vector<Edge> edges;

    // 输入不等式信息并构造差分约束的边
    for (int i = 0; i < m; i++)
    {
        int type, a, b, c;
        cin >> type;
        if (type == 1)
        {
            cin >> a >> b >> c;
            edges.push_back({b, a, -c}); // 农场a比农场b至少多c -> x_b - x_a <= -c
        }
        else if (type == 2)
        {
            cin >> a >> b >> c;
            edges.push_back({a, b, c}); // 农场a比农场b至多多c -> x_a - x_b <= c
        }
        else if (type == 3)
        {
            cin >> a >> b;
            edges.push_back({a, b, 0}); // 农场a = 农场b -> x_a - x_b <= 0
            edges.push_back({b, a, 0}); // 农场b = 农场a -> x_b - x_a <= 0
        }
    }

    // 添加虚拟源点到所有节点
    for (int i = 1; i <= n; i++)
    {
        edges.push_back({0, i, 0}); // 虚拟源点到所有节点，权重为0
    }

    if (solveConstraints(n, m, edges))
    {
        cout << "Yes" << endl; // 存在可行解
    }
    else
    {
        cout << "No" << endl; // 不存在可行解
    }

    return 0;
}
