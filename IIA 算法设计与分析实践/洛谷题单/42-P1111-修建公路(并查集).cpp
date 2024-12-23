#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

// 并查集类
class UnionFind
{
private:
    vector<int> parent; // 每个节点的父节点
    int count;          // 当前集合的数量

public:
    // 构造函数：初始化并查集
    UnionFind(int n) : parent(n + 1), count(n)
    {
        for (int i = 1; i <= n; ++i)
        {
            parent[i] = i; // 每个节点的初始父节点是自己
        }
    }

    // 查找某节点的祖先节点（路径压缩优化）
    int find(int x)
    {
        if (parent[x] != x)
        {
            parent[x] = find(parent[x]); // 路径压缩
        }
        return parent[x];
    }

    // 合并两个集合
    void unite(int x, int y)
    {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY)
        {
            parent[rootX] = rootY; // 合并两个集合
            --count;               // 每合并一个集合，集合数量减少 1
        }
    }

    // 获取当前集合的数量
    int getCount() const
    {
        return count;
    }
};

// 定义表示公路的类
class Road
{
public:
    int village1, village2, time; // 两个村庄编号和修建时间

    // 构造函数
    Road(int v1, int v2, int t) : village1(v1), village2(v2), time(t) {}

    // 重载小于号，用于按修建时间排序
    bool operator<(const Road &rhs) const
    {
        return time < rhs.time;
    }
};

int main()
{
    int n, m; // n 表示村庄数量，m 表示公路数量
    cin >> n >> m;

    // 存储所有公路信息
    vector<Road> roads;
    for (int i = 0; i < m; ++i)
    {
        int x, y, t;
        cin >> x >> y >> t;
        roads.emplace_back(x, y, t);
    }

    // 按修建时间升序排序所有公路
    sort(roads.begin(), roads.end());

    // 初始化并查集
    UnionFind uf(n);

    // 遍历所有公路，逐步连接村庄
    for (const auto &road : roads)
    {
        uf.unite(road.village1, road.village2);

        // 如果所有村庄已经连通
        if (uf.getCount() == 1)
        {
            // 输出最后一条公路的修建时间
            cout << road.time;
            return 0;
        }
    }

    // 如果遍历完所有公路后，仍然无法连通所有村庄，输出 -1
    cout << "-1";
    return 0;
}
