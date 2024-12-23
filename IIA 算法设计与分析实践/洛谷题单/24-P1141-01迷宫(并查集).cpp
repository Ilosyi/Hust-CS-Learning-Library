#include <iostream>
#include <vector>
using namespace std;

// 并查集结构
class UnionFind
{
public:
    vector<int> parent, size;

    UnionFind(int n)
    {
        parent.resize(n);
        size.resize(n, 1);
        for (int i = 0; i < n; ++i)
        {
            parent[i] = i;
        }
    }

    // 查找根节点，带路径压缩
    int find(int x)
    {
        if (parent[x] != x)
        {
            parent[x] = find(parent[x]);
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
            if (size[rootX] < size[rootY])
            {
                swap(rootX, rootY);
            }
            parent[rootY] = rootX;
            size[rootX] += size[rootY];
        }
    }

    // 获取某集合的大小
    int getSize(int x)
    {
        return size[find(x)];
    }
};

int main()
{
    int n, m;
    cin >> n >> m;
    vector<string> maze(n);
    for (int i = 0; i < n; ++i)
    {
        cin >> maze[i];
    }

    // 定义方向数组
    const int dx[4] = {0, 0, 1, -1};
    const int dy[4] = {1, -1, 0, 0};

    // 初始化并查集
    UnionFind uf(n * n);

    // 遍历迷宫，合并相邻格子
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int d = 0; d < 4; ++d)
            {
                int ni = i + dx[d];
                int nj = j + dy[d];
                if (ni >= 0 && ni < n && nj >= 0 && nj < n)
                {
                    // 若当前格子和目标格子满足移动条件，则合并
                    if (maze[i][j] != maze[ni][nj])
                    {
                        uf.unite(i * n + j, ni * n + nj);
                    }
                }
            }
        }
    }

    // 处理每个查询
    for (int k = 0; k < m; ++k)
    {
        int x, y;
        cin >> x >> y;
        --x, --y; // 转换为 0 索引
        cout << uf.getSize(x * n + y) << endl;
    }

    return 0;
}
