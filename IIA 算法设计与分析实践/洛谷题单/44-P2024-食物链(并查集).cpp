#include <iostream>
#include <vector>
using namespace std;

// 简化版并查集
class DisjointSet
{
private:
    vector<int> parent; // 父节点数组

public:
    // 构造函数：初始化并查集
    DisjointSet(int size)
    {
        parent.resize(size);
        for (int i = 0; i < size; i++)
        {
            parent[i] = i; // 每个元素初始是自己的父节点
        }
    }

    // 查找：路径压缩
    int find(int x)
    {
        if (parent[x] != x)
        {
            parent[x] = find(parent[x]); // 递归查找父节点并压缩路径
        }
        return parent[x];
    }

    // 合并：直接合并，无需按秩
    void unite(int x, int y)
    {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY)
        {
            parent[rootY] = rootX; // 将 rootY 挂到 rootX 下
        }
    }
};

int solve_food_chain(int n, int k, vector<vector<int>> &statements)
{
    DisjointSet dsu(3 * n + 1); // 创建大小为 3*n 的并查集
    int false_statements = 0;   // 假话计数器

    for (auto &statement : statements)
    {
        int t = statement[0]; // 关系类型：1 或 2
        int x = statement[1]; // 动物 x
        int y = statement[2]; // 动物 y

        // 条件 2：x 或 y 超出范围
        if (x > n || y > n)
        {
            false_statements++;
            continue;
        }

        if (t == 1)
        { // 第一种关系：x 和 y 是同类
            // 检查冲突：如果 x 的 A 类和 y 的 B 类、C 类有冲突，则是假话
            if (dsu.find(x) == dsu.find(y + n) || dsu.find(x) == dsu.find(y + 2 * n))
            {
                false_statements++;
            }
            else
            {
                // 合并 A 类的 x 和 y
                dsu.unite(x, y);
                // 合并 B 类的 x 和 y
                dsu.unite(x + n, y + n);
                // 合并 C 类的 x 和 y
                dsu.unite(x + 2 * n, y + 2 * n);
            }
        }
        else if (t == 2)
        { // 第二种关系：x 吃 y
            // 条件 3：x 吃自己
            if (x == y)
            {
                false_statements++;
                continue;
            }

            // 检查冲突：x 和 y 是同类，或者 x 吃 y 冲突
            if (dsu.find(x) == dsu.find(y) || dsu.find(x) == dsu.find(y + 2 * n))
            {
                false_statements++;
            }
            else
            {
                // 合并 A 类的 x 和 B 类的 y
                dsu.unite(x, y + n);
                // 合并 B 类的 x 和 C 类的 y
                dsu.unite(x + n, y + 2 * n);
                // 合并 C 类的 x 和 A 类的 y
                dsu.unite(x + 2 * n, y);
            }
        }
    }

    return false_statements;
}

int main()
{
    int n, k; // n 为动物数量，k 为语句数量
    cin >> n >> k;

    vector<vector<int>> statements(k, vector<int>(3)); // 存储语句
    for (int i = 0; i < k; i++)
    {
        cin >> statements[i][0] >> statements[i][1] >> statements[i][2];
    }

    cout << solve_food_chain(n, k, statements) << endl; // 输出假话数量

    return 0;
}
