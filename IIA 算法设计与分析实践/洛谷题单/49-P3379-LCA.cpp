#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

const int MAXN = 500000; // 最大节点数量
const int LOG = 20;      // 最大深度 log2(500000)

vector<int> tree[MAXN + 1]; // 邻接表表示树
int fa[MAXN + 1][LOG];      // fa[u][k] 表示 u 的第 2^k 个祖先
int depth[MAXN + 1];        // 每个节点的深度
int n, m, root;

// 深度优先搜索预处理父节点信息和深度
void dfs(int node, int parent)
{
    fa[node][0] = parent; // node 的第 1 个祖先是 parent
    depth[node] = depth[parent] + 1;

    // 预处理倍增法的祖先表
    for (int k = 1; k < LOG; ++k)
    {
        if (fa[node][k - 1] != 0)
        {
            fa[node][k] = fa[fa[node][k - 1]][k - 1];
        }
        else
        {
            break;
        }
    }

    // 遍历子节点
    for (int child : tree[node])
    {
        if (child != parent)
        { // 避免回到父节点
            dfs(child, node);
        }
    }
}

// 查询 LCA
int lca(int a, int b)
{
    if (depth[a] < depth[b])
        swap(a, b); // 保证 a 比 b 深

    // 让 a 和 b 处于相同的深度
    for (int k = LOG - 1; k >= 0; --k)
    {
        if (depth[a] - (1 << k) >= depth[b])
        {
            a = fa[a][k];
        }
    }

    if (a == b)
        return a; // 如果 b 是 a 的祖先，则直接返回

    // 同步向上找公共祖先
    for (int k = LOG - 1; k >= 0; --k)
    {
        if (fa[a][k] != fa[b][k])
        {
            a = fa[a][k];
            b = fa[b][k];
        }
    }

    return fa[a][0]; // 返回 LCA
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 输入节点数、询问数和根节点
    cin >> n >> m >> root;

    // 构建树
    for (int i = 1; i < n; ++i)
    {
        int x, y;
        cin >> x >> y;
        tree[x].push_back(y);
        tree[y].push_back(x);
    }

    // 初始化 DFS，预处理倍增法祖先表
    depth[0] = 0; // 虚拟根节点的深度为 0
    dfs(root, 0);

    // 处理每个查询
    while (m--)
    {
        int a, b;
        cin >> a >> b;
        cout << lca(a, b) << '\n';
    }

    return 0;
}