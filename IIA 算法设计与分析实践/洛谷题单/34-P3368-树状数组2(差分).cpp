
#include <iostream>
#include <vector>
using namespace std;

// 树状数组
class FenwickTree {
    vector<int> tree;
    int n;

    // 返回最低位的1
    int lowbit(int x) {
        return x & -x;
    }

public:
    // 构造函数
    FenwickTree(int size) : n(size), tree(size + 1, 0) {}

    // 单点更新
    void update(int x, int v) {
        for (int i = x; i <= n; i += lowbit(i))
            tree[i] += v;
    }

    // 查询前缀和
    int query(int x) {
        int res = 0;
        for (int i = x; i > 0; i -= lowbit(i))
            res += tree[i];
        return res;
    }
};

// 支持区间更新和单点查询的树状数组
class RangeFenwickTree {
    FenwickTree ft;

public:
    // 构造函数
    RangeFenwickTree(int size) : ft(size) {}

    // 区间更新：对区间 [l, r] 增加 k
    void RangeUpdate(int l, int r, int k) {
        ft.update(l, k);       // 左端点加 k
        ft.update(r + 1, -k);  // 右端点加 -k
    }

    // 单点查询：查询位置 x 的值
    int PointQuery(int x) {
        return ft.query(x);  // 累加前缀和
    }
};

int main() {
    int n, m; // n 个数，m 个操作
    cin >> n >> m;

    RangeFenwickTree tree(n);

    // 初始化数组
    for (int i = 1; i <= n; i++) {
        int x;
        cin >> x;
        tree.RangeUpdate(i, i, x); // 初始值直接通过区间更新加入
    }

    // 执行操作
    for (int i = 0; i < m; i++) {
        int op;
        cin >> op;

        if (op == 1) {
            int x, y, k; // 区间 [x, y] 加上 k
            cin >> x >> y >> k;
            tree.RangeUpdate(x, y, k);
        }
        else if (op == 2) {
            int x; // 单点查询
            cin >> x;
            cout << tree.PointQuery(x) << endl;
        }
    }

    return 0;
}
