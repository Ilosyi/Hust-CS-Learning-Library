#include <iostream>
#include <vector>
using namespace std;

class SegmentTree {
public:
    struct Node {
        int left, right;         // 节点对应的区间范围
        long long sum;           // 区间的总和
        long long lazy_add;      // 区间加法懒惰标记
        long long lazy_mul;      // 区间乘法懒惰标记
    };

    vector<Node> tree;          // 线段树
    int mod;                    // 模数

    // 构造函数，初始化线段树
    SegmentTree(int n, int mod_val) : mod(mod_val) {
        tree.resize(n * 4);     // 线段树大小为 4 倍区间长度
        build(1, 1, n);         // 从根节点开始建树
    }

    // 析构函数
    ~SegmentTree() {
        tree.clear();
    }

    // 建树：初始化节点
    void build(int node, int start, int end) {
        tree[node].left = start;
        tree[node].right = end;
        tree[node].sum = 0;
		tree[node].lazy_add = 0;// 加法懒惰标记初始化为 0
		tree[node].lazy_mul = 1;// 乘法懒惰标记初始化为 1

        if (start == end) return; // 叶子节点，直接返回

        int mid = (start + end) / 2;
        build(node * 2, start, mid);       // 构建左子树
        build(node * 2 + 1, mid + 1, end); // 构建右子树
    }

    // 推懒惰标记
    void push(int node) {
        if (tree[node].lazy_mul != 1 || tree[node].lazy_add != 0) 
        {
            int lchild = node * 2, rchild = node * 2 + 1;

            // 将当前节点的懒惰标记传递给左右子节点
            apply_lazy(lchild, tree[node].lazy_mul, tree[node].lazy_add);
            apply_lazy(rchild, tree[node].lazy_mul, tree[node].lazy_add);

            // 清除当前节点的懒惰标记
            tree[node].lazy_mul = 1;
            tree[node].lazy_add = 0;
        }
    }

    // 应用懒惰标记
    void apply_lazy(int node, long long mul, long long add) {
		tree[node].lazy_mul = (tree[node].lazy_mul * mul) % mod;// 更新乘法和加法懒惰标记
		tree[node].lazy_add = (tree[node].lazy_add * mul + add) % mod;// 更新乘法和加法懒惰标记

		int length = tree[node].right - tree[node].left + 1;// 区间长度
        tree[node].sum = (tree[node].sum * mul + add * length) % mod;
    }

    // 区间更新：乘法和加法
    void update(int node, int start, int end, long long mul, long long add) {
        if (start <= tree[node].left && end >= tree[node].right) {
            apply_lazy(node, mul, add); // 当前区间完全包含在目标区间内，直接更新
            return;
        }

        push(node); // 推懒惰标记

        int mid = (tree[node].left + tree[node].right) / 2;
        if (start <= mid) update(node * 2, start, end, mul, add);     // 更新左子树
        if (end > mid) update(node * 2 + 1, start, end, mul, add);   // 更新右子树

        // 更新当前节点的值
        tree[node].sum = (tree[node * 2].sum + tree[node * 2 + 1].sum) % mod;
    }

    // 区间查询
    long long query(int node, int start, int end) {
        if (start <= tree[node].left && end >= tree[node].right) { // 当前区间完全包含在目标区间内
            return tree[node].sum % mod;
        }

        push(node); // 推懒惰标记

        int mid = (tree[node].left + tree[node].right) / 2;
        long long result = 0;
        if (start <= mid) result += query(node * 2, start, end);     // 查询左子树
        if (end > mid) result += query(node * 2 + 1, start, end);   // 查询右子树

        return result % mod;
    }
};

int main() {
    int n, q, mod;//该数列数字的个数、操作的总个数和模数
    cin >> n >> q >> mod;

    vector<int> arr(n + 1);
    for (int i = 1; i <= n; ++i) cin >> arr[i];//初始值

    SegmentTree segTree(n, mod);

    // 初始化线段树
    for (int i = 1; i <= n; ++i) {
        segTree.update(1, i, i, 1, arr[i]); // 直接对每个点赋初值
    }

    for(int i=0;i<q;i++)
    {
        int op, x, y, k;
        cin >> op >> x >> y;

        if (op == 1) {         // 区间乘法
            cin >> k;
            segTree.update(1, x, y, k, 0);
        }
        else if (op == 2) {  // 区间加法
            cin >> k;
            segTree.update(1, x, y, 1, k);
        }
        else if (op == 3) {  // 区间求和
            cout << segTree.query(1, x, y) << endl;
        }
    }

    return 0;
}
