#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
// 树状数组
class FenwickTree
{
    vector<int> tree;
    int n; // 数组大小
public:
    // 构造函数
    FenwickTree(int n) : n(n), tree(n + 1, 0) {}
    int lowbit(int x)
    {
        return x & -x;
    }
    // 单点更新
    void update(int x, int v)
    {
        for (int i = x; i <= n; i += lowbit(i))
            tree[i] += v;
    }
    // 查询前缀和
    int query(int x)
    {
        int res = 0;
        for (int i = x; i>0; i -= lowbit(i))
            res += tree[i];
        return res;
    }
    // 区间查询
    int query(int left, int right)
    {
        return query(right) - query(left - 1);
    }
};
int main()
{
    int n, m; // n个数，m个操作
    cin >> n >> m;
    FenwickTree tree(n);
    for (int i = 1; i <= n; i++)
    {
        int x;
        cin >> x;
        tree.update(i, x);
    }
    for(int i=0;i<m;i++)
    {
        int op;
        cin>>op;
        if(op==1)
        {
            int x,k;// 第x个数加上k
            cin>>x>>k;
            tree.update(x,k);
        }
        else if(op==2)
        {
            int x,y;// 区间[x,y]的和
            cin>>x>>y;
            cout<<tree.query(x,y)<<endl;
        }
    }
    return 0;
}
