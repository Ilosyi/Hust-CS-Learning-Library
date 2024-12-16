#include <iostream>
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

// 定义请求结构体
struct Request
{
    int b, e, t; // b: 区间开始, e: 区间结束, t: 需要的树的数量
    // 重载小于运算符，用于排序
    bool operator<(const Request &other) const
    {
        if (e == other.e)
            return b > other.b; // 若结束位置相同，按开始位置从大到小排序
        return e < other.e; // 按区间结束位置排序
    }
};

// 计算最少需要的树的数量
int min_trees(int n, vector<Request> &requests)
{
    // 按区间的结束位置排序请求
    sort(requests.begin(), requests.end());

    // 树的种植情况
    set<int> trees; // 使用集合来记录种树的位置，减少空间使用
    vector<int> prefix_sum(n + 1, 0); // 前缀和数组，用于快速计算区间内已种的树数量

    for (const auto &req : requests)
    {
        int b = req.b, e = req.e, t = req.t; // 读取请求的区间和需要的树的数量

        // 区间 [b, e] 已经种了的树数量
        int already_planted = prefix_sum[e] - prefix_sum[b - 1];

        // 需要额外种的树数量
        int extra_needed = max(0, t - already_planted);

        // 从右往左在区间 [b, e] 种树，尽量靠右种
        for (int i = e; i >= b; --i)
        {
            if (extra_needed == 0) break; // 提前退出循环
            if (trees.count(i) == 0)
            {
                trees.insert(i);
                extra_needed -= 1;

                // 仅更新受影响的区间部分的前缀和
                for (int j = i; j <= e; ++j)
                {
                    prefix_sum[j] += 1;
                }
            }
        }
    }

    // 返回种树的总数
    return trees.size();
}

int main()
{
    int n, m;
    cin >> n >> m; // 读取位置数量和请求的数量

    vector<Request> requests(m);
    for (int i = 0; i < m; ++i)
    {
        cin >> requests[i].b >> requests[i].e >> requests[i].t;
    }

    cout << min_trees(n, requests) << endl; // 输出最少需要的树的数量
    return 0;
}
