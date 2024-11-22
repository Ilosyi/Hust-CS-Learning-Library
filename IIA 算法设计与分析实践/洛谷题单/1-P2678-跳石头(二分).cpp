#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 判断是否可以移除 M 个岩石，使得任意两个岩石之间的最小距离不小于 mid
bool isCanDo(int mid, const std::vector<int>& rocks, int M)
{
    int removeNum = 0;
    int lastRock = 0;  // 上一个保留的岩石位置
    for (int i = 1; i < rocks.size(); ++i) {
        // 当前岩石与上一个保留岩石的距离
		if (rocks[i] - rocks[lastRock] < mid)//如果小于mid
        {
			removeNum += 1;//就需要移除这个岩石，此时不更新lastRock，相当于下次比较时，这个岩石已经被移除了
            if (removeNum > M) {
				return false;//如果移除的岩石数量大于M，返回false
            }
        }
        else {
            lastRock = i;
        }
    }
    return true;
}
int main() {
    // 输入数据
	int L, N, M;//总距离，岩石数量，移除岩石数量
    cin >> L >> N >> M;
	vector<int> rocks(N+2,0);//记录岩石位置
    for (int i = 1; i <=N; ++i) 
    {
        cin >> rocks[i];
    }
    rocks[N + 1] = L;//终点位置
    sort(rocks.begin(), rocks.end());
    // 二分查找
    int left = 1, right = L, Result = 0;
    while (left <= right) 
    {
        int mid = left + (right - left) / 2;
        if (isCanDo(mid, rocks, M))//如果mid可行
        {
            Result = mid;  // 更新答案
            left = mid + 1;  // 扩大范围
        }
        else {
            right = mid - 1;  // 缩小范围
        }
    }
	cout << Result << endl;
    return 0;
}
