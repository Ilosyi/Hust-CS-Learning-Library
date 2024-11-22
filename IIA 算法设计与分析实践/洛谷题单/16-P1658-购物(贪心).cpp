#include <iostream>
#include <algorithm>
using namespace std;

int x, n, a[1001], sum = 0, ans = 0;
bool has_one = false;

int main() {
    cin >> x >> n;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        if (a[i] == 1) has_one = true; // 特判是否有面值为1的硬币
    }
    //如果没有面值为 1，就无法组合出所有值（例如1就无法组合）。

    if (!has_one) { // 如果没有面值为1，输出无解
        cout << -1 << endl;
        return 0;
    }
    sort(a, a + n); // 硬币按从小到大排序
    //按照从小到大的顺序检查每个整数 i∈[1, X]，判断当前是否能够被已有硬币组合出。
    for (int i = 1; i <= x; i++) {
       
//如果当前金额i小于或等于已有硬币的总和sum，
// 那么金额i可以被已有硬币覆盖，无需新增硬币。
//如果当前金额i大于已有硬币的总和sum即i> sum），
// 说明当前的硬币组合无法覆盖金额i，因此我们需要新增一个硬币。
        if (sum >= i) continue; // 如果当前总和已经覆盖i，跳过
        for (int j = n - 1; j >= 0; j--) {
            if (a[j] <= i) { // 选择最大的符合条件的硬币
                ans++;       // 多带一枚硬币
                sum += a[j]; // 更新可覆盖的总范围
                break;
            }
        }
        if (sum >= x) break; // 如果已经覆盖到x，提前结束
    }
    cout << ans << endl;
    return 0;
}
