#include<iostream>
#include<algorithm>
#include<vector>
#define INT_MAX 2147483647
using namespace std;

struct Lamp {
    int power;   // 功率
    int position; // 位置
};

int ans = INT_MAX;  // 最小功耗结果（初始值为无穷大）
int current_energy = 0; // 当前累积能量消耗
int total_power = 0;    // 剩余功率总和
int n, pl;              // 路灯总数和老张初始位置（1-based索引）
Lamp lamps[55];         // 存储每盏灯的信息
bool visited[55] = { false };       // 标记灯是否关闭

// DFS 搜索函数
void dfs(int now) {
    bool hasNext = false; // 标记是否还有灯未关闭

    // 剪枝：当前能量消耗超过最优解时直接返回
    if (current_energy >= ans) return;

    // 向右探索
    for (int i = now + 1; i <= n; i++) {
        if (!visited[i]) { // 找到未关闭的灯
            
            visited[i] = true;

            // 更新当前能量和剩余功率
            int distance = lamps[i].position - lamps[now].position;
            current_energy += distance * total_power;//distance即时间,相乘结果为功耗
			total_power -= lamps[i].power;//当前剩余功率

            dfs(i); // 递归探索

            // 回溯：恢复状态
            total_power += lamps[i].power;
            current_energy -= distance * total_power;
            visited[i] = false;
            hasNext = true;
            break;
        }
    }

    // 向左探索
    for (int i = now - 1; i >= 1; i--) 
    {
        if (!visited[i]) { // 找到未关闭的灯
			
            visited[i] = true;

            // 更新当前能量和剩余功率
            int distance = lamps[now].position - lamps[i].position;
            current_energy += distance * total_power;
            total_power -= lamps[i].power;

            dfs(i); // 递归探索

            // 回溯：恢复状态
            total_power += lamps[i].power;
            current_energy -= distance * total_power;
            visited[i] = false;

            hasNext = true;//需要继续探索
            break;
        }
    }

    // 如果没有灯可以继续探索，更新最优解
    if (!hasNext) {
        ans = min(ans, current_energy);
    }
}

int main() {
    // 输入灯的数量和老张的初始位置
    cin >> n >> pl;
    // 输入每盏灯的位置和功率,位置默认升序
    for (int i = 1; i <= n; i++) {
        cin >> lamps[i].position >> lamps[i].power;
        total_power += lamps[i].power; // 累加所有灯的功率
    }
    // 老张所在的灯视为已经关闭
    visited[pl] = true;
    total_power -= lamps[pl].power;

    // 开始 DFS 搜索
    dfs(pl);

    // 输出最小功耗
    cout << ans << endl;

    return 0;
}
