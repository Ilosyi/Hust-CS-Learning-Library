#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 方向数组，分别表示上下左右四个方向的坐标偏移量
const int dirs[4][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

// 计算从坐标 (r, c) 出发的最长滑坡长度
// 参数说明：
// - height：存储各位置高度值的二维向量，代表地形高度信息
// - dp：存储已经计算过的各位置最长滑坡长度的二维向量，用于避免重复计算
// - r：当前位置的行坐标
// - c：当前位置的列坐标
int dfs(const vector<vector<int>>& height, vector<vector<int>>& dp, int x, int y) {
    // 如果当前位置的最长滑坡长度已经计算过（dp[r][c] 不为 -1），直接返回该值
    if (dp[x][y] != -1) return dp[x][y];

    // 初始化当前位置的最长滑坡长度为 1，因为至少包含自身这个点
    dp[x][y] = 1;

    // 遍历四个方向
    for (const auto& d : dirs) {
        int newX = x + d[0];  // 新位置的行坐标
        int newY = y + d[1];  // 新位置的列坐标

        // 判断新坐标是否在合法范围内，并且新位置的高度小于当前位置的高度
        if (newX >= 0 && newX < height.size() && newY >= 0 && newY < height[0].size() && height[newX][newY] < height[x][y]) {
            // 更新当前位置的最长滑坡长度，取当前长度和递归计算得到的新长度 + 1 中的较大值
            dp[x][y] = max(dp[x][y], dfs(height, dp, newX, newY) + 1);
        }
    }

    return dp[x][y];
}

int main() {
    int R, C;  // R 表示矩阵的行数，C 表示矩阵的列数
    // 输入矩阵的行数和列数
    cin >> R >> C;

    // 存储地形高度信息的二维向量，根据输入的行数和列数进行初始化
    vector<vector<int>> height(R, vector<int>(C));
    // 存储各位置最长滑坡长度的二维向量，初始化为 -1
    vector<vector<int>> dp(R, vector<int>(C, -1));

    // 输入每个位置的地形高度值
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            cin >> height[r][c];
        }
    }

    int maxLen = 0;  // 用于记录最长滑坡长度的最大值
    // 遍历整个矩阵的每个位置，计算并更新最长滑坡长度的最大值
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            maxLen = max(maxLen, dfs(height, dp, r, c));
        }
    }

    // 输出最长滑坡长度
    cout << maxLen << endl;
    return 0;
}
