#include <algorithm>
#include <iostream>
#include <cstring>
#include <vector>
using namespace std;

int n, m, maxScore=0;  // n: 砖块的层数, m: 最大敲掉的砖块数, maxScore: 最终的最大得分

// 使用vector动态分配内存
vector<vector<int>> brickValue, transformedValue, prefixSum;
vector<vector<vector<int>>> dp;  // dp[i][j][k] 表示在第i层，选了j列，移除了k个砖块的最大得分


// 动态规划函数，用于计算最大得分
void DP() {
    // 动态分配三维DP数组的内存
    dp.resize(n + 1, vector<vector<int>>(n + 1, vector<int>(m + 1, -1)));
  
	// 基础状态：在0层，选了0列，移除多少个砖块都是0分
    for (int i = 0; i <= m; i++) {
        dp[0][0][i] = 0;
    }  
    auto buff = dp;
    // 使用动态规划计算最大得分
    for (int i = 1; i <= n; i++) {  // 遍历每一层
        for (int j = i; j >= 0; j--) {  // 遍历每一列
            for (int k = j; k <= m; k++) {  // 遍历敲掉的砖块数量
                if (buff[i - 1][j - 1>=0?j-1:0][k - j] >= 0) {  // 如果前一个状态有效
                    // 更新当前状态的最大得分
                    dp[i][j][k] =  buff[i - 1][max(0, j - 1)][k - j] + prefixSum[i][j];
					//可以观察到，若我们选择敲掉(i,j),则第i行（原来的第n-j+1列）的砖块都会被敲掉，且还需要敲掉i-1行的j-1列
                    buff[i][j][k] = max(buff[i][j + 1][k], dp[i][j][k]);
					//正向依赖关系,不能直接在dp数组上进行更新，dp数组的值是在计算过程中会被覆盖的
                }
            }
        }
    }
    // 遍历DP数组，找到最大得分
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            maxScore = max(maxScore, dp[i][j][m]);  // 选取最大得分
        }
    }
    // 输出最大得分
    cout<<maxScore;
}

int main() {
    cin>>n>>m;  // 读取砖块的层数n和最多可以敲掉的砖块数m

    // 动态分配二维vector内存
    brickValue.resize(n + 1, vector<int>(n + 1, 0));
    transformedValue.resize(n + 1, vector<int>(n + 1, 0));
    prefixSum.resize(n + 1, vector<int>(n + 1, 0));

    // 读取砖块的分值
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n - i + 1; j++) {
            cin>>brickValue[i][j];
        }
    }

    // 对砖块的分值进行转置和前缀和预处理
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            transformedValue[i][j] = brickValue[j][n - i + 1];  // 转置砖块的分值
            prefixSum[i][j] = prefixSum[i][j - 1] + transformedValue[i][j];  // 计算每列的前缀和
        }
    }
    DP();  // 求解问题
    return 0;
}
