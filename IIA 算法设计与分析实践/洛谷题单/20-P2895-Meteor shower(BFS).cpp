#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

const int MAX_COORD = 301;       // 坐标限制为 [0, 300]，但可能超出
const int INF = INT_MAX;         // 表示某格子永远不会被流星砸中
const int dx[4] = {1, -1, 0, 0}; // 四个方向：右、左、下、上
const int dy[4] = {0, 0, 1, -1};

int impactTime[MAX_COORD * 2][MAX_COORD * 2]; // 每个格子的最早受流星影响时间

void initializeImpactTime()
{
    // 初始化所有格子为安全状态（永远不会被流星砸中）
    for (int i = 0; i < MAX_COORD * 2; ++i)
    {
        for (int j = 0; j < MAX_COORD * 2; ++j)
        {
            impactTime[i][j] = INF;
        }
    }
}

void updateImpactTime(int x, int y, int t)
{
    // 更新流星砸中 (x, y) 及其四周的格子
    for (int i = 0; i < 4; ++i)
    {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && ny >= 0 && nx < MAX_COORD * 2 && ny < MAX_COORD * 2)
        {
            impactTime[nx][ny] = min(impactTime[nx][ny], t);
        }
    }
    // 更新流星直接砸中的格子
    if (x >= 0 && y >= 0 && x < MAX_COORD * 2 && y < MAX_COORD * 2)
    {
        impactTime[x][y] = min(impactTime[x][y], t);
    }
}

int bfs()
{
    // 使用队列进行 BFS，存储 (x, y, t)，即当前位置和当前时间
    queue<pair<int, int>> q;
    queue<int> timeQ;

    q.push({0, 0});
    timeQ.push(0);

    // 用于记录访问过的状态
    vector<vector<bool>> visited(MAX_COORD * 2, vector<bool>(MAX_COORD * 2, false));
    visited[0][0] = true;

    // BFS 搜索
    while (!q.empty())
    {
        auto [x, y] = q.front();
        int t = timeQ.front();
        q.pop();
        timeQ.pop();

        // 如果该格子永远不会被流星砸中，则找到答案
        if (impactTime[x][y] == INF)
        {
            return t;
        }

        // 扩展到四个方向
        for (int i = 0; i < 4; ++i)
        {
            int nx = x + dx[i];
            int ny = y + dy[i];

            // 如果新位置在范围内，且未被访问过，且安全
            if (nx >= 0 && ny >= 0 && nx < MAX_COORD * 2 && ny < MAX_COORD * 2 &&
                !visited[nx][ny] && t + 1 < impactTime[nx][ny])
            {
                visited[nx][ny] = true;
                q.push({nx, ny});
                timeQ.push(t + 1);
            }
        }
    }

    // 如果无法找到安全的格子，返回 -1
    return -1;
}

int main()
{
    int m;
    cin >> m;

    // 初始化 impactTime
    initializeImpactTime();

    // 读取流星数据，并更新每个流星的影响范围
    for (int i = 0; i < m; ++i)
    {
        int x, y, t;
        cin >> x >> y >> t;
        updateImpactTime(x, y, t);
    }

    // 使用 BFS 搜索最短时间
    int result = bfs();

    // 输出结果
    cout << result << endl;

    return 0;
}
