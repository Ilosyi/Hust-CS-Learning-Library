#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

const int MAXN = 20;  // 单词最多为 20 个
int n;                // 单词数量
vector<string> words; // 单词列表
int used[MAXN];       // 每个单词的使用次数
int maxLength = 0;    // 最长龙的长度

int calculateOverlap(const string &a, const string &b)
{
    int lenA = a.length(), lenB = b.length();

    // 从单词 a 的尾部倒序查找与单词 b 头部的匹配
    for (int i = lenA - 1; i >= 1; --i)
    { // i 表示从 a 的第 i 个字符开始尝试匹配
        bool canOverlap = true;
        int overlapLength = 0;

        // 比较 a 从 i 开始的部分与 b 的前缀
        for (int k = i, j = 0; k < lenA && j < lenB; ++k, ++j)
        {
            if (a[k] != b[j])
            { // 如果字符不匹配，则无法重叠
                canOverlap = false;
                break;
            }
            ++overlapLength;
        }

        // 检查重叠是否合法
        if (canOverlap && i + overlapLength == lenA && overlapLength < lenB)
        {
            return overlapLength; // 返回重叠长度
        }
    }

    return 0; // 如果没有合法重叠部分，则返回 0
}

void dfs(const string &current, int length)
{
    // 更新最大长度
    maxLength = max(maxLength, length);

    // 尝试将每个单词接到当前单词龙的尾部
    for (int i = 0; i < n; ++i)
    {
        // 如果该单词尚未被使用两次
        if (used[i] < 2)
        {
            // 计算当前单词与候选单词之间的重叠长度
            int overlapLength = calculateOverlap(current, words[i]);

            if (overlapLength > 0)
            { // 如果可以重叠
                // 使用当前单词
                ++used[i];
                // 递归调用，继续拼接单词龙
                dfs(current + words[i].substr(overlapLength), length + words[i].size() - overlapLength);
                // 回溯，恢复状态
                --used[i];
            }
        }
    }
}

int main()
{
    // 输入单词数量
    cin >> n;

    // 输入所有单词
    words.resize(n);
    for (int i = 0; i < n; ++i)
    {
        cin >> words[i];
    }

    // 输入起始字符
    char startChar;
    cin >> startChar;

    // 从起始字符开头的单词开始 DFS
    for (int i = 0; i < n; ++i)
    {
        if (words[i][0] == startChar)
        {                                   // 找到以 startChar 开头的单词
            ++used[i];                      // 标记该单词已使用一次
            dfs(words[i], words[i].size()); // 开始 DFS
            --used[i];                      // 回溯
        }
    }

    // 输出最长单词龙的长度
    cout << maxLength << endl;

    return 0;
}