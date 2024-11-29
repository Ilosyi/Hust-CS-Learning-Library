#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
/*代码解释：
1. 初始化：首先初始化 suffixArray 数组，表示每个后缀的起始位置。
rank 数组存储每个后缀的排名（根据字典序排序）。
2. 倍增排序：从长度为1的后缀开始，逐渐扩展排序范围。每次排序时，
不仅考虑当前的字符，还考虑后续字符（即后缀的第二部分）。
倍增的关键是每次使用 len 表示当前考虑的后缀的长度，并且排序时比较的是后缀的前后两部分。
3. 排序与重计算秩：在每次排序后，重计算每个后缀的排名。
如果两个后缀的当前排名相同，进一步通过比较它们的后半部分来区分它们。
4. 输出结果：最终输出后缀数组，数组中每个元素代表该位置的后缀在原字符串中的起始位置。
*/
// 后缀数组构建函数
vector<int> buildSuffixArray(const string& s) {
    int n = s.size();
    vector<int> suffixArray(n), rank(n), tmpRank(n);

    // 初始化后缀数组和秩数组
    for (int i = 0; i < n; ++i) {
        suffixArray[i] = i;
        rank[i] = s[i];
    }

    // 排序，按长度为1的后缀数组排序
    for (int len = 1; len < n; len <<= 1) //倍增，len每次乘2
    {
        cout << "-------------------" << endl;
		cout << "第" << len << "次排序开始" << endl;
        // 自定义比较函数：首先按前半段（rank）排序，如果相等则按后半段（rank+len）排序
        auto cmp = [&](int a, int b) {
            if (rank[a] != rank[b]) return rank[a] < rank[b];
            int ra = (a + len < n) ? rank[a + len] : -1;
            int rb = (b + len < n) ? rank[b + len] : -1;
            return ra < rb;
            };
        //banana
        // 排序
        sort(suffixArray.begin(), suffixArray.end(), cmp);
		//输出本次排序后SuffixArray
        cout << "Suffix Array: " << endl;
		for (int i = 0; i < n; i++) {
			cout << suffixArray[i] << " ";
			for (int j = suffixArray[i]; j < n; j++)
				cout << s[j];
			cout << endl;
		}
        
        // 重新计算秩
		tmpRank[suffixArray[0]] = 0;//第一个后缀的秩为0
        for (int i = 1; i < n; ++i) {
			tmpRank[suffixArray[i]] = tmpRank[suffixArray[i - 1]];//先令秩相同
            if (cmp(suffixArray[i - 1], suffixArray[i])) {
				tmpRank[suffixArray[i]]++;//若前一个后缀小于后一个后缀，则秩加1
            }
        }
        rank = tmpRank;
		//输出本次排序后Rank
		cout << "Rank: ";
        for (int i = 0; i < n; i++) {
			cout << rank[i] << " ";
        }
		cout <<endl<< "第" << len << "次排序完成" << endl;
		cout << "-------------------" << endl;
    }

    return suffixArray;
}

int main() {
    string s;
    cin >> s;
    vector<int> suffixArray = buildSuffixArray(s);

    cout << "Suffix Array: ";
    for (int i : suffixArray) {
        cout << i << " ";//此处按照题意需要+1后再输出
    }
    cout << endl;

    return 0;
}
