#include <iostream>
#include <vector>
using namespace std;

//输入高精度整数N去掉其中任意 k个数字后剩下的数字按原左右次序将组成一个新的非负整数
//寻找一种方案使得剩下的数字组成的新数最小
//思路：从左到右找到第一个比后面数字大的数字，删除该数字，重复k次
//WA 50074897
//2
//删除前导0
int main()
{
	string N;
	int k;
	cin >> N >> k;
	vector<int> num;
	for (int i = 0; i < N.size(); i++)//将字符串转换为数字
	{
		num.push_back(N[i] - '0');
	}
	//如果k大于等于N的长度，那么删除所有数字，输出0
	if (k >= N.size())
	{
		cout << 0;
		return 0;
	}
	for (int i = 0; i < k; i++)//删除k个数字
	{
		int j = 0;
		while (j < num.size() - 1 && num[j] <= num[j + 1])
		{
			j++;
		}
		//cout << num[j] << endl;
		//删除第j个数字
		num.erase(num.begin() + j);//例如:1 7 5 4 3 8
	}
	//删除前导0
	while (num.size() > 1 && num[0] == 0)//当长度大于1且第一个数字为0时删除第一个数字
	{
		num.erase(num.begin());
	}
	for (int i = 0; i < num.size(); i++)
	{
		cout << num[i];
	}
	
	return 0;

}
