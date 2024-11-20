#include<iostream>
#include<algorithm>
#include<vector>
#include<map>
using namespace std;

struct Building
{
    int left, right;//建筑物左右边界
    int height;//建筑物高度
};
class SegmentTree//线段树
{
public:
    struct Node
    {
        int left, right;//区间左右边界
        int max_height;//区间内最大高度
        int lazy_tag;//懒惰标记
    };
    vector<Node> tree;

    //构建线段树
    void build(int node, int start, int end)
    {
        tree[node].left = start;
        tree[node].right = end;
        tree[node].max_height = 0;
        tree[node].lazy_tag = 0;
        if (start == end)//叶子节点
            return;
        int mid = (start + end) / 2;
        build(node * 2, start, mid);//左子树
        build(node * 2 + 1, mid + 1, end);//右子树
    }
    //更新区间内最大高度
    void push(int node)
    {
        //如果有懒惰标记
        if (tree[node].lazy_tag)
        {
            //更新区间内最大高度和懒惰标记
            tree[node * 2].lazy_tag = max(tree[node * 2].lazy_tag, tree[node].lazy_tag);
            tree[node * 2].max_height = max(tree[node * 2].max_height, tree[node].lazy_tag);
            tree[node * 2 + 1].lazy_tag = max(tree[node * 2 + 1].lazy_tag, tree[node].lazy_tag);
            tree[node * 2 + 1].max_height = max(tree[node * 2 + 1].max_height, tree[node].lazy_tag);
            tree[node].lazy_tag = 0;//清除懒惰标记
        }
    }
    //区间修改
    void update(int node, int start, int end, int height)
    {
        if (start <= tree[node].left && end >= tree[node].right)//区间包含
        {
			tree[node].lazy_tag = max(tree[node].lazy_tag, height);
            tree[node].max_height = max(tree[node].max_height, height);
            return;
        }
        push(node);//更新区间内最大高度
        int mid = (tree[node].left + tree[node].right) / 2;
        if (end <= mid) update(node * 2, start, end, height);//左子树
        else if (start > mid) update(node * 2 + 1, start, end, height);//右子树
        else
        {
            update(node * 2, start, mid, height);//左子树
            update(node * 2 + 1, mid + 1, end, height);//右子树
        }
        tree[node].max_height = max(tree[node * 2].max_height, tree[node * 2 + 1].max_height);//更新区间内最大高度
    }
    //区间查询
    int query(int node, int start, int end)
    {
        if (start <= tree[node].left && end >= tree[node].right)//区间包含
            return tree[node].max_height;

        push(node);//更新区间内最大高度

        int mid = (tree[node].left + tree[node].right) / 2;

        if (end <= mid) return query(node * 2, start, end);//左子树
        else if (start > mid) return query(node * 2 + 1, start, end);//右子树
        else
            return max(query(node * 2, start, mid), query(node * 2 + 1, mid + 1, end));//左右子树
    }
    //构造函数
    SegmentTree(int n)
    {
        tree.resize(n*4);
        build(1, 1, n);
    }
	//析构函数
	~SegmentTree()
	{
		tree.clear();
	}
};
int main() {
    // 输入建筑物信息
    vector<Building> buildings;
    int left, right, height;
	int maxRight = 0;
    while (cin >> left >> height >> right) {
        buildings.push_back({ left, right - 1, height }); // [left, right) 的半开区间
		if (right > maxRight) maxRight = right;
    }
    //按高度顺序排序
	/*sort(buildings.begin(), buildings.end(), [](const Building& a, const Building& b) {
		return a.height < b.height;
		});*/
	
    SegmentTree tree(10000);//大小为building的最右边界
	for (int i = 0; i < buildings.size(); i++)
	{
		tree.update(1, buildings[i].left, buildings[i].right, buildings[i].height);
	}
	vector<int> res(maxRight+1,0);
	for (int i = 1; i <= maxRight; i++)
	{
        res[i]=tree.query(1, i, i);
	}
	// 输出结果
    for (int i = 1; i <= maxRight; i++)
    {
		if (res[i] != res[i - 1]) cout << i << " " << res[i] << " ";
            
    }

    return 0;
}
