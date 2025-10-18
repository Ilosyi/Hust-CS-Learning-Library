## 实验课随堂作业
要求：
1）第一二题推导过程或答案用文字形式写在答题框内；

2）第三题编程题请以“姓名+学号.sml”格式的单个文件作为附件**形式提交。

注意注意注意：

1-3题写完后，请在本作业下立即提交，图文作业中有选做题名为“随堂小测选做题”，该题为选做题，有余力的同学可以尝试。

## 题目：
### 一、分析题

从功能和性能两方面比较下列两个函数的异同点：
```
fun take( [ ], i) = [ ]
    | take(x::xs, i) = 
         if i > 0 then x::take(xs, i-1)
         else [ ];

fun rtake ([ ], _, taken) = taken
     | rtake (x::xs,i,taken) =
          if i>0 then rtake(xs, i-1, x::taken)
          else taken;
```

### 二、定义如下三个函数：

```
fun hd (x::_) = x;
fun next(xlist, y::ys) : int list =
              if hd xlist < y   
              then next(y::xlist, ys)
              else   let fun swap [x] = y::x::ys
                                |  swap (x::xk::xs)  = if xk > y
                                         then x::swap(xk::xs)
                                        else (y::xk::xs)@(x::xs)
                        in swap (xlist)
                        end;
fun nextperm(y::ys) = next ([y], ys);
```
调用上述代码，nextperm[2,3,1,4]的计算结果是什么？用“=>”详细描述其计算过程。

### 三、编写函数subsetSumOption: int list * int -> int list option，

  要求：对函数subsetSumOption(L, s)：如果L中存在子集L'，满足其中所有元素之和为s，则结果为SOME L'；否则结果为NONE。

### 注意注意注意：

先做“课堂作业”里的1-3题，且1-3题写完后请在相应“随堂小测”作业下立即提交；

本题为“选做题”，本题为选做题，有余力的同学可以尝试。

注意注意注意

要求：

1）编程题请以“姓名+学号+Q4.sml”格式用单个文件作为附件形式提交。

### 四、快速排序的原理是分治法：
(1)从输入数据中选择某个值a；

(2)将剩下的数据分为两部分：一部分小于或等于a，另一部分大于a；

(3)分别递归的排序两个部分，并将小的部分放在大的前面。

根据该原理，快速排序需要调用快排和分区两个函数：quickSort和partition，其类型定义分别为：
quickSort: int list -> int list
partition: int list * int list * int list -> int list
或 partition: int list * int list * int list -> int list * int list
编写这两个函数的实现代码（其中，要求函数partition的三个参数分别表示：待分区的表，小于或等于a的表，大于a的表）。
