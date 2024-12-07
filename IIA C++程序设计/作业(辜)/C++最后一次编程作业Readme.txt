最后一次编程作业一共三道题

在第四章编程作业的工程HomeworkWithGTest4Student上继续完成，具体包括：
1：include\ch6_ch8_ch11_homework下面的QUEUE.h ，基本上就是把第四章编程作业的MyArray重新实现了一遍，但是多了几个运算符重载
2：include\ch6_ch8_ch11_homework下面的DUALQUEUE.h 在QUEUE基础上实现双队列
3：include\ch13_homework下面的MAT.h，实现一个矩阵模板

QUEUE实现请放在src\ch6_ch8_ch11_homework\QUEUE.cpp里
DUALQUEUE实现请放在src\ch6_ch8_ch11_homework\DUALQUEUE.cpp里
类模板的实现全部在MAT.h里，原因课堂上解释过

另外在tests写好了三个类的测试用例，分别在QUEUETest.cpp、DUALQUEUETest.cpp和MATTest.cpp里，请大家提交前要通过所有测试用例。

Tips：
1：比较有挑战性的是DUALQUEUE类，特别是拷贝/移动构造和拷贝/移动赋值，这里要利用父类QUEUE已经实现好的拷贝/移动构造和拷贝/移动赋值。
因此在DUALQUEUE是访问不到QUEUE的内部指针的

例如：DUALQUEUE的深拷贝构造：只需要深拷贝构造基类部分和对象q，而深拷贝构造基类部分和对象q在QUEUE都实现好了。

对于移动构造，只需要移动构造基类部分和对象q，而移动构造构造基类部分和对象q在QUEUE都实现好了。
需要注意的是，在调用QUEUE的移动构造时，需要把DUALQUEUE的移动构造函数参数s强制转换成QUEUE类型的右值引用：static_cast<QUEUE &&>(s)
对于DUALQUEUE的移动赋值，二条语句就解决问题了：
	    this->QUEUE::operator=(static_cast<QUEUE &&>(s));  //直接调用基类QUEUE的移动operator=
		this->q = static_cast<QUEUE &&>(s.q);			   //直接调用基类QUEUE的移动operator=
		
另外还定义了如下宏在DUALQUEUE里使用
	//获取入队列元素个数
	#define IN_QUEUE_ELEMS (QUEUE::operator int())
	//获取入队列容量
	#define IN_QUEUE_CAPACITY (QUEUE::capacity())
	//获取出队列元素个数
	#define OUT_QUEUE_ELEMS (int(q))
	//获取入队列容量
	#define OUT_QUEUE_CAPACITY (q.capacity())
	//获取双队列元素个数
	#define DUALQUEUE_ELEMS (int(*this))
	//获取双队列容量
	#define DUALQUEUE_CAPACITY (capacity())
	
2：MAT类如何初始化T **也有一定的挑战性， 因为r和c都是变量，所以这样初始化是不行的：e(new T[r][c]) ,C++要求除第一维可以是变量外，其他维必须是常量，因此只能在构造函数体里这样做：

	e = new T* [r];
    for(int i = 0;i < r;i++ ){
        e[i] = new T[c];
    }
为了减轻负担，MAT的移动构造、拷贝赋值、移动赋值都不需要实现了（因为前面MyArray、QUEUE、DUALQUEUE都已经受过足够的训练了。但是深拷贝构造需要实现，因为矩阵转置、+，*返回的都是MAT对象，为了测试，需要深拷贝构造