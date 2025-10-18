// StopWait.cpp : 定义控制台应用程序的入口点。

#include "stdafx.h"
#include "Global.h"
#include "RdtSender.h"
#include "RdtReceiver.h"
#include "GBNRdtSender.h"
#include "GBNRdtReceiver.h"
#include "SRRdtSender.h"
#include "SRRdtReceiver.h"
#include "FileComparator.h"
#include"TCPRdtSender.h"
#include"TCPRdtReceiver.h"
#define CHOSE 2  // 1: GBN, 2: SR, 3: TCP, 4: Stop-and-Wait
int main(int argc, char* argv[])
{
    std::cout << "=== 可靠数据传输协议测试 ===" << std::endl;
    RdtSender* ps = nullptr;
    RdtReceiver* pr = nullptr;

#if CHOSE == 3
    // 使用TCP
    ps = new TCPRdtSender();
    pr = new TCPRdtReceiver();
#elif CHOSE == 2
    // 使用SR协议
    ps = new SRRdtSender();
    pr = new SRRdtReceiver();
#elif CHOSE == 1
    // 使用GBN协议
    ps = new GBNRdtSender();
    pr = new GBNRdtReceiver();
#elif CHOSE == 4
    // 使用停等协议
    ps = new StopWaitRdtSender();
    pr = new StopWaitRdtReceiver();
#else
#error "CHOSE must be 1, 2, 3, or 4"
#endif

    // pns->setRunMode(0);  //VERBOS模式
    pns->setRunMode(1);  //安静模式
    pns->init();
    pns->setRtdSender(ps);
    pns->setRtdReceiver(pr);
    pns->setInputFile(".\\input.txt");
    pns->setOutputFile(".\\output.txt");

    std::cout << "开始传输..." << std::endl;
    pns->start();
    std::cout << "传输完成!" << std::endl;

    // 使用文件比较器验证传输结果
    FileComparator comparator;
    bool success = comparator.validateTransmission(".\\input.txt", ".\\output.txt");

    // 清理资源
    delete ps;
    delete pr;
    delete pUtils;  // 指向唯一的工具类实例，只在main函数结束前delete
    delete pns;     // 指向唯一的模拟网络环境类实例，只在main函数结束前delete

    std::cout << "\n程序结束，按任意键退出..." << std::endl;
    system("pause");

    return success ? 0 : 1;  // 返回值表示传输是否成功
}
