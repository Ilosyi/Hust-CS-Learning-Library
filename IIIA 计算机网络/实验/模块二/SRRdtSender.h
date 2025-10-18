#pragma once
#ifndef SR_RDT_SENDER_H
#define SR_RDT_SENDER_H

#include "RdtSender.h"
#include "DataStructure.h"
#include <vector>
#include "WindowVisualizer.h"  // 添加可视化器
class SRRdtSender : public RdtSender {
private:
    int windowSize;              // 窗口大小
    int seqNumBits;              // 序号编码位数（例如3位可以表示0-7）
    int seqNumRange;             // 序号范围（2^seqNumBits）
    int base;                    // 发送窗口起始位置
    int nextSeqNum;              // 下一个可用的序号

    std::vector<bool> acked;     // 标记每个数据包是否已确认
    std::vector<Packet> window;  // 发送窗口缓存
    //bool waitingState;           // 是否处于等待状态（窗口满）

    // 辅助函数：判断序号是否在窗口内
    bool inWindow(int seqNum, int base, int windowSize, int range);
    WindowVisualizer* visualizer;  // 添加可视化器
public:
    SRRdtSender();
    virtual ~SRRdtSender();

    bool getWaitingState();
    bool send(const Message& message);       // 发送数据
    void receive(const Packet& ackPkt);      // 接收确认
    void timeoutHandler(int seqNum);         // 超时处理
};

#endif
