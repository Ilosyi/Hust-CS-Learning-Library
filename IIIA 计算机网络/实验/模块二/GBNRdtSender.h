#pragma once
#ifndef GBN_RDT_SENDER_H
#define GBN_RDT_SENDER_H
#include "RdtSender.h"
#include <vector>
#include "WindowVisualizer.h"  // 添加可视化器

class GBNRdtSender : public RdtSender
{
private:
    static const int WINDOW_SIZE = 4;       // 窗口大小
    static const int SEQ_SIZE = 8;          // 序号空间大小 (2^3 = 8)

    int base;                               // 窗口基序号
    int nextSeqNum;                         // 下一个要发送的序号
    std::vector<Packet> sndpkt;            // 发送缓冲区，存储已发送但未确认的数据包
    bool waitingState;                      // 是否处于等待状态（窗口满）
    WindowVisualizer* visualizer;  // 窗口可视化器

public:
    GBNRdtSender();
    virtual ~GBNRdtSender();

    bool getWaitingState();
    bool send(const Message& message);
    void receive(const Packet& ackPkt);
    void timeoutHandler(int seqNum);

private:
    bool isInWindow(int seqNum);           // 判断序号是否在当前窗口内
    int getWindowSize();                   // 获取当前窗口中已发送但未确认的数据包数量
};

#endif
