#pragma once
#ifndef TCP_RDT_SENDER_H
#define TCP_RDT_SENDER_H

#include "RdtSender.h"
#include "DataStructure.h"
#include <vector>
#include "WindowVisualizer.h"
#include<map>
class TCPRdtSender : public RdtSender {
private:
    int windowSize;              // 窗口大小
    int base;                    // 窗口起始序号（最早未确认的报文）
    int nextSeqNum;              // 下一个可用序号

    std::map<int, Packet> window;  //  改用 map，避免固定大小限制
    int duplicateAckCount;       // 冗余ACK计数器
    int lastAckNum;              // 上一次收到的ACK序号

    bool waitingState;           // 窗口是否已满
    WindowVisualizer* visualizer;  // 添加可视化器
public:
    TCPRdtSender();
    virtual ~TCPRdtSender();

    bool getWaitingState();
    bool send(const Message& message);
    void receive(const Packet& ackPkt);
    void timeoutHandler(int seqNum);
};

#endif
