#pragma once
#ifndef SR_RDT_RECEIVER_H
#define SR_RDT_RECEIVER_H

#include "RdtReceiver.h"
#include "DataStructure.h"
#include <vector>
#include "WindowVisualizer.h" 
class SRRdtReceiver : public RdtReceiver {
private:
    int windowSize;              // 窗口大小
    int seqNumBits;              // 序号编码位数
    int seqNumRange;             // 序号范围
    int base;                    // 接收窗口起始位置

    std::vector<bool> received;  // 标记每个序号是否已接收
    std::vector<Packet> buffer;  // 缓存乱序到达的数据包

    // 辅助函数：判断序号是否在窗口内
    bool inWindow(int seqNum, int base, int windowSize, int range);

    // 发送确认包
    void sendAck(int seqNum);
    //可视化器指针
		WindowVisualizer* visualizer;
   

public:
    SRRdtReceiver();
    virtual ~SRRdtReceiver();

    void receive(const Packet& packet);  // 接收数据包
};

#endif
