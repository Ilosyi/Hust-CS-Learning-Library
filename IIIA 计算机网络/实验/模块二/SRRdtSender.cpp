#include "stdafx.h"
#include "Global.h"
#include "SRRdtSender.h"
#include <cstring>

// 构造函数，初始化发送窗口和相关参数
SRRdtSender::SRRdtSender() : windowSize(4), seqNumBits(3), base(0), nextSeqNum(0) {
    // 根据序号位数计算序号范围，2^3=8
    seqNumRange = 1 << seqNumBits;

    // 根据序号范围初始化确认标记和窗口缓存
    // acked[i]为true表示序号i的报文已确认
    acked.resize(seqNumRange, false);
    window.resize(seqNumRange);

    // 初始化窗口可视化工具
    visualizer = new WindowVisualizer("sr_sender_log.txt");
    visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);
}

// 析构函数，释放动态分配的内存
SRRdtSender::~SRRdtSender() {
    delete visualizer;
}

// 检查发送方是否处于等待状态（窗口已满）
bool SRRdtSender::getWaitingState() {
    int windowUsed = (nextSeqNum - base + seqNumRange) % seqNumRange;
    return windowUsed >= windowSize;
}

// 检查给定的序号是否在发送窗口内
bool SRRdtSender::inWindow(int seqNum, int base, int windowSize, int range) {
    int pos = (seqNum - base + range) % range;
    return pos >= 0 && pos < windowSize;
}

// 发送应用层数据
bool SRRdtSender::send(const Message& message) {
    // 检查窗口是否已满，如果满则拒绝发送
    if (getWaitingState()) {
        pUtils->printPacket("发送方窗口已满，拒绝发送", Packet());
        return false;
    }

    // 构造数据包，设置序列号
    Packet packet;
    packet.seqnum = nextSeqNum;
    packet.acknum = -1;
    packet.checksum = 0;
    memcpy(packet.payload, message.data, sizeof(message.data));
    packet.checksum = pUtils->calculateCheckSum(packet);

    // 将数据包存入缓存，并标记为未确认
    window[nextSeqNum] = packet;
    acked[nextSeqNum] = false;

    // 发送数据包
    pUtils->printPacket("发送方发送报文", packet);
    pns->sendToNetworkLayer(RECEIVER, packet);

    // 为该报文启动独立的定时器
    pns->startTimer(SENDER, Configuration::TIME_OUT, nextSeqNum);

    // 更新下一个可用序号
    nextSeqNum = (nextSeqNum + 1) % seqNumRange;

    // 显示窗口状态
    visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);

    return true;
}

// 接收来自接收方的ACK报文
void SRRdtSender::receive(const Packet& ackPkt) {
    // 检查校验和，如果损坏则丢弃
    int checkSum = pUtils->calculateCheckSum(ackPkt);
    if (checkSum != ackPkt.checksum) {
        pUtils->printPacket("发送方收到损坏的确认，丢弃", ackPkt);
        return;
    }

    int ackNum = ackPkt.acknum;
    int oldBase = base;

    // 检查ACK序号是否在发送窗口内
    if (inWindow(ackNum, base, windowSize, seqNumRange)) {
        pUtils->printPacket("发送方正确收到确认", ackPkt);

        // 如果该序号尚未被确认，则标记为已确认
        if (!acked[ackNum]) {
            acked[ackNum] = true;
            pns->stopTimer(SENDER, ackNum); // 停止对应的定时器
        }

        // 滑动窗口：如果窗口基序号被确认，持续向前滑动
        while (acked[base]) {
            acked[base] = false; // 重置，为下一轮循环使用
            base = (base + 1) % seqNumRange;
        }

        // 记录窗口滑动事件并显示新的窗口状态
        if (base != oldBase) {
            visualizer->logWindowMove(oldBase, base, ackNum);
            visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);
        }
    }
    else {
        pUtils->printPacket("发送方收到窗口外的确认，丢弃", ackPkt);
    }
}

// 定时器超时处理函数
void SRRdtSender::timeoutHandler(int seqNum) {
    // 检查超时的报文是否仍在窗口内且未被确认
    if (inWindow(seqNum, base, windowSize, seqNumRange) && !acked[seqNum]) {
        pUtils->printPacket("发送方定时器超时，重发报文", window[seqNum]);

        // 停止并重新启动该报文的定时器
        pns->stopTimer(SENDER, seqNum);
        pns->startTimer(SENDER, Configuration::TIME_OUT, seqNum);

        // 重发单个超时的报文
        pns->sendToNetworkLayer(RECEIVER, window[seqNum]);

        // 记录重传事件并显示新的窗口状态
        // 使用SR专用的日志方法
        visualizer->logSinglePacketTimeout(seqNum);
        visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);
    }
}