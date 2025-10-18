#include "stdafx.h"
#include "Global.h"
#include "SRRdtReceiver.h"
#include <cstring>

/**
 * 构造函数，初始化接收窗口和相关参数
 */
SRRdtReceiver::SRRdtReceiver() : windowSize(4), seqNumBits(3), base(0) {
    // 根据序号位数计算序号范围，例如3位序号范围为2^3=8
    seqNumRange = 1 << seqNumBits;

    // 根据序号范围初始化缓存和接收标记数组
    // received[i] 为 true 表示序号为i的报文已接收并缓存
    received.resize(seqNumRange, false);
    buffer.resize(seqNumRange);

    //  初始化接收方可视化器
    visualizer = new WindowVisualizer("sr_receiver_log.txt");
    visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
}

/**
 * 析构函数，释放可视化器资源
 */
SRRdtReceiver::~SRRdtReceiver() {
    if (visualizer != nullptr) {
        delete visualizer;
        visualizer = nullptr;
    }
}

/**
 * 检查给定的序号是否在接收窗口内
 * 使用数学方法判断，避免了循环，提高了效率
 */
bool SRRdtReceiver::inWindow(int seqNum, int base, int windowSize, int range) {
    int pos = (seqNum - base + range) % range;
    return pos >= 0 && pos < windowSize;
}

/**
 * 封装的发送 ACK 函数
 */
void SRRdtReceiver::sendAck(int seqNum) {
    Packet ackPkt;
    ackPkt.acknum = seqNum;
    ackPkt.seqnum = -1; // ACK包的seqnum字段无效
    ackPkt.checksum = 0;
    memset(ackPkt.payload, 0, sizeof(ackPkt.payload));
    ackPkt.checksum = pUtils->calculateCheckSum(ackPkt);

    pUtils->printPacket("接收方发送确认", ackPkt);
    pns->sendToNetworkLayer(SENDER, ackPkt);
}

/**
 * 接收数据包的核心函数
 */
void SRRdtReceiver::receive(const Packet& packet) {
    // 检查校验和，如果损坏则直接丢弃
    int checkSum = pUtils->calculateCheckSum(packet);
    if (checkSum != packet.checksum) {
        pUtils->printPacket("接收方收到损坏的报文，丢弃", packet);

        //  记录接收损坏报文事件（窗口状态不变）
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
        return;
    }

    int seqNum = packet.seqnum;

    // ========== 情况1：报文序号在接收窗口内：[base, base + windowSize - 1] ==========
    if (inWindow(seqNum, base, windowSize, seqNumRange)) {
        pUtils->printPacket("接收方收到窗口内报文", packet);

        // 如果是首次收到的报文，则缓存并标记
        if (!received[seqNum]) {
            buffer[seqNum] = packet;
            received[seqNum] = true;

            //  记录接收新报文事件
            visualizer->logSRReceiverReceivePacket(seqNum, true);
        }
        else {
            //  重复接收报文
            visualizer->logSRReceiverReceivePacket(seqNum, false);
        }

        // 发送对该报文的独立确认，即使是重复收到的
        sendAck(seqNum);

        int oldBase = base;  // 保存旧的base值

        // 如果收到的报文正好是窗口的起始序号（base）
        if (seqNum == base) {
            // 尝试将连续的、已接收的报文向上交付
            while (received[base]) {
                pUtils->printPacket("接收方交付报文", buffer[base]);
                Message msg;
                memcpy(msg.data, buffer[base].payload, sizeof(buffer[base].payload));
                pns->delivertoAppLayer(RECEIVER, msg);

                // 交付后重置标记并滑动窗口
                received[base] = false;
                base = (base + 1) % seqNumRange;
            }

            //  如果窗口滑动了，记录滑动事件
            if (base != oldBase) {
                visualizer->logSRReceiverWindowSlide(oldBase, base, seqNum);
            }
        }

        //  显示更新后的接收窗口状态
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
    }
    // ========== 情况2：报文序号在窗口之外，但可能属于已确认的报文 ==========
    else if (inWindow(seqNum, (base - windowSize + seqNumRange) % seqNumRange, windowSize, seqNumRange)) {
        pUtils->printPacket("接收方收到已确认过的报文，重发确认", packet);

        //  记录重复接收历史报文事件
        visualizer->logSRReceiverReceiveOldPacket(seqNum);

        sendAck(seqNum); // 必须重发ACK，以应对ACK丢失的情况

        //  显示窗口状态（未改变）
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
    }
    // ========== 情况3：报文序号在未来窗口之外，直接丢弃 ==========
    else {
        pUtils->printPacket("接收方收到窗口外的报文，丢弃", packet);

        //  记录丢弃事件
        visualizer->logSRReceiverDiscardPacket(seqNum);

        //  显示窗口状态（未改变）
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
    }
}
