#include "stdafx.h"
#include "Global.h"
#include "GBNRdtReceiver.h"

// 构造函数，初始化期望接收的序列号 expectedSeqNum = 0
GBNRdtReceiver::GBNRdtReceiver() : expectedSeqNum(0)
{
    // 初始化上一次发送的 ACK 报文
    lastAckPkt.seqnum = -1;   // 接收方 ACK 包的 seqnum 固定无效（不用）
    lastAckPkt.acknum = -1;   // 初始时未确认任何分组
    lastAckPkt.checksum = 0;  // 校验和初始化

    // 填充 payload，使得数据内容无实际意义（只是为了保持数据包结构完整）
    memset(lastAckPkt.payload, '.', Configuration::PAYLOAD_SIZE);

    // 计算校验和
    lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
}

GBNRdtReceiver::~GBNRdtReceiver()
{
    // 析构函数，无特殊处理
}

// 接收数据包的核心函数
void GBNRdtReceiver::receive(const Packet& packet)
{
    // 计算收到的分组的校验和
    int checkSum = pUtils->calculateCheckSum(packet);

    // -------- 1. 校验和错误，说明数据包损坏 --------
    if (checkSum != packet.checksum) {
        pUtils->printPacket("GBN接收方收到错误数据包", packet);

        // 若之前有发送过 ACK，则重发最近的 ACK（累计确认机制）
        if (lastAckPkt.acknum >= 0) {
            pUtils->printPacket("GBN接收方重发最近ACK", lastAckPkt);
            pns->sendToNetworkLayer(SENDER, lastAckPkt);
        }
        return; // 丢弃损坏的数据包
    }

    // -------- 2. 收到期望的数据包（正确且有序） --------
    if (packet.seqnum == expectedSeqNum) {
        pUtils->printPacket("GBN接收方收到期望数据包", packet);

        // 将数据交付给上层应用
        Message msg;
        memcpy(msg.data, packet.payload, sizeof(packet.payload));
        pns->delivertoAppLayer(RECEIVER, msg);

        // 向发送方发送 ACK 确认
        sendAck(expectedSeqNum);

        // 更新期望的序列号（循环取模 SEQ_SIZE）
        expectedSeqNum = (expectedSeqNum + 1) % SEQ_SIZE;
    }
    // -------- 3. 收到失序的数据包（非期望序列号） --------
    else {
        pUtils->printPacket("GBN接收方收到失序数据包", packet);

        // 重发最近一次的 ACK（提醒发送方：期望的序列号还未到）
        if (lastAckPkt.acknum >= 0) {
            pUtils->printPacket("GBN接收方重发最近ACK", lastAckPkt);
            pns->sendToNetworkLayer(SENDER, lastAckPkt);
        }
    }
}

// 封装的发送 ACK 函数
void GBNRdtReceiver::sendAck(int ackNum)
{
    Packet ackPkt;

    // ACK 包的 seqnum 固定为无效值（-1），仅 acknum 有意义
    ackPkt.seqnum = -1;
    ackPkt.acknum = ackNum;
    ackPkt.checksum = 0;

    // payload 同样填充为无实际意义的内容
    memset(ackPkt.payload, '.', Configuration::PAYLOAD_SIZE);

    // 计算 ACK 包的校验和
    ackPkt.checksum = pUtils->calculateCheckSum(ackPkt);

    // 保存最近一次发送的 ACK，便于丢包/出错时重发
    lastAckPkt = ackPkt;

    // 打印并发送 ACK
    pUtils->printPacket("GBN接收方发送ACK", ackPkt);
    pns->sendToNetworkLayer(SENDER, ackPkt);
}
