#include "stdafx.h"
#include "Global.h"
#include "TCPRdtReceiver.h"
#include <cstring>

// 构造函数，初始化TCP接收方
TCPRdtReceiver::TCPRdtReceiver() : expectedSeqNum(1) {
    // 初始化最后的ACK报文
    lastAckPkt.acknum = 0;        // 初始ACK为0，表示期待序号1
    lastAckPkt.seqnum = -1;       // 接收方不使用此字段
    lastAckPkt.checksum = 0;
    memset(lastAckPkt.payload, 0, sizeof(lastAckPkt.payload));
    lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
}

TCPRdtReceiver::~TCPRdtReceiver() {
}

// 接收数据报文
void TCPRdtReceiver::receive(const Packet& packet) {
    // 检查校验和
    int checkSum = pUtils->calculateCheckSum(packet);
    if (checkSum != packet.checksum) {
        pUtils->printPacket("接收方收到损坏的报文", packet);

        // 发送冗余ACK（最后正确接收的报文序号）
        pUtils->printPacket("接收方发送冗余ACK", lastAckPkt);
        pns->sendToNetworkLayer(SENDER, lastAckPkt);
        return;
    }

    int seqNum = packet.seqnum;

    // 情况1：收到期待的报文（按序到达）
    if (seqNum == expectedSeqNum) {
        pUtils->printPacket("接收方正确收到期待的报文", packet);

        // 向上交付给应用层
        Message msg;
        memcpy(msg.data, packet.payload, sizeof(packet.payload));
        pns->delivertoAppLayer(RECEIVER, msg);

        // 构造ACK报文
        // TCP的ACK n 表示：已正确接收序号为n的报文，期待接收n+1
        lastAckPkt.acknum = expectedSeqNum;
        lastAckPkt.checksum = 0;
        lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);

        pUtils->printPacket("接收方发送ACK", lastAckPkt);
        pns->sendToNetworkLayer(SENDER, lastAckPkt);

        // 更新期待的下一个序号
        expectedSeqNum++;
    }
    // 情况2：收到失序报文（序号不是期待的）
    else {
        pUtils->printPacket("接收方收到失序报文", packet);

        // 发送冗余ACK（最后按序接收的报文序号）
        pUtils->printPacket("接收方发送冗余ACK", lastAckPkt);
        pns->sendToNetworkLayer(SENDER, lastAckPkt);
    }
}
