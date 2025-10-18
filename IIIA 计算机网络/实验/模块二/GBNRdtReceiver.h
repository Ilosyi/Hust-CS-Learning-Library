#pragma once
#ifndef GBN_RDT_RECEIVER_H
#define GBN_RDT_RECEIVER_H
#include "RdtReceiver.h"

class GBNRdtReceiver : public RdtReceiver
{
private:
    static const int SEQ_SIZE = 8;          // 序号空间大小 (2^3 = 8)

    int expectedSeqNum;                     // 期望接收的下一个序号
    Packet lastAckPkt;                     // 最近发送的ACK包

public:
    GBNRdtReceiver();
    virtual ~GBNRdtReceiver();

    void receive(const Packet& packet);

private:
    void sendAck(int ackNum);              // 发送ACK
};

#endif
