#pragma once
#ifndef TCP_RDT_RECEIVER_H
#define TCP_RDT_RECEIVER_H

#include "RdtReceiver.h"
#include "DataStructure.h"

class TCPRdtReceiver : public RdtReceiver {
private:
    int expectedSeqNum;          // 期待接收的下一个报文序号
    Packet lastAckPkt;           // 上一次发送的ACK报文

public:
    TCPRdtReceiver();
    virtual ~TCPRdtReceiver();

    void receive(const Packet& packet);
};

#endif
