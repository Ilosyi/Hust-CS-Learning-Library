#pragma once
#ifndef TCP_RDT_RECEIVER_H
#define TCP_RDT_RECEIVER_H

#include "RdtReceiver.h"
#include "DataStructure.h"

class TCPRdtReceiver : public RdtReceiver {
private:
    int expectedSeqNum;          // �ڴ����յ���һ���������
    Packet lastAckPkt;           // ��һ�η��͵�ACK����

public:
    TCPRdtReceiver();
    virtual ~TCPRdtReceiver();

    void receive(const Packet& packet);
};

#endif
