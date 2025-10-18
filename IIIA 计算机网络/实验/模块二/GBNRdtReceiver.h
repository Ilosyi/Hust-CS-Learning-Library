#pragma once
#ifndef GBN_RDT_RECEIVER_H
#define GBN_RDT_RECEIVER_H
#include "RdtReceiver.h"

class GBNRdtReceiver : public RdtReceiver
{
private:
    static const int SEQ_SIZE = 8;          // ��ſռ��С (2^3 = 8)

    int expectedSeqNum;                     // �������յ���һ�����
    Packet lastAckPkt;                     // ������͵�ACK��

public:
    GBNRdtReceiver();
    virtual ~GBNRdtReceiver();

    void receive(const Packet& packet);

private:
    void sendAck(int ackNum);              // ����ACK
};

#endif
