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
    int windowSize;              // ���ڴ�С
    int base;                    // ������ʼ��ţ�����δȷ�ϵı��ģ�
    int nextSeqNum;              // ��һ���������

    std::map<int, Packet> window;  //  ���� map������̶���С����
    int duplicateAckCount;       // ����ACK������
    int lastAckNum;              // ��һ���յ���ACK���

    bool waitingState;           // �����Ƿ�����
    WindowVisualizer* visualizer;  // ��ӿ��ӻ���
public:
    TCPRdtSender();
    virtual ~TCPRdtSender();

    bool getWaitingState();
    bool send(const Message& message);
    void receive(const Packet& ackPkt);
    void timeoutHandler(int seqNum);
};

#endif
