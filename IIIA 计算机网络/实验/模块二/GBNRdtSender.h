#pragma once
#ifndef GBN_RDT_SENDER_H
#define GBN_RDT_SENDER_H
#include "RdtSender.h"
#include <vector>
#include "WindowVisualizer.h"  // ��ӿ��ӻ���

class GBNRdtSender : public RdtSender
{
private:
    static const int WINDOW_SIZE = 4;       // ���ڴ�С
    static const int SEQ_SIZE = 8;          // ��ſռ��С (2^3 = 8)

    int base;                               // ���ڻ����
    int nextSeqNum;                         // ��һ��Ҫ���͵����
    std::vector<Packet> sndpkt;            // ���ͻ��������洢�ѷ��͵�δȷ�ϵ����ݰ�
    bool waitingState;                      // �Ƿ��ڵȴ�״̬����������
    WindowVisualizer* visualizer;  // ���ڿ��ӻ���

public:
    GBNRdtSender();
    virtual ~GBNRdtSender();

    bool getWaitingState();
    bool send(const Message& message);
    void receive(const Packet& ackPkt);
    void timeoutHandler(int seqNum);

private:
    bool isInWindow(int seqNum);           // �ж�����Ƿ��ڵ�ǰ������
    int getWindowSize();                   // ��ȡ��ǰ�������ѷ��͵�δȷ�ϵ����ݰ�����
};

#endif
