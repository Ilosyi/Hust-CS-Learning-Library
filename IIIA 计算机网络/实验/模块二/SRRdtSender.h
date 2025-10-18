#pragma once
#ifndef SR_RDT_SENDER_H
#define SR_RDT_SENDER_H

#include "RdtSender.h"
#include "DataStructure.h"
#include <vector>
#include "WindowVisualizer.h"  // ��ӿ��ӻ���
class SRRdtSender : public RdtSender {
private:
    int windowSize;              // ���ڴ�С
    int seqNumBits;              // ��ű���λ��������3λ���Ա�ʾ0-7��
    int seqNumRange;             // ��ŷ�Χ��2^seqNumBits��
    int base;                    // ���ʹ�����ʼλ��
    int nextSeqNum;              // ��һ�����õ����

    std::vector<bool> acked;     // ���ÿ�����ݰ��Ƿ���ȷ��
    std::vector<Packet> window;  // ���ʹ��ڻ���
    //bool waitingState;           // �Ƿ��ڵȴ�״̬����������

    // �����������ж�����Ƿ��ڴ�����
    bool inWindow(int seqNum, int base, int windowSize, int range);
    WindowVisualizer* visualizer;  // ��ӿ��ӻ���
public:
    SRRdtSender();
    virtual ~SRRdtSender();

    bool getWaitingState();
    bool send(const Message& message);       // ��������
    void receive(const Packet& ackPkt);      // ����ȷ��
    void timeoutHandler(int seqNum);         // ��ʱ����
};

#endif
