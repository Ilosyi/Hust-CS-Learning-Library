#pragma once
#ifndef SR_RDT_RECEIVER_H
#define SR_RDT_RECEIVER_H

#include "RdtReceiver.h"
#include "DataStructure.h"
#include <vector>
#include "WindowVisualizer.h" 
class SRRdtReceiver : public RdtReceiver {
private:
    int windowSize;              // ���ڴ�С
    int seqNumBits;              // ��ű���λ��
    int seqNumRange;             // ��ŷ�Χ
    int base;                    // ���մ�����ʼλ��

    std::vector<bool> received;  // ���ÿ������Ƿ��ѽ���
    std::vector<Packet> buffer;  // �������򵽴�����ݰ�

    // �����������ж�����Ƿ��ڴ�����
    bool inWindow(int seqNum, int base, int windowSize, int range);

    // ����ȷ�ϰ�
    void sendAck(int seqNum);
    //���ӻ���ָ��
		WindowVisualizer* visualizer;
   

public:
    SRRdtReceiver();
    virtual ~SRRdtReceiver();

    void receive(const Packet& packet);  // �������ݰ�
};

#endif
