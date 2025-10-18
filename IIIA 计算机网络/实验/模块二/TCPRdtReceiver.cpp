#include "stdafx.h"
#include "Global.h"
#include "TCPRdtReceiver.h"
#include <cstring>

// ���캯������ʼ��TCP���շ�
TCPRdtReceiver::TCPRdtReceiver() : expectedSeqNum(1) {
    // ��ʼ������ACK����
    lastAckPkt.acknum = 0;        // ��ʼACKΪ0����ʾ�ڴ����1
    lastAckPkt.seqnum = -1;       // ���շ���ʹ�ô��ֶ�
    lastAckPkt.checksum = 0;
    memset(lastAckPkt.payload, 0, sizeof(lastAckPkt.payload));
    lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
}

TCPRdtReceiver::~TCPRdtReceiver() {
}

// �������ݱ���
void TCPRdtReceiver::receive(const Packet& packet) {
    // ���У���
    int checkSum = pUtils->calculateCheckSum(packet);
    if (checkSum != packet.checksum) {
        pUtils->printPacket("���շ��յ��𻵵ı���", packet);

        // ��������ACK�������ȷ���յı�����ţ�
        pUtils->printPacket("���շ���������ACK", lastAckPkt);
        pns->sendToNetworkLayer(SENDER, lastAckPkt);
        return;
    }

    int seqNum = packet.seqnum;

    // ���1���յ��ڴ��ı��ģ����򵽴
    if (seqNum == expectedSeqNum) {
        pUtils->printPacket("���շ���ȷ�յ��ڴ��ı���", packet);

        // ���Ͻ�����Ӧ�ò�
        Message msg;
        memcpy(msg.data, packet.payload, sizeof(packet.payload));
        pns->delivertoAppLayer(RECEIVER, msg);

        // ����ACK����
        // TCP��ACK n ��ʾ������ȷ�������Ϊn�ı��ģ��ڴ�����n+1
        lastAckPkt.acknum = expectedSeqNum;
        lastAckPkt.checksum = 0;
        lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);

        pUtils->printPacket("���շ�����ACK", lastAckPkt);
        pns->sendToNetworkLayer(SENDER, lastAckPkt);

        // �����ڴ�����һ�����
        expectedSeqNum++;
    }
    // ���2���յ�ʧ���ģ���Ų����ڴ��ģ�
    else {
        pUtils->printPacket("���շ��յ�ʧ����", packet);

        // ��������ACK���������յı�����ţ�
        pUtils->printPacket("���շ���������ACK", lastAckPkt);
        pns->sendToNetworkLayer(SENDER, lastAckPkt);
    }
}
