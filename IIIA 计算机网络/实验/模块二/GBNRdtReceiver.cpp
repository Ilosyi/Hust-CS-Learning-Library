#include "stdafx.h"
#include "Global.h"
#include "GBNRdtReceiver.h"

// ���캯������ʼ���������յ����к� expectedSeqNum = 0
GBNRdtReceiver::GBNRdtReceiver() : expectedSeqNum(0)
{
    // ��ʼ����һ�η��͵� ACK ����
    lastAckPkt.seqnum = -1;   // ���շ� ACK ���� seqnum �̶���Ч�����ã�
    lastAckPkt.acknum = -1;   // ��ʼʱδȷ���κη���
    lastAckPkt.checksum = 0;  // У��ͳ�ʼ��

    // ��� payload��ʹ������������ʵ�����壨ֻ��Ϊ�˱������ݰ��ṹ������
    memset(lastAckPkt.payload, '.', Configuration::PAYLOAD_SIZE);

    // ����У���
    lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
}

GBNRdtReceiver::~GBNRdtReceiver()
{
    // ���������������⴦��
}

// �������ݰ��ĺ��ĺ���
void GBNRdtReceiver::receive(const Packet& packet)
{
    // �����յ��ķ����У���
    int checkSum = pUtils->calculateCheckSum(packet);

    // -------- 1. У��ʹ���˵�����ݰ��� --------
    if (checkSum != packet.checksum) {
        pUtils->printPacket("GBN���շ��յ��������ݰ�", packet);

        // ��֮ǰ�з��͹� ACK�����ط������ ACK���ۼ�ȷ�ϻ��ƣ�
        if (lastAckPkt.acknum >= 0) {
            pUtils->printPacket("GBN���շ��ط����ACK", lastAckPkt);
            pns->sendToNetworkLayer(SENDER, lastAckPkt);
        }
        return; // �����𻵵����ݰ�
    }

    // -------- 2. �յ����������ݰ�����ȷ������ --------
    if (packet.seqnum == expectedSeqNum) {
        pUtils->printPacket("GBN���շ��յ��������ݰ�", packet);

        // �����ݽ������ϲ�Ӧ��
        Message msg;
        memcpy(msg.data, packet.payload, sizeof(packet.payload));
        pns->delivertoAppLayer(RECEIVER, msg);

        // ���ͷ����� ACK ȷ��
        sendAck(expectedSeqNum);

        // �������������кţ�ѭ��ȡģ SEQ_SIZE��
        expectedSeqNum = (expectedSeqNum + 1) % SEQ_SIZE;
    }
    // -------- 3. �յ�ʧ������ݰ������������кţ� --------
    else {
        pUtils->printPacket("GBN���շ��յ�ʧ�����ݰ�", packet);

        // �ط����һ�ε� ACK�����ѷ��ͷ������������кŻ�δ����
        if (lastAckPkt.acknum >= 0) {
            pUtils->printPacket("GBN���շ��ط����ACK", lastAckPkt);
            pns->sendToNetworkLayer(SENDER, lastAckPkt);
        }
    }
}

// ��װ�ķ��� ACK ����
void GBNRdtReceiver::sendAck(int ackNum)
{
    Packet ackPkt;

    // ACK ���� seqnum �̶�Ϊ��Чֵ��-1������ acknum ������
    ackPkt.seqnum = -1;
    ackPkt.acknum = ackNum;
    ackPkt.checksum = 0;

    // payload ͬ�����Ϊ��ʵ�����������
    memset(ackPkt.payload, '.', Configuration::PAYLOAD_SIZE);

    // ���� ACK ����У���
    ackPkt.checksum = pUtils->calculateCheckSum(ackPkt);

    // �������һ�η��͵� ACK�����ڶ���/����ʱ�ط�
    lastAckPkt = ackPkt;

    // ��ӡ������ ACK
    pUtils->printPacket("GBN���շ�����ACK", ackPkt);
    pns->sendToNetworkLayer(SENDER, ackPkt);
}
