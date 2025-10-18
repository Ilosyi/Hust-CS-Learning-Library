#include "stdafx.h"
#include "Global.h"
#include "SRRdtSender.h"
#include <cstring>

// ���캯������ʼ�����ʹ��ں���ز���
SRRdtSender::SRRdtSender() : windowSize(4), seqNumBits(3), base(0), nextSeqNum(0) {
    // �������λ��������ŷ�Χ��2^3=8
    seqNumRange = 1 << seqNumBits;

    // ������ŷ�Χ��ʼ��ȷ�ϱ�Ǻʹ��ڻ���
    // acked[i]Ϊtrue��ʾ���i�ı�����ȷ��
    acked.resize(seqNumRange, false);
    window.resize(seqNumRange);

    // ��ʼ�����ڿ��ӻ�����
    visualizer = new WindowVisualizer("sr_sender_log.txt");
    visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);
}

// �����������ͷŶ�̬������ڴ�
SRRdtSender::~SRRdtSender() {
    delete visualizer;
}

// ��鷢�ͷ��Ƿ��ڵȴ�״̬������������
bool SRRdtSender::getWaitingState() {
    int windowUsed = (nextSeqNum - base + seqNumRange) % seqNumRange;
    return windowUsed >= windowSize;
}

// ������������Ƿ��ڷ��ʹ�����
bool SRRdtSender::inWindow(int seqNum, int base, int windowSize, int range) {
    int pos = (seqNum - base + range) % range;
    return pos >= 0 && pos < windowSize;
}

// ����Ӧ�ò�����
bool SRRdtSender::send(const Message& message) {
    // ��鴰���Ƿ��������������ܾ�����
    if (getWaitingState()) {
        pUtils->printPacket("���ͷ������������ܾ�����", Packet());
        return false;
    }

    // �������ݰ����������к�
    Packet packet;
    packet.seqnum = nextSeqNum;
    packet.acknum = -1;
    packet.checksum = 0;
    memcpy(packet.payload, message.data, sizeof(message.data));
    packet.checksum = pUtils->calculateCheckSum(packet);

    // �����ݰ����뻺�棬�����Ϊδȷ��
    window[nextSeqNum] = packet;
    acked[nextSeqNum] = false;

    // �������ݰ�
    pUtils->printPacket("���ͷ����ͱ���", packet);
    pns->sendToNetworkLayer(RECEIVER, packet);

    // Ϊ�ñ������������Ķ�ʱ��
    pns->startTimer(SENDER, Configuration::TIME_OUT, nextSeqNum);

    // ������һ���������
    nextSeqNum = (nextSeqNum + 1) % seqNumRange;

    // ��ʾ����״̬
    visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);

    return true;
}

// �������Խ��շ���ACK����
void SRRdtSender::receive(const Packet& ackPkt) {
    // ���У��ͣ����������
    int checkSum = pUtils->calculateCheckSum(ackPkt);
    if (checkSum != ackPkt.checksum) {
        pUtils->printPacket("���ͷ��յ��𻵵�ȷ�ϣ�����", ackPkt);
        return;
    }

    int ackNum = ackPkt.acknum;
    int oldBase = base;

    // ���ACK����Ƿ��ڷ��ʹ�����
    if (inWindow(ackNum, base, windowSize, seqNumRange)) {
        pUtils->printPacket("���ͷ���ȷ�յ�ȷ��", ackPkt);

        // ����������δ��ȷ�ϣ�����Ϊ��ȷ��
        if (!acked[ackNum]) {
            acked[ackNum] = true;
            pns->stopTimer(SENDER, ackNum); // ֹͣ��Ӧ�Ķ�ʱ��
        }

        // �������ڣ�������ڻ���ű�ȷ�ϣ�������ǰ����
        while (acked[base]) {
            acked[base] = false; // ���ã�Ϊ��һ��ѭ��ʹ��
            base = (base + 1) % seqNumRange;
        }

        // ��¼���ڻ����¼�����ʾ�µĴ���״̬
        if (base != oldBase) {
            visualizer->logWindowMove(oldBase, base, ackNum);
            visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);
        }
    }
    else {
        pUtils->printPacket("���ͷ��յ��������ȷ�ϣ�����", ackPkt);
    }
}

// ��ʱ����ʱ������
void SRRdtSender::timeoutHandler(int seqNum) {
    // ��鳬ʱ�ı����Ƿ����ڴ�������δ��ȷ��
    if (inWindow(seqNum, base, windowSize, seqNumRange) && !acked[seqNum]) {
        pUtils->printPacket("���ͷ���ʱ����ʱ���ط�����", window[seqNum]);

        // ֹͣ�����������ñ��ĵĶ�ʱ��
        pns->stopTimer(SENDER, seqNum);
        pns->startTimer(SENDER, Configuration::TIME_OUT, seqNum);

        // �ط�������ʱ�ı���
        pns->sendToNetworkLayer(RECEIVER, window[seqNum]);

        // ��¼�ش��¼�����ʾ�µĴ���״̬
        // ʹ��SRר�õ���־����
        visualizer->logSinglePacketTimeout(seqNum);
        visualizer->visualizeSenderWindow(base, nextSeqNum, windowSize, seqNumRange);
    }
}