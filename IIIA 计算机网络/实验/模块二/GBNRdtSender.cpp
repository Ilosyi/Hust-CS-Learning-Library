#include "stdafx.h"
#include "Global.h"
#include "GBNRdtSender.h"

// ���캯������ʼ�����ͷ�״̬
GBNRdtSender::GBNRdtSender() : base(0), nextSeqNum(0), waitingState(false)
{
    // ��ʼ��������ݰ��Ļ���������СΪ���кſռ�Ĵ�С
    // GBNЭ����Ҫ���������ѷ��͵�δȷ�ϵ����ݰ����Ա���ʱ�ش�
    sndpkt.resize(SEQ_SIZE);

    // �������ӻ����ߣ����ڼ�¼����ʾ�������ڵ��ƶ�
    visualizer = new WindowVisualizer("gbn_window_log.txt");

    // ��ʾ��ʼ����״̬���������Ժ����
    visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);
}

// �����������ͷŶ�̬������ڴ�
GBNRdtSender::~GBNRdtSender()
{
    delete visualizer;
}

// ��ȡ���ͷ��ȴ�״̬
// ������ʹ�������������true����ʾ�����ٴ��ϲ��������
bool GBNRdtSender::getWaitingState()
{
    return waitingState;
}

// ����Ӧ�ò�����
bool GBNRdtSender::send(const Message& message)
{
    // ��鷢�ʹ����Ƿ����������������������ܾ�����
    if (getWindowSize() >= WINDOW_SIZE) {
        waitingState = true; // ���õȴ�״̬Ϊtrue
        pUtils->printPacket("GBN���ͷ�:��������,�ܾ�����", Packet());
        return false;
    }

    // �����µ����ݰ�
    Packet packet;
    packet.seqnum = nextSeqNum; // ���ñ������
    packet.acknum = -1; // ACK�ֶ���Ч
    packet.checksum = 0;
    memcpy(packet.payload, message.data, sizeof(message.data)); // ����Ӧ�ò�����
    packet.checksum = pUtils->calculateCheckSum(packet); // ����У���

    // �����ݰ����뻺�������Ա��ش�
    sndpkt[nextSeqNum] = packet;

    // ��ӡ�����ͱ��ĵ������
    pUtils->printPacket("GBN���ͷ����ͱ���", packet);
    pns->sendToNetworkLayer(RECEIVER, packet);

    // ����Ǵ����е�һ��δ���͵ı��ģ�������ʱ��
    // ��ʱ��ֻΪ���ڻ��������Ӧ�ı�������
    if (base == nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // ������һ���������кţ���ѭ��ȡģ
    nextSeqNum = (nextSeqNum + 1) % SEQ_SIZE;

    // ���µȴ�״̬��������������������ȴ�
    waitingState = (getWindowSize() >= WINDOW_SIZE);

    // ���ӻ�����¼��ǰ�ķ��ʹ���״̬
    visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);

    return true;
}

// ��������������ACK����
void GBNRdtSender::receive(const Packet& ackPkt)
{
    // �����յ���ACK���ĵ�У���
    int checkSum = pUtils->calculateCheckSum(ackPkt);

    // ���ACK�����Ƿ���
    if (checkSum != ackPkt.checksum) {
        pUtils->printPacket("GBN���ͷ��յ�����ACK", ackPkt);
        return; // �����𻵵�ACK����
    }

    pUtils->printPacket("GBN���ͷ��յ�ACK", ackPkt);

    // ��ȡACK���ĵ�ȷ�Ϻ�
    int ackNum = ackPkt.acknum;
    int oldBase = base;

    // ���ACK�Ƿ��ڵ�ǰ���ʹ����ڣ����Ƿ�Ϊ��Чȷ�ϣ�
    // ������ж��߼����������кŵ�ѭ������
    bool validAck = false;
    if (base <= nextSeqNum) {
        validAck = (ackNum >= base && ackNum < nextSeqNum);
    }
    else {
        validAck = (ackNum >= base || ackNum < nextSeqNum);
    }

    // ����յ���ACK��Ч
    if (validAck) {
        // ֹͣ��ǰ��ʱ������Ϊ�����֮ǰ�ı��Ķ���ȷ��
        pns->stopTimer(SENDER, base);

        // ���ڻ��������´��ڻ����Ϊ(ackNum + 1)
        // GBN���ۻ�ȷ�ϣ�һ��ACKȷ����ackNum����֮ǰ�����б���
        base = (ackNum + 1) % SEQ_SIZE;

        // ��¼�����ƶ��¼�
        visualizer->logWindowMove(oldBase, base, ackNum);

        // �������������δȷ�ϵı��ģ�����������ʱ��
        // ��ʱ������Ϊ�����е�һ��δȷ�ϵı�������
        if (base != nextSeqNum) {
            pns->startTimer(SENDER, Configuration::TIME_OUT, base);
        }

        // ���µȴ�״̬��������ڲ��������ģ����������±���
        waitingState = (getWindowSize() >= WINDOW_SIZE);

        // ���ӻ�����¼�µĴ���״̬
        visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);
    }
}

// ��ʱ����ʱ������
void GBNRdtSender::timeoutHandler(int seqNum)
{
    // ��¼��ʱ�¼�
    visualizer->logTimeoutRetransmit(base, nextSeqNum);

    // ��ӡ��ʱ��ʾ��Ϣ
    Packet timeoutInfo;
    timeoutInfo.seqnum = seqNum;
    timeoutInfo.acknum = -1;
    timeoutInfo.checksum = 0;
    memset(timeoutInfo.payload, 0, sizeof(timeoutInfo.payload));
    strcpy(timeoutInfo.payload, "TIMEOUT");
    pUtils->printPacket("GBN���ͷ���ʱ�ش�", timeoutInfo);

    // ֹͣ��ǰ��ʱ��
    pns->stopTimer(SENDER, seqNum);

    // GBNЭ����ģ���ʱ��Go-Back-N�����ش������ѷ��͵�δȷ�ϵ����ݰ�
    // �����Ӵ��ڻ����base����һ���������nextSeqNum�����б��Ĳ��ش�
    for (int i = base; i != nextSeqNum; i = (i + 1) % SEQ_SIZE) {
        pUtils->printPacket("GBN���ͷ��ش����ݰ�", sndpkt[i]);
        pns->sendToNetworkLayer(RECEIVER, sndpkt[i]);
    }

    // ����������ʱ�����ٴ�Ϊ�����е�һ��δȷ�ϵı�������
    if (base != nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // ���ӻ�����¼�ش���Ĵ���״̬
    visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);
}

// ������к��Ƿ��ڵ�ǰ������
// ע�⣺�˺��������߼���δʹ�ã�������Ϊ�˷�����Ի�������չ
bool GBNRdtSender::isInWindow(int seqNum)
{
    if (base <= nextSeqNum) {
        return (seqNum >= base && seqNum < nextSeqNum);
    }
    else {
        return (seqNum >= base || seqNum < nextSeqNum);
    }
}

// ��ȡ��ǰ���ڴ�С
// �����ѷ��͵�δȷ�ϵı�������
int GBNRdtSender::getWindowSize()
{
    if (nextSeqNum >= base) {
        return nextSeqNum - base;
    }
    else {
        return nextSeqNum + SEQ_SIZE - base;
    }
}