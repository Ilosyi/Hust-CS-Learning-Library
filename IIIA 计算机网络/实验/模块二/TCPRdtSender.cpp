#include "stdafx.h"
#include "Global.h"
#include "TCPRdtSender.h"
#include <cstring>

/**
 * ���캯������ʼ��TCP���ͷ�
 *
 * ��ʼ������˵����
 * - windowSize(4): ���ʹ��ڴ�СΪ4����ͬʱ��෢��4��δȷ�ϵı���
 * - base(1): ������ʼ���Ϊ1������δȷ�ϵı�����ţ�
 * - nextSeqNum(1): ��һ�������͵ı������Ϊ1
 * - duplicateAckCount(0): ����ACK��������ʼ��Ϊ0
 * - lastAckNum(0): �ϴ��յ���ACK��ų�ʼ��Ϊ0
 * - waitingState(false): ��ʼ����δ���������ڵȴ�״̬
 * - visualizer(nullptr): ���ӻ�����ָ���ʼ��Ϊ��
 */
TCPRdtSender::TCPRdtSender()
    : windowSize(4), base(1), nextSeqNum(1),
    duplicateAckCount(0), lastAckNum(0), waitingState(false), visualizer(nullptr) {

    //  ʹ��std::map�洢�����еı��ģ�����ҪԤ�ȷ���ռ�
    // map���������Զ���չ�������˹̶���С�����Խ�����

    // ��ʼ�����ڿ��ӻ����ߣ����ڼ�¼����״̬����־�ļ�
    visualizer = new WindowVisualizer("tcp_window_log.txt");
    if (visualizer != nullptr) {
        // ��¼��ʼ����״̬������Ϊ�գ�
        visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
            duplicateAckCount, lastAckNum);
    }
}

/**
 * ����������������Դ
 *
 * ���ܣ�
 * - ��ȫ�ͷſ��ӻ����߶�����ڴ�
 * - ��ָ����Ϊnullptr����ֹ����ָ��
 */
TCPRdtSender::~TCPRdtSender() {
    if (visualizer != nullptr) {
        delete visualizer;      // �ͷŶ�̬������ڴ�
        visualizer = nullptr;   // ��ֹ�ظ��ͷŵ��µ�����
    }
}

/**
 * ��ȡ���ͷ��ȴ�״̬
 *
 * @return true - �����������ܾ������±���
 *         false - ����δ�������Լ�������
 */
bool TCPRdtSender::getWaitingState() {
    return waitingState;
}

/**
 * ����Ӧ�ò���Ϣ
 *
 * @param message Ӧ�ò��·�����Ϣ����
 * @return true - ���ͳɹ�
 *         false - �����������ܾ�����
 *
 * �������̣�
 * 1. ��鴰���Ƿ�������nextSeqNum >= base + windowSize��
 * 2. ����TCP���ĶΣ�������š�����У��ͣ�
 * 3. �����Ļ��浽�����У����ڿ��ܵ��ش���
 * 4. ���ͱ��ĵ������
 * 5. ����Ǵ����е�һ�����ģ�������ʱ��
 * 6. ���´���״̬����¼��־
 */
bool TCPRdtSender::send(const Message& message) {
    // ��鴰���Ƿ�����
    // ��������������nextSeqNum >= base + windowSize
    // ���磺base=1, windowSize=4���򴰿ڷ�Χ��[1,2,3,4]��nextSeqNum=5ʱ������
    if (nextSeqNum >= base + windowSize) {
        waitingState = true;  // ���Ϊ�ȴ�״̬
        return false;         // �ܾ�����
    }

    // ����TCP���Ķ�
    Packet packet;
    packet.seqnum = nextSeqNum;     // �������кţ������Ķα�ţ�
    packet.acknum = -1;              // ���ͷ���ʹ��ȷ�Ϻ��ֶ�
    packet.checksum = 0;             // �ȳ�ʼ��У���Ϊ0

    // ����Ӧ�ò����ݵ������غ�
    memcpy(packet.payload, message.data, sizeof(message.data));

    // ���㲢����У��ͣ����ڽ��շ���ⱨ���Ƿ��𻵣�
    packet.checksum = pUtils->calculateCheckSum(packet);

    //  ʹ��map�洢���ģ�keyΪ��ţ�valueΪ���Ķ���
    // map���Զ������ڴ棬������̶���С��������Խ��
    window[nextSeqNum] = packet;

    // ��ӡ������Ϣ��ͨ������㷢�ͱ���
    pUtils->printPacket("���ͷ����ͱ���", packet);
    pns->sendToNetworkLayer(RECEIVER, packet);

    // ������Ǵ����еĵ�һ�����ģ�base == nextSeqNum����������ʱ��
    // TCPʹ�õ�һ��ʱ����ֻΪ����δȷ�ϵı��ģ�base����ʱ
    if (base == nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // ������һ�������͵����
    nextSeqNum++;

    // ���´���״̬������������ˣ����õȴ���־
    waitingState = (nextSeqNum >= base + windowSize);

    // ��¼���ͺ�Ĵ���״̬����־
    if (visualizer != nullptr) {
        visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
            duplicateAckCount, lastAckNum);
    }

    return true;  // ���ͳɹ�
}

/**
 * ���ղ�����ACK����
 *
 * @param ackPkt ���յ���ACK����
 *
 * TCP�ۻ�ȷ�ϻ��ƣ�
 * - ACK n ��ʾ������ȷ���յ���š�n�����б��ģ��ڴ��������Ϊn+1�ı���
 *
 * �������������
 * 1. ��ACK��ackNum >= base����������ǰ������������ȷ�ϵı���
 * 2. ����ACK��ackNum == lastAckNum�������Ӽ��������ܴ��������ش�
 * 3. ��ACK��ackNum < lastAckNum����ֱ�Ӷ���
 */
void TCPRdtSender::receive(const Packet& ackPkt) {
    // ��һ������֤ACK���ĵ�������
    int checkSum = pUtils->calculateCheckSum(ackPkt);
    if (checkSum != ackPkt.checksum) {
        // У��Ͳ�ƥ�䣬˵��ACK�ڴ������𻵣�����
        pUtils->printPacket("���ͷ��յ��𻵵�ACK������", ackPkt);
        return;
    }

    int ackNum = ackPkt.acknum;  // ��ȡȷ�Ϻ�

    // ========== ���1���յ��µ�ACK��ȷ�����µ����ݣ� ==========
    if (ackNum >= base) {
        pUtils->printPacket("���ͷ��յ���ACK", ackPkt);

        int oldBase = base;  // ����ɵ�baseֵ�������жϴ����Ƿ񻬶�

        // ֹͣ�ɵĶ�ʱ����Ϊ�ɵ�base�������õĶ�ʱ����
        try {
            pns->stopTimer(SENDER, base);
        }
        catch (...) {
            // ��ʱ�������Ѿ�ֹͣ�򲻴��ڣ������쳣����������
        }

        // �������ڣ�����base��ackNum+1
        // ���磺�յ�ACK 3����ʾ���1,2,3����ȷ�ϣ�base����Ϊ4
        base = ackNum + 1;

        //  ������ȷ�ϵı��ģ��ͷ��ڴ�
        // ������oldBase��ackNum��������ţ���map��ɾ����Ӧ�ı���
        for (int i = oldBase; i <= ackNum; i++) {
            window.erase(i);  // map��erase�������Զ��ͷ��ڴ�
        }

        // ��������ACK���������յ���ACK˵��û�ж�����
        duplicateAckCount = 0;
        lastAckNum = ackNum;  // �����ϴ��յ���ACK���

        // ������ڷǿգ�����δȷ�ϵı��ģ���Ϊ�µ�base������ʱ��
        if (base < nextSeqNum) {
            pns->startTimer(SENDER, Configuration::TIME_OUT, base);
        }

        // ���´���״̬
        waitingState = (nextSeqNum >= base + windowSize);

        // �������ȷʵ�����ˣ�base�����仯������¼����־
        if (oldBase != base && visualizer != nullptr) {
            visualizer->logWindowMove(oldBase, base, ackNum);
            visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
                duplicateAckCount, lastAckNum);
        }
    }
    // ========== ���2���յ�����ACK���ظ�ȷ�ϣ� ==========
    else if (ackNum == lastAckNum) {
        duplicateAckCount++;  // ����ACK������+1
        pUtils->printPacket("���ͷ��յ�����ACK", ackPkt);

        // ��¼����ACK�¼�����־
        if (visualizer != nullptr) {
            visualizer->logDuplicateAck(ackNum, duplicateAckCount);
        }

        // TCP�����ش����ƣ��յ�3������ACK���ܹ�4����ͬ��ACK��
        if (duplicateAckCount == 3) {
            //  ��ȫ��飺ȷ��window�д���base��Ӧ�ı���
            // ���base�����ѱ�ɾ���������ϲ�Ӧ�÷�������ֱ�ӷ���
            if (window.find(base) == window.end()) {
                return;
            }

            // ��¼�����ش��¼�����־
            if (visualizer != nullptr) {
                visualizer->logFastRetransmit(base, duplicateAckCount);
            }

            pUtils->printPacket("�յ�3������ACK,���������ش�", window[base]);

            // ֹͣ�ɶ�ʱ��
            try {
                pns->stopTimer(SENDER, base);
            }
            catch (...) {}

            // �ش�����δȷ�ϵı��ģ�baseλ�õı��ģ�
            pns->sendToNetworkLayer(RECEIVER, window[base]);

            // ����������ʱ��
            pns->startTimer(SENDER, Configuration::TIME_OUT, base);

            // ��¼�����ش���Ĵ���״̬
            if (visualizer != nullptr) {
                visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
                    duplicateAckCount, lastAckNum);
            }

            // ��������ACK������
            duplicateAckCount = 0;
        }
        else {
            // ����ACK����δ�ﵽ3����ֻ��¼����״̬
            if (visualizer != nullptr) {
                visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
                    duplicateAckCount, lastAckNum);
            }
        }
    }
    // ========== ���3���յ��ɵ�ACK����lastAckNum��С�� ==========
    else {
        // ��ACK���ã�ֱ�Ӷ���
        pUtils->printPacket("���ͷ��յ���ACK������", ackPkt);
    }
}

/**
 * ��ʱ����ʱ������
 *
 * @param seqNum ��ʱ����ţ�TCPʹ�õ�һ��ʱ����seqNumӦ�õ���base��
 *
 * TCP��ʱ�ش����ԣ�
 * - ֻ�ش�����δȷ�ϵı��ģ�baseλ�õı��ģ�
 * - ����GBN�����ش���������
 *
 * �������̣�
 * 1. ��鴰���Ƿ�Ϊ��
 * 2. ���base�����Ƿ����
 * 3. �ش�baseλ�õı���
 * 4. ����������ʱ��
 * 5. ��������ACK������
 */
void TCPRdtSender::timeoutHandler(int seqNum) {
    // ���1�������Ƿ�Ϊ��
    // ���base >= nextSeqNum��˵�����б��Ķ��ѷ��Ͳ�ȷ�ϣ������ش�
    if (base >= nextSeqNum) {
        return;
    }

    // ���2��window���Ƿ����base��Ӧ�ı���
    // ��������ѱ�ɾ���������ϲ�Ӧ�÷�������ֱ�ӷ���
    if (window.find(base) == window.end()) {
        return;
    }

    // ��ӡ��ʱ��Ϣ���ش�baseλ�õı���
    pUtils->printPacket("��ʱ����ʱ���ش�����δȷ�ϱ���", window[base]);

    // ֹͣ�ɶ�ʱ��
    try {
        pns->stopTimer(SENDER, base);
    }
    catch (...) {
        // ��ʱ�������Ѿ�ֹͣ�������쳣����������
    }

    // �ش�����δȷ�ϵı��ģ�TCP�ĳ�ʱ�ش����ԣ�
    pns->sendToNetworkLayer(RECEIVER, window[base]);

    // ����������ʱ����ֻ���ڴ��ڷǿ�ʱ��
    if (base < nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // ��������ACK����������ʱ˵��֮ǰ������ACK������Ч��
    duplicateAckCount = 0;

    // ��¼��ʱ�ش��¼��ʹ���״̬����־
    if (visualizer != nullptr) {
        visualizer->logTCPTimeoutRetransmit(base, nextSeqNum);
        visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
            duplicateAckCount, lastAckNum);
    }
}
