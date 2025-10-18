#include "stdafx.h"
#include "Global.h"
#include "SRRdtReceiver.h"
#include <cstring>

/**
 * ���캯������ʼ�����մ��ں���ز���
 */
SRRdtReceiver::SRRdtReceiver() : windowSize(4), seqNumBits(3), base(0) {
    // �������λ��������ŷ�Χ������3λ��ŷ�ΧΪ2^3=8
    seqNumRange = 1 << seqNumBits;

    // ������ŷ�Χ��ʼ������ͽ��ձ������
    // received[i] Ϊ true ��ʾ���Ϊi�ı����ѽ��ղ�����
    received.resize(seqNumRange, false);
    buffer.resize(seqNumRange);

    //  ��ʼ�����շ����ӻ���
    visualizer = new WindowVisualizer("sr_receiver_log.txt");
    visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
}

/**
 * �����������ͷſ��ӻ�����Դ
 */
SRRdtReceiver::~SRRdtReceiver() {
    if (visualizer != nullptr) {
        delete visualizer;
        visualizer = nullptr;
    }
}

/**
 * ������������Ƿ��ڽ��մ�����
 * ʹ����ѧ�����жϣ�������ѭ���������Ч��
 */
bool SRRdtReceiver::inWindow(int seqNum, int base, int windowSize, int range) {
    int pos = (seqNum - base + range) % range;
    return pos >= 0 && pos < windowSize;
}

/**
 * ��װ�ķ��� ACK ����
 */
void SRRdtReceiver::sendAck(int seqNum) {
    Packet ackPkt;
    ackPkt.acknum = seqNum;
    ackPkt.seqnum = -1; // ACK����seqnum�ֶ���Ч
    ackPkt.checksum = 0;
    memset(ackPkt.payload, 0, sizeof(ackPkt.payload));
    ackPkt.checksum = pUtils->calculateCheckSum(ackPkt);

    pUtils->printPacket("���շ�����ȷ��", ackPkt);
    pns->sendToNetworkLayer(SENDER, ackPkt);
}

/**
 * �������ݰ��ĺ��ĺ���
 */
void SRRdtReceiver::receive(const Packet& packet) {
    // ���У��ͣ��������ֱ�Ӷ���
    int checkSum = pUtils->calculateCheckSum(packet);
    if (checkSum != packet.checksum) {
        pUtils->printPacket("���շ��յ��𻵵ı��ģ�����", packet);

        //  ��¼�����𻵱����¼�������״̬���䣩
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
        return;
    }

    int seqNum = packet.seqnum;

    // ========== ���1����������ڽ��մ����ڣ�[base, base + windowSize - 1] ==========
    if (inWindow(seqNum, base, windowSize, seqNumRange)) {
        pUtils->printPacket("���շ��յ������ڱ���", packet);

        // ������״��յ��ı��ģ��򻺴沢���
        if (!received[seqNum]) {
            buffer[seqNum] = packet;
            received[seqNum] = true;

            //  ��¼�����±����¼�
            visualizer->logSRReceiverReceivePacket(seqNum, true);
        }
        else {
            //  �ظ����ձ���
            visualizer->logSRReceiverReceivePacket(seqNum, false);
        }

        // ���ͶԸñ��ĵĶ���ȷ�ϣ���ʹ���ظ��յ���
        sendAck(seqNum);

        int oldBase = base;  // ����ɵ�baseֵ

        // ����յ��ı��������Ǵ��ڵ���ʼ��ţ�base��
        if (seqNum == base) {
            // ���Խ������ġ��ѽ��յı������Ͻ���
            while (received[base]) {
                pUtils->printPacket("���շ���������", buffer[base]);
                Message msg;
                memcpy(msg.data, buffer[base].payload, sizeof(buffer[base].payload));
                pns->delivertoAppLayer(RECEIVER, msg);

                // ���������ñ�ǲ���������
                received[base] = false;
                base = (base + 1) % seqNumRange;
            }

            //  ������ڻ����ˣ���¼�����¼�
            if (base != oldBase) {
                visualizer->logSRReceiverWindowSlide(oldBase, base, seqNum);
            }
        }

        //  ��ʾ���º�Ľ��մ���״̬
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
    }
    // ========== ���2����������ڴ���֮�⣬������������ȷ�ϵı��� ==========
    else if (inWindow(seqNum, (base - windowSize + seqNumRange) % seqNumRange, windowSize, seqNumRange)) {
        pUtils->printPacket("���շ��յ���ȷ�Ϲ��ı��ģ��ط�ȷ��", packet);

        //  ��¼�ظ�������ʷ�����¼�
        visualizer->logSRReceiverReceiveOldPacket(seqNum);

        sendAck(seqNum); // �����ط�ACK����Ӧ��ACK��ʧ�����

        //  ��ʾ����״̬��δ�ı䣩
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
    }
    // ========== ���3�����������δ������֮�⣬ֱ�Ӷ��� ==========
    else {
        pUtils->printPacket("���շ��յ�������ı��ģ�����", packet);

        //  ��¼�����¼�
        visualizer->logSRReceiverDiscardPacket(seqNum);

        //  ��ʾ����״̬��δ�ı䣩
        visualizer->visualizeSRReceiverWindow(base, windowSize, seqNumRange, received);
    }
}
