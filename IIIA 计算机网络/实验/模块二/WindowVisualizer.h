#pragma once
#ifndef WINDOW_VISUALIZER_H
#define WINDOW_VISUALIZER_H

#include <vector>
#include <string>
#include <fstream>

class WindowVisualizer {
public:
    WindowVisualizer(const std::string& logFileName = "window_log.txt");
    ~WindowVisualizer();

    // ���ӻ����ʹ���
    void visualizeSenderWindow(int base, int nextSeqNum, int windowSize, int seqSize);

    // ���ӻ����մ���
    void visualizeReceiverWindow(int expectedSeqNum, int seqSize);

    // ��¼�����ƶ��¼�
    void logWindowMove(int oldBase, int newBase, int ackNum);

    // ��¼��ʱ�ش��¼�
    void logTimeoutRetransmit(int base, int nextSeqNum);

    // ��¼SRЭ��ĵ������ĳ�ʱ�ش�
    void logSinglePacketTimeout(int seqNum);

    // ����/�����ļ���־
    void enableFileLogging(bool enable);

    // ������TCPר�÷���
    void logFastRetransmit(int seqNum, int duplicateCount);  // �����ش���־
    void logDuplicateAck(int ackNum, int currentCount);      // ����ACK��־
    void visualizeTCPWindow(int base, int nextSeqNum, int windowSize,
         int duplicateAckCount, int lastAckNum);  // TCP���ڿ��ӻ�
    // �� WindowVisualizer.h �����
	void logTCPTimeoutRetransmit(int base, int nextSeqNum);// TCP��ʱ�ش���־
    // ========== SR���շ�ר�ÿ��ӻ����� ==========

/**
 * ���ӻ�SR���շ���������״̬
 * @param base ���մ�����ʼλ��
 * @param windowSize ���ڴ�С
 * @param seqNumRange ��ſռ��С
 * @param received ���ձ�����飨��Щ����ѽ��գ�
 */
    void visualizeSRReceiverWindow(int base, int windowSize, int seqNumRange,
        const std::vector<bool>& received);

    /**
     * ��¼���շ����յ������¼�
     * @param seqNum �������
     * @param isNew �Ƿ����״ν��գ�true=�±��ģ�false=�ظ����ģ�
     */
    void logSRReceiverReceivePacket(int seqNum, bool isNew);

    /**
     * ��¼���շ����ڻ����¼�
     * @param oldBase �ɵ�base
     * @param newBase �µ�base
     * @param triggerSeqNum ���������ı������
     */
    void logSRReceiverWindowSlide(int oldBase, int newBase, int triggerSeqNum);

    /**
     * ��¼���յ���ȷ�Ϲ��ľɱ���
     * @param seqNum �ɱ������
     */
    void logSRReceiverReceiveOldPacket(int seqNum);

    /**
     * ��¼���������ⱨ��
     * @param seqNum �������ı������
     */
    void logSRReceiverDiscardPacket(int seqNum);

#endif

private:
    std::ofstream logFile;
    bool fileLoggingEnabled;

    // ������������ӡ������̨���ļ�
    void output(const std::string& message);

    // �������������ƴ���ͼ��
    std::string drawWindow(const std::vector<int>& windowSeqs, int seqSize,
        const std::vector<char>& markers);
};
