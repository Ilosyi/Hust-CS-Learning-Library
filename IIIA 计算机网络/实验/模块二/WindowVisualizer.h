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

    // 可视化发送窗口
    void visualizeSenderWindow(int base, int nextSeqNum, int windowSize, int seqSize);

    // 可视化接收窗口
    void visualizeReceiverWindow(int expectedSeqNum, int seqSize);

    // 记录窗口移动事件
    void logWindowMove(int oldBase, int newBase, int ackNum);

    // 记录超时重传事件
    void logTimeoutRetransmit(int base, int nextSeqNum);

    // 记录SR协议的单个报文超时重传
    void logSinglePacketTimeout(int seqNum);

    // 启用/禁用文件日志
    void enableFileLogging(bool enable);

    // 新增：TCP专用方法
    void logFastRetransmit(int seqNum, int duplicateCount);  // 快速重传日志
    void logDuplicateAck(int ackNum, int currentCount);      // 冗余ACK日志
    void visualizeTCPWindow(int base, int nextSeqNum, int windowSize,
         int duplicateAckCount, int lastAckNum);  // TCP窗口可视化
    // 在 WindowVisualizer.h 中添加
	void logTCPTimeoutRetransmit(int base, int nextSeqNum);// TCP超时重传日志
    // ========== SR接收方专用可视化方法 ==========

/**
 * 可视化SR接收方滑动窗口状态
 * @param base 接收窗口起始位置
 * @param windowSize 窗口大小
 * @param seqNumRange 序号空间大小
 * @param received 接收标记数组（哪些序号已接收）
 */
    void visualizeSRReceiverWindow(int base, int windowSize, int seqNumRange,
        const std::vector<bool>& received);

    /**
     * 记录接收方接收到报文事件
     * @param seqNum 报文序号
     * @param isNew 是否是首次接收（true=新报文，false=重复报文）
     */
    void logSRReceiverReceivePacket(int seqNum, bool isNew);

    /**
     * 记录接收方窗口滑动事件
     * @param oldBase 旧的base
     * @param newBase 新的base
     * @param triggerSeqNum 触发滑动的报文序号
     */
    void logSRReceiverWindowSlide(int oldBase, int newBase, int triggerSeqNum);

    /**
     * 记录接收到已确认过的旧报文
     * @param seqNum 旧报文序号
     */
    void logSRReceiverReceiveOldPacket(int seqNum);

    /**
     * 记录丢弃窗口外报文
     * @param seqNum 被丢弃的报文序号
     */
    void logSRReceiverDiscardPacket(int seqNum);

#endif

private:
    std::ofstream logFile;
    bool fileLoggingEnabled;

    // 辅助函数：打印到控制台和文件
    void output(const std::string& message);

    // 辅助函数：绘制窗口图形
    std::string drawWindow(const std::vector<int>& windowSeqs, int seqSize,
        const std::vector<char>& markers);
};
