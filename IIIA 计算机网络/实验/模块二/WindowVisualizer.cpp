#include "stdafx.h"
#include "WindowVisualizer.h"
#include <iostream>
#include <sstream>
#include <iomanip>

WindowVisualizer::WindowVisualizer(const std::string& logFileName)
    : fileLoggingEnabled(true) {
    logFile.open(logFileName.c_str());
    if (!logFile.is_open()) {
        std::cerr << "警告: 无法打开日志文件 " << logFileName << std::endl;
        fileLoggingEnabled = false;
    }
    else {
        output("=== 滑动窗口日志 ===\n");
    }
}

WindowVisualizer::~WindowVisualizer() {
    if (logFile.is_open()) {
        logFile.flush();
        logFile.close();
    }
}

void WindowVisualizer::output(const std::string& message) {
    // 输出到控制台
    std::cout << message;

    // 输出到文件
    if (fileLoggingEnabled && logFile.is_open()) {
        logFile << message;
        logFile.flush();  // 立即写入文件
    }
}

void WindowVisualizer::visualizeSenderWindow(int base, int nextSeqNum,
    int windowSize, int seqSize) {
    std::ostringstream oss;
    oss << "\n【发送方滑动窗口】\n";
    oss << "窗口基序号(base): " << base << "\n";
    oss << "下一个发送序号(nextSeqNum): " << nextSeqNum << "\n";
    oss << "窗口大小: " << windowSize << "\n";

    // 绘制窗口
    oss << "序号空间: ";
    for (int i = 0; i < seqSize; i++) {
        oss << std::setw(3) << i << " ";
    }
    oss << "\n窗口状态: ";

    for (int i = 0; i < seqSize; i++) {
        bool inWindow = false;
        char marker = ' ';

        if (base <= nextSeqNum) {
            inWindow = (i >= base && i < nextSeqNum);
        }
        else {
            inWindow = (i >= base || i < nextSeqNum);
        }

        if (i == base) {
            marker = 'B';  // Base
        }
        else if (i == nextSeqNum) {
            marker = 'N';  // Next
        }
        else if (inWindow) {
            marker = '*';  // 在窗口内
        }
        else {
            marker = '-';  // 不在窗口内
        }

        oss << " [" << marker << "] ";
    }
    oss << "\n";

    // 已发送未确认的序号
    oss << "已发送未确认: ";
    int count = 0;
    for (int i = base; i != nextSeqNum; i = (i + 1) % seqSize) {
        if (count > 0) oss << ", ";
        oss << i;
        count++;
    }
    if (count == 0) oss << "无";
    oss << "\n";

    oss << "----------------------------------------\n";
    output(oss.str());
}

void WindowVisualizer::visualizeReceiverWindow(int expectedSeqNum, int seqSize) {
    std::ostringstream oss;
    oss << "\n【接收方状态】\n";
    oss << "期望序号(expectedSeqNum): " << expectedSeqNum << "\n";
    oss << "序号空间: ";
    for (int i = 0; i < seqSize; i++) {
        oss << std::setw(3) << i << " ";
    }
    oss << "\n接收状态: ";

    for (int i = 0; i < seqSize; i++) {
        if (i == expectedSeqNum) {
            oss << " [E] ";  // Expected
        }
        else if (i < expectedSeqNum) {
            oss << " [√] ";  // 已接收
        }
        else {
            oss << " [ ] ";  // 未接收
        }
    }
    oss << "\n----------------------------------------\n";
    output(oss.str());
}

void WindowVisualizer::logWindowMove(int oldBase, int newBase, int ackNum) {
    std::ostringstream oss;
    oss << "\n>>> 窗口滑动事件 <<<\n";
    oss << "收到ACK: " << ackNum << "\n";
    oss << "旧base: " << oldBase << " -> 新base: " << newBase << "\n";
    oss << "窗口向前移动了 " << ((newBase - oldBase + 8) % 8) << " 个位置\n";
    output(oss.str());
}

void WindowVisualizer::logTimeoutRetransmit(int base, int nextSeqNum) {
    std::ostringstream oss;
    oss << "\n!!! 超时重传事件 !!!\n";
    oss << "重传窗口: [" << base << ", " << nextSeqNum << ")\n";
    oss << "重传序号: ";

    int seqSize = 8;  // 假设序号空间为8
    for (int i = base; i != nextSeqNum; i = (i + 1) % seqSize) {
        oss << i << " ";
    }
    oss << "\n";
    output(oss.str());
}

void WindowVisualizer::logSinglePacketTimeout(int seqNum) {
    std::ostringstream oss;
    oss << "\n!!! SR超时重传事件 !!!\n";
    oss << "超时报文序号: " << seqNum << "\n";
    oss << "仅重传该报文（SR特性）\n";
    output(oss.str());
}

void WindowVisualizer::enableFileLogging(bool enable) {
    fileLoggingEnabled = enable;
}
void WindowVisualizer::logFastRetransmit(int seqNum, int duplicateCount) {
    std::ostringstream oss;
    oss << "\n";
    oss << "╔════════════════════════════════════════╗\n";
    oss << "║      TCP 快速重传事件触发           ║\n";
    oss << "╚════════════════════════════════════════╝\n";
    oss << "触发条件: 收到 " << duplicateCount << " 个冗余ACK\n";
    oss << "重传报文序号: " << seqNum << "\n";
    oss << "重传原因: 检测到报文丢失，快速恢复\n";
    oss << "性能优势: 无需等待超时，立即重传\n";
    oss << "----------------------------------------\n";

    // 只写入文件，不输出到控制台
    if (fileLoggingEnabled && logFile.is_open()) {
        logFile << oss.str();
 
    }
}

void WindowVisualizer::logDuplicateAck(int ackNum, int currentCount) {
    std::ostringstream oss;
    oss << "\n>>> 收到冗余ACK <<<\n";
    oss << "ACK序号: " << ackNum << "\n";
    oss << "当前冗余ACK计数: " << currentCount << "\n";
    if (currentCount < 3) {
        oss << "状态: 等待更多冗余ACK (" << (3 - currentCount) << " 个后触发快速重传)\n";
    }
    else {
        oss << "状态:  即将触发快速重传！\n";
    }
    oss << "----------------------------------------\n";

    // 只写入文件，不输出到控制台
    if (fileLoggingEnabled && logFile.is_open()) {
        logFile << oss.str();
    }
}

void WindowVisualizer::visualizeTCPWindow(int base, int nextSeqNum, int windowSize,
    int duplicateAckCount, int lastAckNum) {
    std::ostringstream oss;
    oss << "\n╔═══════════════════════════════════════════════════╗\n";
    oss << "║          TCP 发送方滑动窗口状态                   ║\n";
    oss << "╚═══════════════════════════════════════════════════╝\n";

    oss << "窗口参数:\n";
    oss << "   base (最早未确认序号): " << base << "\n";
    oss << "   nextSeqNum (下一个发送序号): " << nextSeqNum << "\n";
    oss << "   窗口大小: " << windowSize << "\n";
    oss << "   已发送未确认数量: " << (nextSeqNum - base) << "\n";
    oss << "   可用发送空间: " << (windowSize - (nextSeqNum - base)) << "\n";

    oss << "\n快速重传状态:\n";
    oss << "   上次ACK序号: " << lastAckNum << "\n";
    oss << "   冗余ACK计数: " << duplicateAckCount << "/3\n";
    if (duplicateAckCount == 0) {
        oss << "   状态:  正常传输\n";
    }
    else if (duplicateAckCount < 3) {
        oss << "   状态:  检测到冗余ACK\n";
    }
    else {
        oss << "   状态:  快速重传已触发\n";
    }

    oss << "\n窗口可视化:\n";
    oss << "序号: ";
    for (int i = base; i < base + windowSize + 2; i++) {
        oss << std::setw(4) << i;
    }
    oss << "\n状态: ";

    for (int i = base; i < base + windowSize + 2; i++) {
        if (i == base) {
            oss << " [B]";  // Base
        }
        else if (i == nextSeqNum) {
            oss << " [N]";  // Next
        }
        else if (i < nextSeqNum) {
            oss << " [*]";  // 已发送未确认
        }
        else if (i < base + windowSize) {
            oss << " [ ]";  // 窗口内可用
        }
        else {
            oss << "  - ";  // 窗口外
        }
    }
    oss << "\n";
    oss << "说明: [B]=base  [N]=nextSeqNum  [*]=已发送未确认  [ ]=可发送  -=窗口外\n";
    oss << "═════════════════════════════════════════════════════\n";

    // 只写入文件，不输出到控制台
    if (fileLoggingEnabled && logFile.is_open()) {
        logFile << oss.str();
    }
}
void WindowVisualizer::logTCPTimeoutRetransmit(int base, int nextSeqNum) {
    std::ostringstream oss;
    oss << "\n!!! TCP 超时重传事件 !!!\n";
    oss << "重传报文序号: " << base << "\n";
    oss << "当前窗口: [" << base << ", " << nextSeqNum << ")\n";
    oss << "窗口内报文数: " << (nextSeqNum - base) << "\n";
    oss << "----------------------------------------\n";

    if (fileLoggingEnabled && logFile.is_open()) {
        logFile << oss.str();
    }
}
/**
 * 可视化SR接收方滑动窗口状态
 */
void WindowVisualizer::visualizeSRReceiverWindow(int base, int windowSize,
    int seqNumRange,
    const std::vector<bool>& received) {
    std::ostringstream oss;
    oss << "\n╔═══════════════════════════════════════════════════╗\n";
    oss << "║          SR 接收方滑动窗口状态                    ║\n";
    oss << "╚═══════════════════════════════════════════════════╝\n";

    oss << "窗口参数:\n";
    oss << "   base (窗口起始序号): " << base << "\n";
    oss << "   窗口大小: " << windowSize << "\n";
    oss << "   序号空间大小: " << seqNumRange << "\n";

    int windowEnd = (base + windowSize) % seqNumRange;
    oss << "   窗口范围: [" << base << ", " << windowEnd << ")\n";

    oss << "\n窗口可视化:\n";
    oss << "序号: ";
    for (int i = 0; i < seqNumRange; i++) {
        oss << std::setw(3) << i << " ";
    }
    oss << "\n状态: ";

    for (int i = 0; i < seqNumRange; i++) {
        bool inWindow = false;
        int pos = (i - base + seqNumRange) % seqNumRange;
        inWindow = (pos >= 0 && pos < windowSize);

        if (i == base) {
            oss << "[B]";
        }
        else if (inWindow) {
            if (received[i]) {  // 直接使用 vector 的 [] 运算符
                oss << "[R]";
            }
            else {
                oss << "[E]";
            }
        }
        else {
            oss << "[-]";
        }
        oss << " ";
    }
    oss << "\n";
    oss << "说明: [B]=base  [R]=已接收  [E]=期待接收  [-]=窗口外\n";

    oss << "\n窗口内已接收: ";
    bool hasReceived = false;
    for (int i = 0; i < windowSize; i++) {
        int seqNum = (base + i) % seqNumRange;
        if (received[seqNum]) {  //  直接使用 [] 运算符
            if (hasReceived) oss << ", ";
            oss << seqNum;
            hasReceived = true;
        }
    }
    if (!hasReceived) oss << "无";
    oss << "\n";

    oss << "═════════════════════════════════════════════════════\n";
    output(oss.str());
}


/**
 * 记录接收方接收到报文事件
 */
void WindowVisualizer::logSRReceiverReceivePacket(int seqNum, bool isNew) {
    std::ostringstream oss;
    if (isNew) {
        oss << "\n>>> 接收方接收新报文 <<<\n";
        oss << "报文序号: " << seqNum << "\n";
        oss << "状态:  首次接收，已缓存\n";
    }
    else {
        oss << "\n>>> 接收方收到重复报文 <<<\n";
        oss << "报文序号: " << seqNum << "\n";
        oss << "状态:  重复接收，已缓存无需再存\n";
    }
    oss << "----------------------------------------\n";
    output(oss.str());
}

/**
 * 记录接收方窗口滑动事件
 */
void WindowVisualizer::logSRReceiverWindowSlide(int oldBase, int newBase, int triggerSeqNum) {
    std::ostringstream oss;
    oss << "\n╔════════════════════════════════════════╗\n";
    oss << "║    SR 接收方窗口滑动事件            ║\n";
    oss << "╚════════════════════════════════════════╝\n";
    oss << "触发报文序号: " << triggerSeqNum << "\n";
    oss << "旧base: " << oldBase << " -> 新base: " << newBase << "\n";

    // 计算滑动距离
    int slideDistance = (newBase - oldBase + 8) % 8;
    oss << "窗口向前滑动了 " << slideDistance << " 个位置\n";

    // 列出交付的报文序号
    oss << "已向上交付报文序号: ";
    for (int i = 0; i < slideDistance; i++) {
        if (i > 0) oss << ", ";
        oss << ((oldBase + i) % 8);
    }
    oss << "\n";
    oss << "----------------------------------------\n";
    output(oss.str());
}

/**
 * 记录接收到已确认过的旧报文
 */
void WindowVisualizer::logSRReceiverReceiveOldPacket(int seqNum) {
    std::ostringstream oss;
    oss << "\n>>> 接收方收到历史报文 <<<\n";
    oss << "报文序号: " << seqNum << "\n";
    oss << "状态:  该报文已交付过，重发ACK\n";
    oss << "原因: ACK可能在传输中丢失，发送方超时重传\n";
    oss << "----------------------------------------\n";
    output(oss.str());
}

/**
 * 记录丢弃窗口外报文
 */
void WindowVisualizer::logSRReceiverDiscardPacket(int seqNum) {
    std::ostringstream oss;
    oss << "\n>>> 接收方丢弃报文 <<<\n";
    oss << "报文序号: " << seqNum << "\n";
    oss << "状态:  序号在窗口范围之外，直接丢弃\n";
    oss << "原因: 可能是延迟到达的未来报文\n";
    oss << "----------------------------------------\n";
    output(oss.str());
}
