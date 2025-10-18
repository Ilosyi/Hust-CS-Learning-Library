#include "stdafx.h"
#include "Global.h"
#include "TCPRdtSender.h"
#include <cstring>

/**
 * 构造函数：初始化TCP发送方
 *
 * 初始化参数说明：
 * - windowSize(4): 发送窗口大小为4，即同时最多发送4个未确认的报文
 * - base(1): 窗口起始序号为1（最早未确认的报文序号）
 * - nextSeqNum(1): 下一个待发送的报文序号为1
 * - duplicateAckCount(0): 冗余ACK计数器初始化为0
 * - lastAckNum(0): 上次收到的ACK序号初始化为0
 * - waitingState(false): 初始窗口未满，不处于等待状态
 * - visualizer(nullptr): 可视化工具指针初始化为空
 */
TCPRdtSender::TCPRdtSender()
    : windowSize(4), base(1), nextSeqNum(1),
    duplicateAckCount(0), lastAckNum(0), waitingState(false), visualizer(nullptr) {

    //  使用std::map存储窗口中的报文，不需要预先分配空间
    // map会根据序号自动扩展，避免了固定大小数组的越界风险

    // 初始化窗口可视化工具，用于记录窗口状态到日志文件
    visualizer = new WindowVisualizer("tcp_window_log.txt");
    if (visualizer != nullptr) {
        // 记录初始窗口状态（窗口为空）
        visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
            duplicateAckCount, lastAckNum);
    }
}

/**
 * 析构函数：清理资源
 *
 * 功能：
 * - 安全释放可视化工具对象的内存
 * - 将指针置为nullptr，防止悬空指针
 */
TCPRdtSender::~TCPRdtSender() {
    if (visualizer != nullptr) {
        delete visualizer;      // 释放动态分配的内存
        visualizer = nullptr;   // 防止重复释放导致的问题
    }
}

/**
 * 获取发送方等待状态
 *
 * @return true - 窗口已满，拒绝发送新报文
 *         false - 窗口未满，可以继续发送
 */
bool TCPRdtSender::getWaitingState() {
    return waitingState;
}

/**
 * 发送应用层消息
 *
 * @param message 应用层下发的消息数据
 * @return true - 发送成功
 *         false - 窗口已满，拒绝发送
 *
 * 工作流程：
 * 1. 检查窗口是否已满（nextSeqNum >= base + windowSize）
 * 2. 构造TCP报文段（设置序号、计算校验和）
 * 3. 将报文缓存到窗口中（用于可能的重传）
 * 4. 发送报文到网络层
 * 5. 如果是窗口中第一个报文，启动定时器
 * 6. 更新窗口状态并记录日志
 */
bool TCPRdtSender::send(const Message& message) {
    // 检查窗口是否已满
    // 窗口满的条件：nextSeqNum >= base + windowSize
    // 例如：base=1, windowSize=4，则窗口范围是[1,2,3,4]，nextSeqNum=5时窗口满
    if (nextSeqNum >= base + windowSize) {
        waitingState = true;  // 标记为等待状态
        return false;         // 拒绝发送
    }

    // 构造TCP报文段
    Packet packet;
    packet.seqnum = nextSeqNum;     // 设置序列号（按报文段编号）
    packet.acknum = -1;              // 发送方不使用确认号字段
    packet.checksum = 0;             // 先初始化校验和为0

    // 拷贝应用层数据到报文载荷
    memcpy(packet.payload, message.data, sizeof(message.data));

    // 计算并设置校验和（用于接收方检测报文是否损坏）
    packet.checksum = pUtils->calculateCheckSum(packet);

    //  使用map存储报文，key为序号，value为报文对象
    // map会自动管理内存，不会像固定大小数组那样越界
    window[nextSeqNum] = packet;

    // 打印调试信息并通过网络层发送报文
    pUtils->printPacket("发送方发送报文", packet);
    pns->sendToNetworkLayer(RECEIVER, packet);

    // 如果这是窗口中的第一个报文（base == nextSeqNum），启动定时器
    // TCP使用单一定时器，只为最早未确认的报文（base）计时
    if (base == nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // 更新下一个待发送的序号
    nextSeqNum++;

    // 更新窗口状态：如果窗口满了，设置等待标志
    waitingState = (nextSeqNum >= base + windowSize);

    // 记录发送后的窗口状态到日志
    if (visualizer != nullptr) {
        visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
            duplicateAckCount, lastAckNum);
    }

    return true;  // 发送成功
}

/**
 * 接收并处理ACK报文
 *
 * @param ackPkt 接收到的ACK报文
 *
 * TCP累积确认机制：
 * - ACK n 表示：已正确接收到序号≤n的所有报文，期待接收序号为n+1的报文
 *
 * 处理三种情况：
 * 1. 新ACK（ackNum >= base）：窗口向前滑动，清理已确认的报文
 * 2. 冗余ACK（ackNum == lastAckNum）：增加计数，可能触发快速重传
 * 3. 旧ACK（ackNum < lastAckNum）：直接丢弃
 */
void TCPRdtSender::receive(const Packet& ackPkt) {
    // 第一步：验证ACK报文的完整性
    int checkSum = pUtils->calculateCheckSum(ackPkt);
    if (checkSum != ackPkt.checksum) {
        // 校验和不匹配，说明ACK在传输中损坏，丢弃
        pUtils->printPacket("发送方收到损坏的ACK，丢弃", ackPkt);
        return;
    }

    int ackNum = ackPkt.acknum;  // 获取确认号

    // ========== 情况1：收到新的ACK（确认了新的数据） ==========
    if (ackNum >= base) {
        pUtils->printPacket("发送方收到新ACK", ackPkt);

        int oldBase = base;  // 保存旧的base值，用于判断窗口是否滑动

        // 停止旧的定时器（为旧的base报文设置的定时器）
        try {
            pns->stopTimer(SENDER, base);
        }
        catch (...) {
            // 定时器可能已经停止或不存在，捕获异常避免程序崩溃
        }

        // 滑动窗口：更新base到ackNum+1
        // 例如：收到ACK 3，表示序号1,2,3都已确认，base更新为4
        base = ackNum + 1;

        //  清理已确认的报文，释放内存
        // 遍历从oldBase到ackNum的所有序号，从map中删除对应的报文
        for (int i = oldBase; i <= ackNum; i++) {
            window.erase(i);  // map的erase操作会自动释放内存
        }

        // 重置冗余ACK计数器（收到新ACK说明没有丢包）
        duplicateAckCount = 0;
        lastAckNum = ackNum;  // 更新上次收到的ACK序号

        // 如果窗口非空（还有未确认的报文），为新的base启动定时器
        if (base < nextSeqNum) {
            pns->startTimer(SENDER, Configuration::TIME_OUT, base);
        }

        // 更新窗口状态
        waitingState = (nextSeqNum >= base + windowSize);

        // 如果窗口确实滑动了（base发生变化），记录到日志
        if (oldBase != base && visualizer != nullptr) {
            visualizer->logWindowMove(oldBase, base, ackNum);
            visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
                duplicateAckCount, lastAckNum);
        }
    }
    // ========== 情况2：收到冗余ACK（重复确认） ==========
    else if (ackNum == lastAckNum) {
        duplicateAckCount++;  // 冗余ACK计数器+1
        pUtils->printPacket("发送方收到冗余ACK", ackPkt);

        // 记录冗余ACK事件到日志
        if (visualizer != nullptr) {
            visualizer->logDuplicateAck(ackNum, duplicateAckCount);
        }

        // TCP快速重传机制：收到3个冗余ACK（总共4个相同的ACK）
        if (duplicateAckCount == 3) {
            //  安全检查：确保window中存在base对应的报文
            // 如果base报文已被删除（理论上不应该发生），直接返回
            if (window.find(base) == window.end()) {
                return;
            }

            // 记录快速重传事件到日志
            if (visualizer != nullptr) {
                visualizer->logFastRetransmit(base, duplicateAckCount);
            }

            pUtils->printPacket("收到3个冗余ACK,触发快速重传", window[base]);

            // 停止旧定时器
            try {
                pns->stopTimer(SENDER, base);
            }
            catch (...) {}

            // 重传最早未确认的报文（base位置的报文）
            pns->sendToNetworkLayer(RECEIVER, window[base]);

            // 重新启动定时器
            pns->startTimer(SENDER, Configuration::TIME_OUT, base);

            // 记录快速重传后的窗口状态
            if (visualizer != nullptr) {
                visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
                    duplicateAckCount, lastAckNum);
            }

            // 重置冗余ACK计数器
            duplicateAckCount = 0;
        }
        else {
            // 冗余ACK数量未达到3个，只记录窗口状态
            if (visualizer != nullptr) {
                visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
                    duplicateAckCount, lastAckNum);
            }
        }
    }
    // ========== 情况3：收到旧的ACK（比lastAckNum还小） ==========
    else {
        // 旧ACK无用，直接丢弃
        pUtils->printPacket("发送方收到旧ACK，丢弃", ackPkt);
    }
}

/**
 * 定时器超时处理函数
 *
 * @param seqNum 超时的序号（TCP使用单一定时器，seqNum应该等于base）
 *
 * TCP超时重传策略：
 * - 只重传最早未确认的报文（base位置的报文）
 * - 不像GBN那样重传整个窗口
 *
 * 工作流程：
 * 1. 检查窗口是否为空
 * 2. 检查base报文是否存在
 * 3. 重传base位置的报文
 * 4. 重新启动定时器
 * 5. 重置冗余ACK计数器
 */
void TCPRdtSender::timeoutHandler(int seqNum) {
    // 检查1：窗口是否为空
    // 如果base >= nextSeqNum，说明所有报文都已发送并确认，无需重传
    if (base >= nextSeqNum) {
        return;
    }

    // 检查2：window中是否存在base对应的报文
    // 如果报文已被删除（理论上不应该发生），直接返回
    if (window.find(base) == window.end()) {
        return;
    }

    // 打印超时信息并重传base位置的报文
    pUtils->printPacket("定时器超时，重传最早未确认报文", window[base]);

    // 停止旧定时器
    try {
        pns->stopTimer(SENDER, base);
    }
    catch (...) {
        // 定时器可能已经停止，捕获异常避免程序崩溃
    }

    // 重传最早未确认的报文（TCP的超时重传策略）
    pns->sendToNetworkLayer(RECEIVER, window[base]);

    // 重新启动定时器（只有在窗口非空时）
    if (base < nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // 重置冗余ACK计数器（超时说明之前的冗余ACK计数无效）
    duplicateAckCount = 0;

    // 记录超时重传事件和窗口状态到日志
    if (visualizer != nullptr) {
        visualizer->logTCPTimeoutRetransmit(base, nextSeqNum);
        visualizer->visualizeTCPWindow(base, nextSeqNum, windowSize,
            duplicateAckCount, lastAckNum);
    }
}
