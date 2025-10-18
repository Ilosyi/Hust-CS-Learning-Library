#include "stdafx.h"
#include "Global.h"
#include "GBNRdtSender.h"

// 构造函数：初始化发送方状态
GBNRdtSender::GBNRdtSender() : base(0), nextSeqNum(0), waitingState(false)
{
    // 初始化存放数据包的缓冲区，大小为序列号空间的大小
    // GBN协议需要缓存所有已发送但未确认的数据包，以备超时重传
    sndpkt.resize(SEQ_SIZE);

    // 创建可视化工具，用于记录和显示滑动窗口的移动
    visualizer = new WindowVisualizer("gbn_window_log.txt");

    // 显示初始窗口状态，帮助调试和理解
    visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);
}

// 析构函数：释放动态分配的内存
GBNRdtSender::~GBNRdtSender()
{
    delete visualizer;
}

// 获取发送方等待状态
// 如果发送窗口已满，返回true，表示不能再从上层接收数据
bool GBNRdtSender::getWaitingState()
{
    return waitingState;
}

// 发送应用层数据
bool GBNRdtSender::send(const Message& message)
{
    // 检查发送窗口是否已满。如果窗口已满，则拒绝发送
    if (getWindowSize() >= WINDOW_SIZE) {
        waitingState = true; // 设置等待状态为true
        pUtils->printPacket("GBN发送方:窗口已满,拒绝发送", Packet());
        return false;
    }

    // 构造新的数据包
    Packet packet;
    packet.seqnum = nextSeqNum; // 设置报文序号
    packet.acknum = -1; // ACK字段无效
    packet.checksum = 0;
    memcpy(packet.payload, message.data, sizeof(message.data)); // 拷贝应用层数据
    packet.checksum = pUtils->calculateCheckSum(packet); // 计算校验和

    // 将数据包存入缓冲区，以便重传
    sndpkt[nextSeqNum] = packet;

    // 打印并发送报文到网络层
    pUtils->printPacket("GBN发送方发送报文", packet);
    pns->sendToNetworkLayer(RECEIVER, packet);

    // 如果是窗口中第一个未发送的报文，启动定时器
    // 定时器只为窗口基序号所对应的报文启动
    if (base == nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // 更新下一个发送序列号，并循环取模
    nextSeqNum = (nextSeqNum + 1) % SEQ_SIZE;

    // 更新等待状态，如果窗口已满，则进入等待
    waitingState = (getWindowSize() >= WINDOW_SIZE);

    // 可视化并记录当前的发送窗口状态
    visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);

    return true;
}

// 接收来自网络层的ACK报文
void GBNRdtSender::receive(const Packet& ackPkt)
{
    // 计算收到的ACK报文的校验和
    int checkSum = pUtils->calculateCheckSum(ackPkt);

    // 检查ACK报文是否损坏
    if (checkSum != ackPkt.checksum) {
        pUtils->printPacket("GBN发送方收到错误ACK", ackPkt);
        return; // 丢弃损坏的ACK报文
    }

    pUtils->printPacket("GBN发送方收到ACK", ackPkt);

    // 获取ACK报文的确认号
    int ackNum = ackPkt.acknum;
    int oldBase = base;

    // 检查ACK是否在当前发送窗口内（即是否为有效确认）
    // 这里的判断逻辑考虑了序列号的循环特性
    bool validAck = false;
    if (base <= nextSeqNum) {
        validAck = (ackNum >= base && ackNum < nextSeqNum);
    }
    else {
        validAck = (ackNum >= base || ackNum < nextSeqNum);
    }

    // 如果收到的ACK有效
    if (validAck) {
        // 停止当前定时器，因为基序号之前的报文都已确认
        pns->stopTimer(SENDER, base);

        // 窗口滑动：更新窗口基序号为(ackNum + 1)
        // GBN是累积确认，一个ACK确认了ackNum及其之前的所有报文
        base = (ackNum + 1) % SEQ_SIZE;

        // 记录窗口移动事件
        visualizer->logWindowMove(oldBase, base, ackNum);

        // 如果窗口内仍有未确认的报文，重新启动定时器
        // 定时器总是为窗口中第一个未确认的报文设置
        if (base != nextSeqNum) {
            pns->startTimer(SENDER, Configuration::TIME_OUT, base);
        }

        // 更新等待状态，如果窗口不再是满的，则允许发送新报文
        waitingState = (getWindowSize() >= WINDOW_SIZE);

        // 可视化并记录新的窗口状态
        visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);
    }
}

// 定时器超时处理函数
void GBNRdtSender::timeoutHandler(int seqNum)
{
    // 记录超时事件
    visualizer->logTimeoutRetransmit(base, nextSeqNum);

    // 打印超时提示信息
    Packet timeoutInfo;
    timeoutInfo.seqnum = seqNum;
    timeoutInfo.acknum = -1;
    timeoutInfo.checksum = 0;
    memset(timeoutInfo.payload, 0, sizeof(timeoutInfo.payload));
    strcpy(timeoutInfo.payload, "TIMEOUT");
    pUtils->printPacket("GBN发送方超时重传", timeoutInfo);

    // 停止当前定时器
    pns->stopTimer(SENDER, seqNum);

    // GBN协议核心：超时后“Go-Back-N”，重传所有已发送但未确认的数据包
    // 遍历从窗口基序号base到下一个发送序号nextSeqNum的所有报文并重传
    for (int i = base; i != nextSeqNum; i = (i + 1) % SEQ_SIZE) {
        pUtils->printPacket("GBN发送方重传数据包", sndpkt[i]);
        pns->sendToNetworkLayer(RECEIVER, sndpkt[i]);
    }

    // 重新启动定时器，再次为窗口中第一个未确认的报文设置
    if (base != nextSeqNum) {
        pns->startTimer(SENDER, Configuration::TIME_OUT, base);
    }

    // 可视化并记录重传后的窗口状态
    visualizer->visualizeSenderWindow(base, nextSeqNum, WINDOW_SIZE, SEQ_SIZE);
}

// 检查序列号是否在当前窗口内
// 注意：此函数在主逻辑中未使用，可能是为了方便调试或其他扩展
bool GBNRdtSender::isInWindow(int seqNum)
{
    if (base <= nextSeqNum) {
        return (seqNum >= base && seqNum < nextSeqNum);
    }
    else {
        return (seqNum >= base || seqNum < nextSeqNum);
    }
}

// 获取当前窗口大小
// 计算已发送但未确认的报文数量
int GBNRdtSender::getWindowSize()
{
    if (nextSeqNum >= base) {
        return nextSeqNum - base;
    }
    else {
        return nextSeqNum + SEQ_SIZE - base;
    }
}