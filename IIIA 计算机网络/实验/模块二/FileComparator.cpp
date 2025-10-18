#pragma once
#include "stdafx.h"
#include "FileComparator.h"
#include <fstream>
#include <iostream>

FileComparator::FileComparator()
{
}

FileComparator::~FileComparator()
{
}

bool FileComparator::compareFiles(const std::string& file1, const std::string& file2)
{
    std::ifstream f1(file1.c_str(), std::ios::binary);
    std::ifstream f2(file2.c_str(), std::ios::binary);

    // 检查文件是否成功打开
    if (!f1.is_open()) {
        std::cout << "错误: 无法打开文件 " << file1 << std::endl;
        return false;
    }
    if (!f2.is_open()) {
        std::cout << "错误: 无法打开文件 " << file2 << std::endl;
        return false;
    }

    // 逐字节比较文件内容
    char c1, c2;
    bool identical = true;
    int position = 0;

    while (f1.get(c1) && f2.get(c2)) {
        if (c1 != c2) {
            identical = false;
            std::cout << "文件在位置 " << position << " 处开始不同" << std::endl;
            std::cout << "文件1字符: '" << c1 << "' (ASCII: " << (int)c1 << ")" << std::endl;
            std::cout << "文件2字符: '" << c2 << "' (ASCII: " << (int)c2 << ")" << std::endl;
            break;
        }
        position++;
    }

    // 检查文件长度是否相同
    if (identical) {
        // 检查是否还有剩余内容
        if (f1.get(c1)) {
            identical = false;
            std::cout << "文件1比文件2长，在位置 " << position << " 后还有内容" << std::endl;
        }
        else if (f2.get(c2)) {
            identical = false;
            std::cout << "文件2比文件1长，在位置 " << position << " 后还有内容" << std::endl;
        }
    }

    f1.close();
    f2.close();

    return identical;
}

long FileComparator::getFileSize(const std::string& filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return -1;
    }
    long size = file.tellg();
    file.close();
    return size;
}

bool FileComparator::validateTransmission(const std::string& inputFile, const std::string& outputFile)
{
    std::cout << "\n=== 文件传输结果验证 ===" << std::endl;

    // 获取并显示文件大小信息
    long inputSize = getFileSize(inputFile);
    long outputSize = getFileSize(outputFile);

    if (inputSize < 0) {
        std::cout << "错误: 无法读取输入文件 " << inputFile << std::endl;
        return false;
    }
    if (outputSize < 0) {
        std::cout << "错误: 无法读取输出文件 " << outputFile << std::endl;
        return false;
    }

    printFileInfo("输入文件", inputSize);
    printFileInfo("输出文件", outputSize);

    // 比较文件内容
    bool filesMatch = compareFiles(inputFile, outputFile);

    // 输出比较结果
    printComparisonResult(filesMatch);

    return filesMatch;
}

bool FileComparator::quickCompare(const std::string& file1, const std::string& file2)
{
    std::ifstream f1(file1.c_str(), std::ios::binary);
    std::ifstream f2(file2.c_str(), std::ios::binary);

    if (!f1.is_open() || !f2.is_open()) {
        return false;
    }

    char c1, c2;
    while (f1.get(c1) && f2.get(c2)) {
        if (c1 != c2) {
            return false;
        }
    }

    // 检查是否都到达文件末尾
    return (!f1.get(c1) && !f2.get(c2));
}

void FileComparator::printFileInfo(const std::string& filename, long size)
{
    std::cout << filename << "大小: " << size << " 字节";

    // 提供更友好的大小显示
    if (size >= 1024 * 1024) {
        std::cout << " (" << (size / (1024.0 * 1024.0)) << " MB)";
    }
    else if (size >= 1024) {
        std::cout << " (" << (size / 1024.0) << " KB)";
    }
    std::cout << std::endl;
}

void FileComparator::printComparisonResult(bool success)
{
    if (success) {
        std::cout << "\n传输成功! 输入文件和输出文件完全一致" << std::endl;
        std::cout << "协议工作正常，数据传输无误" << std::endl;
    }
    else {
        std::cout << "\n传输失败! 输入文件和输出文件不一致" << std::endl;
        std::cout << "请检查协议实现或网络环境设置" << std::endl;
    }
}
