#pragma once
#ifndef FILE_COMPARATOR_H
#define FILE_COMPARATOR_H

#include <string>
#include <iostream>

class FileComparator
{
public:
    // 构造函数
    FileComparator();

    // 析构函数
    ~FileComparator();

    // 比较两个文件是否相同
    bool compareFiles(const std::string& file1, const std::string& file2);

    // 获取文件大小
    long getFileSize(const std::string& filename);

    // 执行完整的文件传输验证（包含输出信息）
    bool validateTransmission(const std::string& inputFile, const std::string& outputFile);

    // 静态方法：快速比较文件（不输出详细信息）
    static bool quickCompare(const std::string& file1, const std::string& file2);

private:
    // 私有方法：输出文件信息
    void printFileInfo(const std::string& filename, long size);

    // 私有方法：输出比较结果
    void printComparisonResult(bool success);
};

#endif
