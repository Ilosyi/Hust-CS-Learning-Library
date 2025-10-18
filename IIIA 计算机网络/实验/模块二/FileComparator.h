#pragma once
#ifndef FILE_COMPARATOR_H
#define FILE_COMPARATOR_H

#include <string>
#include <iostream>

class FileComparator
{
public:
    // ���캯��
    FileComparator();

    // ��������
    ~FileComparator();

    // �Ƚ������ļ��Ƿ���ͬ
    bool compareFiles(const std::string& file1, const std::string& file2);

    // ��ȡ�ļ���С
    long getFileSize(const std::string& filename);

    // ִ���������ļ�������֤�����������Ϣ��
    bool validateTransmission(const std::string& inputFile, const std::string& outputFile);

    // ��̬���������ٱȽ��ļ����������ϸ��Ϣ��
    static bool quickCompare(const std::string& file1, const std::string& file2);

private:
    // ˽�з���������ļ���Ϣ
    void printFileInfo(const std::string& filename, long size);

    // ˽�з���������ȽϽ��
    void printComparisonResult(bool success);
};

#endif
