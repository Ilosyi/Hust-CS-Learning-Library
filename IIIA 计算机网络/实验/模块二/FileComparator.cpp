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

    // ����ļ��Ƿ�ɹ���
    if (!f1.is_open()) {
        std::cout << "����: �޷����ļ� " << file1 << std::endl;
        return false;
    }
    if (!f2.is_open()) {
        std::cout << "����: �޷����ļ� " << file2 << std::endl;
        return false;
    }

    // ���ֽڱȽ��ļ�����
    char c1, c2;
    bool identical = true;
    int position = 0;

    while (f1.get(c1) && f2.get(c2)) {
        if (c1 != c2) {
            identical = false;
            std::cout << "�ļ���λ�� " << position << " ����ʼ��ͬ" << std::endl;
            std::cout << "�ļ�1�ַ�: '" << c1 << "' (ASCII: " << (int)c1 << ")" << std::endl;
            std::cout << "�ļ�2�ַ�: '" << c2 << "' (ASCII: " << (int)c2 << ")" << std::endl;
            break;
        }
        position++;
    }

    // ����ļ������Ƿ���ͬ
    if (identical) {
        // ����Ƿ���ʣ������
        if (f1.get(c1)) {
            identical = false;
            std::cout << "�ļ�1���ļ�2������λ�� " << position << " ��������" << std::endl;
        }
        else if (f2.get(c2)) {
            identical = false;
            std::cout << "�ļ�2���ļ�1������λ�� " << position << " ��������" << std::endl;
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
    std::cout << "\n=== �ļ���������֤ ===" << std::endl;

    // ��ȡ����ʾ�ļ���С��Ϣ
    long inputSize = getFileSize(inputFile);
    long outputSize = getFileSize(outputFile);

    if (inputSize < 0) {
        std::cout << "����: �޷���ȡ�����ļ� " << inputFile << std::endl;
        return false;
    }
    if (outputSize < 0) {
        std::cout << "����: �޷���ȡ����ļ� " << outputFile << std::endl;
        return false;
    }

    printFileInfo("�����ļ�", inputSize);
    printFileInfo("����ļ�", outputSize);

    // �Ƚ��ļ�����
    bool filesMatch = compareFiles(inputFile, outputFile);

    // ����ȽϽ��
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

    // ����Ƿ񶼵����ļ�ĩβ
    return (!f1.get(c1) && !f2.get(c2));
}

void FileComparator::printFileInfo(const std::string& filename, long size)
{
    std::cout << filename << "��С: " << size << " �ֽ�";

    // �ṩ���ѺõĴ�С��ʾ
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
        std::cout << "\n����ɹ�! �����ļ�������ļ���ȫһ��" << std::endl;
        std::cout << "Э�鹤�����������ݴ�������" << std::endl;
    }
    else {
        std::cout << "\n����ʧ��! �����ļ�������ļ���һ��" << std::endl;
        std::cout << "����Э��ʵ�ֻ����绷������" << std::endl;
    }
}
