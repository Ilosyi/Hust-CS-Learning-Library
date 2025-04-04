package hust.cs.javacourse.search.parse.impl;

import hust.cs.javacourse.search.index.AbstractTermTuple;
import hust.cs.javacourse.search.index.impl.Term;
import hust.cs.javacourse.search.index.impl.TermTuple;
import hust.cs.javacourse.search.parse.AbstractTermTupleScanner;
import hust.cs.javacourse.search.util.Config;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

public class TermTupleScanner extends AbstractTermTupleScanner {
    private final Deque<String> buffer = new ArrayDeque<>(); // 缓冲区，用于存储分割后的单词
    private int curPos = 0; // 当前位置计数器

    /**
     * 缺省构造函数
     */
    public TermTupleScanner() {
        super();
    }

    /**
     * 构造函数
     * @param input：指定输入流对象，应该关联到一个文本文件
     */
    public TermTupleScanner(BufferedReader input) {
        super(input);
    }

    /**
     * 获得下一个三元组
     * @return: 下一个三元组；如果到了流的末尾，返回null
     */
    @Override
    public AbstractTermTuple next() {
        // 如果缓冲区为空，读取新行并分割
        while (buffer.isEmpty()) {
            String line;
            try {
                line = input.readLine(); // 读取一行
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
            if (line == null) {
                return null; // 如果流结束，返回 null
            }
            if (line.isBlank()) {
                continue; // 如果行是空行或空白行，跳过
            }
            // 将行按正则表达式分割，并根据配置决定是否转换为小写
            String[] parts = line.trim().split(Config.STRING_SPLITTER_REGEX);
            //trim()去掉首尾空格
            //split()方法根据正则表达式分割字符串
            if (Config.IGNORE_CASE) {
                for (String part : parts) {
                    buffer.add(part.toLowerCase()); // 忽略大小写，转换为小写
                }
            } else {
                buffer.addAll(Arrays.asList(parts)); // 不忽略大小写，直接添加
            }
        }

        // 从缓冲区取出第一个单词，生成三元组
        String termContent = buffer.removeFirst();
        //removeFirst()方法从缓冲区中删除并返回第一个元素
        return new TermTuple(new Term(termContent), curPos++);
    }
}
