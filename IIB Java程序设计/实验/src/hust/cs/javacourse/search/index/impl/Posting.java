package hust.cs.javacourse.search.index.impl;

import hust.cs.javacourse.search.index.AbstractPosting;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.StringJoiner;

public class Posting extends AbstractPosting {

    /**
     * 缺省构造函数
     */
    public Posting() {
        super();
    }

    /**
     * 构造函数
     * @param docId ：包含单词的文档id
     * @param freq  ：单词在文档里出现的次数
     * @param positions   ：单词在文档里出现的位置
     */
    public Posting(int docId, int freq, List<Integer> positions) {
        super(docId, freq, positions);
    }

    /**
     * 判断二个Posting内容是否相同
     * @param obj ：要比较的另外一个Posting
     * @return 如果内容相等返回true，否则返回false
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true; // 如果是同一个对象，直接返回 true
        if (obj == null || getClass() != obj.getClass()) return false; // 如果对象为 null 或类型不同，返回 false
        Posting posting = (Posting) obj;

        // 比较 docId 和 freq
        if (this.docId != posting.docId || this.freq != posting.freq) return false;

        // 如果 positions 列表大小不同，直接返回 false
        if (this.positions.size() != posting.positions.size()) return false;

        // 确保 positions 列表已经排序
        this.sort();
        posting.sort();

        // 遍历比较 positions 列表的每个元素
        for (int i = 0; i < this.positions.size(); i++) {
            if (!this.positions.get(i).equals(posting.positions.get(i))) {
                return false;
            }
        }

        return true;
    }

    /**
     * 返回Posting的字符串表示
     * @return 字符串
     */
    @Override
    public String toString() {
        StringJoiner sj = new StringJoiner(", ", "{", "}");
        sj.add("\"docID\": " + this.docId);
        sj.add("\"freq\": " + this.freq);
        StringJoiner positionsJoiner = new StringJoiner(", ", "\"positions\": [", "]");
        for (int pos : this.positions) {
            positionsJoiner.add(String.valueOf(pos));
        }
        sj.add(positionsJoiner.toString());
        return sj.toString();
    }

    /**
     * 返回包含单词的文档id
     * @return ：文档id
     */
    @Override
    public int getDocId() {
        return docId;
    }

    /**
     * 设置包含单词的文档id
     * @param docId：包含单词的文档id
     */
    @Override
    public void setDocId(int docId) {
        this.docId = docId;
    }

    /**
     * 返回单词在文档里出现的次数
     * @return ：出现次数
     */
    @Override
    public int getFreq() {
        return freq;
    }

    /**
     * 设置单词在文档里出现的次数
     * @param freq:单词在文档里出现的次数
     */
    @Override
    public void setFreq(int freq) {
        this.freq = freq;
    }

    /**
     * 返回单词在文档里出现的位置列表
     * @return ：位置列表
     */
    @Override
    public List<Integer> getPositions() {
        return positions;
    }

    /**
     * 设置单词在文档里出现的位置列表
     * @param positions：单词在文档里出现的位置列表
     */
    @Override
    public void setPositions(List<Integer> positions) {
        this.positions = positions;
    }

    /**
     * 比较二个Posting对象的大小（根据docId）
     * @param o： 另一个Posting对象
     * @return ：二个Posting对象的docId的差值
     */
    @Override
    public int compareTo(AbstractPosting o) {
        return this.docId - o.getDocId();
    }

    /**
     * 对内部positions从小到大排序
     */
    @Override
    public void sort() {
        Collections.sort(positions);
    }

    /**
     * 将Posting对象序列化到二进制文件中
     * @param out ：输出流
     * @throws IOException ：如果发生I/O错误
     */
    @Override
    public void writeObject(ObjectOutputStream out) throws IOException {
        out.writeInt(docId);
        out.writeInt(freq);
        out.writeObject(positions);
    }

    /**
     * 从二进制文件中反序列化Posting对象
     * @param in ：输入流
     * @throws IOException ：如果发生I/O错误
     * @throws ClassNotFoundException ：如果类未找到
     */
    @Override
    public void readObject(ObjectInputStream in) throws IOException {
        this.docId = in.readInt();
        this.freq = in.readInt();
        try{
            this.positions = (List<Integer>) in.readObject();
        }catch(ClassNotFoundException e){}

    }
}
