package hust.cs.javacourse.search.index.impl;

import hust.cs.javacourse.search.index.AbstractPosting;
import hust.cs.javacourse.search.index.AbstractPostingList;
import hust.cs.javacourse.search.index.FileSerializable;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class PostingList extends AbstractPostingList {

    /**
     * 添加Posting,要求不能有内容重复的posting
     * @param posting：Posting对象
     */
    /**contains方法的核心是 AbstractPosting 类的 equals 方法**/
    @Override
    public void add(AbstractPosting posting) {
        if (!this.list.contains(posting)) {
            this.list.add(posting);
        }
    }

    /**
     * 获得PosingList的字符串表示
     * @return ： PosingList的字符串表示
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("PostingList [\n");
        for (AbstractPosting posting : this.list) {
            sb.append("    ").append(posting.toString()).append("\n");
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * 添加Posting列表,,要求不能有内容重复的posting
     * @param postings：Posting列表
     */
    @Override
    public void add(List<AbstractPosting> postings) {
        for (AbstractPosting posting : postings) {
            this.add(posting);
        }
    }

    /**
     * 返回指定下标位置的Posting
     * @param index ：下标
     * @return： 指定下标位置的Posting
     */
    @Override
    public AbstractPosting get(int index) {
        return this.list.get(index);
    }

    /**
     * 返回指定Posting对象的下标
     * @param posting：指定的Posting对象
     * @return ：如果找到返回对应下标；否则返回-1
     */
    @Override
    public int indexOf(AbstractPosting posting) {
        return this.list.indexOf(posting);
    }

    /**
     * 返回指定文档id的Posting对象的下标
     * @param docId ：文档id
     * @return ：如果找到返回对应下标；否则返回-1
     */
    @Override
    public int indexOf(int docId) {
        for (int i = 0; i < this.list.size(); i++) {
            if (this.list.get(i).getDocId() == docId) {
                return i;
            }
        }
        return -1;
    }

    /**
     * 是否包含指定Posting对象
     * @param posting： 指定的Posting对象
     * @return : 如果包含返回true，否则返回false
     */
    @Override
    public boolean contains(AbstractPosting posting) {
        return this.list.contains(posting);
    }

    /**
     * 删除指定下标的Posting对象
     * @param index：指定的下标
     */
    @Override
    public void remove(int index) {
        this.list.remove(index);
    }

    /**
     * 删除指定的Posting对象
     * @param posting ：定的Posting对象
     */
    @Override
    public void remove(AbstractPosting posting) {
        this.list.remove(posting);
    }

    /**
     * 返回PostingList的大小，即包含的Posting的个数
     * @return ：PostingList的大小
     */
    @Override
    public int size() {
        return this.list.size();
    }

    /**
     * 清除PostingList
     */
    @Override
    public void clear() {
        this.list.clear();
    }

    /**
     * PostingList是否为空
     * @return 为空返回true;否则返回false
     */
    @Override
    public boolean isEmpty() {
        return this.list.isEmpty();
    }

    /**
     * 根据文档id的大小对PostingList进行从小到大的排序
     */
    @Override
    public void sort() {
        this.list.sort(new Comparator<AbstractPosting>() {
            @Override
            public int compare(AbstractPosting o1, AbstractPosting o2) {
                return o1.getDocId() - o2.getDocId();
            }
        });
    }

    /**
     * 将PostingList序列化到二进制文件中
     * @param out ：输出流
     * @throws IOException ：如果发生I/O错误
     */
    @Override
    public void writeObject(ObjectOutputStream out) throws IOException {
        out.writeInt(this.list.size());
        for (AbstractPosting posting : this.list) {
            posting.writeObject(out);
        }
    }

    /**
     * 从二进制文件中反序列化PostingList
     * @param in ：输入流
     * @throws IOException ：如果发生I/O错误
     */
    @Override
    public void readObject(ObjectInputStream in) throws IOException {
        int size = in.readInt();
        this.list.clear();
        for (int i = 0; i < size; i++) {
            Posting posting = new Posting();
            posting.readObject(in);
            this.list.add(posting);
        }
    }
}
