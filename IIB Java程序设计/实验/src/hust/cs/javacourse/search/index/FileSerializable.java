package hust.cs.javacourse.search.index;

import java.io.*;

/**
 * 定义文件序列化接口
 */
public interface FileSerializable extends java.io.Serializable{
    /**
     * 写到二进制文件
     * @param out :输出流对象
     */
    public abstract void writeObject(ObjectOutputStream out) throws IOException;

    /**
     * 从二进制文件读
     * @param in ：输入流对象
     */
    public  abstract void readObject(ObjectInputStream in) throws IOException;
}
