package hust.cs.javacourse.search.index.impl;

import hust.cs.javacourse.search.index.AbstractDocument;
import hust.cs.javacourse.search.index.AbstractDocumentBuilder;
import hust.cs.javacourse.search.index.AbstractTermTuple;
import hust.cs.javacourse.search.parse.AbstractTermTupleStream;
import hust.cs.javacourse.search.parse.impl.LengthTermTupleFilter;
import hust.cs.javacourse.search.parse.impl.PatternTermTupleFilter;
import hust.cs.javacourse.search.parse.impl.StopWordTermTupleFilter;
import hust.cs.javacourse.search.parse.impl.TermTupleScanner;

import java.io.*;

/**
 * AbstractDocumentBuilder的具体实现类
 */
public class DocumentBuilder extends AbstractDocumentBuilder {

    /**
     * 缺省构造函数
     */
    public DocumentBuilder() {
    }

    /**
     * 由解析文本文档得到的TermTupleStream,构造Document对象.
     * @param docId             : 文档id
     * @param docPath           : 文档绝对路径
     * @param termTupleStream   : 文档对应的TermTupleStream
     * @return ：Document对象
     */
    @Override
    public AbstractDocument build(int docId, String docPath, AbstractTermTupleStream termTupleStream) {
        AbstractDocument document = new Document(docId, docPath);
        AbstractTermTuple tuple;
        while ((tuple = termTupleStream.next()) != null) {
            document.getTuples().add(tuple); // 将三元组添加到文档中
        }
        return document;
    }

    /**
     * 由给定的File,构造Document对象.
     * 该方法利用输入参数file构造出AbstractTermTupleStream子类对象后,内部调用
     *      AbstractDocument build(int docId, String docPath, AbstractTermTupleStream termTupleStream)
     * @param docId     : 文档id
     * @param docPath   : 文档绝对路径
     * @param file      : 文档对应File对象
     * @return          : Document对象
     */
    @Override
    public AbstractDocument build(int docId, String docPath, File file) {
        AbstractTermTupleStream ts;
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            // 创建 TermTupleScanner
            // 添加过滤器
            ts = new LengthTermTupleFilter(new PatternTermTupleFilter
                    (new StopWordTermTupleFilter
                            (new TermTupleScanner(reader))));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e); // 如果文件未找到，抛出运行时异常
        }
        return this.build(docId, docPath, ts);
    }
}
