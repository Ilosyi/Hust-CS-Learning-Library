package hust.cs.javacourse.search.parse.impl;

import hust.cs.javacourse.search.index.AbstractTermTuple;
import hust.cs.javacourse.search.parse.AbstractTermTupleFilter;
import hust.cs.javacourse.search.parse.AbstractTermTupleStream;

import java.util.Arrays;

import static hust.cs.javacourse.search.util.StopWords.STOP_WORDS;

public class StopWordTermTupleFilter extends AbstractTermTupleFilter {
    /**
     * 构造函数
     * @param input：Filter的输入，类型为AbstractTermTupleStream
     */
    public StopWordTermTupleFilter(AbstractTermTupleStream input) {
        super(input);
    }

    /**
     * 获得下一个三元组
     * @return: 下一个三元组；如果到了流的末尾，返回null
     */
    @Override
    public AbstractTermTuple next() {
        AbstractTermTuple tuple;
        while ((tuple = input.next()) != null) {
            if (!Arrays.asList(STOP_WORDS).contains(tuple.term.getContent())) {
                return tuple; // 如果不是停用词，返回三元组
            }
        }
        return null; // 如果流结束，返回 null
    }
}
