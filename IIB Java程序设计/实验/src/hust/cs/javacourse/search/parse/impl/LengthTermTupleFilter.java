package hust.cs.javacourse.search.parse.impl;

import hust.cs.javacourse.search.index.AbstractTermTuple;
import hust.cs.javacourse.search.parse.AbstractTermTupleFilter;
import hust.cs.javacourse.search.parse.AbstractTermTupleStream;
import hust.cs.javacourse.search.util.Config;

/**
 * 基于单词长度的三元组过滤器
 */
public class LengthTermTupleFilter extends AbstractTermTupleFilter {

    /**
     * 构造函数
     * @param input：Filter的输入，类型为AbstractTermTupleStream
     */
    public LengthTermTupleFilter(AbstractTermTupleStream input) {
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
            int length = tuple.term.getContent().length();
            if (length >= Config.TERM_FILTER_MINLENGTH && length <= Config.TERM_FILTER_MAXLENGTH) {
                return tuple; // 如果单词长度在范围内，返回三元组
            }
        }
        return null; // 如果流结束，返回 null
    }
}
