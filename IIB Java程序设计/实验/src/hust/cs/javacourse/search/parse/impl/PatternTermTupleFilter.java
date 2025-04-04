package hust.cs.javacourse.search.parse.impl;

import hust.cs.javacourse.search.index.AbstractTermTuple;
import hust.cs.javacourse.search.parse.AbstractTermTupleFilter;
import hust.cs.javacourse.search.parse.AbstractTermTupleStream;
import hust.cs.javacourse.search.util.Config;

/**
 * 基于正则表达式的三元组过滤器
 */
public class PatternTermTupleFilter extends AbstractTermTupleFilter {

    /**
     * 构造函数
     * @param input：Filter的输入，类型为AbstractTermTupleStream
     */
    public PatternTermTupleFilter(AbstractTermTupleStream input) {
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
            if (tuple.term.getContent().matches(Config.TERM_FILTER_PATTERN)) {
                return tuple; // 如果单词符合正则表达式，返回三元组
            }
        }
        return null; // 如果流结束，返回 null
    }
}
