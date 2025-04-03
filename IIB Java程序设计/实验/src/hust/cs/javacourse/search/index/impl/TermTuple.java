package hust.cs.javacourse.search.index.impl;

import hust.cs.javacourse.search.index.AbstractTerm;
import hust.cs.javacourse.search.index.AbstractTermTuple;

public class TermTuple extends AbstractTermTuple {

    /**
     * 缺省构造函数
     */
    public TermTuple() {
        super();
    }

    /**
     * 构造函数
     * @param term ：单词
     * @param curPos ：单词出现的当前位置
     */
    public TermTuple(AbstractTerm term, int curPos) {
        this.term = term;
        this.curPos = curPos;
    }

    /**
     * 判断二个三元组内容是否相同
     * @param obj ：要比较的另外一个三元组
     * @return 如果内容相等（三个属性内容都相等）返回true，否则返回false
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true; // 如果是同一个对象，直接返回 true
        if (obj == null || getClass() != obj.getClass()) return false; // 如果对象为 null 或类型不同，返回 false
        TermTuple termTuple = (TermTuple) obj; // 强制转换为 TermTuple 类型
        // 比较 freq
        // 比较 curPos
        return this.curPos == termTuple.curPos && (this.term != null ? this.term.equals(termTuple.term) : termTuple.term == null); // 比较 term
    }

    /**
     * 获得三元组的字符串表示
     * @return ： 三元组的字符串表示
     */
    @Override
    public String toString() {
        return "TermTuple{" +
                "term=" + term.toString() +
                ", freq=" + freq +
                ", curPos=" + curPos +
                '}';
    }
}
