package homework.ch11_13.p4;

public interface ComponentIterator{
    /**
     *是否还有元素
     * @return -如果元素还没有迭代完，返回true;否则返回false
     */
    boolean hasNext();
    /**
     * 获取下一个组件
     * @return -下一个组件
     */
    Component next();

}
