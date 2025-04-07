package homework.ch11_13.p4;

import java.util.Iterator;

public class AtomicComponent extends Component {


    public AtomicComponent() {
        super();
    }

    public AtomicComponent(int id, String name, double price) {
        super(id, name, price);
    }

    /**
     * 添加子组件，对于没有子组件的AtomicComponent如内存条，
     * 调用这个方法应该抛出UnsupportedOperationException.
     * 相同的子组件不能重复加入
     *
     * @param component-要添加的组件
     */
    @Override
    public void add(Component component)
            throws UnsupportedOperationException {
        throw new UnsupportedOperationException("Atomic components cannot have children.");
    }

    /**
     * 删除子组件，对于没有子组件的AtomicComponent如内存条，
     * 调用这个方法应该抛出UnsupportedOperationException.
     *
     * @param component-要删除的组件
     */
    @Override
    public void remove(Component component)
            throws UnsupportedOperationException {
        throw new UnsupportedOperationException("Atomic components cannot have children.");
    }
    /**
     * 计算组件的价格。对于复合组件应该计算其子组件的价格之和
     *
     * @return 组件的价格
     */
    @Override
    public double calcPrice() {
        return price;
    }
    /**
     * 返回组件的迭代器，只需要返回一个NullIterator对象即可。
     *
     * @return 迭代器
     */
    @Override
    public ComponentIterator createIterator() {
        return new NullIterator();
    }

}
