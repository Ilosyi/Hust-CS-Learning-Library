package homework.ch11_13.p4;

/**
 * CompositeComponent 表示一个复合组件，可以包含若干子组件（包括复合组件和原子组件）
 */
public class CompositeComponent extends Component {

    // 用于保存所有子组件的列表（使用支持迭代的 ComponentList）
    protected ComponentList children;

    /**
     * 缺省构造函数
     */
    public CompositeComponent() {
        children = new ComponentList();
    }

    /**
     * 构造函数，初始化 id、name、price 和子组件列表
     */
    public CompositeComponent(int id, String name, double price) {
        super(id, name, price);
        children = new ComponentList();
    }

    /**
     * 向当前组件中添加一个子组件
     * 注意：相同子组件不能重复加入
     *
     * @param component 要添加的子组件
     * @throws UnsupportedOperationException 如果操作不被支持（在原子组件中会抛）
     */
    @Override
    public void add(Component component) throws UnsupportedOperationException {
        if (children.contains(component)) {
            return; // 避免重复添加
        }
        children.add(component);
    }

    /**
     * 从当前组件中删除一个子组件
     *
     * @param component 要删除的组件
     * @throws UnsupportedOperationException 如果操作不被支持（在原子组件中会抛）
     */
    @Override
    public void remove(Component component) throws UnsupportedOperationException {
        children.remove(component);
    }

    /**
     * 计算当前组件的价格
     * 对于复合组件，其价格 = 所有子组件价格之和
     */
    @Override
    public double calcPrice() {
        double total = 0;
        for (Component c : children) {
            total += c.calcPrice(); // 递归计算子组件价格
        }
        this.price = total; // 更新当前组件的价格
        return total;
    }

    /**
     * 创建当前组件的迭代器（返回一个 ComponentList 的迭代器）
     * 用于支持 CompositeIterator 的深度优先遍历
     */
    @Override
    public ComponentIterator createIterator() {
        return new CompositeIterator(children.createIterator());
    }

    /**
     * 返回当前组件的描述字符串
     * 格式: id-name:price
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString()); // 调用父类的 toString() 方法

        // 如果有子组件，递归调用toString打印子组件信息
        if (children != null && !children.isEmpty()) {
            sb.append(" [");
            for (Component child : children) {
                sb.append(child.toString()).append(", ");
            }
            sb.delete(sb.length() - 2, sb.length()); // 删除最后的逗号和空格
            sb.append("]");
        }

        return sb.toString();
    }
}
