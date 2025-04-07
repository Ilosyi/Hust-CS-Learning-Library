package homework.ch11_13.p4;

import java.util.Iterator;

public abstract class Component {
    protected int id;
    protected String name;
    protected double price;

    public Component() {
        this.id = 0;
        this.name = "Unknown";
        this.price = 0.0;
    }
    public Component(int id, String name, double price) {
        this.id = id;
        this.name = name;
        this.price = price;
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public double getPrice() {
        return calcPrice();
    }
    public void setId(int id) {
        this.id = id;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    /**
     *添加子组件，对于没有子组件的AtomicComponent如内存条，
     * 调用这个方法应该抛出UnsupportedOperationException.
     * 相同的子组件不能重复加入
     * @param component-要删除的组件
     */
    public abstract void add(Component component)throws UnsupportedOperationException;
    /**
     * 删除子组件，对于没有子组件的AtomicComponent如内存条，
     * 调用这个方法应该抛出UnsupportedOperationException.
     * @param component-要删除的组件
     */
    public abstract void remove(Component component)throws UnsupportedOperationException;

    /**
     * 计算组件的价格。对于复合组件应该计算其子组件的价格之和
     * @return 组件的价格
     */
    public abstract double calcPrice();
    /**
     * 返回组件的迭代器
     * @return 组件的迭代器
     */
    public abstract ComponentIterator createIterator();
    /**
     * 基于组件id判断二个组件对象是否相等
     * @param obj-要比较的对象
     * @return true-相等，false-不相等
     */
    @Override
    public boolean equals(Object obj)
    {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Component component = (Component) obj;
        return id == component.id;
    }

    /**
     * 返回组件的信息
     * @return 组件的信息
     */
    @Override
    public String toString()
    {
        return "Component{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", price=" + price +
                '}';
    }

}
