package homework.ch11_13.p4;

public class Test {
    public static void main(String[] args) {
        // 1）创建一台计算机对象（组件树）
        Component computer = ComponentFactory.create();

        // 2）打印 toString()，查看组合结构
        System.out.println(computer);

        // 3）利用迭代器遍历所有组件（不打印结构，只打印每个节点信息）
        System.out.println("遍历所有组件（不含子组件结构）：");
        System.out.println("id: " + computer.getId() + ", name: " +
                computer.getName() + ", price: " + computer.getPrice());

        ComponentIterator it = computer.createIterator(); // 使用自定义迭代器
        while (it.hasNext()) {

            Component c = it.next();
            // 注意不要用 c.toString()，否则会递归打印整个子树
            System.out.println("id: " + c.getId() + ", name: " +
                    c.getName() + ", price: " + c.getPrice());
        }

        // 4）打印整台计算机的总价格（由内部递归计算）
        System.out.println("整台计算机总价格为: " + computer.calcPrice());
    }
}
