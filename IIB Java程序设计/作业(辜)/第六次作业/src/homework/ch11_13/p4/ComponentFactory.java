package homework.ch11_13.p4;

/**
 * ComponentFactory 是一个组件对象工厂类，
 * 用于封装复杂组件树（如一台计算机）的创建逻辑
 */
public class ComponentFactory {

    /**
     * 缺省构造函数
     */
    public ComponentFactory() {}

    /**
     * 创建一个复杂的 Component 结构，代表一台计算机
     * 使用组合模式构建：计算机 -> 主板/内存/硬盘/... -> 子模块
     *
     * @return Component 代表整台计算机的组件对象（复合组件）
     */
    public static Component create() {
        int id = 0;
        //创建计算对象
        Component computer = new CompositeComponent(id++, "Think Pad", 0.0);

        //创建键盘对象
        Component keyboard = new AtomicComponent(id++, "Keyboard", 20.0);
        //创建鼠标对象
        Component mouse = new AtomicComponent(id++, "Mouse", 20.0);
        //创建显示器对象
        Component monitor = new AtomicComponent(id++, "Monitor", 1000.0);
        computer.add(keyboard);     //键盘加入computer
        computer.add(mouse);        //鼠标加入computer
        computer.add(monitor);      //显示器加入computer

        //创建主机对象
        Component mainFrame= new CompositeComponent(id++, "Main frame", 0.0);
        //创建硬盘对象
        Component hardDisk = new AtomicComponent(id++, "Hard disk",1000);
        //创建电源对象
        Component powerSupplier = new AtomicComponent(id++, "Power supplier",500);
        mainFrame.add(hardDisk);
        mainFrame.add(powerSupplier);

        //创建主板对象
        Component mainBoard = new CompositeComponent(id++, "Main board", 0.0);
        //创建CPU对象
        Component cpu = new AtomicComponent(id++, "CPU", 1500.0);
        //创建显卡对象
        Component videoCard = new AtomicComponent(id++, "Video card", 900);
        //创建网卡对象
        Component networkCard = new AtomicComponent(id++, "Network card", 100);
        mainBoard.add(cpu);         //cpu加入主板
        mainBoard.add(videoCard);   //videoCard加入主板
        mainBoard.add(networkCard); //networkCard加入主板

        mainFrame.add(mainBoard);   //mainBoard加入主机
        computer.add(mainFrame);    //将主机加入computer
        return computer;
    }
}
