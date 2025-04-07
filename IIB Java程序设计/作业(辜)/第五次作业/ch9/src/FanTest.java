public class FanTest {
    public static void main(String[] args) {
        // 创建一个Fan对象，具有最大速度、半径为10、黄色、打开状态的属性
        Fan fan = new Fan(Fan.FAST, true, 10.0, "yellow");
        Fan fan2 = new Fan(Fan.MEDIUM, false, 5.0, "blue");
        System.out.println(fan.toString());
        System.out.println(fan2.toString());
    }
}
