public class Fan {
    // 1.三个整型常量SLOW、MEDIUM和FAST表示风扇的速度，分别为1、2和3
    public static final int SLOW = 1;
    public static final int MEDIUM = 2;
    public static final int FAST = 3;

    //2.一个名为speed的int型私有数据域，表示风扇的速度（默认值为SLOW）
    private int speed;
    //3.一个名为on的boolean型私有数据域，表示风扇是否打开（默认值为false）
    private boolean on;
    //4.一个名为radius的double型私有数据域，表示风扇的半径（默认值为5）
    private double radius;
    //5.一个名为color的String型私有数据域，表示风扇的颜色（默认值为blue）
    private String color;

    // 7.无参构造函数
    public Fan() {
        this.speed = SLOW;
        this.on = false;
        this.radius = 5.0;
        this.color = "blue";
    }

    // 有参构造函数
    public Fan(int speed, boolean on, double radius, String color) {
        this.speed = speed;
        this.on = on;
        this.radius = radius;
        this.color = color;
    }

    // 6.Getter和Setter方法
    public int getSpeed() {
        return speed;
    }

    public void setSpeed(int speed) {
        this.speed = speed;
    }

    public boolean isOn() {
        return on;
    }

    public void setOn(boolean on) {
        this.on = on;    }

    public double getRadius() {
        return radius;
    }

    public void setRadius(double radius) {
        this.radius = radius;
    }

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    // 8.一个名为toString的方法返回风扇的描述
    @Override
    public String toString() {
        if (on) {
            return "speed=" + speed + ", color=" + color + ", radius=" + radius;
        } else {
            return "Fan is off: color=" + color + ", radius=" + radius;
        }
    }
}
