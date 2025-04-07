package homework.ch11_13.p3;

import java.util.Objects;

public class Person implements Cloneable {
    private String name;
    private int age;

    public Person() {
        this.name = "Unknown";
        this.age = 0;
    }
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + "}";
    }

    @Override
    public boolean equals(Object obj) {
        // 先判断是否是同一个对象
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Person person = (Person) obj;
        return age == person.age && Objects.equals(name, person.name);
    }

    @Override
    public Object clone() throws CloneNotSupportedException{
        Person p= (Person) super.clone();
        p.name=new String(this.name);
        p.age=this.age;
        return p;
    }
}
