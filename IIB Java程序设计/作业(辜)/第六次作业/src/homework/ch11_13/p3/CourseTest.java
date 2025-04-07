package homework.ch11_13.p3;

public class CourseTest {


    public static void main(String[] args) throws CloneNotSupportedException {

        Person teacher = new Faculty(
                "James Gosling",
                65,
                0000,
                "Professor",
                "http://nighthacks.com/jag/bio/index.html"
    );
        Course javaCourse = new Course("Java Language Programming", teacher);
        Person student1 = new Student(
                "aaa",
                20,
                20170101,
                "CS",
                "CS1704"
    );
        Person student2 = new Student(
                "bbb",
                20,
                20170102,
                "CS",
                "CS1705"
    );
        Person student3 = new Student(
                "ccc",
                20,
                20170103,
                "CS",
                "CS1706"
    );
        javaCourse.register(student1);
        javaCourse.register(student2);
        javaCourse.register(student3);
        System.out.println(javaCourse); // 打印课程详细信息
        javaCourse.unregister(student3);
        System.out.println(javaCourse); // 打印课程详细信息

        // 测试是否为深拷贝
        Course javaCourse2 = (Course) javaCourse.clone();
        System.out.println(javaCourse.equals(javaCourse2)); // 测试对象内容是否相等
        System.out.println(javaCourse.getCourseName() != javaCourse2.getCourseName()); // 测试所有引用类型数据成员是否指向相同对象
        System.out.println(javaCourse.getTeacher() != javaCourse2.getTeacher());
        System.out.println(javaCourse.getStudents() != javaCourse2.getStudents());
        System.out.println(javaCourse2);
    }

}
