package homework.ch11_13.p3;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Course implements Cloneable {
    private String courseName;
    private List<Person> students;
    private Person teacher;


    public Course(String courseName, Person teacher) {
        this.courseName = courseName;
        this.teacher = teacher;
        this.students = new ArrayList<>();
    }

    public String getCourseName() {
        return courseName;
    }

    public Person getTeacher() {
        return teacher;
    }

    public List<Person> getStudents() {
        return students;
    }

    public void register(Person student) {
        if (!students.contains(student)) {
            students.add(student);
        }
    }

    public void unregister(Person student) {
        students.remove(student);

    }

    public int getNumberOfStudent() {
        return students.size();
    }

    @Override
    public String toString() {
        //描述Course对象信息的字符串(应该包括课程名称、教师的详细信息，每个学生的详细信息，学生总数)
        //Course类实现toString、equals、clone方法时应该调用每个数据成员的相应方法以实现toString、内容比较和深拷贝克隆
        StringBuilder sb = new StringBuilder();
        sb.append("Course Name: ").append(courseName).append("\n");
        sb.append(teacher.toString()).append("\n");
        sb.append("Students: \n");
        for (Person student : students) {
            sb.append(student.toString()).append("\n");
        }
        sb.append("Total Students: ").append(students.size()).append("\n");
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
      //注意复用数据成员相应的equals
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Course course = (Course) obj;
        if(!Objects.equals(courseName, course.courseName)) return false;
        if(!teacher.equals(course.teacher)) return false;
        if(students.size() != course.students.size()) return false;
        //比较学生列表,顺序不重要,利用ArrayList的containsAll方法
        if(!students.containsAll(course.students)) return false;
        return true;
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        Course cloned = (Course) super.clone();
        cloned.teacher = (Faculty) teacher.clone();
        cloned.courseName = new String(this.courseName);
        cloned.students = new ArrayList<>();
        for (Person student : students) {
            Student p= (Student) student.clone();
            cloned.students.add(p);
        }
        return cloned;
    }
}
