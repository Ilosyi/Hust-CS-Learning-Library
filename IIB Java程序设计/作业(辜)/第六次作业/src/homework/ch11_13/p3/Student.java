package homework.ch11_13.p3;

import java.util.Objects;

public class Student extends Person implements Cloneable {
    private int studentId;
    private String department;
    private String classNo;

    public Student() {
        super();
        this.studentId = 0;
        this.department = "Unknown";
        this.classNo = "Unknown";
    }
    public Student(String name, int age, int studentId, String department, String classNo) {
        super(name, age);
        this.studentId = studentId;
        this.department = department;
        this.classNo = classNo;
    }
    public int getStudentId() {
        return studentId;
    }
    public void setStudentId(int studentId) {
        this.studentId = studentId;
    }
    public String getDepartment() {
        return department;
    }
    public void setDepartment(String department) {
        this.department = department;
    }
    public String getClassNo() {
        return classNo;
    }
    public void setClassNo(String classNo) {
        this.classNo = classNo;
    }
    @Override
    public String toString() {
        return "Student{" +
                "studentId=" + studentId +
                ", department='" + department + '\'' +
                ", classNo='" + classNo + '\'' +
                "} " + '\n'+super.toString();
    }
    @Override
    public boolean equals(Object obj) {
        //复用父类的equals方法
        if (this == obj) return true;
        if (!super.equals(obj)) return false;
        if (getClass() != obj.getClass()) return false;
        Student student = (Student) obj;
        return studentId == student.studentId && Objects.equals(department, student.department) && Objects.equals(classNo, student.classNo);
    }
    @Override
    public Object clone() throws CloneNotSupportedException{
        Student s= (Student) super.clone();
        s.studentId=this.studentId;
        s.department=new String(this.department);
        s.classNo=new String(this.classNo);
        return s;
    }
}
