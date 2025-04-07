package homework.ch11_13.p3;

import java.util.Objects;

public class Faculty extends Person implements Cloneable {
    private int facultyId;
    private String title;

    private String email;

    public Faculty(String name, int age, int facultyId, String title, String email) {
        super(name, age);
        this.facultyId = facultyId;
        this.title = title;
        this.email = email;
    }
    public Faculty() {
        // 初始化默认值
        super();
        this.facultyId = 0;
        this.title = "Unknown";
    }

    public int getFacultyId() {
        return facultyId;
    }

    public void setFacultyId(int facultyId) {
        this.facultyId = facultyId;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    @Override
    public String toString() {
        return "Faculty{" +
                "facultyId=" + facultyId +
                ", title='" + title + '\'' +
                ", email='" + email + '\'' +
                "} " + super.toString();
    }

    @Override
    public boolean equals(Object obj) {
        // 先判断是否是同一个对象
        if (this == obj) return true;
        //复用父类的equals方法
        if (!super.equals(obj)) return false;
        if (getClass() != obj.getClass()) return false;
        Faculty faculty = (Faculty) obj;
        return facultyId == faculty.facultyId && Objects.equals(title, faculty.title) && Objects.equals(email, faculty.email);
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        Faculty f = (Faculty) super.clone();
        f.facultyId = this.facultyId;
        f.title = new String(this.title);
        f.email = new String(this.email);
        return f;
    }
}
