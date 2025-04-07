package homework.ch11_13.p1;

import homework.ch11_13.p1.impl.Task1;
import homework.ch11_13.p1.impl.Task2;
import homework.ch11_13.p1.impl.Task3;
import homework.ch11_13.p1.impl.TaskServiceImpl;

public class Test {
    public static void main(String[] args) {
        // 创建任务服务
        TaskServiceImpl taskService = new TaskServiceImpl();

        // 添加任务
        taskService.addTask(new Task1());
        taskService.addTask(new Task2());
        taskService.addTask(new Task3());

        // 执行任务
        taskService.exeuteTasks();
    }
}
