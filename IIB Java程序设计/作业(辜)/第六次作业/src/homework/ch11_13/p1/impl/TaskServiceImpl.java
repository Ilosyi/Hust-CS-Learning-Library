package homework.ch11_13.p1.impl;

import homework.ch11_13.p1.Task;
import homework.ch11_13.p1.TaskService;

import java.util.ArrayList;

public class TaskServiceImpl implements TaskService {
    private final ArrayList<Task> tasks;

    public TaskServiceImpl() {
        tasks = new ArrayList<>();
    }

    @Override
    public void exeuteTasks() {
        // 遍历任务列表，执行每个任务
        for (Task task : tasks) {
            task.execute();
        }
    }

    @Override
    public void addTask(Task t) {
        tasks.add(t);
    }
}
