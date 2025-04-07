package homework.ch11_13.p1;

/**
 * 任务服务接口
 * @author crackryan
 */
public interface TaskService {
    /**
     * 执行任务接口列表中的每个任务
     */
    public void exeuteTasks();
    /**
     * 添加任务
     * @param t 新添加的任务
     */
    public void addTask(Task t);
}
