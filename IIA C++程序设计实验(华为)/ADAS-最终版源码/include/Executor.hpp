#pragma once
#include <string>

namespace adas
{
    enum class ExecutorType
    {
        NORMAL,
        SPORTS_CAR,
        BUS,
        // 区分不同类型汽车
    };

    struct Pose // 位置和朝向
    {
        int x;
        int y;
        char heading;
    };
    // Executor 类是一个抽象类，定义了执行指令和查询当前位置的接口
    class Executor
    {
    public:
        // Caller should delete *executor when it is no longer needed.
        // static Executor* NewExecutor(const Pose& pose = {0, 0, 'N'}) noexcept;
        static Executor *NewExecutor(const Pose &pose = {0, 0, 'N'},
        const ExecutorType executorType = ExecutorType::NORMAL) noexcept;
        // 同时NewExecutor方法添加参数
        // const ExecutorType executorType = ExecutorType::NORMAL
        // 而且默认为普通类型，这样不影响跑车和Bus的测试用例编写

        // 默认构造函数和析构函数
        Executor(void) = default;
        virtual ~Executor(void) = default;

        // 不能拷贝
        Executor(const Executor &) = delete;
        // 不能赋值
        Executor &operator=(const Executor &) = delete;

    public:
        // 执行指令的纯虚函数接口
        virtual void Execute(const std::string &command) noexcept = 0;
        // 获取当前位置和朝向的纯虚函数接口
        virtual Pose Query(void) const noexcept = 0;
    };

} // namespace adas
