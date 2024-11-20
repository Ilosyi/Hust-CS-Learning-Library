#pragma once
#include "../include/Executor.hpp"
#include <string>
#include "../src/core/PoseHandler.hpp"
#include "../src/cmder/Command.hpp"
#include "../src/cmder/CmderOrchestrator.hpp"
#include<memory>
namespace adas
{

    // Executor的具体实现
    class ExecutorImpl final : public Executor
    {
    public:
       // Executor *NewExecutor(const Pose &pose, ExecutorType executorType) noexcept;
        // 构造函数
        explicit ExecutorImpl(const Pose &pose, CmderOrchestrator* orchestrator) noexcept;
        // 默认析构函数
        ~ExecutorImpl() noexcept = default;

        // 不能拷贝
        ExecutorImpl(const ExecutorImpl &) = delete;
        // 不能赋值
        ExecutorImpl &operator=(const ExecutorImpl &) = delete;

    public:
        // 查询当前的车辆姿态，是父类抽象方法Query的具体实现
        Pose Query() const noexcept override;
        // 第二阶段新增加的纯虚函数，执行一个用字符串表示的指令，是父类抽象方法Execute的具体实现
        void Execute(const std::string &command) noexcept override;

    private:
        PoseHandler poseHandler; // 状态管理类
        std::unique_ptr<CmderOrchestrator> orchestrator; 
    };

} // namespace adas