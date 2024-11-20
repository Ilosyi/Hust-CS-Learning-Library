#include "ExecutorImpl.hpp"
#include "../src/cmder/CmderFactory.hpp"
#include "../src/core/Singleton.hpp"
#include <algorithm>
#include "cmder/BusOrchestrator.hpp"
#include "cmder/CmderFactory.hpp"
#include "cmder/NormalOrchestrator.hpp"
#include "cmder/SportsCarOrchestrator.hpp"
namespace adas
{

    // 构造函数
    ExecutorImpl::ExecutorImpl(const Pose &pose, CmderOrchestrator *orchestrator) noexcept
        : poseHandler(pose), orchestrator(orchestrator) {}

    Executor *Executor::NewExecutor(const Pose &pose ,const ExecutorType executorType ) noexcept
    {
        CmderOrchestrator *orchestrator = nullptr;
        switch (executorType)
        {
        case ExecutorType::NORMAL:
            orchestrator = new (std::nothrow) NormalOrchestrator();
            break;
        case ExecutorType::SPORTS_CAR:
        {
            orchestrator = new (std::nothrow) SportsCarOrchestrator();
            break;
        }
        case ExecutorType::BUS:
            orchestrator = new (std::nothrow) BusOrchestrator();
            break;
        }
        return new (std::nothrow) ExecutorImpl(pose, orchestrator);
    }
    /*
           std::nothrow 是 C++ 标准库的一个常量，用于指示在分配内存时不抛出任何异常。
           它是 std::nothrow_t 类型的实例，通常用在 new 运算符和 std::nothrow 命名空间中，
           以请求内存分配器在分配失败时返回一个空指针，而不是抛出 std::bad_alloc 异常。
       */
    // Query()方法用于查询当前位置和朝向
    Pose ExecutorImpl::Query(void) const noexcept
    {
        return poseHandler.Query(); // 返回当前位置和朝向
    }

  

    // Execute()方法用于执行指令，根据指令调用相应的处理方法
    // void ExecutorImpl::Execute(const std::string &commands) noexcept
    // {
    //     // 使用指令工厂获取字符串对应的命令列表
    //     // Cmders 类型是 std::list<Cmder>
    //     // Cmder类型是 std::function<void(PoseHandler& poseHandler)>
    //     const auto cmders = Singleton<CmderFactory>::Instance().GetCmders(commands);

    //     // 遍历命令列表里的每个命令cmder
    //     std::for_each(
    //         cmders.begin(),
    //         cmders.end(),
    //         [this](const Cmder&cmder) noexcept {
    //         cmder(poseHandler).DoOperate(poseHandler); // 执行cmder(poseHandler)就是进行移动或转向操作
    //         }
    //     );
    // }
    void ExecutorImpl::Execute(const std::string &commands) noexcept
    {
        const auto cmders = Singleton<CmderFactory>::Instance().GetCmders(commands);
        std::for_each(cmders.begin(), cmders.end(),
                      [this](const Cmder &cmder) noexcept
                      {
                          cmder(poseHandler, *orchestrator).DoOperate(poseHandler);
                      });
    }
    // 原子指令抽象前 Cmder 对应了 std::function<void(PoseHandler&)>，
    //执行 cmder(poseHandler) 就是进行移动或转向操作；
    //原子指令抽象后 Cmder 对应了 std::function<ActionGroup(PoseHandler&)>，
    //执行 cmder(poseHandler).DoOperate(poseHandler) 才是进行移动或转向操作。

} // namespace adas
