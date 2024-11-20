#pragma once

#include "ExecutorImpl.hpp"
#include "../src/core/PoseHandler.hpp"
#include <functional>
#include "CmderOrchestrator.hpp"
namespace adas
{
    class MoveCommand final
    {
    public:
        // 现在MoveCommand不再直接调用PoseHandler执行动作
        // 而是返回一个ActionGroup对象，里面包含了可执行的命令
        // 要执行命令需要通过枚举ActionType来表示
        ActionGroup operator()(PoseHandler &poseHandler,const CmderOrchestrator
        & orchestrator) const noexcept
        {
           return orchestrator.Move(poseHandler);
        }
    };

    class TurnLeftCommand final
    {
    public:
        ActionGroup operator()(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator) const noexcept
        {
            return orchestrator.TurnLeft(poseHandler);
        }
    };

    class TurnRightCommand final
    {
    public:
        ActionGroup operator()(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator) const noexcept
        {
            return orchestrator.TurnRight(poseHandler);
        }
    };

    class FastCommand final
    {
    public:
        ActionGroup operator()(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator) const noexcept
        {
            ActionGroup actionGroup;

            actionGroup.PushAction(ActionType::BE_FAST_ACTION);//切换加速状态
            return actionGroup;
        }
    };

    class ReverseCommand final
    {
    public:
        ActionGroup operator()(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator) const noexcept
        {
            ActionGroup actionGroup;
            actionGroup.PushAction(ActionType::BE_REVERSE_ACTION);//切换倒车状态
            return actionGroup;
        }
    };
    class TurnRoundCommand final
    {
    public:
        ActionGroup operator()(PoseHandler &poseHandler, const CmderOrchestrator &orchestrator) const noexcept
        {
            return orchestrator.TurnRound(poseHandler);
        }
    };
} // namespace adas
