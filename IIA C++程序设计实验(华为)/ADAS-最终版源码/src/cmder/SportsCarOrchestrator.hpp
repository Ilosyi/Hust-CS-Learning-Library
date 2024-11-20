#pragma once
#include "NormalOrchestrator.hpp"

namespace adas
{
    class SportsCarOrchestrator : public NormalOrchestrator
    {
    public:
        // ActionType GetStepAction(const PoseHandler &poseHandler) const noexcept // 获取步进动作
        // {
        //     return poseHandler.IsReverse() ? ActionType::BACKWARD_1_STEP_ACTION : ActionType::FORWARD_1_STEP_ACTION;
        // }

        // ActionGroup OnFast(const PoseHandler &poseHandler) const noexcept
        // {
        //     if (poseHandler.IsFast())
        //     {
        //         return ActionGroup({GetStepAction(poseHandler)}); // 如果是加速状态，额外执行一次action
        //     }
        //     return ActionGroup(); // 否则返回空ActionGroup
        // }
        ActionGroup Move(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);
            actionGroup += OnFast(poseHandler);
            actionGroup.PushAction(GetStepAction(poseHandler));
            actionGroup.PushAction(GetStepAction(poseHandler));
            return actionGroup;
        }
        ActionGroup TurnRight(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);

            const auto turnAction = poseHandler.IsReverse() ? // 如果是倒车状态，执行左转动作，否则执行右转动作
            ActionType::REVERSE_TURN_RIGHT_ACTION : ActionType::TURN_RIGHT_ACTION;

            actionGroup.PushAction(turnAction);
            actionGroup.PushAction(GetStepAction(poseHandler)); // 再执行一次前进/后退动作
            return actionGroup;
        }
        ActionGroup TurnLeft(const PoseHandler &poseHandler) const noexcept override{
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);// 如果是加速状态，额外执行一次前进/后退动作

            const auto turnAction = poseHandler.IsReverse() ? // 如果是倒车状态，执行右转动作，否则执行左转动作
            ActionType::REVERSE_TURN_LEFT_ACTION : ActionType::TURN_LEFT_ACTION;

            actionGroup.PushAction(turnAction);
            actionGroup.PushAction(GetStepAction(poseHandler));//再执行一次前进/后退动作
            return actionGroup;
        }
        
    };
}