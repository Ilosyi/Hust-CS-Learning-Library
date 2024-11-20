#pragma once

#include "CmderOrchestrator.hpp"

namespace adas
{

    class NormalOrchestrator : public CmderOrchestrator
    {
    public:
        ActionGroup Move(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);
            actionGroup.PushAction(GetStepAction(poseHandler));
            return actionGroup;
        }

        ActionGroup TurnLeft(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);

            const auto turnAction = poseHandler.IsReverse() ? ActionType::REVERSE_TURN_LEFT_ACTION : ActionType::TURN_LEFT_ACTION;

            actionGroup.PushAction(turnAction);
            return actionGroup;
        }

        ActionGroup TurnRight(const PoseHandler &poseHandler) const noexcept override
        {
            ActionGroup actionGroup;
            actionGroup += OnFast(poseHandler);

            const auto turnAction = poseHandler.IsReverse() ? ActionType::REVERSE_TURN_RIGHT_ACTION : ActionType::TURN_RIGHT_ACTION;

            actionGroup.PushAction(turnAction);
            return actionGroup;
        }

        ActionGroup TurnRound(const PoseHandler &poseHandler) const noexcept override
        {
            if (poseHandler.IsReverse())
            {
                return ActionGroup(); // 在倒车状态下不执行掉头操作
            }

            ActionGroup actionGroup;
            if (poseHandler.IsFast())
            {
                actionGroup.PushAction(ActionType::FORWARD_1_STEP_ACTION);
                actionGroup.PushAction(ActionType::TURN_LEFT_ACTION);
                actionGroup.PushAction(ActionType::FORWARD_1_STEP_ACTION);
                actionGroup.PushAction(ActionType::TURN_LEFT_ACTION);
            }
            else
            {
                actionGroup.PushAction(ActionType::TURN_LEFT_ACTION);
                actionGroup.PushAction(ActionType::FORWARD_1_STEP_ACTION);
                actionGroup.PushAction(ActionType::TURN_LEFT_ACTION);
            }
            return actionGroup;
        }

    protected:
        ActionType GetStepAction(const PoseHandler &poseHandler) const noexcept // 获取步进动作
        {
            return poseHandler.IsReverse() ? ActionType::BACKWARD_1_STEP_ACTION : ActionType::FORWARD_1_STEP_ACTION;
        }

        ActionGroup OnFast(const PoseHandler &poseHandler) const noexcept
        {
            if (poseHandler.IsFast())
            {
                return ActionGroup({GetStepAction(poseHandler)}); // 如果是加速状态，额外执行一次action
            }
            return ActionGroup(); // 否则返回空ActionGroup
        }
    };

} // namespace adas