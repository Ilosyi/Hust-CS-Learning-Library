#include "ActionGroup.hpp"
#include "CmderFactory.hpp"

#include <vector>
#include <algorithm>

namespace adas
{

    class ForwardAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.Forward();
        }
    };

    class BackwardAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.Backward();
        }
    };

    class TurnLeftAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.TurnLeft();
        }
    };

    class ReverseTurnLeftAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.TurnRight();
        }
    };

    class TurnRightAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.TurnRight();
        }
    };

    class ReverseTurnRightAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.TurnLeft();
        }
    };

    class BeFastAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.Fast();
        }
    };

    class BeReverseAction final
    {
    public:
        void operator()(PoseHandler &poseHandler) const noexcept
        {
            poseHandler.Reverse();
        }
    };
    ActionGroup::ActionGroup(const std::list<ActionType> &actions) noexcept : actions(actions) {}// 有参构造函数

   //+=运算符重载
    ActionGroup &ActionGroup::operator+=(const ActionGroup &rhs) noexcept
    {
        actions.insert(actions.end(), rhs.actions.begin(), rhs.actions.end());
        return *this;
    }

    void ActionGroup::PushAction(const ActionType &ActionType) noexcept // 添加动作
    {
        actions.push_back(ActionType);
    }

    void ActionGroup::DoOperate(PoseHandler &poseHandler) const noexcept// 执行动作
    {
        static std::vector<std::function<void(PoseHandler & poseHandler)>> actionVec = {
            ForwardAction(),
            BackwardAction(),
            TurnLeftAction(),
            ReverseTurnLeftAction(),
            TurnRightAction(),
            ReverseTurnRightAction(),
            BeFastAction(),
            BeReverseAction()};

        std::for_each(actions.begin(), actions.end(), [&poseHandler](const ActionType &actionType) mutable noexcept// 使用 lambda 表达式调用指令列表中的每个指令
        { actionVec[static_cast<uint16_t>(actionType)](poseHandler); });
    }
}