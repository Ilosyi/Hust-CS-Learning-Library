#include "PoseHandler.hpp"

namespace adas
{

    PoseHandler::PoseHandler(const Pose &pose) noexcept
        : point(pose.x, pose.y),
          facing(&Direction::GetDirection(pose.heading)),
          fast(false) {} // 初始化 fast 状态为 false

    void PoseHandler::Forward() noexcept
    {
        // 将当前坐标加上方向向量
        point += facing->Move();
    }
    void PoseHandler::Backward() noexcept
    {
        point -= facing->Move();
    }
    void PoseHandler::TurnLeft() noexcept
    {
        // 左转
        facing = &(facing->LeftOne());
    }

    void PoseHandler::TurnRight() noexcept
    {
        // 右转
        facing = &(facing->RightOne());
    }

    void PoseHandler::Fast() noexcept
    {
        // 切换加速模式
        fast = !fast;
    }
    void PoseHandler::Reverse() noexcept
    {
        // 切换后退模式
        reverse = !reverse;
    }

    bool PoseHandler::IsFast() const noexcept
    {
        // 是否处于快速模式
        return fast;
    }
    bool PoseHandler::IsReverse() const noexcept
    {
        // 是否处于后退模式
        return reverse;
    }
    Pose PoseHandler::Query(void) const noexcept
    {
        // 返回当前位置和方向信息
        return {point.GetX(), point.GetY(), facing->GetHeading()};
    }

} // namespace adas
