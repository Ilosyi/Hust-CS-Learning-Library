#include "PoseEq.hpp"
#include <tuple>

// 重载Pose的==操作符，用于比较两个Pose对象是否相等
namespace adas
{
    bool operator==(const Pose &lhs, const Pose &rhs)
    {
        return std::tie(lhs.x, lhs.y, lhs.heading) == std::tie(rhs.x, rhs.y, rhs.heading);
    }
}