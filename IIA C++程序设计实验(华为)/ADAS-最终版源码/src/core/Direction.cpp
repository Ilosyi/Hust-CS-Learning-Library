#include "Direction.hpp"

namespace adas
{

    // 4种方向
    static const Direction directions[4] = {
        {0, 'E'}, // East
        {1, 'S'}, // South
        {2, 'W'}, // West
        {3, 'N'}  // North
    };

    // 4种前进坐标
    static const Point points[4] = {
        {1, 0},  // East: 向右移动
        {0, -1}, // South: 向下移动
        {-1, 0}, // West: 向左移动
        {0, 1}   // North: 向上移动
    };

    // 静态函数 - 根据方向字符返回对应的方向
    const Direction &Direction::GetDirection(const char heading) noexcept
    {
        // 遍历方向数组，查找与指定方向字符匹配的方向
        for (const auto &dir : directions)
        {
            if (dir.heading == heading)
            {
                return dir; // 找到匹配方向，返回引用
            }
        }
        // 如果未找到匹配项，默认返回东（E）方向
        return directions[3];
    }

    // 构造函数 - 初始化方向索引和字符
    Direction::Direction(const unsigned index, const char heading) noexcept
        : index(index), heading(heading)
    {
        // 使用初始化列表赋值，无额外逻辑
    }

    // 前进 - 返回当前方向对应的前进坐标
    const Point &Direction::Move() const noexcept
    {
        // 返回points数组中与当前方向索引对应的点
        return points[index % 4];
    }

    // 左转 - 返回左转后对应的方向
    const Direction &Direction::LeftOne() const noexcept
    {
        // 左转将索引逆时针移动1，相当于 (index + 3) % 4
        return directions[(index + 3) % 4];
    }

    // 右转 - 返回右转后对应的方向
    const Direction &Direction::RightOne() const noexcept
    {
        // 右转将索引顺时针移动1，相当于 (index + 1) % 4
        return directions[(index + 1) % 4];
    }

    // 获取方向字符
    const char Direction::GetHeading() const noexcept
    {
        // 返回当前方向的字符
        return heading;
    }

}
