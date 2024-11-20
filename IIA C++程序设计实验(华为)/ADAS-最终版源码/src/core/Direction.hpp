#pragma once

#include "Point.hpp"

namespace adas
{

    /**
     * 方向类
     */
    class Direction final
    {
    public:
        // 根据方向字符获取方向
        static const Direction &GetDirection(const char heading) noexcept;

    public:
        Direction(const unsigned index, const char heading) noexcept;

    public:
        const Point &Move(void) const noexcept;         // 前进
        const Direction &LeftOne(void) const noexcept;  // 左转
        const Direction &RightOne(void) const noexcept; // 右转

        const char GetHeading(void) const noexcept; // 获取方向字符

    private:
        unsigned index; // 方向索引 0 1 2 3
        char heading;   // 方向字符 E S W N
    };

}