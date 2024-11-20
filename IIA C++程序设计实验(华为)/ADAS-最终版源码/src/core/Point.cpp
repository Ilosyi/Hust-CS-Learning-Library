#include "Point.hpp"

namespace adas
{

    Point::Point(const int x, const int y) noexcept : x(x), y(y)
    {
        // 构造函数实现，使用初始化列表初始化x和y
    }

    Point::Point(const Point &rhs) noexcept : x(rhs.x), y(rhs.y)
    {
        // 拷贝构造函数实现，使用初始化列表初始化x和y
    }

    Point &Point::operator=(const Point &rhs) noexcept
    {
        // 拷贝赋值运算符实现
        if (this != &rhs)
        {
            x = rhs.x;
            y = rhs.y;
        }
        return *this;
    }

    Point &Point::operator+=(const Point &rhs) noexcept
    {
        // 移动运算符实现
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
    Point &Point::operator-=(const Point &rhs) noexcept
    {
        // 移动运算符实现
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    int Point::GetX(void) const noexcept
    {
        // 获取X坐标的实现
        return x;
    }

    int Point::GetY(void) const noexcept
    {
        // 获取Y坐标的实现
        return y;
    }

}