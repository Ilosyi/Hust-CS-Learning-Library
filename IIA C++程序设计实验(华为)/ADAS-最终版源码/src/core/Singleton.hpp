#pragma once

namespace adas
{
    template <typename T>
    class Singleton final
    {
        // 单例模式
    public:
        static T &Instance(void) noexcept // 单体类
        {
            static T instance;
            return instance;
        }
        // 删除构造函数
        Singleton(const Singleton &) = delete;            // 删除拷贝构造函数
        Singleton &operator=(const Singleton &) = delete; // 删除赋值构造函数
    private:
        Singleton() noexcept = default;  // 默认构造函数
        ~Singleton() noexcept = default; // 默认析构函数
    }; //

} // namespace adas
