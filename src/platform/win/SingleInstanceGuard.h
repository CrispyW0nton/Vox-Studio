#pragma once

#include <Windows.h>

namespace voxstudio::platform::win {

class SingleInstanceGuard final {
public:
    explicit SingleInstanceGuard(const wchar_t* mutexName) noexcept;
    ~SingleInstanceGuard() noexcept;

    SingleInstanceGuard(const SingleInstanceGuard&) = delete;
    SingleInstanceGuard& operator=(const SingleInstanceGuard&) = delete;
    SingleInstanceGuard(SingleInstanceGuard&&) = delete;
    SingleInstanceGuard& operator=(SingleInstanceGuard&&) = delete;

    [[nodiscard]] bool isPrimaryInstance() const noexcept;

private:
    HANDLE m_mutexHandle{nullptr};
    bool m_isPrimaryInstance{true};
};

} // namespace voxstudio::platform::win
