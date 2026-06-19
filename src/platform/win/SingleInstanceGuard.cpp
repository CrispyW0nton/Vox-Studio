#include "platform/win/SingleInstanceGuard.h"

namespace voxstudio::platform::win {

SingleInstanceGuard::SingleInstanceGuard(const wchar_t* mutexName) noexcept {
    m_mutexHandle = CreateMutexW(nullptr, TRUE, mutexName);
    if (m_mutexHandle == nullptr) {
        m_isPrimaryInstance = true;
        return;
    }

    m_isPrimaryInstance = GetLastError() != ERROR_ALREADY_EXISTS;
}

SingleInstanceGuard::~SingleInstanceGuard() noexcept {
    if (m_mutexHandle != nullptr) {
        CloseHandle(m_mutexHandle);
    }
}

bool SingleInstanceGuard::isPrimaryInstance() const noexcept {
    return m_isPrimaryInstance;
}

} // namespace voxstudio::platform::win
