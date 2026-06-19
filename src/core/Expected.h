#pragma once

#include "core/Error.h"

#include <utility>
#include <variant>

namespace voxstudio::core {

template <typename TValue, typename TError = Error>
class Expected final {
public:
    Expected(const TValue& value)
        : m_storage(value) {}

    Expected(TValue&& value) noexcept
        : m_storage(std::move(value)) {}

    Expected(const TError& error)
        : m_storage(error) {}

    Expected(TError&& error) noexcept
        : m_storage(std::move(error)) {}

    [[nodiscard]] bool hasValue() const noexcept {
        return std::holds_alternative<TValue>(m_storage);
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return hasValue();
    }

    [[nodiscard]] TValue& value() & {
        return std::get<TValue>(m_storage);
    }

    [[nodiscard]] const TValue& value() const& {
        return std::get<TValue>(m_storage);
    }

    [[nodiscard]] TValue&& value() && {
        return std::get<TValue>(std::move(m_storage));
    }

    [[nodiscard]] TError& error() & {
        return std::get<TError>(m_storage);
    }

    [[nodiscard]] const TError& error() const& {
        return std::get<TError>(m_storage);
    }

private:
    std::variant<TValue, TError> m_storage;
};

} // namespace voxstudio::core
