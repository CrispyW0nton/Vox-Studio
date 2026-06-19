#pragma once

#include "core/Expected.h"
#include "net/elevenlabs/Models.h"

#include <memory>
#include <string>

#include <nlohmann/json_fwd.hpp>

namespace voxstudio::net::elevenlabs {

struct HttpResponse final {
    int statusCode{0};
    std::string body;
};

class IHttpTransport {
public:
    virtual ~IHttpTransport() = default;

    [[nodiscard]] virtual core::Expected<HttpResponse, ApiError>
    get(const std::string& path, const std::string& apiKey) const = 0;
};

class CprHttpTransport final : public IHttpTransport {
public:
    CprHttpTransport();
    explicit CprHttpTransport(std::string baseUrl);

    [[nodiscard]] core::Expected<HttpResponse, ApiError>
    get(const std::string& path, const std::string& apiKey) const override;

private:
    std::string m_baseUrl;
};

class Client final {
public:
    explicit Client(std::string apiKey);
    Client(std::string apiKey, std::unique_ptr<IHttpTransport> transport);

    [[nodiscard]] core::Expected<UserInfo, ApiError> getUser() const;
    [[nodiscard]] core::Expected<SubscriptionInfo, ApiError> getSubscription() const;
    [[nodiscard]] core::Expected<VoicesResponse, ApiError> getVoices() const;
    [[nodiscard]] core::Expected<ModelsResponse, ApiError> getModels() const;

private:
    [[nodiscard]] core::Expected<nlohmann::json, ApiError> getJson(const std::string& path) const;

    std::string m_apiKey;
    std::unique_ptr<IHttpTransport> m_transport;
};

} // namespace voxstudio::net::elevenlabs
