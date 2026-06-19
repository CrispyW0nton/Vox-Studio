#include "net/elevenlabs/Client.h"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <memory>
#include <utility>

namespace voxstudio::net::elevenlabs {
namespace {

constexpr auto kDefaultBaseUrl = "https://api.elevenlabs.io";
constexpr auto kUserPath = "/v1/user";
constexpr auto kSubscriptionPath = "/v1/user/subscription";
constexpr auto kVoicesPath = "/v1/voices";
constexpr auto kModelsPath = "/v1/models";
constexpr std::chrono::seconds kRequestTimeout{15};

[[nodiscard]] std::string joinedUrl(const std::string& baseUrl, const std::string& path) {
    if (baseUrl.empty()) {
        return path;
    }
    if (baseUrl.back() == '/' && !path.empty() && path.front() == '/') {
        return baseUrl.substr(0, baseUrl.size() - 1) + path;
    }
    if (baseUrl.back() != '/' && !path.empty() && path.front() != '/') {
        return baseUrl + "/" + path;
    }
    return baseUrl + path;
}

} // namespace

CprHttpTransport::CprHttpTransport()
    : CprHttpTransport(kDefaultBaseUrl) {}

CprHttpTransport::CprHttpTransport(std::string baseUrl)
    : m_baseUrl(std::move(baseUrl)) {}

core::Expected<HttpResponse, ApiError> CprHttpTransport::get(const std::string& path,
                                                             const std::string& apiKey) const {
    try {
        const auto response = cpr::Get(
            cpr::Url{joinedUrl(m_baseUrl, path)},
            cpr::Header{{"Accept", "application/json"}, {"xi-api-key", apiKey}},
            cpr::Timeout{kRequestTimeout});

        if (response.error.code != cpr::ErrorCode::OK) {
            return makeApiError(ApiErrorCode::TransportFailure, response.error.message);
        }

        return HttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::TransportFailure, exception.what());
    }
}

Client::Client(std::string apiKey)
    : Client(std::move(apiKey), std::make_unique<CprHttpTransport>()) {}

Client::Client(std::string apiKey, std::unique_ptr<IHttpTransport> transport)
    : m_apiKey(std::move(apiKey))
    , m_transport(std::move(transport)) {}

core::Expected<UserInfo, ApiError> Client::getUser() const {
    auto json = getJson(kUserPath);
    if (!json) {
        return json.error();
    }

    return parseUserInfo(json.value());
}

core::Expected<SubscriptionInfo, ApiError> Client::getSubscription() const {
    auto json = getJson(kSubscriptionPath);
    if (!json) {
        return json.error();
    }

    return parseSubscriptionInfo(json.value());
}

core::Expected<VoicesResponse, ApiError> Client::getVoices() const {
    auto json = getJson(kVoicesPath);
    if (!json) {
        return json.error();
    }

    return parseVoicesResponse(json.value());
}

core::Expected<ModelsResponse, ApiError> Client::getModels() const {
    auto json = getJson(kModelsPath);
    if (!json) {
        return json.error();
    }

    return parseModelsResponse(json.value());
}

core::Expected<nlohmann::json, ApiError> Client::getJson(const std::string& path) const {
    if (m_apiKey.empty()) {
        return makeApiError(ApiErrorCode::MissingApiKey, "ElevenLabs API key is missing.");
    }
    if (m_transport == nullptr) {
        return makeApiError(ApiErrorCode::TransportFailure, "HTTP transport is not configured.");
    }

    auto response = m_transport->get(path, m_apiKey);
    if (!response) {
        return response.error();
    }

    if (response.value().statusCode < 200 || response.value().statusCode >= 300) {
        return makeApiError(ApiErrorCode::HttpError, response.value().body,
                            response.value().statusCode);
    }

    try {
        return nlohmann::json::parse(response.value().body);
    } catch (const nlohmann::json::exception& exception) {
        return makeApiError(ApiErrorCode::JsonParseError, exception.what());
    }
}

} // namespace voxstudio::net::elevenlabs
