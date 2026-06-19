#pragma once

#include "core/Expected.h"

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace voxstudio::net::elevenlabs {

enum class ApiErrorCode {
    MissingApiKey,
    TransportFailure,
    HttpError,
    JsonParseError,
    UnexpectedResponse,
};

struct ApiError final {
    ApiErrorCode code{ApiErrorCode::UnexpectedResponse};
    std::string message;
    int statusCode{0};
};

struct UserInfo final {
    std::string userId;
    std::string email;
    std::string firstName;
    bool isNewUser{false};
};

struct SubscriptionInfo final {
    std::string tier;
    int characterCount{0};
    int characterLimit{0};
    bool canExtendCharacterLimit{false};
};

struct VoiceInfo final {
    std::string voiceId;
    std::string name;
    std::string category;
    std::string description;
    std::string previewUrl;
    std::string defaultSettingsJson;
    std::map<std::string, std::string> labels;
    bool isOwner{false};
};

struct VoicesResponse final {
    std::vector<VoiceInfo> voices;
    bool hasMore{false};
    int totalCount{0};
    std::string nextPageToken;
};

struct ModelInfo final {
    std::string modelId;
    std::string name;
    bool canDoTextToSpeech{false};
    bool canDoVoiceConversion{false};
    bool canUseStyle{false};
    bool canUseSpeakerBoost{false};
};

struct ModelsResponse final {
    std::vector<ModelInfo> models;
};

[[nodiscard]] ApiError makeApiError(ApiErrorCode code, std::string message, int statusCode = 0);

[[nodiscard]] core::Expected<UserInfo, ApiError> parseUserInfo(const nlohmann::json& json);
[[nodiscard]] core::Expected<SubscriptionInfo, ApiError>
parseSubscriptionInfo(const nlohmann::json& json);
[[nodiscard]] core::Expected<VoiceInfo, ApiError> parseVoiceInfo(const nlohmann::json& json);
[[nodiscard]] core::Expected<VoicesResponse, ApiError>
parseVoicesResponse(const nlohmann::json& json);
[[nodiscard]] core::Expected<ModelsResponse, ApiError>
parseModelsResponse(const nlohmann::json& json);

} // namespace voxstudio::net::elevenlabs
