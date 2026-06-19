#include "net/elevenlabs/Models.h"

#include <nlohmann/json.hpp>

#include <exception>
#include <utility>

namespace voxstudio::net::elevenlabs {
namespace {

[[nodiscard]] std::string optionalString(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || json.at(key).is_null()) {
        return {};
    }

    if (json.at(key).is_string()) {
        return json.at(key).get<std::string>();
    }

    return {};
}

[[nodiscard]] int optionalInt(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || json.at(key).is_null()) {
        return 0;
    }

    if (json.at(key).is_number_integer()) {
        return json.at(key).get<int>();
    }

    return 0;
}

[[nodiscard]] bool optionalBool(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || json.at(key).is_null()) {
        return false;
    }

    if (json.at(key).is_boolean()) {
        return json.at(key).get<bool>();
    }

    return false;
}

[[nodiscard]] std::string optionalObjectDump(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || json.at(key).is_null()) {
        return {};
    }

    if (json.at(key).is_object()) {
        return json.at(key).dump();
    }

    return {};
}

[[nodiscard]] std::map<std::string, std::string> optionalStringMap(const nlohmann::json& json,
                                                                   const char* key) {
    if (!json.contains(key) || !json.at(key).is_object()) {
        return {};
    }

    std::map<std::string, std::string> result;
    for (const auto& item : json.at(key).items()) {
        if (item.value().is_string()) {
            result.emplace(item.key(), item.value().get<std::string>());
        }
    }
    return result;
}

} // namespace

ApiError makeApiError(const ApiErrorCode code, std::string message, const int statusCode) {
    return ApiError{code, std::move(message), statusCode};
}

core::Expected<VoiceInfo, ApiError> parseVoiceInfo(const nlohmann::json& json) {
    if (!json.is_object()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "Voice entry is not an object.");
    }

    VoiceInfo voice;
    voice.voiceId = optionalString(json, "voice_id");
    voice.name = optionalString(json, "name");
    voice.category = optionalString(json, "category");
    voice.description = optionalString(json, "description");
    voice.previewUrl = optionalString(json, "preview_url");
    voice.defaultSettingsJson = optionalObjectDump(json, "settings");
    voice.labels = optionalStringMap(json, "labels");
    voice.isOwner = optionalBool(json, "is_owner");

    if (voice.voiceId.empty() || voice.name.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse,
                            "Voice entry is missing voice_id or name.");
    }

    return voice;
}

namespace {

[[nodiscard]] core::Expected<ModelInfo, ApiError> parseModelInfo(const nlohmann::json& json) {
    if (!json.is_object()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "Model entry is not an object.");
    }

    ModelInfo model;
    model.modelId = optionalString(json, "model_id");
    model.name = optionalString(json, "name");
    model.canDoTextToSpeech = optionalBool(json, "can_do_text_to_speech");
    model.canDoVoiceConversion = optionalBool(json, "can_do_voice_conversion");
    model.canUseStyle = optionalBool(json, "can_use_style");
    model.canUseSpeakerBoost = optionalBool(json, "can_use_speaker_boost");

    if (model.modelId.empty() || model.name.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse,
                            "Model entry is missing model_id or name.");
    }

    return model;
}

} // namespace

core::Expected<UserInfo, ApiError> parseUserInfo(const nlohmann::json& json) {
    try {
        if (!json.is_object()) {
            return makeApiError(ApiErrorCode::UnexpectedResponse,
                                "User response is not an object.");
        }

        UserInfo user;
        user.userId = optionalString(json, "user_id");
        user.email = optionalString(json, "email");
        user.firstName = optionalString(json, "first_name");
        user.isNewUser = optionalBool(json, "is_new_user");
        return user;
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::JsonParseError, exception.what());
    }
}

core::Expected<SubscriptionInfo, ApiError> parseSubscriptionInfo(const nlohmann::json& json) {
    try {
        if (!json.is_object()) {
            return makeApiError(ApiErrorCode::UnexpectedResponse,
                                "Subscription response is not an object.");
        }

        SubscriptionInfo subscription;
        subscription.tier = optionalString(json, "tier");
        subscription.characterCount = optionalInt(json, "character_count");
        subscription.characterLimit = optionalInt(json, "character_limit");
        subscription.canExtendCharacterLimit = optionalBool(json, "can_extend_character_limit");
        return subscription;
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::JsonParseError, exception.what());
    }
}

core::Expected<VoicesResponse, ApiError> parseVoicesResponse(const nlohmann::json& json) {
    try {
        if (!json.is_object() || !json.contains("voices") || !json.at("voices").is_array()) {
            return makeApiError(ApiErrorCode::UnexpectedResponse,
                                "Voices response is missing voices array.");
        }

        VoicesResponse response;
        for (const auto& voiceJson : json.at("voices")) {
            auto voice = parseVoiceInfo(voiceJson);
            if (!voice) {
                return voice.error();
            }
            response.voices.push_back(std::move(voice).value());
        }
        response.hasMore = optionalBool(json, "has_more");
        response.totalCount = optionalInt(json, "total_count");
        response.nextPageToken = optionalString(json, "next_page_token");

        return response;
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::JsonParseError, exception.what());
    }
}

core::Expected<ModelsResponse, ApiError> parseModelsResponse(const nlohmann::json& json) {
    try {
        const nlohmann::json* modelsJson = nullptr;
        if (json.is_array()) {
            modelsJson = &json;
        } else if (json.is_object() && json.contains("models") && json.at("models").is_array()) {
            modelsJson = &json.at("models");
        } else {
            return makeApiError(ApiErrorCode::UnexpectedResponse,
                                "Models response is not an array.");
        }

        ModelsResponse response;
        for (const auto& modelJson : *modelsJson) {
            auto model = parseModelInfo(modelJson);
            if (!model) {
                return model.error();
            }
            response.models.push_back(std::move(model).value());
        }

        return response;
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::JsonParseError, exception.what());
    }
}

} // namespace voxstudio::net::elevenlabs
