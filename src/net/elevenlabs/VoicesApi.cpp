#include "net/elevenlabs/VoicesApi.h"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <utility>

namespace voxstudio::net::elevenlabs {
namespace {

constexpr auto kDefaultBaseUrl = "https://api.elevenlabs.io";
constexpr auto kAddVoicePath = "/v1/voices/add";
constexpr auto kSearchVoicesPath = "/v2/voices?page_size=100&include_total_count=true";
constexpr auto kFilesPartName = "files[]";
constexpr std::chrono::seconds kRequestTimeout{30};

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

[[nodiscard]] bool isUnreservedUrlCharacter(const unsigned char value) noexcept {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
           (value >= '0' && value <= '9') || value == '-' || value == '_' || value == '.' ||
           value == '~';
}

[[nodiscard]] std::string urlEncode(const std::string& value) {
    std::ostringstream stream;
    stream << std::uppercase << std::hex;
    for (const unsigned char character : value) {
        if (isUnreservedUrlCharacter(character)) {
            stream << static_cast<char>(character);
        } else {
            stream << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(character);
        }
    }
    return stream.str();
}

[[nodiscard]] std::string voicePath(const std::string& voiceId) {
    return "/v1/voices/" + urlEncode(voiceId);
}

[[nodiscard]] std::string editVoicePath(const std::string& voiceId) {
    return voicePath(voiceId) + "/edit";
}

[[nodiscard]] std::string listVoicesPath(const std::string& nextPageToken) {
    if (nextPageToken.empty()) {
        return kSearchVoicesPath;
    }

    return std::string{kSearchVoicesPath} + "&next_page_token=" + urlEncode(nextPageToken);
}

[[nodiscard]] cpr::Header apiHeaders(const std::string& apiKey) {
    return cpr::Header{{"Accept", "application/json"}, {"xi-api-key", apiKey}};
}

[[nodiscard]] core::Expected<nlohmann::json, ApiError>
parseJsonResponse(const core::Expected<HttpResponse, ApiError>& response) {
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

[[nodiscard]] core::Expected<bool, ApiError> validateApiKey(const std::string& apiKey) {
    if (apiKey.empty()) {
        return makeApiError(ApiErrorCode::MissingApiKey, "ElevenLabs API key is missing.");
    }
    return true;
}

[[nodiscard]] nlohmann::json labelsJson(const std::map<std::string, std::string>& labels) {
    nlohmann::json labelsObject = nlohmann::json::object();
    for (const auto& [key, value] : labels) {
        if (!key.empty() && !value.empty()) {
            labelsObject[key] = value;
        }
    }
    return labelsObject;
}

void appendSharedParts(std::vector<MultipartPart>& parts,
                       const std::string& name,
                       const std::string& description,
                       const std::map<std::string, std::string>& labels,
                       const bool removeBackgroundNoise) {
    parts.push_back(MultipartPart{.name = "name", .value = name});
    parts.push_back(MultipartPart{.name = "remove_background_noise",
                                  .value = removeBackgroundNoise ? "true" : "false"});
    if (!description.empty()) {
        parts.push_back(MultipartPart{.name = "description", .value = description});
    }
    if (!labels.empty()) {
        parts.push_back(MultipartPart{.name = "labels", .value = labelsJson(labels).dump()});
    }
}

void appendFiles(std::vector<MultipartPart>& parts,
                 const std::vector<std::filesystem::path>& files) {
    for (const auto& file : files) {
        parts.push_back(MultipartPart{.name = kFilesPartName, .filePath = file, .isFile = true});
    }
}

} // namespace

CprVoiceHttpTransport::CprVoiceHttpTransport()
    : CprVoiceHttpTransport(kDefaultBaseUrl) {}

CprVoiceHttpTransport::CprVoiceHttpTransport(std::string baseUrl)
    : m_baseUrl(std::move(baseUrl)) {}

core::Expected<HttpResponse, ApiError>
CprVoiceHttpTransport::get(const std::string& path, const std::string& apiKey) const {
    try {
        const auto response = cpr::Get(cpr::Url{joinedUrl(m_baseUrl, path)}, apiHeaders(apiKey),
                                       cpr::Timeout{kRequestTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return makeApiError(ApiErrorCode::TransportFailure, response.error.message);
        }
        return HttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::TransportFailure, exception.what());
    }
}

core::Expected<HttpResponse, ApiError>
CprVoiceHttpTransport::postMultipart(const std::string& path,
                                     const std::string& apiKey,
                                     const std::vector<MultipartPart>& parts) const {
    try {
        std::vector<cpr::Part> cprParts;
        cprParts.reserve(parts.size());
        for (const auto& part : parts) {
            if (part.isFile) {
                cprParts.emplace_back(part.name, cpr::Files{cpr::File{part.filePath.string()}});
            } else {
                cprParts.emplace_back(part.name, part.value);
            }
        }

        const auto response =
            cpr::Post(cpr::Url{joinedUrl(m_baseUrl, path)}, apiHeaders(apiKey),
                      cpr::Multipart{std::move(cprParts)}, cpr::Timeout{kRequestTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return makeApiError(ApiErrorCode::TransportFailure, response.error.message);
        }
        return HttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::TransportFailure, exception.what());
    }
}

core::Expected<HttpResponse, ApiError>
CprVoiceHttpTransport::deleteRequest(const std::string& path, const std::string& apiKey) const {
    try {
        const auto response =
            cpr::Delete(cpr::Url{joinedUrl(m_baseUrl, path)}, apiHeaders(apiKey),
                        cpr::Timeout{kRequestTimeout});
        if (response.error.code != cpr::ErrorCode::OK) {
            return makeApiError(ApiErrorCode::TransportFailure, response.error.message);
        }
        return HttpResponse{static_cast<int>(response.status_code), response.text};
    } catch (const std::exception& exception) {
        return makeApiError(ApiErrorCode::TransportFailure, exception.what());
    }
}

VoicesApi::VoicesApi(std::string apiKey)
    : VoicesApi(std::move(apiKey), std::make_unique<CprVoiceHttpTransport>()) {}

VoicesApi::VoicesApi(std::string apiKey, std::unique_ptr<IVoiceHttpTransport> transport)
    : m_apiKey(std::move(apiKey))
    , m_transport(std::move(transport)) {}

core::Expected<VoicesResponse, ApiError> VoicesApi::listVoices() const {
    VoicesResponse allVoices;
    std::string nextPageToken;

    for (int page = 0; page < 100; ++page) {
        auto json = getJson(listVoicesPath(nextPageToken));
        if (!json) {
            return json.error();
        }

        auto pageResponse = parseVoicesResponse(json.value());
        if (!pageResponse) {
            return pageResponse.error();
        }

        allVoices.voices.insert(allVoices.voices.end(),
                                std::make_move_iterator(pageResponse.value().voices.begin()),
                                std::make_move_iterator(pageResponse.value().voices.end()));
        allVoices.totalCount = pageResponse.value().totalCount;
        allVoices.hasMore = pageResponse.value().hasMore;
        allVoices.nextPageToken = pageResponse.value().nextPageToken;

        if (!pageResponse.value().hasMore) {
            return allVoices;
        }
        if (pageResponse.value().nextPageToken.empty()) {
            return makeApiError(ApiErrorCode::UnexpectedResponse,
                                "Voice list response is missing next_page_token.");
        }
        nextPageToken = pageResponse.value().nextPageToken;
    }

    return makeApiError(ApiErrorCode::UnexpectedResponse,
                        "Voice list pagination exceeded the safety limit.");
}

core::Expected<VoiceInfo, ApiError> VoicesApi::getVoice(const std::string& voiceId) const {
    if (voiceId.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "Voice id must not be empty.");
    }

    auto json = getJson(voicePath(voiceId));
    if (!json) {
        return json.error();
    }
    return parseVoiceInfo(json.value());
}

core::Expected<VoiceCloneResponse, ApiError>
VoicesApi::addVoice(const VoiceCloneRequest& request) const {
    auto json = postMultipartJson(kAddVoicePath, cloneRequestParts(request));
    if (!json) {
        return json.error();
    }
    return parseVoiceCloneResponse(json.value());
}

core::Expected<StatusResponse, ApiError>
VoicesApi::editVoice(const std::string& voiceId, const VoiceEditRequest& request) const {
    if (voiceId.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "Voice id must not be empty.");
    }

    auto json = postMultipartJson(editVoicePath(voiceId), editRequestParts(request));
    if (!json) {
        return json.error();
    }
    return parseStatusResponse(json.value());
}

core::Expected<StatusResponse, ApiError> VoicesApi::deleteVoice(const std::string& voiceId) const {
    if (voiceId.empty()) {
        return makeApiError(ApiErrorCode::UnexpectedResponse, "Voice id must not be empty.");
    }

    auto json = deleteJson(voicePath(voiceId));
    if (!json) {
        return json.error();
    }
    return parseStatusResponse(json.value());
}

core::Expected<nlohmann::json, ApiError> VoicesApi::getJson(const std::string& path) const {
    auto validKey = validateApiKey(m_apiKey);
    if (!validKey) {
        return validKey.error();
    }
    if (m_transport == nullptr) {
        return makeApiError(ApiErrorCode::TransportFailure, "HTTP transport is not configured.");
    }

    return parseJsonResponse(m_transport->get(path, m_apiKey));
}

core::Expected<nlohmann::json, ApiError>
VoicesApi::postMultipartJson(const std::string& path,
                             const std::vector<MultipartPart>& parts) const {
    auto validKey = validateApiKey(m_apiKey);
    if (!validKey) {
        return validKey.error();
    }
    if (m_transport == nullptr) {
        return makeApiError(ApiErrorCode::TransportFailure, "HTTP transport is not configured.");
    }

    return parseJsonResponse(m_transport->postMultipart(path, m_apiKey, parts));
}

core::Expected<nlohmann::json, ApiError> VoicesApi::deleteJson(const std::string& path) const {
    auto validKey = validateApiKey(m_apiKey);
    if (!validKey) {
        return validKey.error();
    }
    if (m_transport == nullptr) {
        return makeApiError(ApiErrorCode::TransportFailure, "HTTP transport is not configured.");
    }

    return parseJsonResponse(m_transport->deleteRequest(path, m_apiKey));
}

core::Expected<VoiceCloneResponse, ApiError>
parseVoiceCloneResponse(const nlohmann::json& json) {
    try {
        if (!json.is_object() || !json.contains("voice_id") || !json.at("voice_id").is_string()) {
            return makeApiError(ApiErrorCode::UnexpectedResponse,
                                "Create voice response is missing voice_id.");
        }

        VoiceCloneResponse response;
        response.voiceId = json.at("voice_id").get<std::string>();
        if (json.contains("requires_verification") &&
            json.at("requires_verification").is_boolean()) {
            response.requiresVerification = json.at("requires_verification").get<bool>();
        }
        return response;
    } catch (const nlohmann::json::exception& exception) {
        return makeApiError(ApiErrorCode::JsonParseError, exception.what());
    }
}

core::Expected<StatusResponse, ApiError> parseStatusResponse(const nlohmann::json& json) {
    try {
        if (!json.is_object() || !json.contains("status") || !json.at("status").is_string()) {
            return makeApiError(ApiErrorCode::UnexpectedResponse,
                                "Status response is missing status.");
        }

        return StatusResponse{json.at("status").get<std::string>()};
    } catch (const nlohmann::json::exception& exception) {
        return makeApiError(ApiErrorCode::JsonParseError, exception.what());
    }
}

std::vector<MultipartPart> cloneRequestParts(const VoiceCloneRequest& request) {
    std::vector<MultipartPart> parts;
    appendSharedParts(parts, request.name, request.description, request.labels,
                      request.removeBackgroundNoise);
    appendFiles(parts, request.sampleFiles);
    return parts;
}

std::vector<MultipartPart> editRequestParts(const VoiceEditRequest& request) {
    std::vector<MultipartPart> parts;
    appendSharedParts(parts, request.name, request.description, request.labels,
                      request.removeBackgroundNoise);
    appendFiles(parts, request.sampleFiles);
    return parts;
}

} // namespace voxstudio::net::elevenlabs
