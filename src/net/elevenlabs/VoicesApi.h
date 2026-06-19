#pragma once

#include "core/Expected.h"
#include "net/elevenlabs/Client.h"
#include "net/elevenlabs/Models.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace voxstudio::net::elevenlabs {

struct MultipartPart final {
    std::string name;
    std::string value;
    std::filesystem::path filePath;
    bool isFile{false};
};

class IVoiceHttpTransport {
public:
    virtual ~IVoiceHttpTransport() = default;

    [[nodiscard]] virtual core::Expected<HttpResponse, ApiError>
    get(const std::string& path, const std::string& apiKey) const = 0;

    [[nodiscard]] virtual core::Expected<HttpResponse, ApiError>
    postMultipart(const std::string& path,
                  const std::string& apiKey,
                  const std::vector<MultipartPart>& parts) const = 0;

    [[nodiscard]] virtual core::Expected<HttpResponse, ApiError>
    deleteRequest(const std::string& path, const std::string& apiKey) const = 0;
};

class CprVoiceHttpTransport final : public IVoiceHttpTransport {
public:
    CprVoiceHttpTransport();
    explicit CprVoiceHttpTransport(std::string baseUrl);

    [[nodiscard]] core::Expected<HttpResponse, ApiError>
    get(const std::string& path, const std::string& apiKey) const override;

    [[nodiscard]] core::Expected<HttpResponse, ApiError>
    postMultipart(const std::string& path,
                  const std::string& apiKey,
                  const std::vector<MultipartPart>& parts) const override;

    [[nodiscard]] core::Expected<HttpResponse, ApiError>
    deleteRequest(const std::string& path, const std::string& apiKey) const override;

private:
    std::string m_baseUrl;
};

struct VoiceCloneRequest final {
    std::string name;
    std::string description;
    std::map<std::string, std::string> labels;
    std::vector<std::filesystem::path> sampleFiles;
    bool removeBackgroundNoise{false};
};

struct VoiceEditRequest final {
    std::string name;
    std::string description;
    std::map<std::string, std::string> labels;
    std::vector<std::filesystem::path> sampleFiles;
    bool removeBackgroundNoise{false};
};

struct VoiceCloneResponse final {
    std::string voiceId;
    bool requiresVerification{false};
};

struct StatusResponse final {
    std::string status;
};

class VoicesApi final {
public:
    explicit VoicesApi(std::string apiKey);
    VoicesApi(std::string apiKey, std::unique_ptr<IVoiceHttpTransport> transport);

    [[nodiscard]] core::Expected<VoicesResponse, ApiError> listVoices() const;
    [[nodiscard]] core::Expected<VoiceInfo, ApiError> getVoice(const std::string& voiceId) const;
    [[nodiscard]] core::Expected<VoiceCloneResponse, ApiError>
    addVoice(const VoiceCloneRequest& request) const;
    [[nodiscard]] core::Expected<StatusResponse, ApiError>
    editVoice(const std::string& voiceId, const VoiceEditRequest& request) const;
    [[nodiscard]] core::Expected<StatusResponse, ApiError>
    deleteVoice(const std::string& voiceId) const;

private:
    [[nodiscard]] core::Expected<nlohmann::json, ApiError> getJson(const std::string& path) const;
    [[nodiscard]] core::Expected<nlohmann::json, ApiError>
    postMultipartJson(const std::string& path, const std::vector<MultipartPart>& parts) const;
    [[nodiscard]] core::Expected<nlohmann::json, ApiError>
    deleteJson(const std::string& path) const;

    std::string m_apiKey;
    std::unique_ptr<IVoiceHttpTransport> m_transport;
};

[[nodiscard]] core::Expected<VoiceCloneResponse, ApiError>
parseVoiceCloneResponse(const nlohmann::json& json);
[[nodiscard]] core::Expected<StatusResponse, ApiError>
parseStatusResponse(const nlohmann::json& json);
[[nodiscard]] std::vector<MultipartPart> cloneRequestParts(const VoiceCloneRequest& request);
[[nodiscard]] std::vector<MultipartPart> editRequestParts(const VoiceEditRequest& request);

} // namespace voxstudio::net::elevenlabs
