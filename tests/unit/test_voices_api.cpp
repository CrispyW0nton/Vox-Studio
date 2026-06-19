#include "net/elevenlabs/VoicesApi.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] std::string loadFixtureText(const std::filesystem::path& relativePath) {
    const std::filesystem::path fixtureRoot{VOXSTUDIO_TEST_FIXTURE_DIR};
    std::ifstream input{fixtureRoot / relativePath};
    REQUIRE(input.good());
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

class FakeVoiceTransport final : public voxstudio::net::elevenlabs::IVoiceHttpTransport {
public:
    explicit FakeVoiceTransport(std::map<std::string, std::string> responses)
        : m_responses(std::move(responses)) {}

    [[nodiscard]] voxstudio::core::Expected<voxstudio::net::elevenlabs::HttpResponse,
                                            voxstudio::net::elevenlabs::ApiError>
    get(const std::string& path, const std::string& apiKey) const override {
        m_getPaths.push_back(path);
        m_apiKeys.push_back(apiKey);
        return responseFor("GET", path);
    }

    [[nodiscard]] voxstudio::core::Expected<voxstudio::net::elevenlabs::HttpResponse,
                                            voxstudio::net::elevenlabs::ApiError>
    postMultipart(const std::string& path,
                  const std::string& apiKey,
                  const std::vector<voxstudio::net::elevenlabs::MultipartPart>& parts)
        const override {
        m_postPaths.push_back(path);
        m_apiKeys.push_back(apiKey);
        m_multipartParts.push_back(parts);
        return responseFor("POST", path);
    }

    [[nodiscard]] voxstudio::core::Expected<voxstudio::net::elevenlabs::HttpResponse,
                                            voxstudio::net::elevenlabs::ApiError>
    deleteRequest(const std::string& path, const std::string& apiKey) const override {
        m_deletePaths.push_back(path);
        m_apiKeys.push_back(apiKey);
        return responseFor("DELETE", path);
    }

    [[nodiscard]] const std::vector<std::string>& getPaths() const noexcept {
        return m_getPaths;
    }

    [[nodiscard]] const std::vector<std::string>& postPaths() const noexcept {
        return m_postPaths;
    }

    [[nodiscard]] const std::vector<std::string>& deletePaths() const noexcept {
        return m_deletePaths;
    }

    [[nodiscard]] const std::vector<std::string>& apiKeys() const noexcept {
        return m_apiKeys;
    }

    [[nodiscard]] const std::vector<std::vector<voxstudio::net::elevenlabs::MultipartPart>>&
    multipartParts() const noexcept {
        return m_multipartParts;
    }

private:
    [[nodiscard]] voxstudio::net::elevenlabs::HttpResponse
    responseFor(const std::string& method, const std::string& path) const {
        const auto methodResponse = m_responses.find(method + " " + path);
        if (methodResponse != m_responses.end()) {
            return voxstudio::net::elevenlabs::HttpResponse{200, methodResponse->second};
        }

        const auto response = m_responses.find(path);
        if (response == m_responses.end()) {
            return voxstudio::net::elevenlabs::HttpResponse{404, R"({"detail":"missing"})"};
        }

        return voxstudio::net::elevenlabs::HttpResponse{200, response->second};
    }

    std::map<std::string, std::string> m_responses;
    mutable std::vector<std::string> m_getPaths;
    mutable std::vector<std::string> m_postPaths;
    mutable std::vector<std::string> m_deletePaths;
    mutable std::vector<std::string> m_apiKeys;
    mutable std::vector<std::vector<voxstudio::net::elevenlabs::MultipartPart>> m_multipartParts;
};

[[nodiscard]] bool containsPart(const std::vector<voxstudio::net::elevenlabs::MultipartPart>& parts,
                                const std::string& name,
                                const std::string& value) {
    return std::ranges::any_of(parts, [&](const auto& part) {
        return part.name == name && part.value == value && !part.isFile;
    });
}

} // namespace

TEST_CASE("Voices API lists all paginated voices", "[net][elevenlabs][voices]") {
    auto transport = std::make_unique<FakeVoiceTransport>(std::map<std::string, std::string>{
        {"GET /v2/voices?page_size=100&include_total_count=true",
         loadFixtureText("elevenlabs/voices_search_page_1.json")},
        {"GET /v2/voices?page_size=100&include_total_count=true&next_page_token=page-2",
         loadFixtureText("elevenlabs/voices_search_page_2.json")}});
    const auto* transportView = transport.get();

    const voxstudio::net::elevenlabs::VoicesApi api{"test-key", std::move(transport)};
    auto voices = api.listVoices();

    REQUIRE(voices.hasValue());
    REQUIRE(voices.value().voices.size() == 2);
    CHECK(voices.value().voices.at(0).voiceId == "voice_001");
    CHECK(voices.value().voices.at(1).voiceId == "voice_002");
    CHECK(voices.value().totalCount == 2);
    CHECK_FALSE(voices.value().hasMore);
    CHECK(transportView->getPaths() ==
          std::vector<std::string>{
              "/v2/voices?page_size=100&include_total_count=true",
              "/v2/voices?page_size=100&include_total_count=true&next_page_token=page-2"});
}

TEST_CASE("Voices API supports get add edit and delete", "[net][elevenlabs][voices]") {
    auto transport = std::make_unique<FakeVoiceTransport>(std::map<std::string, std::string>{
        {"GET /v1/voices/voice_new", loadFixtureText("elevenlabs/voice_detail.json")},
        {"POST /v1/voices/add", loadFixtureText("elevenlabs/voice_created.json")},
        {"POST /v1/voices/voice_new/edit", loadFixtureText("elevenlabs/status_ok.json")},
        {"DELETE /v1/voices/voice_new", loadFixtureText("elevenlabs/status_ok.json")}});
    const auto* transportView = transport.get();

    const voxstudio::net::elevenlabs::VoicesApi api{"test-key", std::move(transport)};

    auto voice = api.getVoice("voice_new");
    REQUIRE(voice.hasValue());
    CHECK(voice.value().previewUrl == "https://example.test/voice_new.mp3");
    CHECK(voice.value().labels.at("age") == "adult");

    voxstudio::net::elevenlabs::VoiceCloneRequest cloneRequest;
    cloneRequest.name = "Clone Hero";
    cloneRequest.description = "A local test clone.";
    cloneRequest.labels = {{"accent", "neutral"}};
    cloneRequest.sampleFiles = {std::filesystem::path{"sample.wav"}};
    cloneRequest.removeBackgroundNoise = true;
    auto created = api.addVoice(cloneRequest);
    REQUIRE(created.hasValue());
    CHECK(created.value().voiceId == "voice_new");

    voxstudio::net::elevenlabs::VoiceEditRequest editRequest;
    editRequest.name = "Clone Hero";
    editRequest.description = "Updated description.";
    auto edited = api.editVoice("voice_new", editRequest);
    REQUIRE(edited.hasValue());
    CHECK(edited.value().status == "ok");

    auto deleted = api.deleteVoice("voice_new");
    REQUIRE(deleted.hasValue());
    CHECK(deleted.value().status == "ok");

    CHECK(transportView->postPaths() ==
          std::vector<std::string>{"/v1/voices/add", "/v1/voices/voice_new/edit"});
    REQUIRE(transportView->multipartParts().size() == 2);
    CHECK(containsPart(transportView->multipartParts().at(0), "name", "Clone Hero"));
    CHECK(containsPart(transportView->multipartParts().at(0), "remove_background_noise", "true"));
    CHECK(transportView->multipartParts().at(0).back().name == "files[]");
    CHECK(transportView->multipartParts().at(0).back().isFile);
}
