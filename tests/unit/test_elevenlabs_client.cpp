#include "net/elevenlabs/Client.h"

#include <catch2/catch_test_macros.hpp>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

class FakeTransport final : public voxstudio::net::elevenlabs::IHttpTransport {
public:
    explicit FakeTransport(std::map<std::string, std::string> responses)
        : m_responses(std::move(responses)) {}

    [[nodiscard]] voxstudio::core::Expected<voxstudio::net::elevenlabs::HttpResponse,
                                            voxstudio::net::elevenlabs::ApiError>
    get(const std::string& path, const std::string& apiKey) const override {
        m_paths.push_back(path);
        m_apiKeys.push_back(apiKey);

        const auto response = m_responses.find(path);
        if (response == m_responses.end()) {
            return voxstudio::net::elevenlabs::HttpResponse{404, R"({"detail":"missing"})"};
        }

        return voxstudio::net::elevenlabs::HttpResponse{200, response->second};
    }

    [[nodiscard]] const std::vector<std::string>& paths() const noexcept {
        return m_paths;
    }

    [[nodiscard]] const std::vector<std::string>& apiKeys() const noexcept {
        return m_apiKeys;
    }

private:
    std::map<std::string, std::string> m_responses;
    mutable std::vector<std::string> m_paths;
    mutable std::vector<std::string> m_apiKeys;
};

} // namespace

TEST_CASE("ElevenLabs client calls typed endpoints through transport", "[net][elevenlabs]") {
    auto transport = std::make_unique<FakeTransport>(std::map<std::string, std::string>{
        {"/v1/user", R"({"user_id":"user_123","email":"dev@example.test"})"},
        {"/v1/user/subscription",
         R"({"tier":"creator","character_count":10,"character_limit":100})"},
        {"/v1/voices", R"({"voices":[{"voice_id":"voice_001","name":"Narrator"}]})"},
        {"/v1/models",
         R"([{"model_id":"model_001","name":"Model","can_do_text_to_speech":true}])"}});
    const auto* transportView = transport.get();

    const voxstudio::net::elevenlabs::Client client{"test-key", std::move(transport)};

    auto user = client.getUser();
    auto subscription = client.getSubscription();
    auto voices = client.getVoices();
    auto models = client.getModels();

    REQUIRE(user.hasValue());
    REQUIRE(subscription.hasValue());
    REQUIRE(voices.hasValue());
    REQUIRE(models.hasValue());
    CHECK(user.value().email == "dev@example.test");
    CHECK(subscription.value().tier == "creator");
    CHECK(voices.value().voices.at(0).voiceId == "voice_001");
    CHECK(models.value().models.at(0).modelId == "model_001");

    CHECK(transportView->paths() ==
          std::vector<std::string>{"/v1/user", "/v1/user/subscription", "/v1/voices",
                                   "/v1/models"});
    CHECK(transportView->apiKeys() ==
          std::vector<std::string>{"test-key", "test-key", "test-key", "test-key"});
}

TEST_CASE("ElevenLabs client reports missing API keys", "[net][elevenlabs]") {
    const voxstudio::net::elevenlabs::Client client{""};

    auto user = client.getUser();

    REQUIRE(!user.hasValue());
    CHECK(user.error().code == voxstudio::net::elevenlabs::ApiErrorCode::MissingApiKey);
}
