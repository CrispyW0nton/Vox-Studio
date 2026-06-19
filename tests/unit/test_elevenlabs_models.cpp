#include "net/elevenlabs/Models.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

namespace {

[[nodiscard]] nlohmann::json loadFixture(const std::filesystem::path& relativePath) {
    const std::filesystem::path fixtureRoot{VOXSTUDIO_TEST_FIXTURE_DIR};
    std::ifstream input{fixtureRoot / relativePath};
    REQUIRE(input.good());
    return nlohmann::json::parse(input);
}

} // namespace

TEST_CASE("ElevenLabs user and subscription fixtures deserialize", "[net][elevenlabs]") {
    const auto userJson = loadFixture("elevenlabs/user.json");
    const auto user = voxstudio::net::elevenlabs::parseUserInfo(userJson);
    REQUIRE(user.hasValue());
    CHECK(user.value().userId == "user_123");
    CHECK(user.value().email == "dev@example.test");
    CHECK(user.value().firstName == "Dev");

    const auto subscriptionJson = loadFixture("elevenlabs/subscription.json");
    const auto subscription =
        voxstudio::net::elevenlabs::parseSubscriptionInfo(subscriptionJson);
    REQUIRE(subscription.hasValue());
    CHECK(subscription.value().tier == "creator");
    CHECK(subscription.value().characterCount == 1234);
    CHECK(subscription.value().characterLimit == 100000);
    CHECK(subscription.value().canExtendCharacterLimit);
}

TEST_CASE("ElevenLabs voices and models fixtures deserialize", "[net][elevenlabs]") {
    const auto voicesJson = loadFixture("elevenlabs/voices.json");
    const auto voices = voxstudio::net::elevenlabs::parseVoicesResponse(voicesJson);
    REQUIRE(voices.hasValue());
    REQUIRE(voices.value().voices.size() == 2);
    CHECK(voices.value().voices.at(0).voiceId == "voice_001");
    CHECK(voices.value().voices.at(0).name == "Narrator");
    CHECK(voices.value().voices.at(0).labels.at("accent") == "neutral");

    const auto modelsJson = loadFixture("elevenlabs/models.json");
    const auto models = voxstudio::net::elevenlabs::parseModelsResponse(modelsJson);
    REQUIRE(models.hasValue());
    REQUIRE(models.value().models.size() == 2);
    CHECK(models.value().models.at(0).modelId == "eleven_multilingual_v2");
    CHECK(models.value().models.at(0).canDoTextToSpeech);
    CHECK(models.value().models.at(0).canDoVoiceConversion);
}

