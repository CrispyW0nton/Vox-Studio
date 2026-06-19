#include "rvc/RvcClient.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

class FakeRvcTransport final : public voxstudio::rvc::IRvcHttpTransport {
public:
    explicit FakeRvcTransport(std::string healthBody,
                              std::vector<std::vector<std::uint8_t>> chunks)
        : m_healthBody(std::move(healthBody))
        , m_chunks(std::move(chunks)) {}

    [[nodiscard]] voxstudio::core::Expected<voxstudio::rvc::RvcHttpResponse> getJson(
        const std::string& path) const override {
        m_healthPath = path;
        return voxstudio::rvc::RvcHttpResponse{200, m_healthBody};
    }

    [[nodiscard]] voxstudio::core::Expected<voxstudio::rvc::RvcHttpResponse> postPcmStream(
        const std::string& path,
        const voxstudio::rvc::RvcConvertRequest& request,
        const voxstudio::rvc::RvcAudioChunkCallback& onChunk) const override {
        m_convertPath = path;
        m_request = request;
        for (const auto& chunk : m_chunks) {
            if (!onChunk(std::span<const std::uint8_t>{chunk.data(), chunk.size()})) {
                return voxstudio::core::makeError(voxstudio::core::ErrorCode::FileSystemFailure,
                                                  "callback stopped");
            }
        }
        return voxstudio::rvc::RvcHttpResponse{200, {}};
    }

    [[nodiscard]] const std::string& healthPath() const noexcept {
        return m_healthPath;
    }

    [[nodiscard]] const std::string& convertPath() const noexcept {
        return m_convertPath;
    }

    [[nodiscard]] const voxstudio::rvc::RvcConvertRequest& request() const noexcept {
        return m_request;
    }

private:
    std::string m_healthBody;
    std::vector<std::vector<std::uint8_t>> m_chunks;
    mutable std::string m_healthPath;
    mutable std::string m_convertPath;
    mutable voxstudio::rvc::RvcConvertRequest m_request;
};

} // namespace

TEST_CASE("RVC client parses health diagnostics", "[rvc][client]") {
    auto transport = std::make_unique<FakeRvcTransport>(
        R"({"ok":true,"engine":"w-okada","cuda_available":true,)"
        R"("cuda_version":"12.1","loaded_model_id":"hero","last_latency_ms":142})",
        std::vector<std::vector<std::uint8_t>>{});
    const auto* transportView = transport.get();

    const voxstudio::rvc::RvcClient client{"http://127.0.0.1:18888", std::move(transport)};
    auto health = client.health();

    REQUIRE(health.hasValue());
    CHECK(health.value().ok);
    CHECK(health.value().engine == "w-okada");
    CHECK(health.value().cudaAvailable);
    CHECK(health.value().cudaVersion == "12.1");
    CHECK(health.value().loadedModelId == "hero");
    CHECK(health.value().lastLatencyMs == 142);
    CHECK(transportView->healthPath() == "/health");
}

TEST_CASE("RVC client streams converted PCM chunks", "[rvc][client]") {
    auto transport = std::make_unique<FakeRvcTransport>(
        R"({"ok":true})",
        std::vector<std::vector<std::uint8_t>>{{0x01, 0x00}, {0x02, 0x00}});
    const auto* transportView = transport.get();

    voxstudio::rvc::RvcConvertRequest request;
    request.modelId = "hero_model";
    request.pcm16Audio = {0x10, 0x00, 0x20, 0x00};
    request.sampleRate = 48000;

    const voxstudio::rvc::RvcClient client{"http://127.0.0.1:18888", std::move(transport)};
    std::vector<std::uint8_t> callbackBytes;
    auto converted = client.convertChunk(
        request,
        [&callbackBytes](std::span<const std::uint8_t> chunk) {
            callbackBytes.insert(callbackBytes.end(), chunk.begin(), chunk.end());
            return true;
        });

    REQUIRE(converted.hasValue());
    CHECK(converted.value().pcm16Audio == std::vector<std::uint8_t>{0x01, 0x00, 0x02, 0x00});
    CHECK(callbackBytes == converted.value().pcm16Audio);
    CHECK(transportView->convertPath() == "/convert_chunk");
    CHECK(transportView->request().modelId == "hero_model");
}

TEST_CASE("RVC client validates conversion requests", "[rvc][client]") {
    const voxstudio::rvc::RvcClient client{
        "http://127.0.0.1:18888",
        std::make_unique<FakeRvcTransport>(R"({"ok":true})",
                                           std::vector<std::vector<std::uint8_t>>{})};

    voxstudio::rvc::RvcConvertRequest request;
    request.pcm16Audio = {0x01, 0x00};
    auto converted = client.convertChunk(request, {});
    REQUIRE_FALSE(converted.hasValue());
}
