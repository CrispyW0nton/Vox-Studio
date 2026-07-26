#include "audio/SpeechPhraseBuffer.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace {

[[nodiscard]] std::vector<std::uint8_t> bytes(const std::size_t count,
                                              const std::uint8_t value) {
    return std::vector<std::uint8_t>(count, value);
}

} // namespace

TEST_CASE("speech phrase buffer emits after a natural pause", "[audio][phrases]") {
    voxstudio::audio::SpeechPhraseBuffer buffer{{
        .preRollBytes = 4,
        .trailingSilenceBytes = 6,
        .retainedTrailingSilenceBytes = 2,
        .minimumSpeechBytes = 4,
        .maximumPhraseBytes = 100,
    }};

    CHECK_FALSE(buffer.append(bytes(4, 1), false).has_value());
    CHECK_FALSE(buffer.append(bytes(6, 2), true).has_value());
    auto phrase = buffer.append(bytes(6, 0), false);

    REQUIRE(phrase.has_value());
    REQUIRE(phrase->size() == 12);
    CHECK((*phrase)[0] == 1);
    CHECK((*phrase)[4] == 2);
    CHECK((*phrase)[10] == 0);
}

TEST_CASE("speech phrase buffer drops short noise and splits long performances",
          "[audio][phrases]") {
    voxstudio::audio::SpeechPhraseBuffer buffer{{
        .preRollBytes = 0,
        .trailingSilenceBytes = 4,
        .retainedTrailingSilenceBytes = 0,
        .minimumSpeechBytes = 4,
        .maximumPhraseBytes = 8,
    }};

    CHECK_FALSE(buffer.append(bytes(2, 9), true).has_value());
    CHECK_FALSE(buffer.append(bytes(4, 0), false).has_value());

    CHECK_FALSE(buffer.append(bytes(4, 7), true).has_value());
    auto phrase = buffer.append(bytes(4, 8), true);
    REQUIRE(phrase.has_value());
    CHECK(phrase->size() == 8);
}
