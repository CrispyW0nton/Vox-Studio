#include "core/TextSegmentation.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>

TEST_CASE("text segmentation follows weighted natural boundaries",
          "[core][text][monologue]") {
    const std::array<std::size_t, 3> weights{10, 30, 20};

    const auto sections = voxstudio::core::splitTextByWeights(
        "Wait here. I need to search the old station before nightfall. Then we leave.",
        weights);

    REQUIRE(sections.size() == weights.size());
    CHECK(sections[0] == "Wait here.");
    CHECK(sections[1] == "I need to search the old station before nightfall.");
    CHECK(sections[2] == "Then we leave.");
}

TEST_CASE("text segmentation keeps every word when sections outnumber sentences",
          "[core][text][monologue]") {
    const std::array<std::size_t, 2> weights{1, 1};

    const auto sections =
        voxstudio::core::splitTextByWeights("One continuous thought without punctuation", weights);

    REQUIRE(sections.size() == weights.size());
    CHECK(sections[0] + " " + sections[1] ==
          "One continuous thought without punctuation");
}
