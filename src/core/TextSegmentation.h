#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace voxstudio::core {

[[nodiscard]] std::vector<std::string>
splitTextByWeights(std::string_view text, std::span<const std::size_t> weights);

} // namespace voxstudio::core
