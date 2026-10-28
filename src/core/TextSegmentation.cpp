#include "core/TextSegmentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>

namespace voxstudio::core {
namespace {

[[nodiscard]] std::vector<std::string> wordsIn(const std::string_view text) {
    std::vector<std::string> words;
    std::size_t start = 0;
    while (start < text.size()) {
        while (start < text.size() &&
               std::isspace(static_cast<unsigned char>(text[start])) != 0) {
            ++start;
        }
        if (start == text.size()) {
            break;
        }
        auto end = start;
        while (end < text.size() &&
               std::isspace(static_cast<unsigned char>(text[end])) == 0) {
            ++end;
        }
        words.emplace_back(text.substr(start, end - start));
        start = end;
    }
    return words;
}

[[nodiscard]] bool endsSentence(const std::string& word) noexcept {
    return !word.empty() &&
           (word.back() == '.' || word.back() == '?' || word.back() == '!');
}

[[nodiscard]] bool endsClause(const std::string& word) noexcept {
    return !word.empty() &&
           (word.back() == ',' || word.back() == ';' || word.back() == ':');
}

[[nodiscard]] std::size_t joinedLength(const std::vector<std::string>& words,
                                       const std::size_t begin,
                                       const std::size_t end) noexcept {
    std::size_t length = end > begin ? end - begin - 1U : 0U;
    for (auto index = begin; index < end; ++index) {
        length += words[index].size();
    }
    return length;
}

[[nodiscard]] std::string joinWords(const std::vector<std::string>& words,
                                    const std::size_t begin,
                                    const std::size_t end) {
    std::string result;
    result.reserve(joinedLength(words, begin, end));
    for (auto index = begin; index < end; ++index) {
        if (!result.empty()) {
            result.push_back(' ');
        }
        result += words[index];
    }
    return result;
}

} // namespace

std::vector<std::string>
splitTextByWeights(const std::string_view text,
                   const std::span<const std::size_t> weights) {
    if (weights.empty()) {
        return {};
    }

    const auto words = wordsIn(text);
    std::vector<std::string> result(weights.size());
    if (words.empty()) {
        return result;
    }
    if (weights.size() == 1U) {
        result.front() = joinWords(words, 0, words.size());
        return result;
    }

    std::size_t wordStart = 0;
    for (std::size_t section = 0; section + 1U < weights.size(); ++section) {
        const auto sectionsAfter = weights.size() - section - 1U;
        if (wordStart >= words.size() ||
            words.size() - wordStart <= sectionsAfter) {
            break;
        }

        const auto remainingWeight =
            std::accumulate(weights.begin() + static_cast<std::ptrdiff_t>(section),
                            weights.end(),
                            std::size_t{0});
        const auto remainingCharacters = joinedLength(words, wordStart, words.size());
        const auto targetCharacters =
            remainingWeight == 0U
                ? static_cast<double>(remainingCharacters) /
                      static_cast<double>(weights.size() - section)
                : static_cast<double>(remainingCharacters) *
                      (static_cast<double>(weights[section]) /
                       static_cast<double>(remainingWeight));

        const auto maximumEnd = words.size() - sectionsAfter;
        auto bestEnd = wordStart + 1U;
        auto bestScore = std::numeric_limits<double>::max();
        for (auto candidateEnd = wordStart + 1U;
             candidateEnd <= maximumEnd;
             ++candidateEnd) {
            const auto candidateCharacters =
                static_cast<double>(joinedLength(words, wordStart, candidateEnd));
            auto score = std::abs(candidateCharacters - targetCharacters);
            const auto& finalWord = words[candidateEnd - 1U];
            if (endsSentence(finalWord)) {
                score -= std::max(targetCharacters * 0.4, 8.0);
            } else if (endsClause(finalWord)) {
                score -= std::max(targetCharacters * 0.2, 4.0);
            }
            if (score < bestScore) {
                bestScore = score;
                bestEnd = candidateEnd;
            }
        }

        result[section] = joinWords(words, wordStart, bestEnd);
        wordStart = bestEnd;
    }
    if (wordStart < words.size()) {
        result.back() = joinWords(words, wordStart, words.size());
    }
    return result;
}

} // namespace voxstudio::core
