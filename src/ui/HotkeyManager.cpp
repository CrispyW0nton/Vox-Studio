#include "ui/HotkeyManager.h"

#include <QKeyEvent>

#include <algorithm>

namespace voxstudio::ui {

HotkeyManager::HotkeyManager(QObject* parent)
    : QObject(parent) {}

void HotkeyManager::setCharacters(std::vector<db::CharacterRecord> characters) {
    std::ranges::sort(characters, {}, &db::CharacterRecord::name);

    m_slots.clear();
    const auto slotCount = characters.size() < 9U ? characters.size() : 9U;
    m_slots.reserve(slotCount);
    for (std::size_t index = 0; index < slotCount; ++index) {
        m_slots.push_back(
            CharacterHotkeySlot{static_cast<int>(index + 1U), characters[index]});
    }

    m_activeSlot.reset();
    if (!m_slots.empty()) {
        m_activeSlot = m_slots.front();
        emit activeCharacterChanged(m_activeSlot.value());
    }
}

std::vector<CharacterHotkeySlot> HotkeyManager::hotkeySlots() const {
    return m_slots;
}

std::optional<CharacterHotkeySlot> HotkeyManager::activeSlot() const {
    return m_activeSlot;
}

bool HotkeyManager::handleKeyPress(const QKeyEvent& event) {
    const int key = event.key();
    if (key < Qt::Key_1 || key > Qt::Key_9) {
        return false;
    }

    const int slotNumber = key - Qt::Key_0;
    const auto slot = std::ranges::find_if(m_slots, [slotNumber](const auto& candidate) {
        return candidate.number == slotNumber;
    });
    if (slot == m_slots.end()) {
        return false;
    }

    m_activeSlot = *slot;
    emit activeCharacterChanged(*slot);
    return true;
}

} // namespace voxstudio::ui
