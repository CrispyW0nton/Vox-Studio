#pragma once

#include "db/ScriptRepository.h"

#include <QObject>

#include <optional>
#include <vector>

class QKeyEvent;

namespace voxstudio::ui {

struct CharacterHotkeySlot final {
    int number{0};
    db::CharacterRecord character;
};

class HotkeyManager final : public QObject {
    Q_OBJECT

public:
    explicit HotkeyManager(QObject* parent = nullptr);

    void setCharacters(std::vector<db::CharacterRecord> characters);
    [[nodiscard]] std::vector<CharacterHotkeySlot> hotkeySlots() const;
    [[nodiscard]] std::optional<CharacterHotkeySlot> activeSlot() const;
    [[nodiscard]] bool handleKeyPress(const QKeyEvent& event);

signals:
    void activeCharacterChanged(CharacterHotkeySlot slot);

private:
    std::vector<CharacterHotkeySlot> m_slots;
    std::optional<CharacterHotkeySlot> m_activeSlot;
};

} // namespace voxstudio::ui
