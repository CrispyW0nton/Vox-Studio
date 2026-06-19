#pragma once

#include "db/ScriptRepository.h"
#include "db/VoiceRepository.h"

#include <QDialog>

#include <string>
#include <vector>

class QTableWidget;

namespace voxstudio::ui {

class CharacterAssignmentDialog final : public QDialog {
    Q_OBJECT

public:
    CharacterAssignmentDialog(const std::vector<std::string>& speakers,
                              const std::vector<db::VoiceRecord>& voices,
                              QWidget* parent = nullptr);

    [[nodiscard]] std::vector<db::CharacterAssignment> assignments() const;

private:
    QTableWidget* m_table{nullptr};
    std::vector<std::string> m_speakers;
    std::vector<db::VoiceRecord> m_voices;
};

} // namespace voxstudio::ui
