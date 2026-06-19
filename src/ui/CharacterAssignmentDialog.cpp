#include "ui/CharacterAssignmentDialog.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QHeaderView>
#include <QLabel>
#include <QTableWidget>
#include <QVBoxLayout>

#include <memory>

namespace voxstudio::ui {

CharacterAssignmentDialog::CharacterAssignmentDialog(const std::vector<std::string>& speakers,
                                                     const std::vector<db::VoiceRecord>& voices,
                                                     QWidget* parent)
    : QDialog(parent)
    , m_speakers(speakers)
    , m_voices(voices) {
    setWindowTitle(QStringLiteral("Character Assignments"));
    setModal(true);

    auto rootLayout = std::make_unique<QVBoxLayout>();
    auto title = std::make_unique<QLabel>(QStringLiteral("Detected Speakers"));
    title->setObjectName(QStringLiteral("CharacterAssignmentTitle"));
    rootLayout->addWidget(title.release());

    auto table = std::make_unique<QTableWidget>(static_cast<int>(m_speakers.size()), 2);
    m_table = table.get();
    m_table->setHorizontalHeaderLabels({QStringLiteral("Speaker"), QStringLiteral("Voice")});
    m_table->horizontalHeader()->setStretchLastSection(true);
    m_table->verticalHeader()->setVisible(false);
    m_table->setSelectionMode(QAbstractItemView::NoSelection);

    for (int row = 0; row < static_cast<int>(m_speakers.size()); ++row) {
        auto speakerItem =
            std::make_unique<QTableWidgetItem>(QString::fromStdString(m_speakers[row]));
        speakerItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
        m_table->setItem(row, 0, speakerItem.release());

        auto voiceCombo = std::make_unique<QComboBox>();
        voiceCombo->addItem(QStringLiteral("No voice"), QString{});
        for (const auto& voice : m_voices) {
            voiceCombo->addItem(QString::fromStdString(voice.name),
                                QString::fromStdString(voice.id));
        }
        m_table->setCellWidget(row, 1, voiceCombo.release());
    }

    rootLayout->addWidget(table.release());

    auto buttons =
        std::make_unique<QDialogButtonBox>(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons.get(), &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons.get(), &QDialogButtonBox::rejected, this, &QDialog::reject);
    rootLayout->addWidget(buttons.release());

    setLayout(rootLayout.release());
    resize(520, 360);
}

std::vector<db::CharacterAssignment> CharacterAssignmentDialog::assignments() const {
    std::vector<db::CharacterAssignment> result;
    result.reserve(m_speakers.size());
    for (int row = 0; row < static_cast<int>(m_speakers.size()); ++row) {
        const auto* combo = qobject_cast<QComboBox*>(m_table->cellWidget(row, 1));
        result.push_back(db::CharacterAssignment{
            m_speakers[static_cast<std::size_t>(row)],
            combo == nullptr ? std::string{} : combo->currentData().toString().toStdString()});
    }
    return result;
}

} // namespace voxstudio::ui
