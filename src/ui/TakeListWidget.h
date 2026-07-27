#pragma once

#include "db/TakeRepository.h"

#include <QString>
#include <QWidget>

#include <vector>

class QLabel;
class QListWidget;
class QPushButton;

namespace voxstudio::ui {

class TakeListWidget final : public QWidget {
    Q_OBJECT

public:
    explicit TakeListWidget(QWidget* parent = nullptr);

    void setTitle(const QString& title);
    void setTakes(std::vector<db::TakeRecord> takes);
    void setBatchExportEnabled(bool enabled);
    [[nodiscard]] std::vector<db::TakeRecord> takes() const;
    [[nodiscard]] std::vector<db::TakeRecord> selectedTakes() const;

signals:
    void playTakeRequested(db::TakeRecord take);
    void starTakeRequested(db::TakeRecord take);
    void revealTakeRequested(db::TakeRecord take);
    void deleteTakeRequested(db::TakeRecord take);
    void exportTakesRequested(std::vector<db::TakeRecord> takes);

private:
    void playSelectedTake();
    void starSelectedTake();
    void revealSelectedTake();
    void deleteSelectedTake();
    void exportSelectedTakes();
    void updateActionState();
    [[nodiscard]] db::TakeRecord* selectedTake();
    [[nodiscard]] const db::TakeRecord* selectedTake() const;
    void refreshList();

    std::vector<db::TakeRecord> m_takes;
    bool m_batchExportEnabled{false};
    QListWidget* m_takeList{nullptr};
    QLabel* m_titleLabel{nullptr};
    QLabel* m_statusLabel{nullptr};
    QPushButton* m_playButton{nullptr};
    QPushButton* m_starButton{nullptr};
    QPushButton* m_revealButton{nullptr};
    QPushButton* m_deleteButton{nullptr};
    QPushButton* m_selectAllButton{nullptr};
    QPushButton* m_exportButton{nullptr};
};

} // namespace voxstudio::ui
