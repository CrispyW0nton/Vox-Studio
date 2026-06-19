#pragma once

#include "db/TakeRepository.h"

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

    void setTakes(std::vector<db::TakeRecord> takes);
    [[nodiscard]] std::vector<db::TakeRecord> takes() const;

signals:
    void playTakeRequested(db::TakeRecord take);
    void starTakeRequested(db::TakeRecord take);
    void deleteTakeRequested(db::TakeRecord take);

private:
    void playSelectedTake();
    void starSelectedTake();
    void deleteSelectedTake();
    [[nodiscard]] db::TakeRecord* selectedTake();
    [[nodiscard]] const db::TakeRecord* selectedTake() const;
    void refreshList();

    std::vector<db::TakeRecord> m_takes;
    QListWidget* m_takeList{nullptr};
    QLabel* m_statusLabel{nullptr};
    QPushButton* m_playButton{nullptr};
    QPushButton* m_starButton{nullptr};
    QPushButton* m_deleteButton{nullptr};
};

} // namespace voxstudio::ui
