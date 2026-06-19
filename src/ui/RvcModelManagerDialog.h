#pragma once

#include "core/Project.h"
#include "db/ScriptRepository.h"
#include "rvc/RvcModelRegistry.h"

#include <QDialog>

#include <optional>
#include <vector>

class QComboBox;
class QLabel;
class QListWidget;
class QPushButton;

namespace voxstudio::ui {

class RvcModelManagerDialog final : public QDialog {
    Q_OBJECT

public:
    explicit RvcModelManagerDialog(QWidget* parent = nullptr);
    RvcModelManagerDialog(std::optional<core::Project> project, QWidget* parent = nullptr);

signals:
    void rvcAssignmentsChanged();

private:
    void refresh();
    void importModel();
    void deleteSelectedModel();
    void assignSelectedModel();
    void setStatusText(const QString& text);
    [[nodiscard]] std::string selectedModelId() const;
    [[nodiscard]] std::string selectedCharacterId() const;

    rvc::RvcModelRegistry m_registry;
    db::ScriptRepository m_scriptRepository;
    std::optional<core::Project> m_project;
    std::vector<rvc::RvcModelRecord> m_models;
    std::vector<db::CharacterRecord> m_characters;
    QListWidget* m_modelList{nullptr};
    QComboBox* m_characterCombo{nullptr};
    QLabel* m_statusLabel{nullptr};
    QPushButton* m_importButton{nullptr};
    QPushButton* m_deleteButton{nullptr};
    QPushButton* m_assignButton{nullptr};
    QPushButton* m_refreshButton{nullptr};
};

} // namespace voxstudio::ui
