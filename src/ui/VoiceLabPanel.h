#pragma once

#include <QByteArray>
#include <QJsonArray>
#include <QProcess>
#include <QWidget>

class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QTreeWidget;

namespace voxstudio::ui {

class VoiceLabPanel final : public QWidget {
    Q_OBJECT
public:
    explicit VoiceLabPanel(QWidget* parent = nullptr);
    [[nodiscard]] bool isRunning() const;
    void setInventory(const QJsonObject& inventory);

protected:
    void showEvent(QShowEvent* event) override;

private:
    void refresh();
    void showSelection();
    void selectPreset();
    void invalidatePreview();
    void start(const QString& action);
    void finished(int code, QProcess::ExitStatus status);
    void setBusy(bool busy);
    void openPath(const QString& path);
    QProcess* m_process;
    QTreeWidget* m_tree;
    QPlainTextEdit* m_details;
    QListWidget* m_links;
    QComboBox* m_presets;
    QLineEdit* m_script;
    QLineEdit* m_checkpoint;
    QLineEdit* m_lines;
    QSpinBox* m_seeds;
    QPlainTextEdit* m_log;
    QLabel* m_note;
    QPushButton* m_refresh;
    QPushButton* m_preview;
    QPushButton* m_render;
    QPushButton* m_stop;
    QPushButton* m_browse;
    QPushButton* m_import;
    QJsonArray m_presetData;
    QByteArray m_output;
    QString m_action;
    QString m_jobFile;
    QString m_revision;
    bool m_inventoryLoaded{false};
};

} // namespace voxstudio::ui
