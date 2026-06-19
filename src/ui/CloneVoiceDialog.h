#pragma once

#include "audio/AudioPreview.h"

#include <QDialog>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

class QCheckBox;
class QDialogButtonBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QTextEdit;

namespace voxstudio::ui {

struct CloneVoiceDialogResult final {
    std::string name;
    std::string description;
    std::map<std::string, std::string> labels;
    std::vector<std::filesystem::path> sampleFiles;
    bool removeBackgroundNoise{false};
};

class CloneVoiceDialog final : public QDialog {
    Q_OBJECT

public:
    explicit CloneVoiceDialog(QWidget* parent = nullptr);

    [[nodiscard]] CloneVoiceDialogResult result() const;

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    void browseFiles();
    void removeSelectedFiles();
    void previewSelectedFile();
    void updateValidation();
    void addFiles(const std::vector<std::filesystem::path>& files);
    [[nodiscard]] std::map<std::string, std::string> parsedLabels() const;
    [[nodiscard]] std::vector<std::filesystem::path> sampleFiles() const;

    QLineEdit* m_nameEdit{nullptr};
    QTextEdit* m_descriptionEdit{nullptr};
    QLineEdit* m_labelsEdit{nullptr};
    QListWidget* m_filesList{nullptr};
    QLabel* m_statusLabel{nullptr};
    QCheckBox* m_removeNoiseCheck{nullptr};
    QCheckBox* m_consentCheck{nullptr};
    QPushButton* m_cloneButton{nullptr};
    audio::AudioPreviewPlayer m_previewPlayer;
    double m_totalDurationSeconds{0.0};
};

} // namespace voxstudio::ui
