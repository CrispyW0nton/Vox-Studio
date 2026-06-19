#pragma once

#include "audio/AudioPreview.h"
#include "core/Project.h"
#include "db/VoiceRepository.h"
#include "net/elevenlabs/Models.h"
#include "net/elevenlabs/VoicesApi.h"

#include <QFutureWatcher>
#include <QWidget>
#include <QString>

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

class QLabel;
class QListWidget;
class QPushButton;

namespace voxstudio::ui {

enum class VoiceLibraryTaskKind {
    Refresh,
    Clone,
    Edit,
    Delete,
};

struct VoiceLibraryTaskResult final {
    VoiceLibraryTaskKind kind{VoiceLibraryTaskKind::Refresh};
    bool success{false};
    QString message;
    std::vector<net::elevenlabs::VoiceInfo> voices;
};

struct VoicePreviewTaskResult final {
    bool success{false};
    QString message;
    std::filesystem::path filePath;
};

class VoiceLibraryPanel final : public QWidget {
    Q_OBJECT

public:
    explicit VoiceLibraryPanel(QWidget* parent = nullptr);

    void setProject(std::optional<core::Project> project);

signals:
    void voiceCacheChanged();

private:
    void refreshVoices();
    void cloneVoice();
    void editSelectedVoice();
    void deleteSelectedVoice();
    void previewSelectedVoice();
    void finishVoiceTask();
    void finishPreviewTask();
    void setBusy(bool isBusy);
    void setVoices(std::vector<net::elevenlabs::VoiceInfo> voices);
    void loadCachedVoices();
    [[nodiscard]] std::optional<net::elevenlabs::VoiceInfo> selectedVoice() const;

    db::VoiceRepository m_voiceRepository;
    std::optional<core::Project> m_project;
    std::vector<net::elevenlabs::VoiceInfo> m_voices;
    QListWidget* m_voiceList{nullptr};
    QLabel* m_statusLabel{nullptr};
    QPushButton* m_refreshButton{nullptr};
    QPushButton* m_cloneButton{nullptr};
    QPushButton* m_previewButton{nullptr};
    QPushButton* m_editButton{nullptr};
    QPushButton* m_deleteButton{nullptr};
    std::unique_ptr<QFutureWatcher<VoiceLibraryTaskResult>> m_voiceTaskWatcher;
    std::unique_ptr<QFutureWatcher<VoicePreviewTaskResult>> m_previewTaskWatcher;
    audio::AudioPreviewPlayer m_previewPlayer;
};

} // namespace voxstudio::ui
