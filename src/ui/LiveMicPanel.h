#pragma once

#include "audio/AudioEngine.h"
#include "audio/Capture.h"
#include "audio/LatencyProbe.h"
#include "core/Project.h"
#include "db/ScriptRepository.h"
#include "db/TakeRepository.h"
#include "db/VoiceRepository.h"
#include "rvc/RvcModelRegistry.h"
#include "rvc/RvcSidecar.h"
#include "voxcpm/VoxCpmSidecar.h"

#include <QByteArray>
#include <QFutureWatcher>
#include <QString>
#include <QThread>
#include <QWidget>

#include <atomic>
#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSlider;
class QSpinBox;

namespace voxstudio::rvc {
class OnnxRvcEngine;
}

namespace voxstudio::ui {

class LiveAudioProcessor;
class TakeListWidget;

struct CloudConversionResult final {
    bool success{false};
    QString message;
    QString playbackWarning;
    QByteArray convertedPcmBytes;
    double inputSeconds{0.0};
    int sampleRate{24000};
    int playbackDurationMs{0};
    QString delivery;
};

struct LocalRvcConversionResult final {
    bool success{false};
    QString message;
    QString playbackWarning;
    QByteArray convertedPcmBytes;
    int latencyMs{0};
    int sampleRate{48000};
    int channels{1};
};

struct PlaybackTargets final {
    audio::AudioEngine* monitor{nullptr};
    audio::AudioEngine* broadcast{nullptr};
    double durationScale{1.0};
};

class LiveMicPanel final : public QWidget {
    Q_OBJECT

public:
    explicit LiveMicPanel(QWidget* parent = nullptr);
    ~LiveMicPanel() override;

    void setProject(std::optional<core::Project> project);

private:
    void refreshDevices();
    void refreshVoices();
    void refreshRvcModels();
    void toggleMonitor(bool enabled);
    void toggleCloudConversion();
    void cancelCloudConversion();
    void toggleLocalRvcConversion();
    void cancelLocalRvcConversion();
    void openRvcModelManager();
    void updateGain(int value);
    void updateVoiceFx();
    void updateOutputRoute(int index);
    void setBroadcastChecked(bool enabled);
    void toggleVoiceChangerPower();
    void selectQuickVoiceSlot();
    void updateVoiceHud();
    void updateTransportState();
    void setHearSelfChecked(bool enabled);
    void setLiveInputChecked(bool enabled);
    void applyMeterUpdate(int level, bool speechActive);
    void enqueueCloudChunk(QByteArray chunk);
    void enqueueLocalRvcChunk(QByteArray chunk);
    void finishCloudConversion();
    void finishLocalRvcConversion();
    void testLatency();
    void setStatusText(const QString& text);
    void startAudioProcessor();
    void stopAudioProcessor(Qt::ConnectionType connectionType);
    void setProcessorPassthrough(bool enabled, Qt::ConnectionType connectionType);
    void setProcessorCloudCapture(bool enabled, Qt::ConnectionType connectionType);
    void setProcessorCloudCapturePaused(bool paused, Qt::ConnectionType connectionType);
    void setProcessorLocalRvcCapture(bool enabled, Qt::ConnectionType connectionType);
    void startNextCloudChunk();
    void finishCloudPlaybackGuard(int playbackDurationMs);
    void startNextLocalRvcChunk();
    void saveCloudRecordingIfReady();
    void saveLocalRvcRecordingIfReady();
    void refreshRecentTakes();
    void playTake(db::TakeRecord take);
    void starTake(db::TakeRecord take);
    void revealTake(db::TakeRecord take);
    void deleteTake(db::TakeRecord take);
    [[nodiscard]] bool ensureRecordingLine();
    [[nodiscard]] audio::CaptureConfig currentCaptureConfig() const;
    [[nodiscard]] std::string currentVoiceId() const;
    [[nodiscard]] std::string currentRvcModelId() const;
    [[nodiscard]] QString currentVoiceName() const;
    [[nodiscard]] QString currentRvcModelName() const;
    [[nodiscard]] int currentPitchShiftSemitones() const;
    [[nodiscard]] bool ensureCaptureRunning();
    [[nodiscard]] PlaybackTargets currentPlaybackTargets() noexcept;

    audio::Capture m_capture;
    audio::AudioEngine m_audioEngine;
    audio::AudioEngine m_broadcastAudioEngine;
    audio::LatencyProbe m_latencyProbe;
    db::ScriptRepository m_scriptRepository;
    db::TakeRepository m_takeRepository;
    db::VoiceRepository m_voiceRepository;
    rvc::RvcModelRegistry m_rvcModelRegistry;
    rvc::RvcSidecar m_rvcSidecar;
    voxcpm::VoxCpmSidecar m_voxCpmSidecar;
    std::shared_ptr<rvc::OnnxRvcEngine> m_nativeRvcEngine;
    std::optional<core::Project> m_project;
    std::vector<audio::AudioDeviceInfo> m_inputDevices;
    std::vector<audio::AudioDeviceInfo> m_outputDevices;
    QThread m_audioThread;
    LiveAudioProcessor* m_audioProcessor{nullptr};
    std::unique_ptr<QFutureWatcher<CloudConversionResult>> m_cloudWatcher;
    std::unique_ptr<QFutureWatcher<LocalRvcConversionResult>> m_localRvcWatcher;
    std::deque<QByteArray> m_pendingCloudChunks;
    std::deque<QByteArray> m_pendingLocalRvcChunks;
    QByteArray m_recordedCloudPcm;
    QByteArray m_recordedLocalRvcPcm;
    std::string m_recordingLineId;
    std::string m_recordingLineVoiceId;
    std::string m_recordingRvcModelId;
    QString m_recordingLineText;
    std::shared_ptr<std::atomic_bool> m_cloudCancelFlag;
    std::shared_ptr<std::atomic_bool> m_localRvcCancelFlag;
    double m_cloudSeconds{0.0};
    double m_localRvcSeconds{0.0};
    int m_cloudOutputSampleRate{24000};
    int m_localRvcOutputSampleRate{48000};
    int m_localRvcOutputChannels{1};
    std::uint64_t m_cloudPlaybackGeneration{0};
    bool m_cloudPlaybackGuardActive{false};
    bool m_cloudActive{false};
    bool m_localRvcActive{false};
    QComboBox* m_inputDeviceCombo{nullptr};
    QComboBox* m_outputDeviceCombo{nullptr};
    QComboBox* m_broadcastOutputDeviceCombo{nullptr};
    QComboBox* m_modeCombo{nullptr};
    QComboBox* m_voiceCombo{nullptr};
    QComboBox* m_rvcModelCombo{nullptr};
    QProgressBar* m_levelMeter{nullptr};
    QLabel* m_vadLabel{nullptr};
    QLabel* m_costLabel{nullptr};
    QLabel* m_statusLabel{nullptr};
    QLabel* m_selectedVoiceBadge{nullptr};
    QLabel* m_selectedVoiceLabel{nullptr};
    QLabel* m_selectedEngineLabel{nullptr};
    QLabel* m_outputRouteLabel{nullptr};
    QLabel* m_voiceVolumeValueLabel{nullptr};
    QLabel* m_bassValueLabel{nullptr};
    QLabel* m_midValueLabel{nullptr};
    QLabel* m_trebleValueLabel{nullptr};
    QLabel* m_pitchValueLabel{nullptr};
    QCheckBox* m_monitorCheck{nullptr};
    QCheckBox* m_recordTakeCheck{nullptr};
    QLineEdit* m_lineIdEdit{nullptr};
    QSlider* m_gainSlider{nullptr};
    QSlider* m_voiceVolumeSlider{nullptr};
    QSlider* m_bassSlider{nullptr};
    QSlider* m_midSlider{nullptr};
    QSlider* m_trebleSlider{nullptr};
    QSlider* m_pitchSlider{nullptr};
    QSpinBox* m_frameMsSpin{nullptr};
    QPushButton* m_refreshButton{nullptr};
    QPushButton* m_latencyButton{nullptr};
    QPushButton* m_voicePowerButton{nullptr};
    QPushButton* m_monitorButton{nullptr};
    QPushButton* m_liveInputButton{nullptr};
    QPushButton* m_broadcastButton{nullptr};
    QPushButton* m_cloudButton{nullptr};
    QPushButton* m_cancelCloudButton{nullptr};
    QPushButton* m_localRvcButton{nullptr};
    QPushButton* m_cancelLocalRvcButton{nullptr};
    QPushButton* m_manageRvcModelsButton{nullptr};
    TakeListWidget* m_recentTakesWidget{nullptr};
    std::array<QPushButton*, 6> m_quickVoiceButtons{};
};

} // namespace voxstudio::ui
