#pragma once

#include "audio/AudioEngine.h"
#include "audio/Capture.h"
#include "audio/LatencyProbe.h"
#include "core/Project.h"
#include "db/VoiceRepository.h"
#include "rvc/RvcModelRegistry.h"
#include "rvc/RvcSidecar.h"

#include <QByteArray>
#include <QFutureWatcher>
#include <QThread>
#include <QWidget>

#include <atomic>
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

struct CloudConversionResult final {
    bool success{false};
    QString message;
    QString playbackWarning;
    QByteArray convertedPcmBytes;
    double inputSeconds{0.0};
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
    void setProcessorLocalRvcCapture(bool enabled, Qt::ConnectionType connectionType);
    void startNextCloudChunk();
    void startNextLocalRvcChunk();
    void saveCloudRecordingIfReady();
    void saveLocalRvcRecordingIfReady();
    [[nodiscard]] audio::CaptureConfig currentCaptureConfig() const;
    [[nodiscard]] std::string currentVoiceId() const;
    [[nodiscard]] std::string currentRvcModelId() const;
    [[nodiscard]] bool ensureCaptureRunning();

    audio::Capture m_capture;
    audio::AudioEngine m_audioEngine;
    audio::LatencyProbe m_latencyProbe;
    db::VoiceRepository m_voiceRepository;
    rvc::RvcModelRegistry m_rvcModelRegistry;
    rvc::RvcSidecar m_rvcSidecar;
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
    std::shared_ptr<std::atomic_bool> m_cloudCancelFlag;
    std::shared_ptr<std::atomic_bool> m_localRvcCancelFlag;
    double m_cloudSeconds{0.0};
    double m_localRvcSeconds{0.0};
    int m_localRvcOutputSampleRate{48000};
    int m_localRvcOutputChannels{1};
    bool m_cloudActive{false};
    bool m_localRvcActive{false};
    QComboBox* m_inputDeviceCombo{nullptr};
    QComboBox* m_outputDeviceCombo{nullptr};
    QComboBox* m_modeCombo{nullptr};
    QComboBox* m_voiceCombo{nullptr};
    QComboBox* m_rvcModelCombo{nullptr};
    QProgressBar* m_levelMeter{nullptr};
    QLabel* m_vadLabel{nullptr};
    QLabel* m_costLabel{nullptr};
    QLabel* m_statusLabel{nullptr};
    QCheckBox* m_monitorCheck{nullptr};
    QCheckBox* m_recordTakeCheck{nullptr};
    QLineEdit* m_lineIdEdit{nullptr};
    QSlider* m_gainSlider{nullptr};
    QSpinBox* m_frameMsSpin{nullptr};
    QPushButton* m_refreshButton{nullptr};
    QPushButton* m_latencyButton{nullptr};
    QPushButton* m_cloudButton{nullptr};
    QPushButton* m_cancelCloudButton{nullptr};
    QPushButton* m_localRvcButton{nullptr};
    QPushButton* m_cancelLocalRvcButton{nullptr};
    QPushButton* m_manageRvcModelsButton{nullptr};
};

} // namespace voxstudio::ui
