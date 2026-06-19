#include "ui/LiveMicPanel.h"

#include "audio/AudioFile.h"
#include "net/elevenlabs/StsApi.h"
#include "rvc/OnnxRvcEngine.h"
#include "rvc/RvcClient.h"
#include "secrets/DpapiVault.h"
#include "ui/LiveAudioProcessor.h"
#include "ui/RvcModelManagerDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace voxstudio::ui {
namespace {

constexpr int kCloudInputSampleRate = 16000;
constexpr int kCloudOutputChannels = 1;
constexpr int kLocalRvcSampleRate = 48000;
constexpr int kLocalRvcChannels = 1;

template <typename TWidget, typename... TArgs>
[[nodiscard]] TWidget* addOwnedWidget(QLayout& layout, TArgs&&... args) {
    auto widget = std::make_unique<TWidget>(std::forward<TArgs>(args)...);
    auto* widgetPointer = widget.get();
    layout.addWidget(widget.release());
    return widgetPointer;
}

[[nodiscard]] int defaultDeviceIndex(const std::vector<audio::AudioDeviceInfo>& devices) {
    const auto found = std::ranges::find_if(devices, &audio::AudioDeviceInfo::isDefault);
    return found == devices.end() ? -1 : found->index;
}

[[nodiscard]] int comboDeviceIndex(const QComboBox* combo) {
    if (combo == nullptr || combo->currentIndex() < 0) {
        return -1;
    }
    return combo->currentData().toInt();
}

[[nodiscard]] QString deviceLabel(const audio::AudioDeviceInfo& device) {
    auto label = QString::fromStdString(device.name);
    if (device.isDefault) {
        label += QStringLiteral(" (default)");
    }
    return label;
}

[[nodiscard]] QString apiErrorText(const net::elevenlabs::ApiError& error) {
    if (error.statusCode > 0) {
        return QStringLiteral("%1 (HTTP %2)")
            .arg(QString::fromStdString(error.message))
            .arg(error.statusCode);
    }
    return QString::fromStdString(error.message);
}

[[nodiscard]] core::Expected<std::string> loadApiKey() {
    const secrets::DpapiVault vault;
    return vault.loadElevenLabsApiKey();
}

[[nodiscard]] std::vector<std::uint8_t> bytesFromByteArray(const QByteArray& bytes) {
    const auto* first = reinterpret_cast<const std::uint8_t*>(bytes.constData());
    return {first, first + bytes.size()};
}

[[nodiscard]] QByteArray byteArrayFromBytes(const std::vector<std::uint8_t>& bytes) {
    return QByteArray{reinterpret_cast<const char*>(bytes.data()),
                      static_cast<qsizetype>(bytes.size())};
}

[[nodiscard]] CloudConversionResult convertCloudChunk(
    const std::string& voiceId,
    const QByteArray& inputPcmBytes,
    audio::AudioEngine* audioEngine,
    const std::shared_ptr<std::atomic_bool>& cancelFlag) {
    if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
        return CloudConversionResult{false, QStringLiteral("Cloud conversion cancelled.")};
    }

    auto apiKey = loadApiKey();
    if (!apiKey) {
        return CloudConversionResult{false, QString::fromStdString(apiKey.error().message)};
    }

    net::elevenlabs::StsRequest request;
    request.voiceId = voiceId;
    request.pcm16Audio = bytesFromByteArray(inputPcmBytes);
    request.outputFormat = "pcm_44100";
    const auto outputSampleRate =
        net::elevenlabs::pcmSampleRateFromOutputFormat(request.outputFormat);

    std::vector<std::uint8_t> pendingBytes;
    QString playbackWarning;
    const auto onChunk = [audioEngine, outputSampleRate, &pendingBytes, &playbackWarning,
                          cancelFlag](std::span<const std::uint8_t> chunk) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return false;
        }

        pendingBytes.insert(pendingBytes.end(), chunk.begin(), chunk.end());
        const auto playableByteCount = pendingBytes.size() - (pendingBytes.size() % 2U);
        if (playableByteCount == 0U || audioEngine == nullptr || outputSampleRate <= 0) {
            return true;
        }

        const auto playableBytes =
            std::span<const std::uint8_t>{pendingBytes.data(), playableByteCount};
        auto queued = audioEngine->queuePcm16LittleEndian(playableBytes,
                                                          outputSampleRate,
                                                          kCloudOutputChannels);
        if (!queued && playbackWarning.isEmpty()) {
            playbackWarning = QString::fromStdString(queued.error().message);
        }
        pendingBytes.erase(pendingBytes.begin(),
                           pendingBytes.begin() +
                               static_cast<std::ptrdiff_t>(playableByteCount));
        return true;
    };

    const net::elevenlabs::StsApi api{std::move(apiKey).value()};
    auto streamed = api.streamSpeech(request, onChunk);
    if (!streamed) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return CloudConversionResult{false, QStringLiteral("Cloud conversion cancelled.")};
        }
        return CloudConversionResult{false, apiErrorText(streamed.error()), playbackWarning};
    }

    return CloudConversionResult{true,
                                 QStringLiteral("Cloud chunk converted."),
                                 playbackWarning,
                                 byteArrayFromBytes(streamed.value().audioBytes),
                                 streamed.value().inputSeconds};
}

[[nodiscard]] LocalRvcConversionResult convertLocalRvcChunk(
    const std::string& endpoint,
    const std::string& modelId,
    const QByteArray& inputPcmBytes,
    audio::AudioEngine* audioEngine,
    const std::shared_ptr<std::atomic_bool>& cancelFlag) {
    if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
        return LocalRvcConversionResult{false, QStringLiteral("Local RVC cancelled.")};
    }

    rvc::RvcConvertRequest request;
    request.modelId = modelId;
    request.pcm16Audio = bytesFromByteArray(inputPcmBytes);
    request.sampleRate = kLocalRvcSampleRate;
    request.channels = kLocalRvcChannels;

    QString playbackWarning;
    const auto onChunk = [audioEngine, &playbackWarning, cancelFlag](
                             std::span<const std::uint8_t> chunk) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return false;
        }
        if (chunk.empty() || audioEngine == nullptr) {
            return true;
        }

        auto queued = audioEngine->queuePcm16LittleEndian(chunk,
                                                          kLocalRvcSampleRate,
                                                          kLocalRvcChannels);
        if (!queued && playbackWarning.isEmpty()) {
            playbackWarning = QString::fromStdString(queued.error().message);
        }
        return true;
    };

    const rvc::RvcClient client{endpoint};
    auto converted = client.convertChunk(request, onChunk);
    if (!converted) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return LocalRvcConversionResult{false, QStringLiteral("Local RVC cancelled.")};
        }
        return LocalRvcConversionResult{false,
                                        QString::fromStdString(converted.error().message),
                                        playbackWarning};
    }

    return LocalRvcConversionResult{true,
                                    QStringLiteral("Local RVC chunk converted."),
                                    playbackWarning,
                                    byteArrayFromBytes(converted.value().pcm16Audio),
                                    converted.value().latencyMs,
                                    converted.value().sampleRate,
                                    converted.value().channels};
}

[[nodiscard]] LocalRvcConversionResult convertNativeRvcChunk(
    const std::shared_ptr<rvc::OnnxRvcEngine>& engine,
    const QByteArray& inputPcmBytes,
    audio::AudioEngine* audioEngine,
    const std::shared_ptr<std::atomic_bool>& cancelFlag) {
    if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
        return LocalRvcConversionResult{false, QStringLiteral("Native RVC cancelled.")};
    }
    if (engine == nullptr) {
        return LocalRvcConversionResult{false,
                                        QStringLiteral("Native ONNX engine is not configured.")};
    }

    rvc::OnnxRvcRequest request;
    request.pcm16Audio = bytesFromByteArray(inputPcmBytes);
    request.sampleRate = kLocalRvcSampleRate;
    request.channels = kLocalRvcChannels;

    auto converted = engine->convertChunk(request);
    if (!converted) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return LocalRvcConversionResult{false, QStringLiteral("Native RVC cancelled.")};
        }
        return LocalRvcConversionResult{false,
                                        QString::fromStdString(converted.error().message)};
    }

    QString playbackWarning;
    if (audioEngine != nullptr && !converted.value().pcm16Audio.empty()) {
        auto queued = audioEngine->queuePcm16LittleEndian(converted.value().pcm16Audio,
                                                          converted.value().sampleRate,
                                                          converted.value().channels);
        if (!queued) {
            playbackWarning = QString::fromStdString(queued.error().message);
        }
    }

    return LocalRvcConversionResult{true,
                                    QStringLiteral("Native RVC chunk converted."),
                                    playbackWarning,
                                    byteArrayFromBytes(converted.value().pcm16Audio),
                                    converted.value().latencyMs,
                                    converted.value().sampleRate,
                                    converted.value().channels};
}

} // namespace

LiveMicPanel::LiveMicPanel(QWidget* parent)
    : QWidget(parent)
    , m_cloudWatcher(std::make_unique<QFutureWatcher<CloudConversionResult>>())
    , m_localRvcWatcher(std::make_unique<QFutureWatcher<LocalRvcConversionResult>>()) {
    auto rootLayout = std::make_unique<QVBoxLayout>();

    auto* titleLabel = addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Live Mic"));
    titleLabel->setObjectName(QStringLiteral("LiveMicTitle"));

    auto deviceLayout = std::make_unique<QGridLayout>();
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Input")).release(), 0, 0);
    m_inputDeviceCombo = addOwnedWidget<QComboBox>(*deviceLayout);
    m_inputDeviceCombo->setObjectName(QStringLiteral("LiveMicInputCombo"));
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Output")).release(), 1, 0);
    m_outputDeviceCombo = addOwnedWidget<QComboBox>(*deviceLayout);
    m_outputDeviceCombo->setObjectName(QStringLiteral("LiveMicOutputCombo"));
    rootLayout->addLayout(deviceLayout.release());

    auto cloudLayout = std::make_unique<QGridLayout>();
    cloudLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Mode")).release(), 0, 0);
    m_modeCombo = addOwnedWidget<QComboBox>(*cloudLayout);
    m_modeCombo->setObjectName(QStringLiteral("LiveMicModeCombo"));
    m_modeCombo->addItem(QStringLiteral("Monitor"));
    m_modeCombo->addItem(QStringLiteral("Cloud"));
    m_modeCombo->addItem(QStringLiteral("Local"));
    cloudLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Voice")).release(), 0, 2);
    m_voiceCombo = addOwnedWidget<QComboBox>(*cloudLayout);
    m_voiceCombo->setObjectName(QStringLiteral("LiveMicVoiceCombo"));
    cloudLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("RVC Model")).release(), 1, 0);
    m_rvcModelCombo = addOwnedWidget<QComboBox>(*cloudLayout);
    m_rvcModelCombo->setObjectName(QStringLiteral("LiveMicRvcModelCombo"));
    m_cloudButton = addOwnedWidget<QPushButton>(*cloudLayout, QStringLiteral("Convert Cloud"));
    m_cloudButton->setObjectName(QStringLiteral("LiveMicCloudButton"));
    m_cancelCloudButton = addOwnedWidget<QPushButton>(*cloudLayout, QStringLiteral("Cancel"));
    m_cancelCloudButton->setObjectName(QStringLiteral("LiveMicCancelCloudButton"));
    m_cancelCloudButton->setEnabled(false);
    m_localRvcButton = addOwnedWidget<QPushButton>(*cloudLayout, QStringLiteral("Start Local"));
    m_localRvcButton->setObjectName(QStringLiteral("LiveMicLocalRvcButton"));
    m_cancelLocalRvcButton =
        addOwnedWidget<QPushButton>(*cloudLayout, QStringLiteral("Cancel Local"));
    m_cancelLocalRvcButton->setObjectName(QStringLiteral("LiveMicCancelLocalRvcButton"));
    m_cancelLocalRvcButton->setEnabled(false);
    m_manageRvcModelsButton = addOwnedWidget<QPushButton>(*cloudLayout,
                                                          QStringLiteral("RVC Models"));
    m_manageRvcModelsButton->setObjectName(QStringLiteral("LiveMicManageRvcModelsButton"));
    m_recordTakeCheck = addOwnedWidget<QCheckBox>(*cloudLayout, QStringLiteral("Record take"));
    m_recordTakeCheck->setObjectName(QStringLiteral("LiveMicRecordTakeCheck"));
    m_lineIdEdit = addOwnedWidget<QLineEdit>(*cloudLayout);
    m_lineIdEdit->setObjectName(QStringLiteral("LiveMicLineIdEdit"));
    m_lineIdEdit->setPlaceholderText(QStringLiteral("Line ID"));
    rootLayout->addLayout(cloudLayout.release());

    auto controlsLayout = std::make_unique<QHBoxLayout>();
    m_monitorCheck = addOwnedWidget<QCheckBox>(*controlsLayout, QStringLiteral("Monitor"));
    m_monitorCheck->setObjectName(QStringLiteral("LiveMicMonitorToggle"));
    controlsLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Gain")).release());
    m_gainSlider = addOwnedWidget<QSlider>(*controlsLayout, Qt::Horizontal);
    m_gainSlider->setObjectName(QStringLiteral("LiveMicGainSlider"));
    m_gainSlider->setRange(0, 400);
    m_gainSlider->setValue(100);
    controlsLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Frame")).release());
    m_frameMsSpin = addOwnedWidget<QSpinBox>(*controlsLayout);
    m_frameMsSpin->setObjectName(QStringLiteral("LiveMicFrameMsSpin"));
    m_frameMsSpin->setRange(10, audio::kMaxRealtimeFrameMs);
    m_frameMsSpin->setSingleStep(10);
    m_frameMsSpin->setValue(10);
    m_frameMsSpin->setSuffix(QStringLiteral(" ms"));
    rootLayout->addLayout(controlsLayout.release());

    auto meterLayout = std::make_unique<QHBoxLayout>();
    meterLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Level")).release());
    m_levelMeter = addOwnedWidget<QProgressBar>(*meterLayout);
    m_levelMeter->setObjectName(QStringLiteral("LiveMicLevelMeter"));
    m_levelMeter->setRange(0, 100);
    m_levelMeter->setValue(0);
    m_vadLabel = addOwnedWidget<QLabel>(*meterLayout, QStringLiteral("VAD idle"));
    m_vadLabel->setObjectName(QStringLiteral("LiveMicVadLabel"));
    m_costLabel = addOwnedWidget<QLabel>(*meterLayout, QStringLiteral("Cloud cost: 0.0 s"));
    m_costLabel->setObjectName(QStringLiteral("LiveMicCostLabel"));
    rootLayout->addLayout(meterLayout.release());

    auto actionsLayout = std::make_unique<QHBoxLayout>();
    m_refreshButton = addOwnedWidget<QPushButton>(*actionsLayout, QStringLiteral("Refresh"));
    m_refreshButton->setObjectName(QStringLiteral("LiveMicRefreshButton"));
    m_latencyButton = addOwnedWidget<QPushButton>(*actionsLayout, QStringLiteral("Test Latency"));
    m_latencyButton->setObjectName(QStringLiteral("LiveMicLatencyButton"));
    rootLayout->addLayout(actionsLayout.release());

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setObjectName(QStringLiteral("LiveMicStatusLabel"));
    m_statusLabel->setWordWrap(true);

    setLayout(rootLayout.release());

    connect(m_refreshButton, &QPushButton::clicked, this, &LiveMicPanel::refreshDevices);
    connect(m_monitorCheck, &QCheckBox::toggled, this, &LiveMicPanel::toggleMonitor);
    connect(m_gainSlider, &QSlider::valueChanged, this, &LiveMicPanel::updateGain);
    connect(m_latencyButton, &QPushButton::clicked, this, &LiveMicPanel::testLatency);
    connect(m_cloudButton, &QPushButton::clicked, this, &LiveMicPanel::toggleCloudConversion);
    connect(m_cancelCloudButton, &QPushButton::clicked, this, &LiveMicPanel::cancelCloudConversion);
    connect(m_localRvcButton, &QPushButton::clicked, this,
            &LiveMicPanel::toggleLocalRvcConversion);
    connect(m_cancelLocalRvcButton, &QPushButton::clicked, this,
            &LiveMicPanel::cancelLocalRvcConversion);
    connect(m_manageRvcModelsButton, &QPushButton::clicked, this,
            &LiveMicPanel::openRvcModelManager);
    connect(m_cloudWatcher.get(), &QFutureWatcher<CloudConversionResult>::finished, this,
            &LiveMicPanel::finishCloudConversion);
    connect(m_localRvcWatcher.get(),
            &QFutureWatcher<LocalRvcConversionResult>::finished,
            this,
            &LiveMicPanel::finishLocalRvcConversion);

    auto processor = std::make_unique<LiveAudioProcessor>();
    m_audioProcessor = processor.get();
    m_audioProcessor->moveToThread(&m_audioThread);
    connect(&m_audioThread, &QThread::finished, m_audioProcessor, &QObject::deleteLater);
    connect(m_audioProcessor,
            &LiveAudioProcessor::meterUpdated,
            this,
            &LiveMicPanel::applyMeterUpdate,
            Qt::QueuedConnection);
    connect(m_audioProcessor,
            &LiveAudioProcessor::statusMessage,
            this,
            &LiveMicPanel::setStatusText,
            Qt::QueuedConnection);
    connect(m_audioProcessor,
            &LiveAudioProcessor::cloudPcmChunkReady,
            this,
            &LiveMicPanel::enqueueCloudChunk,
            Qt::QueuedConnection);
    connect(m_audioProcessor,
            &LiveAudioProcessor::localRvcPcmChunkReady,
            this,
            &LiveMicPanel::enqueueLocalRvcChunk,
            Qt::QueuedConnection);
    processor.release();
    m_audioThread.setObjectName(QStringLiteral("LiveMicProcessor"));
    m_audioThread.start(QThread::HighPriority);

    refreshDevices();
    refreshVoices();
    refreshRvcModels();
}

LiveMicPanel::~LiveMicPanel() {
    cancelCloudConversion();
    cancelLocalRvcConversion();
    if (m_cloudWatcher->isRunning()) {
        m_cloudWatcher->waitForFinished();
    }
    if (m_localRvcWatcher->isRunning()) {
        m_localRvcWatcher->waitForFinished();
    }
    stopAudioProcessor(Qt::BlockingQueuedConnection);
    m_capture.stop();
    m_audioThread.quit();
    m_audioThread.wait();
}

void LiveMicPanel::setProject(std::optional<core::Project> project) {
    m_project = std::move(project);
    refreshVoices();
    refreshRvcModels();
}

void LiveMicPanel::refreshDevices() {
    m_inputDeviceCombo->clear();
    m_outputDeviceCombo->clear();

    auto inputs = audio::Capture::listInputDevices();
    if (inputs) {
        m_inputDevices = std::move(inputs).value();
        for (const auto& device : m_inputDevices) {
            m_inputDeviceCombo->addItem(deviceLabel(device), device.index);
        }
        const auto defaultIndex = defaultDeviceIndex(m_inputDevices);
        if (defaultIndex >= 0) {
            m_inputDeviceCombo->setCurrentIndex(defaultIndex);
        }
    }

    auto outputs = audio::Capture::listOutputDevices();
    if (outputs) {
        m_outputDevices = std::move(outputs).value();
        for (const auto& device : m_outputDevices) {
            m_outputDeviceCombo->addItem(deviceLabel(device), device.index);
        }
        const auto defaultIndex = defaultDeviceIndex(m_outputDevices);
        if (defaultIndex >= 0) {
            m_outputDeviceCombo->setCurrentIndex(defaultIndex);
        }
    }

    if (!inputs || !outputs) {
        const auto inputError = inputs ? QString{} : QString::fromStdString(inputs.error().message);
        const auto outputError =
            outputs ? QString{} : QString::fromStdString(outputs.error().message);
        setStatusText(QStringLiteral("%1 %2").arg(inputError, outputError).trimmed());
        return;
    }

    setStatusText(QStringLiteral("%1 input device(s), %2 output device(s).")
                      .arg(static_cast<int>(m_inputDevices.size()))
                      .arg(static_cast<int>(m_outputDevices.size())));
}

void LiveMicPanel::refreshVoices() {
    m_voiceCombo->clear();
    if (!m_project.has_value()) {
        m_voiceCombo->setEnabled(false);
        m_cloudButton->setEnabled(false);
        return;
    }

    auto voices = m_voiceRepository.listVoices(m_project->rootPath());
    if (!voices) {
        m_voiceCombo->setEnabled(false);
        m_cloudButton->setEnabled(false);
        setStatusText(QString::fromStdString(voices.error().message));
        return;
    }

    for (const auto& voice : voices.value()) {
        m_voiceCombo->addItem(QString::fromStdString(voice.name),
                              QString::fromStdString(voice.id));
    }
    const bool hasVoices = m_voiceCombo->count() > 0;
    m_voiceCombo->setEnabled(hasVoices);
    m_cloudButton->setEnabled(hasVoices);
    if (!hasVoices) {
        setStatusText(QStringLiteral("Sync or clone voices before cloud conversion."));
    }
}

void LiveMicPanel::refreshRvcModels() {
    m_rvcModelCombo->clear();
    auto models = m_rvcModelRegistry.listModels();
    if (!models) {
        m_rvcModelCombo->setEnabled(false);
        m_localRvcButton->setEnabled(false);
        setStatusText(QString::fromStdString(models.error().message));
        return;
    }

    for (const auto& model : models.value()) {
        m_rvcModelCombo->addItem(QString::fromStdString(model.displayName),
                                 QString::fromStdString(model.id));
    }
    const bool hasModels = m_rvcModelCombo->count() > 0;
    m_rvcModelCombo->setEnabled(hasModels);
    m_localRvcButton->setEnabled(hasModels);
}

void LiveMicPanel::toggleMonitor(const bool enabled) {
    if (!enabled) {
        if (m_cloudActive || m_localRvcActive) {
            m_capture.setMonitorEnabled(false);
            setProcessorPassthrough(false, Qt::QueuedConnection);
            setStatusText(m_cloudActive ? QStringLiteral("Cloud capture active.")
                                        : QStringLiteral("Local RVC active."));
            return;
        }

        stopAudioProcessor(Qt::BlockingQueuedConnection);
        m_capture.stop();
        m_levelMeter->setValue(0);
        m_vadLabel->setText(QStringLiteral("VAD idle"));
        setStatusText(QStringLiteral("Monitor stopped."));
        return;
    }

    if (!ensureCaptureRunning()) {
        m_monitorCheck->blockSignals(true);
        m_monitorCheck->setChecked(false);
        m_monitorCheck->blockSignals(false);
        return;
    }

    const bool passthrough = !m_cloudActive && !m_localRvcActive;
    m_capture.setMonitorEnabled(passthrough);
    setProcessorPassthrough(passthrough, Qt::QueuedConnection);
    setStatusText(passthrough ? QStringLiteral("Monitoring microphone.")
                              : QStringLiteral("Voice conversion active."));
}

void LiveMicPanel::toggleCloudConversion() {
    if (m_localRvcActive) {
        setStatusText(QStringLiteral("Stop Local RVC before starting cloud conversion."));
        return;
    }

    if (m_cloudActive) {
        m_cloudActive = false;
        m_cloudButton->setText(QStringLiteral("Convert Cloud"));
        setProcessorCloudCapture(false, Qt::BlockingQueuedConnection);
        if (m_monitorCheck->isChecked()) {
            m_capture.setMonitorEnabled(true);
            setProcessorPassthrough(true, Qt::QueuedConnection);
        } else {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
        setStatusText(QStringLiteral("Finishing queued cloud conversion."));
        saveCloudRecordingIfReady();
        return;
    }

    const auto voiceId = currentVoiceId();
    if (voiceId.empty()) {
        setStatusText(QStringLiteral("Select a target voice first."));
        return;
    }
    if (!ensureCaptureRunning()) {
        return;
    }

    m_cloudCancelFlag = std::make_shared<std::atomic_bool>(false);
    m_pendingCloudChunks.clear();
    m_recordedCloudPcm.clear();
    m_cloudSeconds = 0.0;
    m_costLabel->setText(QStringLiteral("Cloud cost: 0.0 s"));
    m_cloudActive = true;
    m_cloudButton->setText(QStringLiteral("Stop Cloud"));
    m_cancelCloudButton->setEnabled(true);
    m_modeCombo->setCurrentText(QStringLiteral("Cloud"));
    m_capture.setMonitorEnabled(false);
    setProcessorPassthrough(false, Qt::BlockingQueuedConnection);
    setProcessorCloudCapture(true, Qt::BlockingQueuedConnection);
    setStatusText(QStringLiteral("Cloud capture active."));
}

void LiveMicPanel::cancelCloudConversion() {
    if (m_cloudCancelFlag != nullptr) {
        m_cloudCancelFlag->store(true, std::memory_order_release);
    }
    m_cloudActive = false;
    m_pendingCloudChunks.clear();
    m_recordedCloudPcm.clear();
    m_cloudButton->setText(QStringLiteral("Convert Cloud"));
    m_cancelCloudButton->setEnabled(false);
    if (m_audioThread.isRunning()) {
        setProcessorCloudCapture(false, Qt::BlockingQueuedConnection);
    }
    if (!m_monitorCheck->isChecked() && !m_localRvcActive) {
        stopAudioProcessor(Qt::BlockingQueuedConnection);
        m_capture.stop();
    }
    setStatusText(QStringLiteral("Cloud conversion cancelled."));
}

void LiveMicPanel::toggleLocalRvcConversion() {
    if (m_localRvcActive) {
        m_localRvcActive = false;
        m_localRvcButton->setText(QStringLiteral("Start Local"));
        m_cancelLocalRvcButton->setEnabled(false);
        setProcessorLocalRvcCapture(false, Qt::BlockingQueuedConnection);
        if (m_monitorCheck->isChecked()) {
            m_capture.setMonitorEnabled(true);
            setProcessorPassthrough(true, Qt::QueuedConnection);
        } else {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
        m_rvcSidecar.stop();
        if (m_pendingLocalRvcChunks.empty() && !m_localRvcWatcher->isRunning()) {
            m_nativeRvcEngine.reset();
        }
        setStatusText(QStringLiteral("Finishing queued Local RVC conversion."));
        saveLocalRvcRecordingIfReady();
        return;
    }

    if (m_cloudActive) {
        setStatusText(QStringLiteral("Stop Cloud before starting Local RVC."));
        return;
    }

    const auto modelId = currentRvcModelId();
    if (modelId.empty()) {
        setStatusText(QStringLiteral("Import and select an RVC model first."));
        return;
    }

    const QSettings settings;
    const auto runtimeMode =
        settings.value(QStringLiteral("rvc/runtime"), QStringLiteral("sidecar")).toString();
    if (runtimeMode == QStringLiteral("native_onnx")) {
        auto engine = std::make_shared<rvc::OnnxRvcEngine>();
        auto runtime = engine->probeRuntime();
        if (!runtime) {
            setStatusText(QString::fromStdString(runtime.error().message));
            return;
        }
        if (!runtime.value().available) {
            setStatusText(QString::fromStdString(runtime.value().message));
            return;
        }

        const auto bundleRoot = rvc::OnnxRvcEngine::defaultNativeModelRoot() / modelId;
        auto bundle = engine->loadModelBundle(bundleRoot);
        if (!bundle) {
            setStatusText(QString::fromStdString(bundle.error().message));
            return;
        }

        auto configured = engine->configureModelBundle(std::move(bundle.value()));
        if (!configured) {
            setStatusText(QString::fromStdString(configured.error().message));
            return;
        }

        auto description = engine->describeConfiguredModel();
        if (!description) {
            setStatusText(QString::fromStdString(description.error().message));
            return;
        }
        auto pipeline = engine->describeConfiguredPipeline();
        if (!pipeline) {
            setStatusText(QString::fromStdString(pipeline.error().message));
            return;
        }
        if (!ensureCaptureRunning()) {
            return;
        }

        m_nativeRvcEngine = std::move(engine);
        m_localRvcCancelFlag = std::make_shared<std::atomic_bool>(false);
        m_pendingLocalRvcChunks.clear();
        m_recordedLocalRvcPcm.clear();
        m_localRvcSeconds = 0.0;
        m_localRvcOutputSampleRate = description.value().bundle.sampleRate;
        m_localRvcOutputChannels = kLocalRvcChannels;
        m_costLabel->setText(QStringLiteral("Native audio: 0.0 s"));
        m_localRvcActive = true;
        m_localRvcButton->setText(QStringLiteral("Stop Local"));
        m_cancelLocalRvcButton->setEnabled(true);
        m_modeCombo->setCurrentText(QStringLiteral("Local"));
        m_capture.setMonitorEnabled(false);
        setProcessorPassthrough(false, Qt::BlockingQueuedConnection);
        setProcessorLocalRvcCapture(true, Qt::BlockingQueuedConnection);
        const auto generatorInputs = description.value().generator.inputs.size();
        const auto generatorOutputs = description.value().generator.outputs.size();
        setStatusText(QStringLiteral("Native ONNX RVC active: generator %1 in/%2 out.")
                          .arg(static_cast<qulonglong>(generatorInputs))
                          .arg(static_cast<qulonglong>(generatorOutputs)));
        return;
    }

    m_nativeRvcEngine.reset();
    auto started = m_rvcSidecar.start();
    if (!started) {
        setStatusText(QString::fromStdString(started.error().message));
        return;
    }
    if (!ensureCaptureRunning()) {
        m_rvcSidecar.stop();
        return;
    }

    m_localRvcCancelFlag = std::make_shared<std::atomic_bool>(false);
    m_pendingLocalRvcChunks.clear();
    m_recordedLocalRvcPcm.clear();
    m_localRvcSeconds = 0.0;
    m_localRvcOutputSampleRate = kLocalRvcSampleRate;
    m_localRvcOutputChannels = kLocalRvcChannels;
    m_costLabel->setText(QStringLiteral("Local audio: 0.0 s"));
    m_localRvcActive = true;
    m_localRvcButton->setText(QStringLiteral("Stop Local"));
    m_cancelLocalRvcButton->setEnabled(true);
    m_modeCombo->setCurrentText(QStringLiteral("Local"));
    m_capture.setMonitorEnabled(false);
    setProcessorPassthrough(false, Qt::BlockingQueuedConnection);
    setProcessorLocalRvcCapture(true, Qt::BlockingQueuedConnection);
    setStatusText(QStringLiteral("Local RVC active at %1.")
                      .arg(QString::fromStdString(started.value().endpoint)));
}

void LiveMicPanel::cancelLocalRvcConversion() {
    if (m_localRvcCancelFlag != nullptr) {
        m_localRvcCancelFlag->store(true, std::memory_order_release);
    }
    m_localRvcActive = false;
    m_pendingLocalRvcChunks.clear();
    m_recordedLocalRvcPcm.clear();
    m_localRvcButton->setText(QStringLiteral("Start Local"));
    m_cancelLocalRvcButton->setEnabled(false);
    if (m_audioThread.isRunning()) {
        setProcessorLocalRvcCapture(false, Qt::BlockingQueuedConnection);
    }
    if (!m_monitorCheck->isChecked() && !m_cloudActive) {
        stopAudioProcessor(Qt::BlockingQueuedConnection);
        m_capture.stop();
    }
    m_rvcSidecar.stop();
    m_nativeRvcEngine.reset();
    setStatusText(QStringLiteral("Local RVC cancelled."));
}

void LiveMicPanel::openRvcModelManager() {
    RvcModelManagerDialog dialog{m_project, this};
    connect(&dialog,
            &RvcModelManagerDialog::rvcAssignmentsChanged,
            this,
            &LiveMicPanel::refreshRvcModels);
    dialog.exec();
    refreshRvcModels();
}

void LiveMicPanel::updateGain(const int value) {
    m_capture.setGain(static_cast<float>(value) / 100.0F);
}

void LiveMicPanel::applyMeterUpdate(const int level, const bool speechActive) {
    m_levelMeter->setValue(level);
    m_vadLabel->setText(speechActive ? QStringLiteral("VAD speech")
                                     : QStringLiteral("VAD idle"));
}

void LiveMicPanel::enqueueCloudChunk(QByteArray chunk) {
    if (chunk.isEmpty() || (m_cloudCancelFlag != nullptr &&
                            m_cloudCancelFlag->load(std::memory_order_acquire))) {
        return;
    }

    m_cloudSeconds += static_cast<double>(chunk.size()) /
                      static_cast<double>(kCloudInputSampleRate * 2);
    m_costLabel->setText(QStringLiteral("Cloud cost: %1 s").arg(m_cloudSeconds, 0, 'f', 1));
    m_pendingCloudChunks.push_back(std::move(chunk));
    startNextCloudChunk();
}

void LiveMicPanel::enqueueLocalRvcChunk(QByteArray chunk) {
    if (chunk.isEmpty() || (m_localRvcCancelFlag != nullptr &&
                            m_localRvcCancelFlag->load(std::memory_order_acquire))) {
        return;
    }

    m_localRvcSeconds += static_cast<double>(chunk.size()) /
                         static_cast<double>(kLocalRvcSampleRate * 2);
    const auto label = m_nativeRvcEngine == nullptr ? QStringLiteral("Local audio: %1 s")
                                                    : QStringLiteral("Native audio: %1 s");
    m_costLabel->setText(label.arg(m_localRvcSeconds, 0, 'f', 1));
    m_pendingLocalRvcChunks.push_back(std::move(chunk));
    startNextLocalRvcChunk();
}

void LiveMicPanel::finishCloudConversion() {
    const auto result = m_cloudWatcher->result();
    if (m_cloudCancelFlag != nullptr && m_cloudCancelFlag->load(std::memory_order_acquire)) {
        setStatusText(QStringLiteral("Cloud conversion cancelled."));
        return;
    }

    if (!result.success) {
        setStatusText(result.message);
    } else {
        if (m_recordTakeCheck->isChecked()) {
            m_recordedCloudPcm.append(result.convertedPcmBytes);
        }
        setStatusText(result.playbackWarning.isEmpty()
                          ? result.message
                          : QStringLiteral("%1 Playback: %2")
                                .arg(result.message, result.playbackWarning));
    }

    startNextCloudChunk();
    saveCloudRecordingIfReady();
}

void LiveMicPanel::finishLocalRvcConversion() {
    const auto result = m_localRvcWatcher->result();
    if (m_localRvcCancelFlag != nullptr &&
        m_localRvcCancelFlag->load(std::memory_order_acquire)) {
        setStatusText(QStringLiteral("Local RVC cancelled."));
        return;
    }

    if (!result.success) {
        setStatusText(result.message);
    } else {
        if (m_recordTakeCheck->isChecked()) {
            m_recordedLocalRvcPcm.append(result.convertedPcmBytes);
        }
        m_localRvcOutputSampleRate = result.sampleRate;
        m_localRvcOutputChannels = result.channels;
        const auto message = QStringLiteral("%1 Latency: %2 ms.")
                                 .arg(result.message)
                                 .arg(result.latencyMs);
        setStatusText(result.playbackWarning.isEmpty()
                          ? message
                          : QStringLiteral("%1 Playback: %2")
                                .arg(message, result.playbackWarning));
    }

    startNextLocalRvcChunk();
    saveLocalRvcRecordingIfReady();
}

void LiveMicPanel::testLatency() {
    const auto result = m_latencyProbe.estimateSharedModeLatency(m_frameMsSpin->value());
    setStatusText(QStringLiteral("Estimated monitor latency: %1 ms (%2 target).")
                      .arg(result.latencyMs)
                      .arg(result.withinTarget ? QStringLiteral("within")
                                               : QStringLiteral("over")));
}

void LiveMicPanel::setStatusText(const QString& text) {
    m_statusLabel->setText(text);
}

void LiveMicPanel::startAudioProcessor() {
    auto* processor = m_audioProcessor;
    auto* capture = &m_capture;
    QMetaObject::invokeMethod(
        processor,
        [processor, capture]() {
            processor->start(capture);
        },
        Qt::QueuedConnection);
}

void LiveMicPanel::stopAudioProcessor(const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor,
        [processor]() {
            processor->stop();
        },
        connectionType);
}

void LiveMicPanel::setProcessorPassthrough(const bool enabled,
                                           const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor,
        [processor, enabled]() {
            processor->setPassthroughEnabled(enabled);
        },
        connectionType);
}

void LiveMicPanel::setProcessorCloudCapture(const bool enabled,
                                            const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor,
        [processor, enabled]() {
            processor->setCloudCaptureEnabled(enabled);
        },
        connectionType);
}

void LiveMicPanel::setProcessorLocalRvcCapture(const bool enabled,
                                               const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor,
        [processor, enabled]() {
            processor->setLocalRvcCaptureEnabled(enabled);
        },
        connectionType);
}

void LiveMicPanel::startNextCloudChunk() {
    if (m_cloudWatcher->isRunning() || m_pendingCloudChunks.empty()) {
        saveCloudRecordingIfReady();
        return;
    }
    if (m_cloudCancelFlag != nullptr && m_cloudCancelFlag->load(std::memory_order_acquire)) {
        m_pendingCloudChunks.clear();
        return;
    }

    const auto voiceId = currentVoiceId();
    if (voiceId.empty()) {
        m_pendingCloudChunks.clear();
        setStatusText(QStringLiteral("Select a target voice first."));
        return;
    }

    auto chunk = std::move(m_pendingCloudChunks.front());
    m_pendingCloudChunks.pop_front();
    auto* audioEngine = &m_audioEngine;
    auto cancelFlag = m_cloudCancelFlag;
    m_cloudWatcher->setFuture(QtConcurrent::run([voiceId, chunk, audioEngine, cancelFlag]() {
        return convertCloudChunk(voiceId, chunk, audioEngine, cancelFlag);
    }));
}

void LiveMicPanel::startNextLocalRvcChunk() {
    if (m_localRvcWatcher->isRunning() || m_pendingLocalRvcChunks.empty()) {
        saveLocalRvcRecordingIfReady();
        return;
    }
    if (m_localRvcCancelFlag != nullptr &&
        m_localRvcCancelFlag->load(std::memory_order_acquire)) {
        m_pendingLocalRvcChunks.clear();
        return;
    }

    const auto modelId = currentRvcModelId();
    if (modelId.empty()) {
        m_pendingLocalRvcChunks.clear();
        setStatusText(QStringLiteral("Select an RVC model first."));
        return;
    }

    auto chunk = std::move(m_pendingLocalRvcChunks.front());
    m_pendingLocalRvcChunks.pop_front();
    auto* audioEngine = &m_audioEngine;
    auto cancelFlag = m_localRvcCancelFlag;
    auto nativeEngine = m_nativeRvcEngine;
    if (nativeEngine != nullptr) {
        m_localRvcWatcher->setFuture(
            QtConcurrent::run([nativeEngine, chunk, audioEngine, cancelFlag]() {
                return convertNativeRvcChunk(nativeEngine, chunk, audioEngine, cancelFlag);
            }));
        return;
    }

    const auto endpoint = m_rvcSidecar.status().endpoint;
    m_localRvcWatcher->setFuture(
        QtConcurrent::run([endpoint, modelId, chunk, audioEngine, cancelFlag]() {
            return convertLocalRvcChunk(endpoint, modelId, chunk, audioEngine, cancelFlag);
        }));
}

void LiveMicPanel::saveCloudRecordingIfReady() {
    if (m_cloudActive || m_cloudWatcher->isRunning() || !m_pendingCloudChunks.empty() ||
        m_recordedCloudPcm.isEmpty()) {
        return;
    }
    if (!m_recordTakeCheck->isChecked()) {
        m_recordedCloudPcm.clear();
        return;
    }
    if (!m_project.has_value() || m_lineIdEdit->text().trimmed().isEmpty()) {
        setStatusText(QStringLiteral("Cloud conversion finished. STS take was not saved."));
        m_recordedCloudPcm.clear();
        return;
    }

    const auto bytes = std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(m_recordedCloudPcm.constData()),
        static_cast<std::size_t>(m_recordedCloudPcm.size())};
    auto audio = audio::pcm16LittleEndianToPcm(bytes, 44100, kCloudOutputChannels);
    if (!audio) {
        setStatusText(QString::fromStdString(audio.error().message));
        m_recordedCloudPcm.clear();
        return;
    }

    core::TakeManager takeManager;
    auto saved = takeManager.saveStsTake(m_project->rootPath(),
                                         m_lineIdEdit->text().trimmed().toStdString(),
                                         currentVoiceId(),
                                         audio.value(),
                                         core::defaultVoiceSettings());
    if (!saved) {
        setStatusText(QString::fromStdString(saved.error().message));
        m_recordedCloudPcm.clear();
        return;
    }

    m_recordedCloudPcm.clear();
    setStatusText(QStringLiteral("Cloud conversion saved as STS take."));
}

void LiveMicPanel::saveLocalRvcRecordingIfReady() {
    if (m_localRvcActive || m_localRvcWatcher->isRunning() ||
        !m_pendingLocalRvcChunks.empty()) {
        return;
    }
    if (m_recordedLocalRvcPcm.isEmpty()) {
        m_nativeRvcEngine.reset();
        return;
    }
    if (!m_recordTakeCheck->isChecked()) {
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }
    if (!m_project.has_value() || m_lineIdEdit->text().trimmed().isEmpty()) {
        setStatusText(QStringLiteral("Local RVC finished. Take was not saved."));
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }

    const auto bytes = std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(m_recordedLocalRvcPcm.constData()),
        static_cast<std::size_t>(m_recordedLocalRvcPcm.size())};
    auto audio = audio::pcm16LittleEndianToPcm(bytes,
                                               m_localRvcOutputSampleRate,
                                               m_localRvcOutputChannels);
    if (!audio) {
        setStatusText(QString::fromStdString(audio.error().message));
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }

    core::TakeManager takeManager;
    auto saved = takeManager.saveRvcLocalTake(m_project->rootPath(),
                                              m_lineIdEdit->text().trimmed().toStdString(),
                                              currentRvcModelId(),
                                              audio.value());
    if (!saved) {
        setStatusText(QString::fromStdString(saved.error().message));
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }

    m_recordedLocalRvcPcm.clear();
    m_nativeRvcEngine.reset();
    setStatusText(QStringLiteral("Local RVC saved as take."));
}

audio::CaptureConfig LiveMicPanel::currentCaptureConfig() const {
    return audio::CaptureConfig{comboDeviceIndex(m_inputDeviceCombo),
                                comboDeviceIndex(m_outputDeviceCombo),
                                m_frameMsSpin->value(),
                                static_cast<float>(m_gainSlider->value()) / 100.0F};
}

std::string LiveMicPanel::currentVoiceId() const {
    if (m_voiceCombo == nullptr || m_voiceCombo->currentIndex() < 0) {
        return {};
    }
    return m_voiceCombo->currentData().toString().toStdString();
}

std::string LiveMicPanel::currentRvcModelId() const {
    if (m_rvcModelCombo == nullptr || m_rvcModelCombo->currentIndex() < 0) {
        return {};
    }
    return m_rvcModelCombo->currentData().toString().toStdString();
}

bool LiveMicPanel::ensureCaptureRunning() {
    if (m_capture.stats().running) {
        return true;
    }

    auto started = m_capture.start(currentCaptureConfig());
    if (!started) {
        setStatusText(QString::fromStdString(started.error().message));
        return false;
    }

    startAudioProcessor();
    return true;
}

} // namespace voxstudio::ui
