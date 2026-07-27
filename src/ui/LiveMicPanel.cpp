#include "ui/LiveMicPanel.h"

#include "audio/AudioFile.h"
#include "net/elevenlabs/StsApi.h"
#include "rvc/OnnxRvcEngine.h"
#include "rvc/RvcClient.h"
#include "secrets/DpapiVault.h"
#include "ui/LiveAudioProcessor.h"
#include "ui/RvcModelManagerDialog.h"
#include "ui/TakeListWidget.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QMessageBox>
#include <QProgressBar>
#include <QProcess>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
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
constexpr int kCloudPlaybackTailMs = 200;
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
    return found == devices.end()
               ? -1
               : static_cast<int>(std::distance(devices.begin(), found));
}

[[nodiscard]] int comboDeviceIndex(const QComboBox* combo) {
    if (combo == nullptr || combo->currentIndex() < 0) {
        return -1;
    }
    return combo->currentData().toInt();
}

[[nodiscard]] int preferredVirtualOutputIndex(
    const std::vector<audio::AudioDeviceInfo>& devices) {
    static constexpr std::array preferredNames{
        "voicemod virtual audio device",
        "cable input",
        "voicemeeter input",
        "virtual audio cable",
    };

    for (const auto preferredName : preferredNames) {
        const auto found = std::ranges::find_if(devices, [preferredName](const auto& device) {
            return QString::fromStdString(device.name)
                .contains(QString::fromLatin1(preferredName), Qt::CaseInsensitive);
        });
        if (found != devices.end()) {
            return static_cast<int>(std::distance(devices.begin(), found));
        }
    }
    return -1;
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

[[nodiscard]] QString voiceBadgeText(QString name) {
    name = name.trimmed();
    if (name.isEmpty()) {
        return QStringLiteral("--");
    }

    QString badge;
    const auto words = name.split(QChar{' '}, Qt::SkipEmptyParts);
    for (const auto& word : words) {
        if (!word.isEmpty()) {
            badge += word.front().toUpper();
        }
        if (badge.size() == 2) {
            return badge;
        }
    }

    return name.left(std::min(name.size(), qsizetype{2})).toUpper();
}

[[nodiscard]] QLabel* addValueLabel(QLayout& layout, const QString& objectName) {
    auto* label = addOwnedWidget<QLabel>(layout, QStringLiteral("0"));
    label->setObjectName(objectName);
    label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    label->setMinimumWidth(40);
    return label;
}

void appendPlaybackWarning(QString& warning, const QString& route, const core::Error& error) {
    if (!warning.isEmpty()) {
        warning += QStringLiteral(" ");
    }
    warning += QStringLiteral("%1: %2").arg(route, QString::fromStdString(error.message));
}

[[nodiscard]] bool queuePcmForTargets(const PlaybackTargets& targets,
                                      const std::span<const std::uint8_t> bytes,
                                      const int sampleRate,
                                      const int channels,
                                      QString& playbackWarning) {
    bool queuedAny = false;
    const auto queue = [&](audio::AudioEngine* engine, const QString& route) {
        if (engine == nullptr || bytes.empty()) {
            return;
        }
        auto queued = engine->queuePcm16LittleEndian(bytes, sampleRate, channels);
        if (!queued) {
            appendPlaybackWarning(playbackWarning, route, queued.error());
            return;
        }
        queuedAny = true;
    };

    queue(targets.monitor, QStringLiteral("Monitor"));
    if (targets.broadcast != targets.monitor) {
        queue(targets.broadcast, QStringLiteral("Broadcast"));
    }
    return queuedAny;
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
    const PlaybackTargets playbackTargets,
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
    request.outputFormat = "pcm_24000";
    const auto outputSampleRate =
        net::elevenlabs::pcmSampleRateFromOutputFormat(request.outputFormat);

    const auto onChunk = [cancelFlag](std::span<const std::uint8_t>) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return false;
        }
        return true;
    };

    const net::elevenlabs::StsApi api{std::move(apiKey).value()};
    auto streamed = api.streamSpeech(request, onChunk);
    if (!streamed) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return CloudConversionResult{false, QStringLiteral("Cloud conversion cancelled.")};
        }
        return CloudConversionResult{false, apiErrorText(streamed.error())};
    }

    QString playbackWarning;
    const auto playbackQueued = queuePcmForTargets(playbackTargets,
                                                   streamed.value().audioBytes,
                                                   outputSampleRate,
                                                   kCloudOutputChannels,
                                                   playbackWarning);
    const auto outputFrames =
        streamed.value().audioBytes.size() / (sizeof(std::int16_t) * kCloudOutputChannels);
    const auto playbackDurationMs =
        playbackQueued
            ? static_cast<int>(std::ceil(
                  ((static_cast<double>(outputFrames) * 1000.0) /
                   static_cast<double>(outputSampleRate)) *
                  playbackTargets.durationScale))
            : 0;
    return CloudConversionResult{true,
                                 QStringLiteral("Character phrase converted."),
                                 playbackWarning,
                                 byteArrayFromBytes(streamed.value().audioBytes),
                                 streamed.value().inputSeconds,
                                 outputSampleRate,
                                 playbackDurationMs};
}

[[nodiscard]] LocalRvcConversionResult convertLocalRvcChunk(
    const std::string& endpoint,
    const std::string& modelId,
    const QByteArray& inputPcmBytes,
    const PlaybackTargets playbackTargets,
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
    const auto onChunk = [playbackTargets, &playbackWarning, cancelFlag](
                             std::span<const std::uint8_t> chunk) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return false;
        }
        if (chunk.empty()) {
            return true;
        }

        (void)queuePcmForTargets(playbackTargets,
                                 chunk,
                                 kLocalRvcSampleRate,
                                 kLocalRvcChannels,
                                 playbackWarning);
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
    const PlaybackTargets playbackTargets,
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
    (void)queuePcmForTargets(playbackTargets,
                             converted.value().pcm16Audio,
                             converted.value().sampleRate,
                             converted.value().channels,
                             playbackWarning);

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
    setObjectName(QStringLiteral("LiveMicPanel"));
    setStyleSheet(QStringLiteral(
        "#LiveMicPanel { background: #111517; color: #f4f7fb; }"
        "QGroupBox { border: 1px solid #30383d; border-radius: 6px; margin-top: 12px;"
        " padding: 12px 10px 10px 10px; font-weight: 600; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        "QComboBox, QLineEdit, QSpinBox { background: #1b2023; color: #f4f7fb;"
        " border: 1px solid #333b41; border-radius: 4px; padding: 5px; }"
        "QPushButton { background: #252b30; color: #f4f7fb; border: 1px solid #3a4248;"
        " border-radius: 5px; padding: 7px 10px; }"
        "QPushButton:checked { background: #10c7d8; color: #071113; border-color: #6eeef7; }"
        "QPushButton:disabled { color: #78828a; background: #1b2023; }"
        "#LiveMicPowerButton { min-width: 74px; min-height: 74px; border-radius: 37px;"
        " font-weight: 700; }"
        "#LiveMicHearButton, #LiveMicBroadcastButton { min-width: 82px; }"
        "#LiveMicSelectedVoiceBadge { min-width: 108px; min-height: 108px;"
        " border-radius: 54px; background: #10c7d8; color: #071113;"
        " font-size: 30px; font-weight: 800; border: 4px solid #80f5ff; }"
        "#LiveMicSelectedVoiceName { font-size: 18px; font-weight: 700; }"
        "#LiveMicSelectedEngineLabel, #LiveMicOutputRouteLabel, #LiveMicCostLabel,"
        " #LiveMicVadLabel { color: #aeb8c0; }"
        "#LiveMicTransport { background: #20262a; border: 1px solid #4a535a;"
        " border-radius: 8px; padding: 10px; }"
        "#LiveMicLevelMeter { min-height: 12px; text-align: center; }"));

    auto rootLayout = std::make_unique<QVBoxLayout>();
    rootLayout->setContentsMargins(18, 16, 18, 16);
    rootLayout->setSpacing(12);

    auto headerLayout = std::make_unique<QHBoxLayout>();
    auto* titleLabel = addOwnedWidget<QLabel>(*headerLayout, QStringLiteral("Live Voice Changer"));
    titleLabel->setObjectName(QStringLiteral("LiveMicTitle"));
    titleLabel->setStyleSheet(QStringLiteral("font-size: 22px; font-weight: 700;"));
    headerLayout->addStretch(1);
    m_refreshButton = addOwnedWidget<QPushButton>(*headerLayout, QStringLiteral("Refresh I/O"));
    m_refreshButton->setObjectName(QStringLiteral("LiveMicRefreshButton"));
    m_latencyButton = addOwnedWidget<QPushButton>(*headerLayout, QStringLiteral("Latency"));
    m_latencyButton->setObjectName(QStringLiteral("LiveMicLatencyButton"));
    rootLayout->addLayout(headerLayout.release());

    auto mainLayout = std::make_unique<QHBoxLayout>();
    mainLayout->setSpacing(14);

    auto centerLayout = std::make_unique<QVBoxLayout>();
    centerLayout->setSpacing(12);

    auto devicesGroup = std::make_unique<QGroupBox>(QStringLiteral("Audio I/O"));
    auto deviceLayout = std::make_unique<QGridLayout>();
    deviceLayout->setColumnStretch(1, 1);
    deviceLayout->setColumnStretch(3, 1);
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Input")).release(), 0, 0);
    m_inputDeviceCombo = addOwnedWidget<QComboBox>(*deviceLayout);
    m_inputDeviceCombo->setObjectName(QStringLiteral("LiveMicInputCombo"));
    deviceLayout->addWidget(m_inputDeviceCombo, 0, 1);
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Monitor")).release(), 0, 2);
    m_outputDeviceCombo = addOwnedWidget<QComboBox>(*deviceLayout);
    m_outputDeviceCombo->setObjectName(QStringLiteral("LiveMicOutputCombo"));
    deviceLayout->addWidget(m_outputDeviceCombo, 0, 3);
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Voice output")).release(), 1, 0);
    auto broadcastOutputCombo = std::make_unique<QComboBox>();
    m_broadcastOutputDeviceCombo = broadcastOutputCombo.get();
    m_broadcastOutputDeviceCombo->setObjectName(QStringLiteral("LiveMicBroadcastOutputCombo"));
    m_broadcastOutputDeviceCombo->setToolTip(
        QStringLiteral("Send the converted voice to a virtual audio line used as a microphone."));
    deviceLayout->addWidget(broadcastOutputCombo.release(), 1, 1, 1, 3);
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Input gain")).release(), 2, 0);
    m_gainSlider = addOwnedWidget<QSlider>(*deviceLayout, Qt::Horizontal);
    m_gainSlider->setObjectName(QStringLiteral("LiveMicGainSlider"));
    m_gainSlider->setRange(0, 400);
    m_gainSlider->setValue(100);
    deviceLayout->addWidget(m_gainSlider, 2, 1);
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Frame")).release(), 2, 2);
    m_frameMsSpin = addOwnedWidget<QSpinBox>(*deviceLayout);
    m_frameMsSpin->setObjectName(QStringLiteral("LiveMicFrameMsSpin"));
    m_frameMsSpin->setRange(10, audio::kMaxRealtimeFrameMs);
    m_frameMsSpin->setSingleStep(10);
    m_frameMsSpin->setValue(10);
    m_frameMsSpin->setSuffix(QStringLiteral(" ms"));
    deviceLayout->addWidget(m_frameMsSpin, 2, 3);
    devicesGroup->setLayout(deviceLayout.release());
    centerLayout->addWidget(devicesGroup.release());

    auto meterLayout = std::make_unique<QHBoxLayout>();
    meterLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Level")).release());
    m_levelMeter = addOwnedWidget<QProgressBar>(*meterLayout);
    m_levelMeter->setObjectName(QStringLiteral("LiveMicLevelMeter"));
    m_levelMeter->setRange(0, 100);
    m_levelMeter->setValue(0);
    m_vadLabel = addOwnedWidget<QLabel>(*meterLayout, QStringLiteral("Mic idle"));
    m_vadLabel->setObjectName(QStringLiteral("LiveMicVadLabel"));
    centerLayout->addLayout(meterLayout.release());

    auto transportFrame = std::make_unique<QFrame>();
    transportFrame->setObjectName(QStringLiteral("LiveMicTransport"));
    auto transportLayout = std::make_unique<QHBoxLayout>();
    transportLayout->setContentsMargins(12, 10, 12, 10);
    transportLayout->setSpacing(14);
    m_voicePowerButton = addOwnedWidget<QPushButton>(*transportLayout, QStringLiteral("Power"));
    m_voicePowerButton->setObjectName(QStringLiteral("LiveMicPowerButton"));
    m_voicePowerButton->setCheckable(true);
    m_voicePowerButton->setToolTip(QStringLiteral("Start or stop the selected voice changer."));
    m_selectedVoiceBadge = addOwnedWidget<QLabel>(*transportLayout, QStringLiteral("--"));
    m_selectedVoiceBadge->setObjectName(QStringLiteral("LiveMicSelectedVoiceBadge"));
    m_selectedVoiceBadge->setAlignment(Qt::AlignCenter);
    auto transportTextLayout = std::make_unique<QVBoxLayout>();
    m_selectedVoiceLabel =
        addOwnedWidget<QLabel>(*transportTextLayout, QStringLiteral("No voice selected"));
    m_selectedVoiceLabel->setObjectName(QStringLiteral("LiveMicSelectedVoiceName"));
    m_selectedEngineLabel =
        addOwnedWidget<QLabel>(*transportTextLayout, QStringLiteral("ElevenLabs performance"));
    m_selectedEngineLabel->setObjectName(QStringLiteral("LiveMicSelectedEngineLabel"));
    m_outputRouteLabel = addOwnedWidget<QLabel>(*transportTextLayout,
                                                QStringLiteral("Output: default"));
    m_outputRouteLabel->setObjectName(QStringLiteral("LiveMicOutputRouteLabel"));
    m_outputRouteLabel->setWordWrap(true);
    transportLayout->addLayout(transportTextLayout.release(), 1);
    m_monitorButton = addOwnedWidget<QPushButton>(*transportLayout, QStringLiteral("Hear"));
    m_monitorButton->setObjectName(QStringLiteral("LiveMicHearButton"));
    m_monitorButton->setCheckable(true);
    m_monitorButton->setChecked(false);
    m_monitorButton->setToolTip(
        QStringLiteral("Play changed character phrases through the selected headphones."));
    m_liveInputButton =
        addOwnedWidget<QPushButton>(*transportLayout, QStringLiteral("Live Input Off"));
    m_liveInputButton->setObjectName(QStringLiteral("LiveMicInputMonitorButton"));
    m_liveInputButton->setCheckable(true);
    m_liveInputButton->setChecked(false);
    m_liveInputButton->setToolTip(
        QStringLiteral("Hear your unchanged microphone while recording a performance."));
    m_broadcastButton =
        addOwnedWidget<QPushButton>(*transportLayout, QStringLiteral("Broadcast Off"));
    m_broadcastButton->setObjectName(QStringLiteral("LiveMicBroadcastButton"));
    m_broadcastButton->setCheckable(true);
    m_broadcastButton->setToolTip(
        QStringLiteral("Send the converted voice to the selected virtual microphone line."));
    m_costLabel =
        addOwnedWidget<QLabel>(*transportLayout, QStringLiteral("Character audio: 0.0 s"));
    m_costLabel->setObjectName(QStringLiteral("LiveMicCostLabel"));
    transportFrame->setLayout(transportLayout.release());
    centerLayout->addWidget(transportFrame.release());

    auto quickSlotsGroup = std::make_unique<QGroupBox>(QStringLiteral("Character Slots"));
    auto quickSlotsLayout = std::make_unique<QHBoxLayout>();
    quickSlotsLayout->setSpacing(8);
    for (std::size_t index = 0; index < m_quickVoiceButtons.size(); ++index) {
        auto* button = addOwnedWidget<QPushButton>(
            *quickSlotsLayout, QStringLiteral("%1").arg(static_cast<int>(index + 1U)));
        button->setObjectName(QStringLiteral("LiveMicQuickVoiceSlot%1")
                                  .arg(static_cast<int>(index + 1U)));
        button->setCheckable(true);
        button->setProperty("voiceIndex", static_cast<int>(index));
        button->setEnabled(false);
        m_quickVoiceButtons[index] = button;
    }
    quickSlotsGroup->setLayout(quickSlotsLayout.release());
    centerLayout->addWidget(quickSlotsGroup.release());

    auto advancedGroup = std::make_unique<QGroupBox>(QStringLiteral("Take Capture"));
    auto advancedLayout = std::make_unique<QGridLayout>();
    m_recordTakeCheck =
        addOwnedWidget<QCheckBox>(*advancedLayout, QStringLiteral("Save converted take"));
    m_recordTakeCheck->setObjectName(QStringLiteral("LiveMicRecordTakeCheck"));
    m_recordTakeCheck->setChecked(true);
    advancedLayout->addWidget(m_recordTakeCheck, 0, 0);
    m_lineIdEdit = addOwnedWidget<QLineEdit>(*advancedLayout);
    m_lineIdEdit->setObjectName(QStringLiteral("LiveMicLineIdEdit"));
    m_lineIdEdit->setPlaceholderText(QStringLiteral("Type the line to perform"));
    advancedLayout->addWidget(m_lineIdEdit, 0, 1);
    m_cloudButton =
        addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Record Performance"));
    m_cloudButton->setObjectName(QStringLiteral("LiveMicCloudButton"));
    advancedLayout->addWidget(m_cloudButton, 1, 0);
    m_cancelCloudButton = addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Cancel"));
    m_cancelCloudButton->setObjectName(QStringLiteral("LiveMicCancelCloudButton"));
    m_cancelCloudButton->setEnabled(false);
    advancedLayout->addWidget(m_cancelCloudButton, 1, 1);
    m_localRvcButton = addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Start Local"));
    m_localRvcButton->setObjectName(QStringLiteral("LiveMicLocalRvcButton"));
    advancedLayout->addWidget(m_localRvcButton, 2, 0);
    m_cancelLocalRvcButton =
        addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Cancel Local"));
    m_cancelLocalRvcButton->setObjectName(QStringLiteral("LiveMicCancelLocalRvcButton"));
    m_cancelLocalRvcButton->setEnabled(false);
    advancedLayout->addWidget(m_cancelLocalRvcButton, 2, 1);
    advancedGroup->setLayout(advancedLayout.release());
    centerLayout->addWidget(advancedGroup.release());

    auto recentTakes = std::make_unique<TakeListWidget>(this);
    recentTakes->setObjectName(QStringLiteral("LiveMicRecentTakes"));
    recentTakes->setTitle(QStringLiteral("Recent Takes"));
    m_recentTakesWidget = recentTakes.get();
    centerLayout->addWidget(recentTakes.release(), 1);
    centerLayout->addStretch(1);
    mainLayout->addLayout(centerLayout.release(), 1);

    auto rightPanel = std::make_unique<QGroupBox>(QStringLiteral("Selected Voice"));
    rightPanel->setMinimumWidth(300);
    rightPanel->setMaximumWidth(380);
    auto rightLayout = std::make_unique<QGridLayout>();
    rightLayout->setColumnStretch(1, 1);
    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Mode")).release(), 0, 0);
    m_modeCombo = addOwnedWidget<QComboBox>(*rightLayout);
    m_modeCombo->setObjectName(QStringLiteral("LiveMicModeCombo"));
    m_modeCombo->addItem(QStringLiteral("Mic Check"));
    m_modeCombo->addItem(QStringLiteral("Performance"));
    m_modeCombo->addItem(QStringLiteral("Local"));
    rightLayout->addWidget(m_modeCombo, 0, 1);
    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Character")).release(), 1, 0);
    m_voiceCombo = addOwnedWidget<QComboBox>(*rightLayout);
    m_voiceCombo->setObjectName(QStringLiteral("LiveMicVoiceCombo"));
    rightLayout->addWidget(m_voiceCombo, 1, 1);
    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("RVC Model")).release(), 2, 0);
    m_rvcModelCombo = addOwnedWidget<QComboBox>(*rightLayout);
    m_rvcModelCombo->setObjectName(QStringLiteral("LiveMicRvcModelCombo"));
    rightLayout->addWidget(m_rvcModelCombo, 2, 1);
    m_manageRvcModelsButton = addOwnedWidget<QPushButton>(*rightLayout,
                                                          QStringLiteral("RVC Models"));
    m_manageRvcModelsButton->setObjectName(QStringLiteral("LiveMicManageRvcModelsButton"));
    rightLayout->addWidget(m_manageRvcModelsButton, 3, 0, 1, 2);

    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Voice volume")).release(), 4, 0);
    m_voiceVolumeSlider = addOwnedWidget<QSlider>(*rightLayout, Qt::Horizontal);
    m_voiceVolumeSlider->setObjectName(QStringLiteral("LiveMicVoiceVolumeSlider"));
    m_voiceVolumeSlider->setRange(0, 200);
    m_voiceVolumeSlider->setValue(100);
    rightLayout->addWidget(m_voiceVolumeSlider, 4, 1);
    m_voiceVolumeValueLabel =
        addValueLabel(*rightLayout, QStringLiteral("LiveMicVoiceVolumeValueLabel"));
    rightLayout->addWidget(m_voiceVolumeValueLabel, 4, 2);

    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Bass")).release(), 5, 0);
    m_bassSlider = addOwnedWidget<QSlider>(*rightLayout, Qt::Horizontal);
    m_bassSlider->setObjectName(QStringLiteral("LiveMicBassSlider"));
    m_bassSlider->setRange(-12, 12);
    m_bassSlider->setValue(0);
    rightLayout->addWidget(m_bassSlider, 5, 1);
    m_bassValueLabel = addValueLabel(*rightLayout, QStringLiteral("LiveMicBassValueLabel"));
    rightLayout->addWidget(m_bassValueLabel, 5, 2);

    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Mid")).release(), 6, 0);
    m_midSlider = addOwnedWidget<QSlider>(*rightLayout, Qt::Horizontal);
    m_midSlider->setObjectName(QStringLiteral("LiveMicMidSlider"));
    m_midSlider->setRange(-12, 12);
    m_midSlider->setValue(0);
    rightLayout->addWidget(m_midSlider, 6, 1);
    m_midValueLabel = addValueLabel(*rightLayout, QStringLiteral("LiveMicMidValueLabel"));
    rightLayout->addWidget(m_midValueLabel, 6, 2);

    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Treble")).release(), 7, 0);
    m_trebleSlider = addOwnedWidget<QSlider>(*rightLayout, Qt::Horizontal);
    m_trebleSlider->setObjectName(QStringLiteral("LiveMicTrebleSlider"));
    m_trebleSlider->setRange(-12, 12);
    m_trebleSlider->setValue(0);
    rightLayout->addWidget(m_trebleSlider, 7, 1);
    m_trebleValueLabel = addValueLabel(*rightLayout, QStringLiteral("LiveMicTrebleValueLabel"));
    rightLayout->addWidget(m_trebleValueLabel, 7, 2);

    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Pitch")).release(), 8, 0);
    m_pitchSlider = addOwnedWidget<QSlider>(*rightLayout, Qt::Horizontal);
    m_pitchSlider->setObjectName(QStringLiteral("LiveMicPitchSlider"));
    m_pitchSlider->setRange(-24, 24);
    m_pitchSlider->setValue(0);
    rightLayout->addWidget(m_pitchSlider, 8, 1);
    m_pitchValueLabel = addValueLabel(*rightLayout, QStringLiteral("LiveMicPitchValueLabel"));
    rightLayout->addWidget(m_pitchValueLabel, 8, 2);
    rightLayout->setRowStretch(9, 1);
    rightPanel->setLayout(rightLayout.release());
    mainLayout->addWidget(rightPanel.release());
    rootLayout->addLayout(mainLayout.release(), 1);

    m_monitorCheck = addOwnedWidget<QCheckBox>(*rootLayout, QStringLiteral("Monitor"));
    m_monitorCheck->setObjectName(QStringLiteral("LiveMicMonitorToggle"));
    m_monitorCheck->setChecked(false);
    m_monitorCheck->setVisible(false);

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setObjectName(QStringLiteral("LiveMicStatusLabel"));
    m_statusLabel->setWordWrap(true);

    setLayout(rootLayout.release());

    connect(m_refreshButton, &QPushButton::clicked, this, &LiveMicPanel::refreshDevices);
    connect(m_monitorCheck, &QCheckBox::toggled, this, &LiveMicPanel::toggleMonitor);
    connect(m_monitorButton, &QPushButton::toggled, this, &LiveMicPanel::setHearSelfChecked);
    connect(m_liveInputButton, &QPushButton::toggled, this,
            &LiveMicPanel::setLiveInputChecked);
    connect(m_broadcastButton, &QPushButton::toggled, this, &LiveMicPanel::setBroadcastChecked);
    connect(m_voicePowerButton, &QPushButton::clicked, this,
            &LiveMicPanel::toggleVoiceChangerPower);
    connect(m_inputDeviceCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &LiveMicPanel::updateOutputRoute);
    connect(m_outputDeviceCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &LiveMicPanel::updateOutputRoute);
    connect(m_broadcastOutputDeviceCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &LiveMicPanel::updateOutputRoute);
    connect(m_modeCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &LiveMicPanel::updateVoiceHud);
    connect(m_voiceCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &LiveMicPanel::updateVoiceHud);
    connect(m_rvcModelCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &LiveMicPanel::updateVoiceHud);
    connect(m_gainSlider, &QSlider::valueChanged, this, &LiveMicPanel::updateGain);
    connect(m_voiceVolumeSlider, &QSlider::valueChanged, this, &LiveMicPanel::updateVoiceFx);
    connect(m_bassSlider, &QSlider::valueChanged, this, &LiveMicPanel::updateVoiceFx);
    connect(m_midSlider, &QSlider::valueChanged, this, &LiveMicPanel::updateVoiceFx);
    connect(m_trebleSlider, &QSlider::valueChanged, this, &LiveMicPanel::updateVoiceFx);
    connect(m_pitchSlider, &QSlider::valueChanged, this, &LiveMicPanel::updateVoiceFx);
    for (auto* quickVoiceButton : m_quickVoiceButtons) {
        connect(quickVoiceButton, &QPushButton::clicked, this, &LiveMicPanel::selectQuickVoiceSlot);
    }
    connect(m_latencyButton, &QPushButton::clicked, this, &LiveMicPanel::testLatency);
    connect(m_cloudButton, &QPushButton::clicked, this, &LiveMicPanel::toggleCloudConversion);
    connect(m_cancelCloudButton, &QPushButton::clicked, this, &LiveMicPanel::cancelCloudConversion);
    connect(m_localRvcButton, &QPushButton::clicked, this,
            &LiveMicPanel::toggleLocalRvcConversion);
    connect(m_cancelLocalRvcButton, &QPushButton::clicked, this,
            &LiveMicPanel::cancelLocalRvcConversion);
    connect(m_manageRvcModelsButton, &QPushButton::clicked, this,
            &LiveMicPanel::openRvcModelManager);
    connect(m_recentTakesWidget, &TakeListWidget::playTakeRequested, this,
            &LiveMicPanel::playTake);
    connect(m_recentTakesWidget, &TakeListWidget::starTakeRequested, this,
            &LiveMicPanel::starTake);
    connect(m_recentTakesWidget, &TakeListWidget::revealTakeRequested, this,
            &LiveMicPanel::revealTake);
    connect(m_recentTakesWidget, &TakeListWidget::deleteTakeRequested, this,
            &LiveMicPanel::deleteTake);
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
    updateVoiceFx();
    updateVoiceHud();
    updateTransportState();
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
    m_recordingLineId.clear();
    m_recordingLineVoiceId.clear();
    m_recordingRvcModelId.clear();
    m_recordingLineText.clear();
    refreshVoices();
    refreshRvcModels();
    refreshRecentTakes();
}

void LiveMicPanel::refreshDevices() {
    const QSignalBlocker inputBlocker{m_inputDeviceCombo};
    const QSignalBlocker outputBlocker{m_outputDeviceCombo};
    const QSignalBlocker broadcastOutputBlocker{m_broadcastOutputDeviceCombo};
    m_inputDeviceCombo->clear();
    m_outputDeviceCombo->clear();
    m_broadcastOutputDeviceCombo->clear();
    m_inputDevices.clear();
    m_outputDevices.clear();
    {
        const QSignalBlocker broadcastButtonBlocker{m_broadcastButton};
        m_broadcastButton->setChecked(false);
    }
    m_broadcastButton->setEnabled(false);

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
            m_broadcastOutputDeviceCombo->addItem(deviceLabel(device), device.index);
        }
        const auto defaultIndex = defaultDeviceIndex(m_outputDevices);
        if (defaultIndex >= 0) {
            m_outputDeviceCombo->setCurrentIndex(defaultIndex);
        }
        const auto virtualIndex = preferredVirtualOutputIndex(m_outputDevices);
        if (virtualIndex >= 0) {
            m_broadcastOutputDeviceCombo->setCurrentIndex(virtualIndex);
        }
        {
            const QSignalBlocker broadcastButtonBlocker{m_broadcastButton};
            m_broadcastButton->setChecked(false);
        }
        m_broadcastButton->setEnabled(!m_outputDevices.empty());
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
    updateOutputRoute(m_outputDeviceCombo->currentIndex());
    updateTransportState();
}

void LiveMicPanel::refreshVoices() {
    m_voiceCombo->clear();
    for (auto* button : m_quickVoiceButtons) {
        button->setText(QStringLiteral("--"));
        button->setEnabled(false);
        button->setChecked(false);
        button->setToolTip(QString{});
    }
    if (!m_project.has_value()) {
        m_voiceCombo->setEnabled(false);
        m_cloudButton->setEnabled(false);
        updateVoiceHud();
        return;
    }

    auto voices = m_voiceRepository.listVoices(m_project->rootPath());
    if (!voices) {
        m_voiceCombo->setEnabled(false);
        m_cloudButton->setEnabled(false);
        setStatusText(QString::fromStdString(voices.error().message));
        updateVoiceHud();
        return;
    }

    auto voiceRecords = std::move(voices).value();
    const auto priority = [](const db::VoiceRecord& voice) {
        return voice.origin == "ivc" || voice.origin == "pvc" ? 0 : 1;
    };
    std::stable_sort(voiceRecords.begin(),
                     voiceRecords.end(),
                     [&priority](const auto& left, const auto& right) {
                         return priority(left) < priority(right);
                     });

    int voiceIndex = 0;
    for (const auto& voice : voiceRecords) {
        const auto name = QString::fromStdString(voice.name);
        m_voiceCombo->addItem(name, QString::fromStdString(voice.id));
        m_voiceCombo->setItemData(
            voiceIndex,
            QStringLiteral("%1 character voice").arg(QString::fromStdString(voice.origin)),
            Qt::ToolTipRole);
        if (voiceIndex < static_cast<int>(m_quickVoiceButtons.size())) {
            auto* button = m_quickVoiceButtons[static_cast<std::size_t>(voiceIndex)];
            button->setText(voiceBadgeText(name));
            button->setToolTip(
                QStringLiteral("%1\n%2 character voice")
                    .arg(name, QString::fromStdString(voice.origin)));
            button->setEnabled(true);
        }
        ++voiceIndex;
    }
    const bool hasVoices = m_voiceCombo->count() > 0;
    m_voiceCombo->setEnabled(hasVoices);
    m_cloudButton->setEnabled(hasVoices);
    if (hasVoices && m_modeCombo != nullptr &&
        m_modeCombo->currentText() == QStringLiteral("Mic Check") &&
        !m_capture.stats().running) {
        m_modeCombo->setCurrentText(QStringLiteral("Performance"));
    }
    if (!hasVoices) {
        setStatusText(QStringLiteral("Sync or clone a voice before a character performance."));
    }
    updateVoiceHud();
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
    updateVoiceHud();
}

void LiveMicPanel::updateVoiceFx() {
    if (m_voiceVolumeSlider == nullptr) {
        return;
    }

    const auto volume = static_cast<float>(m_voiceVolumeSlider->value()) / 100.0F;
    const auto bass = m_bassSlider == nullptr ? 0 : m_bassSlider->value();
    const auto mid = m_midSlider == nullptr ? 0 : m_midSlider->value();
    const auto treble = m_trebleSlider == nullptr ? 0 : m_trebleSlider->value();
    const auto pitch = currentPitchShiftSemitones();
    const audio::OutputFxSettings settings{volume,
                                           static_cast<float>(bass),
                                           static_cast<float>(mid),
                                           static_cast<float>(treble),
                                           pitch};
    m_audioEngine.setOutputFxSettings(settings);
    m_broadcastAudioEngine.setOutputFxSettings(settings);

    if (m_voiceVolumeValueLabel != nullptr) {
        m_voiceVolumeValueLabel->setText(QString::number(m_voiceVolumeSlider->value()));
    }
    if (m_bassValueLabel != nullptr) {
        m_bassValueLabel->setText(QString::number(bass));
    }
    if (m_midValueLabel != nullptr) {
        m_midValueLabel->setText(QString::number(mid));
    }
    if (m_trebleValueLabel != nullptr) {
        m_trebleValueLabel->setText(QString::number(treble));
    }
    if (m_pitchValueLabel != nullptr) {
        m_pitchValueLabel->setText(QString::number(pitch));
    }
}

void LiveMicPanel::updateOutputRoute(int) {
    const auto outputIndex = comboDeviceIndex(m_outputDeviceCombo);
    auto routed = m_audioEngine.setOutputDeviceIndex(outputIndex);
    if (!routed) {
        setStatusText(QString::fromStdString(routed.error().message));
    }
    const auto broadcastOutputIndex = comboDeviceIndex(m_broadcastOutputDeviceCombo);
    auto broadcastRouted = m_broadcastAudioEngine.setOutputDeviceIndex(broadcastOutputIndex);
    if (!broadcastRouted) {
        setStatusText(QString::fromStdString(broadcastRouted.error().message));
    }
    updateVoiceHud();
}

void LiveMicPanel::setBroadcastChecked(const bool enabled) {
    if (m_broadcastButton != nullptr) {
        const QSignalBlocker blocker{m_broadcastButton};
        m_broadcastButton->setChecked(enabled);
    }
    if (!enabled) {
        m_broadcastAudioEngine.clear();
        setStatusText(QStringLiteral("Virtual microphone output muted."));
    } else if (m_broadcastOutputDeviceCombo == nullptr ||
               m_broadcastOutputDeviceCombo->currentIndex() < 0) {
        setStatusText(QStringLiteral("Select a virtual audio output before broadcasting."));
    } else {
        setStatusText(QStringLiteral("Converted voice will feed %1.")
                          .arg(m_broadcastOutputDeviceCombo->currentText()));
    }
    updateTransportState();
}

void LiveMicPanel::toggleVoiceChangerPower() {
    if (m_cloudActive) {
        toggleCloudConversion();
        updateTransportState();
        return;
    }
    if (m_localRvcActive) {
        toggleLocalRvcConversion();
        updateTransportState();
        return;
    }

    const auto selectedMode = m_modeCombo == nullptr ? QString{} : m_modeCombo->currentText();
    if (selectedMode == QStringLiteral("Mic Check")) {
        setHearSelfChecked(!m_capture.stats().running);
        return;
    }
    if (selectedMode == QStringLiteral("Local") && !currentRvcModelId().empty()) {
        setHearSelfChecked(true);
        toggleLocalRvcConversion();
        updateTransportState();
        return;
    }

    if (!currentVoiceId().empty()) {
        m_modeCombo->setCurrentText(QStringLiteral("Performance"));
        setHearSelfChecked(true);
        toggleCloudConversion();
        updateTransportState();
        return;
    }

    if (!currentRvcModelId().empty()) {
        m_modeCombo->setCurrentText(QStringLiteral("Local"));
        setHearSelfChecked(true);
        toggleLocalRvcConversion();
        updateTransportState();
        return;
    }

    setStatusText(QStringLiteral("Clone or import a voice before starting the voice changer."));
    updateTransportState();
}

void LiveMicPanel::selectQuickVoiceSlot() {
    const auto* button = qobject_cast<QPushButton*>(sender());
    if (button == nullptr || m_voiceCombo == nullptr) {
        return;
    }

    const auto voiceIndex = button->property("voiceIndex").toInt();
    if (voiceIndex < 0 || voiceIndex >= m_voiceCombo->count()) {
        return;
    }

    m_voiceCombo->setCurrentIndex(voiceIndex);
    m_modeCombo->setCurrentText(QStringLiteral("Performance"));
    updateVoiceHud();
}

void LiveMicPanel::updateVoiceHud() {
    const auto mode =
        m_modeCombo == nullptr ? QStringLiteral("Mic Check") : m_modeCombo->currentText();
    const auto voiceName = currentVoiceName();
    const auto rvcName = currentRvcModelName();
    const auto selectedName =
        mode == QStringLiteral("Local") && !rvcName.isEmpty() ? rvcName : voiceName;

    if (m_selectedVoiceLabel != nullptr) {
        m_selectedVoiceLabel->setText(selectedName.isEmpty() ? QStringLiteral("No voice selected")
                                                             : selectedName);
    }
    if (m_selectedVoiceBadge != nullptr) {
        m_selectedVoiceBadge->setText(voiceBadgeText(selectedName));
    }
    if (m_selectedEngineLabel != nullptr) {
        const auto engineText =
            mode == QStringLiteral("Local") ? QStringLiteral("Local RVC engine")
            : mode == QStringLiteral("Performance")
                ? QStringLiteral("ElevenLabs performance (converts after each pause)")
                : QStringLiteral("Direct microphone monitor");
        m_selectedEngineLabel->setText(engineText);
    }
    if (m_outputRouteLabel != nullptr) {
        const auto monitorRoute =
            m_outputDeviceCombo != nullptr && m_outputDeviceCombo->currentIndex() >= 0
                ? m_outputDeviceCombo->currentText()
                : QStringLiteral("default");
        const auto broadcastRoute =
            m_broadcastOutputDeviceCombo != nullptr &&
                    m_broadcastOutputDeviceCombo->currentIndex() >= 0
                ? m_broadcastOutputDeviceCombo->currentText()
                : QStringLiteral("none");
        m_outputRouteLabel->setText(
            QStringLiteral("Monitor: %1 | Voice output: %2").arg(monitorRoute, broadcastRoute));
    }

    for (std::size_t index = 0; index < m_quickVoiceButtons.size(); ++index) {
        auto* button = m_quickVoiceButtons[index];
        if (button == nullptr) {
            continue;
        }
        const QSignalBlocker blocker{button};
        button->setChecked(m_voiceCombo != nullptr &&
                           m_voiceCombo->currentIndex() == static_cast<int>(index));
    }
    updateTransportState();
}

void LiveMicPanel::updateTransportState() {
    if (m_voicePowerButton != nullptr) {
        const QSignalBlocker blocker{m_voicePowerButton};
        const bool micCheckMode =
            m_modeCombo != nullptr &&
            m_modeCombo->currentText() == QStringLiteral("Mic Check");
        const bool microphoneCheckActive =
            micCheckMode && !m_cloudActive && !m_localRvcActive && m_capture.stats().running;
        const bool powerActive = m_cloudActive || m_localRvcActive || microphoneCheckActive;
        m_voicePowerButton->setChecked(powerActive);
        m_voicePowerButton->setText(powerActive ? QStringLiteral("On")
                                                 : QStringLiteral("Power"));
    }
    if (m_liveInputButton != nullptr) {
        const QSignalBlocker blocker{m_liveInputButton};
        m_liveInputButton->setText(m_liveInputButton->isChecked()
                                       ? QStringLiteral("Live Input On")
                                       : QStringLiteral("Live Input Off"));
    }
    if (m_monitorButton != nullptr && m_monitorCheck != nullptr) {
        const QSignalBlocker blocker{m_monitorButton};
        m_monitorButton->setChecked(m_monitorCheck->isChecked());
        const bool micCheckMode =
            m_modeCombo != nullptr &&
            m_modeCombo->currentText() == QStringLiteral("Mic Check");
        if (micCheckMode) {
            m_monitorButton->setText(m_monitorCheck->isChecked()
                                         ? QStringLiteral("Mic Check On")
                                         : QStringLiteral("Mic Check"));
        } else {
            m_monitorButton->setText(m_monitorCheck->isChecked()
                                         ? QStringLiteral("Hear Result On")
                                         : QStringLiteral("Hear Result"));
        }
    }
    if (m_broadcastButton != nullptr) {
        const QSignalBlocker blocker{m_broadcastButton};
        m_broadcastButton->setText(m_broadcastButton->isChecked()
                                       ? QStringLiteral("Broadcast On")
                                       : QStringLiteral("Broadcast Off"));
    }
}

void LiveMicPanel::setHearSelfChecked(const bool enabled) {
    if (m_monitorButton != nullptr) {
        const QSignalBlocker blocker{m_monitorButton};
        m_monitorButton->setChecked(enabled);
    }
    if (m_monitorCheck != nullptr) {
        const QSignalBlocker blocker{m_monitorCheck};
        m_monitorCheck->setChecked(enabled);
    }

    if (m_cloudActive || m_localRvcActive) {
        if (!enabled) {
            m_audioEngine.clear();
            setStatusText(QStringLiteral("Converted monitoring muted."));
        } else {
            setStatusText(QStringLiteral("Converted voice monitoring enabled."));
        }
        updateTransportState();
        return;
    }

    const bool micCheckMode =
        m_modeCombo != nullptr &&
        m_modeCombo->currentText() == QStringLiteral("Mic Check");
    if (micCheckMode) {
        toggleMonitor(enabled);
        return;
    }

    setStatusText(enabled
                      ? QStringLiteral("Changed-voice playback is armed for the next performance.")
                      : QStringLiteral("Changed-voice playback muted."));
    updateTransportState();
}

void LiveMicPanel::setLiveInputChecked(const bool enabled) {
    if (m_liveInputButton != nullptr) {
        const QSignalBlocker blocker{m_liveInputButton};
        m_liveInputButton->setChecked(enabled);
    }

    if (m_cloudActive || m_localRvcActive) {
        m_capture.setMonitorEnabled(enabled);
        setProcessorPassthrough(enabled, Qt::QueuedConnection);
        setStatusText(enabled
                          ? QStringLiteral("Live input monitor enabled; changed phrases remain audible.")
                          : QStringLiteral("Live input monitor muted; changed phrases remain audible."));
    } else {
        setStatusText(enabled ? QStringLiteral("Live input monitor will stay on while recording.")
                              : QStringLiteral("Live input monitor will stay muted while recording."));
    }
    updateTransportState();
}

void LiveMicPanel::toggleMonitor(const bool enabled) {
    if (!enabled) {
        if (m_cloudActive || m_localRvcActive) {
            m_audioEngine.clear();
            m_capture.setMonitorEnabled(false);
            setProcessorPassthrough(false, Qt::QueuedConnection);
            setStatusText(QStringLiteral("Converted monitoring muted."));
            updateTransportState();
            return;
        }

        stopAudioProcessor(Qt::BlockingQueuedConnection);
        m_capture.stop();
        m_levelMeter->setValue(0);
        m_vadLabel->setText(QStringLiteral("Mic idle"));
        setStatusText(QStringLiteral("Microphone check stopped."));
        updateTransportState();
        return;
    }

    if (!ensureCaptureRunning()) {
        m_monitorCheck->blockSignals(true);
        m_monitorCheck->setChecked(false);
        m_monitorCheck->blockSignals(false);
        if (m_monitorButton != nullptr) {
            const QSignalBlocker blocker{m_monitorButton};
            m_monitorButton->setChecked(false);
        }
        updateTransportState();
        return;
    }

    const bool passthrough = !m_cloudActive && !m_localRvcActive;
    m_capture.setMonitorEnabled(passthrough);
    setProcessorPassthrough(passthrough, Qt::QueuedConnection);
    setStatusText(passthrough ? QStringLiteral("Microphone check active. Speak and watch the meter.")
                              : QStringLiteral("Converted voice monitoring enabled."));
    updateTransportState();
}

void LiveMicPanel::toggleCloudConversion() {
    if (m_localRvcActive) {
        setStatusText(QStringLiteral("Stop Local RVC before starting a cloud performance."));
        return;
    }

    if (m_cloudActive) {
        m_cloudActive = false;
        m_cloudButton->setText(QStringLiteral("Record Performance"));
        m_cancelCloudButton->setEnabled(false);
        setProcessorCloudCapture(false, Qt::BlockingQueuedConnection);
        const bool keepLiveInput =
            m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
        setStatusText(QStringLiteral("Finishing the last character phrase."));
        updateTransportState();
        saveCloudRecordingIfReady();
        return;
    }

    const auto voiceId = currentVoiceId();
    if (voiceId.empty()) {
        setStatusText(QStringLiteral("Select a target voice first."));
        return;
    }
    if (!ensureRecordingLine()) {
        return;
    }
    if (!ensureCaptureRunning()) {
        return;
    }

    m_cloudCancelFlag = std::make_shared<std::atomic_bool>(false);
    m_pendingCloudChunks.clear();
    m_recordedCloudPcm.clear();
    ++m_cloudPlaybackGeneration;
    m_cloudPlaybackGuardActive = false;
    m_cloudOutputSampleRate = 24000;
    m_cloudSeconds = 0.0;
    m_costLabel->setText(QStringLiteral("Character audio: 0.0 s"));
    m_cloudActive = true;
    m_cloudButton->setText(QStringLiteral("Stop & Save"));
    m_cancelCloudButton->setEnabled(true);
    m_modeCombo->setCurrentText(QStringLiteral("Performance"));
    const bool liveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
    m_capture.setMonitorEnabled(liveInput);
    setProcessorPassthrough(liveInput, Qt::BlockingQueuedConnection);
    setProcessorCloudCapturePaused(false, Qt::BlockingQueuedConnection);
    setProcessorCloudCapture(true, Qt::BlockingQueuedConnection);
    setStatusText(
        liveInput
            ? QStringLiteral(
                  "Performance active. Live input is on; the character plays after each pause.")
            : QStringLiteral(
                  "Performance active. Speak naturally; the character plays after each pause."));
    updateTransportState();
}

void LiveMicPanel::cancelCloudConversion() {
    if (m_cloudCancelFlag != nullptr) {
        m_cloudCancelFlag->store(true, std::memory_order_release);
    }
    m_cloudActive = false;
    ++m_cloudPlaybackGeneration;
    m_cloudPlaybackGuardActive = false;
    m_pendingCloudChunks.clear();
    m_recordedCloudPcm.clear();
    m_audioEngine.clear();
    m_broadcastAudioEngine.clear();
    m_cloudButton->setText(QStringLiteral("Record Performance"));
    m_cancelCloudButton->setEnabled(false);
    if (m_audioThread.isRunning()) {
        setProcessorCloudCapturePaused(false, Qt::BlockingQueuedConnection);
        setProcessorCloudCapture(false, Qt::BlockingQueuedConnection);
    }
    if (!m_localRvcActive) {
        const bool keepLiveInput =
            m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
    }
    setStatusText(QStringLiteral("Cloud conversion cancelled."));
    updateTransportState();
}

void LiveMicPanel::toggleLocalRvcConversion() {
    if (m_localRvcActive) {
        m_localRvcActive = false;
        m_localRvcButton->setText(QStringLiteral("Start Local"));
        m_cancelLocalRvcButton->setEnabled(false);
        setProcessorLocalRvcCapture(false, Qt::BlockingQueuedConnection);
        const bool keepLiveInput =
            m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
        m_rvcSidecar.stop();
        if (m_pendingLocalRvcChunks.empty() && !m_localRvcWatcher->isRunning()) {
            m_nativeRvcEngine.reset();
        }
        setStatusText(QStringLiteral("Finishing queued Local RVC conversion."));
        updateTransportState();
        saveLocalRvcRecordingIfReady();
        return;
    }

    if (m_cloudActive) {
        setStatusText(QStringLiteral("Stop Performance before starting Local RVC."));
        return;
    }

    const auto modelId = currentRvcModelId();
    if (modelId.empty()) {
        setStatusText(QStringLiteral("Import and select an RVC model first."));
        return;
    }
    if (!ensureRecordingLine()) {
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
        const bool liveInput =
            m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(liveInput);
        setProcessorPassthrough(liveInput, Qt::BlockingQueuedConnection);
        setProcessorLocalRvcCapture(true, Qt::BlockingQueuedConnection);
        const auto generatorInputs = description.value().generator.inputs.size();
        const auto generatorOutputs = description.value().generator.outputs.size();
        setStatusText(QStringLiteral("Native ONNX RVC active: generator %1 in/%2 out.")
                          .arg(static_cast<qulonglong>(generatorInputs))
                          .arg(static_cast<qulonglong>(generatorOutputs)));
        updateTransportState();
        return;
    }

    m_nativeRvcEngine.reset();
    auto started = m_rvcSidecar.start();
    if (!started) {
        setStatusText(QString::fromStdString(started.error().message));
        return;
    }
    setStatusText(QStringLiteral("Loading local voice \"%1\"...")
                      .arg(currentRvcModelName()));
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    const rvc::RvcClient warmupClient{started.value().endpoint};
    auto warmed = warmupClient.loadModel(modelId);
    for (int attempt = 0; !warmed && attempt < 4; ++attempt) {
        QThread::msleep(250);
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        warmed = warmupClient.loadModel(modelId);
    }
    if (!warmed) {
        m_rvcSidecar.stop();
        setStatusText(QString::fromStdString(warmed.error().message));
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
    const bool liveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
    m_capture.setMonitorEnabled(liveInput);
    setProcessorPassthrough(liveInput, Qt::BlockingQueuedConnection);
    setProcessorLocalRvcCapture(true, Qt::BlockingQueuedConnection);
    QString localStatus =
        QStringLiteral("Local RVC active at %1.")
            .arg(QString::fromStdString(started.value().endpoint));
    const auto& health = warmed;
    if (health && health.value().engine == "compat-pass-through") {
        localStatus += QStringLiteral(
            " This test sidecar passes audio through; use Cloud for cloned voices.");
    }
    setStatusText(localStatus);
    updateTransportState();
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
    if (!m_cloudActive) {
        const bool keepLiveInput =
            m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
    }
    m_rvcSidecar.stop();
    m_nativeRvcEngine.reset();
    setStatusText(QStringLiteral("Local RVC cancelled."));
    updateTransportState();
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
    m_vadLabel->setText(speechActive || level >= 3 ? QStringLiteral("Mic detected")
                                                   : QStringLiteral("Listening..."));
}

void LiveMicPanel::enqueueCloudChunk(QByteArray chunk) {
    if (chunk.isEmpty() || (m_cloudCancelFlag != nullptr &&
                            m_cloudCancelFlag->load(std::memory_order_acquire))) {
        return;
    }

    m_cloudSeconds += static_cast<double>(chunk.size()) /
                      static_cast<double>(kCloudInputSampleRate * 2);
    m_pendingCloudChunks.push_back(std::move(chunk));
    m_costLabel->setText(
        QStringLiteral("Character audio: %1 s | %2 waiting")
            .arg(m_cloudSeconds, 0, 'f', 1)
            .arg(static_cast<int>(m_pendingCloudChunks.size())));
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
        m_cloudPlaybackGuardActive = false;
        setStatusText(QStringLiteral("Cloud conversion cancelled."));
        return;
    }

    if (!result.success) {
        setStatusText(result.message);
    } else {
        if (m_recordTakeCheck->isChecked()) {
            m_recordedCloudPcm.append(result.convertedPcmBytes);
        }
        m_cloudOutputSampleRate = result.sampleRate;
        setStatusText(result.playbackWarning.isEmpty()
                          ? result.message
                          : QStringLiteral("%1 Playback: %2")
                                .arg(result.message, result.playbackWarning));
    }

    finishCloudPlaybackGuard(result.success ? result.playbackDurationMs : 0);
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

void LiveMicPanel::setProcessorCloudCapturePaused(
    const bool paused,
    const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor,
        [processor, paused]() {
            processor->setCloudCapturePaused(paused);
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
    if (m_cloudWatcher->isRunning() || m_cloudPlaybackGuardActive ||
        m_pendingCloudChunks.empty()) {
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
    m_cloudPlaybackGuardActive = true;
    ++m_cloudPlaybackGeneration;
    setProcessorCloudCapturePaused(true, Qt::BlockingQueuedConnection);
    m_costLabel->setText(
        QStringLiteral("Character audio: %1 s | %2 waiting")
            .arg(m_cloudSeconds, 0, 'f', 1)
            .arg(static_cast<int>(m_pendingCloudChunks.size())));
    const auto playbackTargets = currentPlaybackTargets();
    auto cancelFlag = m_cloudCancelFlag;
    m_cloudWatcher->setFuture(QtConcurrent::run([voiceId, chunk, playbackTargets, cancelFlag]() {
        return convertCloudChunk(voiceId, chunk, playbackTargets, cancelFlag);
    }));
}

void LiveMicPanel::finishCloudPlaybackGuard(const int playbackDurationMs) {
    const auto generation = m_cloudPlaybackGeneration;
    const auto delayMs =
        playbackDurationMs > 0 ? playbackDurationMs + kCloudPlaybackTailMs : 0;
    QTimer::singleShot(delayMs, this, [this, generation]() {
        if (generation != m_cloudPlaybackGeneration) {
            return;
        }

        m_cloudPlaybackGuardActive = false;
        if (m_cloudActive) {
            setProcessorCloudCapturePaused(false, Qt::BlockingQueuedConnection);
        }
        startNextCloudChunk();
        saveCloudRecordingIfReady();
    });
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
    const auto playbackTargets = currentPlaybackTargets();
    auto cancelFlag = m_localRvcCancelFlag;
    auto nativeEngine = m_nativeRvcEngine;
    if (nativeEngine != nullptr) {
        m_localRvcWatcher->setFuture(
            QtConcurrent::run([nativeEngine, chunk, playbackTargets, cancelFlag]() {
                return convertNativeRvcChunk(nativeEngine, chunk, playbackTargets, cancelFlag);
            }));
        return;
    }

    const auto endpoint = m_rvcSidecar.status().endpoint;
    m_localRvcWatcher->setFuture(
        QtConcurrent::run([endpoint, modelId, chunk, playbackTargets, cancelFlag]() {
            return convertLocalRvcChunk(endpoint, modelId, chunk, playbackTargets, cancelFlag);
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
    if (!m_project.has_value() || m_recordingLineId.empty()) {
        setStatusText(QStringLiteral("Cloud conversion finished. STS take was not saved."));
        m_recordedCloudPcm.clear();
        return;
    }

    const auto bytes = std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(m_recordedCloudPcm.constData()),
        static_cast<std::size_t>(m_recordedCloudPcm.size())};
    auto audio =
        audio::pcm16LittleEndianToPcm(bytes, m_cloudOutputSampleRate, kCloudOutputChannels);
    if (!audio) {
        setStatusText(QString::fromStdString(audio.error().message));
        m_recordedCloudPcm.clear();
        return;
    }

    core::TakeManager takeManager;
    auto saved = takeManager.saveStsTake(m_project->rootPath(),
                                         m_recordingLineId,
                                         m_recordingLineVoiceId,
                                         audio.value(),
                                         core::defaultVoiceSettings());
    if (!saved) {
        setStatusText(QString::fromStdString(saved.error().message));
        m_recordedCloudPcm.clear();
        return;
    }

    m_recordedCloudPcm.clear();
    refreshRecentTakes();
    setStatusText(
        QStringLiteral("Saved changed-voice take for \"%1\".").arg(m_recordingLineText));
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
    if (!m_project.has_value() || m_recordingLineId.empty()) {
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
                                              m_recordingLineId,
                                              m_recordingRvcModelId,
                                              audio.value());
    if (!saved) {
        setStatusText(QString::fromStdString(saved.error().message));
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }

    m_recordedLocalRvcPcm.clear();
    m_nativeRvcEngine.reset();
    refreshRecentTakes();
    setStatusText(QStringLiteral("Saved local take for \"%1\".").arg(m_recordingLineText));
}

void LiveMicPanel::refreshRecentTakes() {
    if (m_recentTakesWidget == nullptr) {
        return;
    }
    if (!m_project.has_value()) {
        m_recentTakesWidget->setTakes({});
        return;
    }

    auto takes = m_takeRepository.listRecentTakes(m_project->rootPath());
    if (!takes) {
        m_recentTakesWidget->setTakes({});
        setStatusText(QString::fromStdString(takes.error().message));
        return;
    }
    m_recentTakesWidget->setTakes(std::move(takes).value());
}

void LiveMicPanel::playTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }

    const auto absolutePath = m_project->rootPath() / std::filesystem::path{take.filePath};
    auto played = m_audioEngine.playFile(absolutePath);
    if (!played) {
        setStatusText(QString::fromStdString(played.error().message));
        return;
    }
    setStatusText(QStringLiteral("Playing %1 take for %2.")
                      .arg(QString::fromStdString(take.source),
                           QString::fromStdString(take.characterName)));
}

void LiveMicPanel::starTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }

    auto starred = m_takeRepository.setActiveTake(m_project->rootPath(), take.lineId, take.id);
    if (!starred) {
        setStatusText(QString::fromStdString(starred.error().message));
        return;
    }
    refreshRecentTakes();
    setStatusText(QStringLiteral("Active take updated."));
}

void LiveMicPanel::revealTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }

    const auto absolutePath = m_project->rootPath() / std::filesystem::path{take.filePath};
    if (!std::filesystem::exists(absolutePath)) {
        setStatusText(QStringLiteral("The take audio file no longer exists."));
        return;
    }

    const auto path = QString::fromStdWString(absolutePath.wstring());
    if (!QProcess::startDetached(QStringLiteral("explorer.exe"),
                                 {QStringLiteral("/select,"),
                                  QDir::toNativeSeparators(path)})) {
        setStatusText(QStringLiteral("Unable to open File Explorer."));
        return;
    }
    setStatusText(QStringLiteral("Opened take in File Explorer."));
}

void LiveMicPanel::deleteTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }
    const auto answer =
        QMessageBox::question(this,
                              QStringLiteral("Delete Take"),
                              QStringLiteral("Delete this recorded take and its audio file?"));
    if (answer != QMessageBox::Yes) {
        return;
    }

    auto deleted = m_takeRepository.deleteTake(m_project->rootPath(), take.lineId, take.id);
    if (!deleted) {
        setStatusText(QString::fromStdString(deleted.error().message));
        return;
    }
    refreshRecentTakes();
    setStatusText(QStringLiteral("Take deleted."));
}

bool LiveMicPanel::ensureRecordingLine() {
    if (m_recordTakeCheck == nullptr || !m_recordTakeCheck->isChecked()) {
        return true;
    }
    if (!m_project.has_value()) {
        setStatusText(QStringLiteral("Open a project before recording a take."));
        return false;
    }

    auto lineText = m_lineIdEdit == nullptr ? QString{} : m_lineIdEdit->text().trimmed();
    if (lineText.isEmpty()) {
        lineText = QStringLiteral("Live microphone performance");
    }

    const auto voiceId = currentVoiceId();
    const auto rvcModelId = currentRvcModelId();
    const bool localMode =
        m_modeCombo != nullptr && m_modeCombo->currentText() == QStringLiteral("Local");
    auto characterName = currentVoiceName().trimmed();
    if (characterName.isEmpty() && localMode) {
        characterName = currentRvcModelName().trimmed();
    }
    if (characterName.isEmpty()) {
        setStatusText(QStringLiteral("Select a character voice first."));
        return false;
    }

    const auto savedVoiceId = voiceId;
    const auto savedRvcModelId = localMode ? rvcModelId : std::string{};
    if (!m_recordingLineId.empty() && m_recordingLineText == lineText &&
        m_recordingLineVoiceId == savedVoiceId &&
        m_recordingRvcModelId == savedRvcModelId) {
        return true;
    }

    auto line = m_scriptRepository.createPerformanceLine(m_project->rootPath(),
                                                         characterName.toStdString(),
                                                         savedVoiceId,
                                                         lineText.toStdString());
    if (!line) {
        setStatusText(QString::fromStdString(line.error().message));
        return false;
    }
    if (localMode && !savedRvcModelId.empty()) {
        auto assigned = m_scriptRepository.updateCharacterRvcModel(
            m_project->rootPath(), line.value().characterId, savedRvcModelId);
        if (!assigned) {
            setStatusText(QString::fromStdString(assigned.error().message));
            return false;
        }
    }

    m_recordingLineId = line.value().id;
    m_recordingLineVoiceId = savedVoiceId;
    m_recordingRvcModelId = savedRvcModelId;
    m_recordingLineText = lineText;
    return true;
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

QString LiveMicPanel::currentVoiceName() const {
    if (m_voiceCombo == nullptr || m_voiceCombo->currentIndex() < 0) {
        return {};
    }
    return m_voiceCombo->currentText();
}

QString LiveMicPanel::currentRvcModelName() const {
    if (m_rvcModelCombo == nullptr || m_rvcModelCombo->currentIndex() < 0) {
        return {};
    }
    return m_rvcModelCombo->currentText();
}

int LiveMicPanel::currentPitchShiftSemitones() const {
    return m_pitchSlider == nullptr ? 0 : m_pitchSlider->value();
}

PlaybackTargets LiveMicPanel::currentPlaybackTargets() noexcept {
    PlaybackTargets targets;
    const auto pitchFactor = std::clamp(
        std::pow(2.0, static_cast<double>(currentPitchShiftSemitones()) / 12.0), 0.25, 4.0);
    targets.durationScale = 1.0 / pitchFactor;
    if (m_monitorCheck != nullptr && m_monitorCheck->isChecked()) {
        targets.monitor = &m_audioEngine;
    }
    if (m_broadcastButton != nullptr && m_broadcastButton->isChecked() &&
        m_broadcastOutputDeviceCombo != nullptr &&
        m_broadcastOutputDeviceCombo->currentIndex() >= 0) {
        targets.broadcast = &m_broadcastAudioEngine;
    }

    if (targets.monitor != nullptr && targets.broadcast != nullptr &&
        comboDeviceIndex(m_outputDeviceCombo) == comboDeviceIndex(m_broadcastOutputDeviceCombo)) {
        targets.broadcast = nullptr;
    }
    return targets;
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
