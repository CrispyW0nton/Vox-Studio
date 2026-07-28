#include "ui/LiveMicPanel.h"

#include "audio/AudioFile.h"
#include "core/TakeManager.h"
#include "core/TextSegmentation.h"
#include "rvc/OnnxRvcEngine.h"
#include "rvc/RvcClient.h"
#include "ui/LiveAudioProcessor.h"
#include "ui/RvcModelManagerDialog.h"
#include "ui/TakeListWidget.h"
#include "voxcpm/VoxCpmClient.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileDialog>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMetaObject>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
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
    return found == devices.end() ? -1 : static_cast<int>(std::distance(devices.begin(), found));
}

[[nodiscard]] bool isVirtualInputDevice(const audio::AudioDeviceInfo& device) {
    static constexpr std::array virtualInputNames{
        "voicemod", "cable output", "voicemeeter output", "virtual audio", "stereo mix",
    };
    const auto name = QString::fromStdString(device.name);
    return std::ranges::any_of(virtualInputNames, [&name](const auto candidate) {
        return name.contains(QString::fromLatin1(candidate), Qt::CaseInsensitive);
    });
}

[[nodiscard]] int preferredPhysicalInputIndex(const std::vector<audio::AudioDeviceInfo>& devices) {
    const auto defaultIndex = defaultDeviceIndex(devices);
    if (defaultIndex >= 0 &&
        !isVirtualInputDevice(devices[static_cast<std::size_t>(defaultIndex)])) {
        return defaultIndex;
    }

    const auto physical = std::ranges::find_if(
        devices, [](const auto& device) { return !isVirtualInputDevice(device); });
    return physical == devices.end() ? defaultIndex
                                     : static_cast<int>(std::distance(devices.begin(), physical));
}

[[nodiscard]] int comboDeviceIndex(const QComboBox* combo) {
    if (combo == nullptr || combo->currentIndex() < 0) {
        return -1;
    }
    return combo->currentData().toInt();
}

[[nodiscard]] int preferredVirtualOutputIndex(const std::vector<audio::AudioDeviceInfo>& devices) {
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

[[nodiscard]] QString safeCaptureName(QString name) {
    name = name.trimmed();
    static const auto illegal = QStringLiteral("<>:\"/\\|?*");
    for (auto index = 0; index < name.size(); ++index) {
        if (name[index].unicode() < 32U || illegal.contains(name[index])) {
            name[index] = QChar{'_'};
        }
    }
    while (name.endsWith(QChar{'.'}) || name.endsWith(QChar{' '})) {
        name.chop(1);
    }
    return name.isEmpty() ? QStringLiteral("Monologue") : name;
}

[[nodiscard]] std::filesystem::path uniqueCapturePath(const std::filesystem::path& folder,
                                                      const QString& captureName) {
    const auto baseName = safeCaptureName(captureName);
    auto candidate =
        folder / std::filesystem::path{(baseName + QStringLiteral(".mp3")).toStdWString()};
    for (int suffix = 2; std::filesystem::exists(candidate); ++suffix) {
        candidate =
            folder / std::filesystem::path{
                         QStringLiteral("%1 (%2).mp3").arg(baseName).arg(suffix).toStdWString()};
    }
    return candidate;
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
                                      const int sampleRate, const int channels,
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

[[nodiscard]] std::vector<std::uint8_t> bytesFromByteArray(const QByteArray& bytes) {
    const auto* first = reinterpret_cast<const std::uint8_t*>(bytes.constData());
    return {first, first + bytes.size()};
}

[[nodiscard]] QByteArray byteArrayFromBytes(const std::vector<std::uint8_t>& bytes) {
    return QByteArray{reinterpret_cast<const char*>(bytes.data()),
                      static_cast<qsizetype>(bytes.size())};
}

[[nodiscard]] CloudConversionResult
convertCloudChunk(const std::string& endpoint, const std::string& voiceId,
                  const QByteArray& inputPcmBytes, const std::string& transcript,
                  const std::shared_ptr<std::atomic_bool>& cancelFlag) {
    if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
        return CloudConversionResult{false, QStringLiteral("VoxCPM2 rendering cancelled.")};
    }

    voxcpm::VoxCpmRenderRequest request;
    request.voiceId = voiceId;
    request.pcm16Audio = bytesFromByteArray(inputPcmBytes);
    request.sampleRate = kCloudInputSampleRate;
    request.channels = kCloudOutputChannels;
    request.transcript = transcript;

    const voxcpm::VoxCpmClient client{endpoint};
    auto rendered = client.renderPerformance(request);
    if (!rendered) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return CloudConversionResult{false, QStringLiteral("VoxCPM2 rendering cancelled.")};
        }
        return CloudConversionResult{false, QString::fromStdString(rendered.error().message)};
    }

    const auto inputSeconds = static_cast<double>(inputPcmBytes.size()) /
                              static_cast<double>(kCloudInputSampleRate * sizeof(std::int16_t));
    auto delivery = QString::fromStdString(rendered.value().delivery).trimmed();
    if (!delivery.isEmpty()) {
        delivery.front() = delivery.front().toUpper();
    }
    const auto trainedAdapter = rendered.value().adapter == "trained";
    auto message =
        delivery.isEmpty()
            ? QStringLiteral("%1 character phrase ready in %2 ms.")
                  .arg(trainedAdapter ? QStringLiteral("Trained") : QStringLiteral("VoxCPM2"))
                  .arg(rendered.value().latencyMs)
            : QStringLiteral("%1 matched %2 delivery in %3 ms.")
                  .arg(trainedAdapter ? QStringLiteral("Trained character")
                                      : QStringLiteral("Character"))
                  .arg(delivery)
                  .arg(rendered.value().latencyMs);
    if (!rendered.value().pronunciations.empty()) {
        message += QStringLiteral(" Pronunciation guide applied.");
    }
    return CloudConversionResult{true,
                                 message,
                                 byteArrayFromBytes(rendered.value().pcm16Audio),
                                 inputSeconds,
                                 rendered.value().sampleRate,
                                 delivery,
                                 voiceId};
}

[[nodiscard]] LocalRvcConversionResult
convertLocalRvcChunk(const std::string& endpoint, const std::string& targetId,
                     const QString& engineName, const QByteArray& inputPcmBytes,
                     const std::shared_ptr<std::atomic_bool>& cancelFlag) {
    if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
        return LocalRvcConversionResult{false, QStringLiteral("%1 cancelled.").arg(engineName)};
    }

    rvc::RvcConvertRequest request;
    request.modelId = targetId;
    request.pcm16Audio = bytesFromByteArray(inputPcmBytes);
    request.sampleRate = kLocalRvcSampleRate;
    request.channels = kLocalRvcChannels;

    const rvc::RvcClient client{endpoint};
    auto converted = client.convertChunk(request, {});
    if (!converted) {
        if (cancelFlag != nullptr && cancelFlag->load(std::memory_order_acquire)) {
            return LocalRvcConversionResult{false, QStringLiteral("%1 cancelled.").arg(engineName)};
        }
        return LocalRvcConversionResult{false,
                                        QString::fromStdString(converted.error().message),
                                        {},
                                        {},
                                        0,
                                        kLocalRvcSampleRate,
                                        kLocalRvcChannels,
                                        targetId};
    }

    return LocalRvcConversionResult{true,
                                    QStringLiteral("%1 block converted.").arg(engineName),
                                    {},
                                    byteArrayFromBytes(converted.value().pcm16Audio),
                                    converted.value().latencyMs,
                                    converted.value().sampleRate,
                                    converted.value().channels,
                                    targetId};
}

[[nodiscard]] LocalRvcConversionResult
convertNativeRvcChunk(const std::shared_ptr<rvc::OnnxRvcEngine>& engine,
                      const std::string& targetId, const QByteArray& inputPcmBytes,
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
        return LocalRvcConversionResult{false, QString::fromStdString(converted.error().message)};
    }

    return LocalRvcConversionResult{true,
                                    QStringLiteral("Native RVC chunk converted."),
                                    {},
                                    byteArrayFromBytes(converted.value().pcm16Audio),
                                    converted.value().latencyMs,
                                    converted.value().sampleRate,
                                    converted.value().channels,
                                    targetId};
}

} // namespace

LiveMicPanel::LiveMicPanel(QWidget* parent)
    : QWidget(parent), m_cloudWatcher(std::make_unique<QFutureWatcher<CloudConversionResult>>()),
      m_localRvcWatcher(std::make_unique<QFutureWatcher<LocalRvcConversionResult>>()) {
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
    deviceLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Voice output")).release(), 1,
                            0);
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
        addOwnedWidget<QLabel>(*transportTextLayout, QStringLiteral("VoxCPM2 performance"));
    m_selectedEngineLabel->setObjectName(QStringLiteral("LiveMicSelectedEngineLabel"));
    m_outputRouteLabel =
        addOwnedWidget<QLabel>(*transportTextLayout, QStringLiteral("Output: default"));
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
        button->setObjectName(
            QStringLiteral("LiveMicQuickVoiceSlot%1").arg(static_cast<int>(index + 1U)));
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
    m_lineIdEdit->setPlaceholderText(QStringLiteral("Optional exact script"));
    advancedLayout->addWidget(m_lineIdEdit, 0, 1);
    m_cloudButton =
        addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Record HQ Phrase"));
    m_cloudButton->setObjectName(QStringLiteral("LiveMicCloudButton"));
    advancedLayout->addWidget(m_cloudButton, 1, 0);
    m_cancelCloudButton = addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Cancel"));
    m_cancelCloudButton->setObjectName(QStringLiteral("LiveMicCancelCloudButton"));
    m_cancelCloudButton->setEnabled(false);
    advancedLayout->addWidget(m_cancelCloudButton, 1, 1);
    m_localRvcButton =
        addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Start Performance Mirror"));
    m_localRvcButton->setObjectName(QStringLiteral("LiveMicLocalRvcButton"));
    advancedLayout->addWidget(m_localRvcButton, 2, 0);
    m_cancelLocalRvcButton =
        addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Cancel Local"));
    m_cancelLocalRvcButton->setObjectName(QStringLiteral("LiveMicCancelLocalRvcButton"));
    m_cancelLocalRvcButton->setEnabled(false);
    advancedLayout->addWidget(m_cancelLocalRvcButton, 2, 1);
    m_installMirrorButton =
        addOwnedWidget<QPushButton>(*advancedLayout, QStringLiteral("Install Mirror Engine"));
    m_installMirrorButton->setObjectName(QStringLiteral("LiveMicInstallMirrorButton"));
    advancedLayout->addWidget(m_installMirrorButton, 3, 0, 1, 2);
    advancedGroup->setLayout(advancedLayout.release());
    centerLayout->addWidget(advancedGroup.release());

    auto monologueGroup = std::make_unique<QGroupBox>(QStringLiteral("Monologue Capture"));
    m_monologueCaptureGroup = monologueGroup.get();
    m_monologueCaptureGroup->setObjectName(QStringLiteral("LiveMicMonologueCaptureGroup"));
    auto monologueLayout = std::make_unique<QGridLayout>();
    monologueLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Capture name")).release(),
                               0, 0);
    m_captureNameEdit = addOwnedWidget<QLineEdit>(*monologueLayout);
    m_captureNameEdit->setObjectName(QStringLiteral("LiveMicCaptureNameEdit"));
    m_captureNameEdit->setPlaceholderText(QStringLiteral("Story or scene name"));
    monologueLayout->addWidget(m_captureNameEdit, 0, 1, 1, 2);
    monologueLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Save folder")).release(), 1,
                               0);
    m_captureFolderEdit = addOwnedWidget<QLineEdit>(*monologueLayout);
    m_captureFolderEdit->setObjectName(QStringLiteral("LiveMicCaptureFolderEdit"));
    monologueLayout->addWidget(m_captureFolderEdit, 1, 1);
    m_browseCaptureFolderButton =
        addOwnedWidget<QPushButton>(*monologueLayout, QStringLiteral("Browse"));
    m_browseCaptureFolderButton->setObjectName(QStringLiteral("LiveMicBrowseCaptureFolderButton"));
    monologueLayout->addWidget(m_browseCaptureFolderButton, 1, 2);
    m_openCaptureFolderButton =
        addOwnedWidget<QPushButton>(*monologueLayout, QStringLiteral("Open in Explorer"));
    m_openCaptureFolderButton->setObjectName(QStringLiteral("LiveMicOpenCaptureFolderButton"));
    monologueLayout->addWidget(m_openCaptureFolderButton, 2, 1, 1, 2);
    monologueGroup->setLayout(monologueLayout.release());
    centerLayout->addWidget(monologueGroup.release());

    QSettings settings;
    auto captureFolder =
        settings.value(QStringLiteral("capture/monologue_folder")).toString().trimmed();
    if (captureFolder.isEmpty()) {
        captureFolder =
            QDir{QStandardPaths::writableLocation(QStandardPaths::MusicLocation)}.filePath(
                QStringLiteral("Vox Studio Captures"));
    }
    m_captureFolderEdit->setText(QDir::toNativeSeparators(captureFolder));

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
    m_modeCombo->addItem(QStringLiteral("Performance Mirror"));
    m_modeCombo->addItem(QStringLiteral("HQ Phrase"));
    m_modeCombo->addItem(QStringLiteral("Monologue"));
    m_modeCombo->addItem(QStringLiteral("Imported RVC"));
    rightLayout->addWidget(m_modeCombo, 0, 1);
    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Character")).release(), 1, 0);
    m_voiceCombo = addOwnedWidget<QComboBox>(*rightLayout);
    m_voiceCombo->setObjectName(QStringLiteral("LiveMicVoiceCombo"));
    rightLayout->addWidget(m_voiceCombo, 1, 1);
    m_rvcModelLabel = addOwnedWidget<QLabel>(*rightLayout, QStringLiteral("RVC Model"));
    rightLayout->addWidget(m_rvcModelLabel, 2, 0);
    m_rvcModelCombo = addOwnedWidget<QComboBox>(*rightLayout);
    m_rvcModelCombo->setObjectName(QStringLiteral("LiveMicRvcModelCombo"));
    rightLayout->addWidget(m_rvcModelCombo, 2, 1);
    m_manageRvcModelsButton =
        addOwnedWidget<QPushButton>(*rightLayout, QStringLiteral("RVC Models"));
    m_manageRvcModelsButton->setObjectName(QStringLiteral("LiveMicManageRvcModelsButton"));
    rightLayout->addWidget(m_manageRvcModelsButton, 3, 0, 1, 2);

    rightLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Voice volume")).release(), 4,
                           0);
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
    connect(m_liveInputButton, &QPushButton::toggled, this, &LiveMicPanel::setLiveInputChecked);
    connect(m_broadcastButton, &QPushButton::toggled, this, &LiveMicPanel::setBroadcastChecked);
    connect(m_voicePowerButton, &QPushButton::clicked, this,
            &LiveMicPanel::toggleVoiceChangerPower);
    connect(m_inputDeviceCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &LiveMicPanel::updateInputRoute);
    connect(m_outputDeviceCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &LiveMicPanel::updateOutputRoute);
    connect(m_broadcastOutputDeviceCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &LiveMicPanel::updateOutputRoute);
    connect(m_modeCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &LiveMicPanel::updateVoiceHud);
    connect(m_voiceCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &LiveMicPanel::handleVoiceSelectionChanged);
    connect(m_rvcModelCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
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
    connect(m_localRvcButton, &QPushButton::clicked, this, &LiveMicPanel::toggleLocalRvcConversion);
    connect(m_cancelLocalRvcButton, &QPushButton::clicked, this,
            &LiveMicPanel::cancelLocalRvcConversion);
    connect(m_installMirrorButton, &QPushButton::clicked, this,
            &LiveMicPanel::installPerformanceMirror);
    connect(m_manageRvcModelsButton, &QPushButton::clicked, this,
            &LiveMicPanel::openRvcModelManager);
    connect(m_browseCaptureFolderButton, &QPushButton::clicked, this,
            &LiveMicPanel::browseMonologueCaptureFolder);
    connect(m_openCaptureFolderButton, &QPushButton::clicked, this,
            &LiveMicPanel::revealMonologueCapture);
    connect(m_recentTakesWidget, &TakeListWidget::playTakeRequested, this, &LiveMicPanel::playTake);
    connect(m_recentTakesWidget, &TakeListWidget::starTakeRequested, this, &LiveMicPanel::starTake);
    connect(m_recentTakesWidget, &TakeListWidget::revealTakeRequested, this,
            &LiveMicPanel::revealTake);
    connect(m_recentTakesWidget, &TakeListWidget::deleteTakeRequested, this,
            &LiveMicPanel::deleteTake);
    connect(m_cloudWatcher.get(), &QFutureWatcher<CloudConversionResult>::finished, this,
            &LiveMicPanel::finishCloudConversion);
    connect(m_localRvcWatcher.get(), &QFutureWatcher<LocalRvcConversionResult>::finished, this,
            &LiveMicPanel::finishLocalRvcConversion);

    auto processor = std::make_unique<LiveAudioProcessor>();
    m_audioProcessor = processor.get();
    m_audioProcessor->moveToThread(&m_audioThread);
    connect(&m_audioThread, &QThread::finished, m_audioProcessor, &QObject::deleteLater);
    connect(m_audioProcessor, &LiveAudioProcessor::meterUpdated, this,
            &LiveMicPanel::applyMeterUpdate, Qt::QueuedConnection);
    connect(m_audioProcessor, &LiveAudioProcessor::statusMessage, this,
            &LiveMicPanel::setStatusText, Qt::QueuedConnection);
    connect(m_audioProcessor, &LiveAudioProcessor::cloudPcmChunkReady, this,
            &LiveMicPanel::enqueueCloudChunk, Qt::QueuedConnection);
    connect(m_audioProcessor, &LiveAudioProcessor::localRvcPcmChunkReady, this,
            &LiveMicPanel::enqueueLocalRvcChunk, Qt::QueuedConnection);
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
        const auto inputIndex = preferredPhysicalInputIndex(m_inputDevices);
        if (inputIndex >= 0) {
            m_inputDeviceCombo->setCurrentIndex(inputIndex);
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
    std::erase_if(voiceRecords,
                  [](const db::VoiceRecord& voice) { return voice.origin == "premade"; });
    const auto priority = [](const db::VoiceRecord& voice) {
        static constexpr std::array<std::string_view, 7> preferredNames{
            "Kreia",    "Xaria - Ancient Dathomir Witch Prototype",
            "Atton",    "Carth",
            "Bao-Dur",  "Alan Watts",
            "Carth HQ",
        };
        const auto found = std::ranges::find(preferredNames, std::string_view{voice.name});
        return found == preferredNames.end()
                   ? static_cast<int>(preferredNames.size())
                   : static_cast<int>(std::distance(preferredNames.begin(), found));
    };
    std::stable_sort(voiceRecords.begin(), voiceRecords.end(),
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
            button->setToolTip(QStringLiteral("%1\n%2 character voice")
                                   .arg(name, QString::fromStdString(voice.origin)));
            button->setEnabled(true);
        }
        ++voiceIndex;
    }
    const bool hasVoices = m_voiceCombo->count() > 0;
    m_voiceCombo->setEnabled(hasVoices);
    m_cloudButton->setEnabled(hasVoices);
    if (hasVoices && m_modeCombo != nullptr &&
        m_modeCombo->currentText() == QStringLiteral("Mic Check") && !m_capture.stats().running) {
        m_modeCombo->setCurrentText(QStringLiteral("Performance Mirror"));
    }
    if (!hasVoices) {
        setStatusText(QStringLiteral("Sync or clone a voice before a character performance."));
    }
    updateVoiceHud();
}

void LiveMicPanel::refreshRvcModels() {
    m_rvcModelCombo->clear();
    m_characterRvcModels.clear();
    auto models = m_rvcModelRegistry.listModels();
    if (!models) {
        m_rvcModelCombo->setEnabled(false);
        setStatusText(QString::fromStdString(models.error().message));
        return;
    }

    std::unordered_set<std::string> installedModelIds;
    for (const auto& model : models.value()) {
        installedModelIds.insert(model.id);
        m_rvcModelCombo->addItem(QString::fromStdString(model.displayName),
                                 QString::fromStdString(model.id));
        if (!model.characterVoiceId.empty()) {
            m_characterRvcModels[model.characterVoiceId] = model.id;
        }
    }
    if (m_project.has_value()) {
        auto characters = m_scriptRepository.listCharacters(m_project->rootPath());
        if (characters) {
            for (const auto& character : characters.value()) {
                if (!character.voiceId.empty() &&
                    installedModelIds.contains(character.rvcModelId)) {
                    m_characterRvcModels[character.voiceId] = character.rvcModelId;
                }
            }
        }
    }
    const bool hasModels = m_rvcModelCombo->count() > 0;
    m_rvcModelCombo->setEnabled(hasModels);
    m_localRvcButton->setEnabled((performanceMirrorMode() && !currentVoiceId().empty()) ||
                                 (importedRvcMode() && hasModels));
    if (m_installMirrorButton != nullptr) {
        const auto status = m_performanceMirrorSidecar.status();
        m_installMirrorButton->setVisible(!status.engineInstalled);
    }
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
    const audio::OutputFxSettings settings{volume, static_cast<float>(bass),
                                           static_cast<float>(mid), static_cast<float>(treble),
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

void LiveMicPanel::updateInputRoute(int) {
    if (!m_capture.stats().running) {
        return;
    }

    const bool directActive = m_localRvcActive;
    const bool cloudActive = m_cloudActive;
    if (directActive) {
        setProcessorLocalRvcCapture(false, Qt::BlockingQueuedConnection);
    }
    if (cloudActive) {
        setProcessorCloudCapture(false, Qt::BlockingQueuedConnection);
    }
    stopAudioProcessor(Qt::BlockingQueuedConnection);
    m_capture.stop();

    if (!ensureCaptureRunning()) {
        if (directActive) {
            cancelLocalRvcConversion();
        } else if (cloudActive) {
            cancelCloudConversion();
        } else {
            setHearSelfChecked(false);
        }
        return;
    }

    const bool convertedMode = directActive || cloudActive;
    const bool liveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
    const bool passthrough =
        convertedMode ? liveInput : m_monitorCheck != nullptr && m_monitorCheck->isChecked();
    m_capture.setMonitorEnabled(passthrough);
    setProcessorPassthrough(passthrough, Qt::BlockingQueuedConnection);
    if (directActive) {
        setProcessorLocalRvcBlockMs(
            m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror ? 360 : 240,
            Qt::BlockingQueuedConnection);
        setProcessorLocalRvcCapture(true, Qt::BlockingQueuedConnection);
    }
    if (cloudActive) {
        setProcessorCloudCapturePaused(false, Qt::BlockingQueuedConnection);
        setProcessorCloudCapture(true, Qt::BlockingQueuedConnection);
    }
    setStatusText(
        QStringLiteral("Microphone switched to %1.").arg(m_inputDeviceCombo->currentText()));
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
    if (selectedMode == QStringLiteral("Performance Mirror") && !currentVoiceId().empty()) {
        setHearSelfChecked(true);
        toggleLocalRvcConversion();
        updateTransportState();
        return;
    }
    if (selectedMode == QStringLiteral("Imported RVC") && !currentRvcModelId().empty()) {
        setHearSelfChecked(true);
        toggleLocalRvcConversion();
        updateTransportState();
        return;
    }

    if (!currentVoiceId().empty() && (selectedMode == QStringLiteral("HQ Phrase") ||
                                      selectedMode == QStringLiteral("Monologue"))) {
        setHearSelfChecked(true);
        toggleCloudConversion();
        updateTransportState();
        return;
    }

    if (!currentVoiceId().empty()) {
        m_modeCombo->setCurrentText(QStringLiteral("Performance Mirror"));
        setHearSelfChecked(true);
        toggleLocalRvcConversion();
        updateTransportState();
        return;
    }

    if (!currentRvcModelId().empty()) {
        m_modeCombo->setCurrentText(QStringLiteral("Imported RVC"));
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
    m_modeCombo->setCurrentText(QStringLiteral("Performance Mirror"));
    updateVoiceHud();
}

void LiveMicPanel::handleVoiceSelectionChanged(int) {
    updateVoiceHud();
    if (m_localRvcActive && (m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror ||
                             m_directVoiceEngine == DirectVoiceEngine::CharacterRvc)) {
        const auto selectedVoiceId = currentVoiceId();
        const auto characterModelId = currentCharacterRvcModelId();
        const auto desiredEngine = characterModelId.empty() ? DirectVoiceEngine::PerformanceMirror
                                                            : DirectVoiceEngine::CharacterRvc;
        const auto desiredTargetId = characterModelId.empty() ? selectedVoiceId : characterModelId;
        if (selectedVoiceId.empty() || desiredTargetId == m_directVoiceId) {
            return;
        }

        if (desiredEngine != m_directVoiceEngine) {
            const bool conversionRunning = m_localRvcWatcher->isRunning();
            m_restartDirectAfterCancel = conversionRunning;
            cancelLocalRvcConversion();
            if (!conversionRunning) {
                m_localRvcCancelFlag.reset();
                toggleLocalRvcConversion();
            }
            return;
        }

        m_directVoiceId = desiredTargetId;
        m_pendingLocalRvcChunks.clear();
        m_recordedLocalRvcPcm.clear();
        m_directHasSpeech = false;
        m_recordingLineId.clear();
        m_recordingLineVoiceId.clear();
        m_recordingRvcModelId = characterModelId;
        m_audioEngine.clear();
        m_broadcastAudioEngine.clear();
        (void)ensureRecordingLine();
        setStatusText(QStringLiteral("%1 is selected. The next live block will use this character.")
                          .arg(currentVoiceName()));
        return;
    }
    if (!m_cloudActive || m_cloudLongTake) {
        return;
    }

    const auto selectedVoiceId = currentVoiceId();
    if (selectedVoiceId.empty() || selectedVoiceId == m_cloudVoiceId) {
        return;
    }

    m_cloudVoiceId = selectedVoiceId;
    m_pendingCloudChunks.clear();
    m_pendingCloudTranscripts.clear();
    m_recordedCloudPcm.clear();
    m_cloudSeconds = 0.0;
    ++m_cloudPlaybackGeneration;
    m_cloudPlaybackGuardActive = false;
    m_audioEngine.clear();
    m_broadcastAudioEngine.clear();
    setProcessorCloudCapturePaused(false, Qt::BlockingQueuedConnection);
    if (!ensureRecordingLine()) {
        return;
    }

    m_costLabel->setText(QStringLiteral("Character audio: 0.0 s"));
    setStatusText(QStringLiteral("%1 is active. Speak your next phrase.").arg(currentVoiceName()));
}

void LiveMicPanel::updateVoiceHud() {
    const auto mode =
        m_modeCombo == nullptr ? QStringLiteral("Mic Check") : m_modeCombo->currentText();
    const auto voiceName = currentVoiceName();
    const auto rvcName = currentRvcModelName();
    const auto selectedName =
        mode == QStringLiteral("Imported RVC") && !rvcName.isEmpty() ? rvcName : voiceName;

    if (m_selectedVoiceLabel != nullptr) {
        m_selectedVoiceLabel->setText(selectedName.isEmpty() ? QStringLiteral("No voice selected")
                                                             : selectedName);
    }
    if (m_selectedVoiceBadge != nullptr) {
        m_selectedVoiceBadge->setText(voiceBadgeText(selectedName));
    }
    if (m_selectedEngineLabel != nullptr) {
        const bool hasCharacterModel = !currentCharacterRvcModelId().empty();
        const auto engineText =
            mode == QStringLiteral("Performance Mirror")
                ? hasCharacterModel
                      ? QStringLiteral(
                            "Character-trained live model: follows your timing, tone, and pauses")
                      : QStringLiteral(
                            "Reference Mirror fallback: character training is not installed")
            : mode == QStringLiteral("Imported RVC") ? QStringLiteral("Imported RVC model")
            : mode == QStringLiteral("Monologue") ? QStringLiteral("VoxCPM2 HQ monologue capture")
            : mode == QStringLiteral("HQ Phrase")
                ? QStringLiteral("VoxCPM2 HQ (matches your delivery after each pause)")
                : QStringLiteral("Direct microphone monitor");
        m_selectedEngineLabel->setText(engineText);
    }
    if (m_monologueCaptureGroup != nullptr) {
        m_monologueCaptureGroup->setVisible(mode == QStringLiteral("Monologue"));
    }
    if (m_cloudButton != nullptr && !m_cloudActive) {
        m_cloudButton->setText(mode == QStringLiteral("Monologue")
                                   ? QStringLiteral("Record Monologue")
                                   : QStringLiteral("Record HQ Phrase"));
        m_cloudButton->setVisible(mode == QStringLiteral("HQ Phrase") ||
                                  mode == QStringLiteral("Monologue"));
        m_cloudButton->setEnabled(!currentVoiceId().empty());
    }
    if (m_cancelCloudButton != nullptr) {
        m_cancelCloudButton->setVisible(mode == QStringLiteral("HQ Phrase") ||
                                        mode == QStringLiteral("Monologue"));
    }
    const bool directMode =
        mode == QStringLiteral("Performance Mirror") || mode == QStringLiteral("Imported RVC");
    if (m_localRvcButton != nullptr) {
        if (!m_localRvcActive) {
            m_localRvcButton->setText(mode == QStringLiteral("Imported RVC")
                                          ? QStringLiteral("Start Imported RVC")
                                          : QStringLiteral("Start Performance Mirror"));
        }
        m_localRvcButton->setVisible(directMode);
        m_localRvcButton->setEnabled(mode == QStringLiteral("Performance Mirror")
                                         ? !currentVoiceId().empty()
                                         : mode == QStringLiteral("Imported RVC") &&
                                               !currentRvcModelId().empty());
    }
    if (m_cancelLocalRvcButton != nullptr) {
        m_cancelLocalRvcButton->setText(mode == QStringLiteral("Performance Mirror")
                                            ? QStringLiteral("Cancel Mirror")
                                            : QStringLiteral("Cancel RVC"));
        m_cancelLocalRvcButton->setVisible(directMode);
    }
    const bool showRvcModel = mode == QStringLiteral("Imported RVC");
    if (m_rvcModelLabel != nullptr) {
        m_rvcModelLabel->setVisible(showRvcModel);
    }
    if (m_rvcModelCombo != nullptr) {
        m_rvcModelCombo->setVisible(showRvcModel);
    }
    if (m_manageRvcModelsButton != nullptr) {
        m_manageRvcModelsButton->setVisible(showRvcModel);
    }
    if (m_installMirrorButton != nullptr) {
        const auto mirrorStatus = m_performanceMirrorSidecar.status();
        m_installMirrorButton->setVisible(mode == QStringLiteral("Performance Mirror") &&
                                          currentCharacterRvcModelId().empty() &&
                                          !mirrorStatus.engineInstalled);
    }
    if (m_outputRouteLabel != nullptr) {
        const auto monitorRoute =
            m_outputDeviceCombo != nullptr && m_outputDeviceCombo->currentIndex() >= 0
                ? m_outputDeviceCombo->currentText()
                : QStringLiteral("default");
        const auto broadcastRoute = m_broadcastOutputDeviceCombo != nullptr &&
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
            m_modeCombo != nullptr && m_modeCombo->currentText() == QStringLiteral("Mic Check");
        const bool microphoneCheckActive =
            micCheckMode && !m_cloudActive && !m_localRvcActive && m_capture.stats().running;
        const bool powerActive = m_cloudActive || m_localRvcActive || microphoneCheckActive;
        m_voicePowerButton->setChecked(powerActive);
        m_voicePowerButton->setText(powerActive ? QStringLiteral("On") : QStringLiteral("Power"));
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
            m_modeCombo != nullptr && m_modeCombo->currentText() == QStringLiteral("Mic Check");
        if (micCheckMode) {
            m_monitorButton->setText(m_monitorCheck->isChecked() ? QStringLiteral("Mic Check On")
                                                                 : QStringLiteral("Mic Check"));
        } else {
            m_monitorButton->setText(m_monitorCheck->isChecked() ? QStringLiteral("Hear Result On")
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
        m_modeCombo != nullptr && m_modeCombo->currentText() == QStringLiteral("Mic Check");
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
        setStatusText(
            enabled ? QStringLiteral("Live input monitor enabled; changed phrases remain audible.")
                    : QStringLiteral("Live input monitor muted; changed phrases remain audible."));
    } else {
        setStatusText(enabled
                          ? QStringLiteral("Live input monitor will stay on while recording.")
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
    setStatusText(passthrough
                      ? QStringLiteral("Microphone check active. Speak and watch the meter.")
                      : QStringLiteral("Converted voice monitoring enabled."));
    updateTransportState();
}

void LiveMicPanel::toggleCloudConversion() {
    if (m_localRvcActive) {
        setStatusText(QStringLiteral("Stop the live voice changer before starting an HQ capture."));
        return;
    }

    if (m_cloudActive) {
        m_cloudActive = false;
        m_cloudButton->setText(m_cloudLongTake ? QStringLiteral("Record Monologue")
                                               : QStringLiteral("Record HQ Phrase"));
        m_cancelCloudButton->setEnabled(false);
        setProcessorCloudCapture(false, Qt::BlockingQueuedConnection);
        const bool keepLiveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
        setStatusText(m_cloudLongTake
                          ? QStringLiteral("Rendering the complete monologue capture.")
                          : QStringLiteral("Finishing the last VoxCPM2 character phrase."));
        updateTransportState();
        QTimer::singleShot(0, this, [this]() {
            if (m_cloudLongTake) {
                prepareMonologueTranscripts();
            }
            startNextCloudChunk();
            saveCloudRecordingIfReady();
        });
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
    m_performanceMirrorSidecar.stop();
    auto sidecar = m_voxCpmSidecar.start();
    if (!sidecar) {
        setStatusText(QString::fromStdString(sidecar.error().message));
        return;
    }

    m_cloudCancelFlag = std::make_shared<std::atomic_bool>(false);
    m_pendingCloudChunks.clear();
    m_pendingCloudTranscripts.clear();
    m_recordedCloudPcm.clear();
    ++m_cloudPlaybackGeneration;
    m_cloudPlaybackGuardActive = false;
    m_cloudOutputSampleRate = 24000;
    m_cloudSeconds = 0.0;
    m_cloudVoiceId = voiceId;
    m_cloudLongTake =
        m_modeCombo != nullptr && m_modeCombo->currentText() == QStringLiteral("Monologue");
    m_cloudConversionFailed = false;
    if (m_cloudLongTake) {
        auto captureName =
            m_captureNameEdit == nullptr ? QString{} : m_captureNameEdit->text().trimmed();
        if (captureName.isEmpty()) {
            captureName = QStringLiteral("Monologue");
            m_captureNameEdit->setText(captureName);
        }
        auto captureFolder =
            m_captureFolderEdit == nullptr ? QString{} : m_captureFolderEdit->text().trimmed();
        if (captureFolder.isEmpty()) {
            captureFolder =
                QDir{QStandardPaths::writableLocation(QStandardPaths::MusicLocation)}.filePath(
                    QStringLiteral("Vox Studio Captures"));
            m_captureFolderEdit->setText(QDir::toNativeSeparators(captureFolder));
        }
        m_activeCaptureName = captureName;
        m_activeCaptureFolder =
            std::filesystem::path{QDir::fromNativeSeparators(captureFolder).toStdWString()};
        QSettings settings;
        settings.setValue(QStringLiteral("capture/monologue_folder"), captureFolder);
    }
    m_costLabel->setText(QStringLiteral("Character audio: 0.0 s"));
    m_cloudActive = true;
    setHearSelfChecked(true);
    m_cloudButton->setText(m_cloudLongTake ? QStringLiteral("Stop & Render")
                                           : QStringLiteral("Stop & Save"));
    m_cancelCloudButton->setEnabled(true);
    if (!m_cloudLongTake) {
        m_modeCombo->setCurrentText(QStringLiteral("HQ Phrase"));
    }
    const bool liveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
    m_capture.setMonitorEnabled(liveInput);
    setProcessorPassthrough(liveInput, Qt::BlockingQueuedConnection);
    setProcessorCloudCapturePaused(false, Qt::BlockingQueuedConnection);
    setProcessorCloudCapture(true, Qt::BlockingQueuedConnection);
    if (m_cloudLongTake) {
        setStatusText(QStringLiteral(
            "Monologue capture is recording. Character playback begins after you stop."));
    } else {
        setStatusText(
            liveInput
                ? QStringLiteral(
                      "VoxCPM2 is active. Live input is on; the character plays after each pause.")
                : QStringLiteral(
                      "VoxCPM2 is active. Speak naturally; the character plays after each pause."));
    }
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
    m_pendingCloudTranscripts.clear();
    m_recordedCloudPcm.clear();
    m_cloudVoiceId.clear();
    m_cloudLongTake = false;
    m_cloudConversionFailed = false;
    m_audioEngine.clear();
    m_broadcastAudioEngine.clear();
    m_cloudButton->setText(m_modeCombo != nullptr &&
                                   m_modeCombo->currentText() == QStringLiteral("Monologue")
                               ? QStringLiteral("Record Monologue")
                               : QStringLiteral("Record HQ Phrase"));
    m_cancelCloudButton->setEnabled(false);
    if (m_audioThread.isRunning()) {
        setProcessorCloudCapturePaused(false, Qt::BlockingQueuedConnection);
        setProcessorCloudCapture(false, Qt::BlockingQueuedConnection);
    }
    if (!m_localRvcActive) {
        const bool keepLiveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
    }
    setStatusText(QStringLiteral("VoxCPM2 rendering cancelled."));
    updateTransportState();
}

void LiveMicPanel::toggleLocalRvcConversion() {
    if (m_localRvcActive) {
        m_localRvcActive = false;
        m_cancelLocalRvcButton->setEnabled(false);
        setProcessorLocalRvcCapture(false, Qt::BlockingQueuedConnection);
        const bool keepLiveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
        if (m_pendingLocalRvcChunks.empty() && !m_localRvcWatcher->isRunning()) {
            m_nativeRvcEngine.reset();
        }
        setStatusText(m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror
                          ? QStringLiteral("Finishing the last mirrored voice block.")
                          : QStringLiteral("Finishing queued RVC conversion."));
        updateVoiceHud();
        updateTransportState();
        saveLocalRvcRecordingIfReady();
        return;
    }

    if (m_cloudActive) {
        setStatusText(QStringLiteral("Stop the HQ capture before starting the live voice."));
        return;
    }

    const bool mirrorMode = performanceMirrorMode();
    if (!mirrorMode && !importedRvcMode()) {
        setStatusText(QStringLiteral("Choose Performance Mirror or Imported RVC first."));
        return;
    }

    const auto characterModelId = mirrorMode ? currentCharacterRvcModelId() : std::string{};
    const bool useCharacterModel = !characterModelId.empty();
    const auto targetId = useCharacterModel ? characterModelId
                          : mirrorMode      ? currentVoiceId()
                                            : currentRvcModelId();
    if (targetId.empty() && mirrorMode) {
        setStatusText(QStringLiteral("Select a character first."));
        return;
    }
    if (targetId.empty()) {
        setStatusText(QStringLiteral("Import and select an RVC model first."));
        return;
    }
    if (!ensureRecordingLine()) {
        return;
    }

    if (mirrorMode && !useCharacterModel) {
        m_voxCpmSidecar.stop();
        m_nativeRvcEngine.reset();
        auto started = m_performanceMirrorSidecar.start();
        if (!started) {
            const auto mirrorStatus = m_performanceMirrorSidecar.status();
            if (m_installMirrorButton != nullptr) {
                m_installMirrorButton->setVisible(!mirrorStatus.engineInstalled);
            }
            setStatusText(QString::fromStdString(started.error().message));
            return;
        }

        setStatusText(QStringLiteral(
            "Warming Performance Mirror on the GPU. The first start can take a minute..."));
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        const rvc::RvcClient warmupClient{started.value().endpoint};
        core::Expected<rvc::RvcHealth> health = core::makeError(
            core::ErrorCode::FileSystemFailure, "Performance Mirror has not responded yet.");
        for (int attempt = 0; attempt < 240; ++attempt) {
            health = warmupClient.health();
            if (health && health.value().cudaAvailable) {
                break;
            }
            if (health && health.value().cudaVersion != "loading" &&
                !health.value().message.empty() &&
                health.value().message.find("warming up") == std::string::npos) {
                break;
            }
            QThread::msleep(250);
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
        if (!health || !health.value().cudaAvailable) {
            setStatusText(health ? QString::fromStdString(health.value().message)
                                 : QString::fromStdString(health.error().message));
            return;
        }

        setStatusText(
            QStringLiteral("Loading %1's live character reference...").arg(currentVoiceName()));
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        auto loaded = warmupClient.loadModel(targetId);
        if (!loaded) {
            auto latestHealth = warmupClient.health();
            if (!latestHealth || latestHealth.value().loadedModelId != targetId) {
                setStatusText(QString::fromStdString(loaded.error().message));
                return;
            }
        }
        m_directVoiceEngine = DirectVoiceEngine::PerformanceMirror;
        m_directVoiceId = targetId;
    } else {
        const QSettings settings;
        const auto runtimeMode =
            settings.value(QStringLiteral("rvc/runtime"), QStringLiteral("sidecar")).toString();
        if (!useCharacterModel && runtimeMode == QStringLiteral("native_onnx")) {
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

            const auto bundleRoot = rvc::OnnxRvcEngine::defaultNativeModelRoot() / targetId;
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
            m_nativeRvcEngine = std::move(engine);
            m_localRvcOutputSampleRate = description.value().bundle.sampleRate;
            m_directVoiceEngine = DirectVoiceEngine::NativeRvc;
            m_directVoiceId = targetId;
        } else {
            m_nativeRvcEngine.reset();
            auto started = m_rvcSidecar.start();
            if (!started) {
                setStatusText(QString::fromStdString(started.error().message));
                return;
            }
            setStatusText(useCharacterModel
                              ? QStringLiteral("Loading %1's trained character model...")
                                    .arg(currentVoiceName())
                              : QStringLiteral("Loading imported voice \"%1\"...")
                                    .arg(currentRvcModelName()));
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
            const rvc::RvcClient warmupClient{started.value().endpoint};
            auto warmed = warmupClient.loadModel(targetId);
            for (int attempt = 0; !warmed && attempt < 4; ++attempt) {
                QThread::msleep(250);
                QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
                warmed = warmupClient.loadModel(targetId);
            }
            if (!warmed) {
                m_rvcSidecar.stop();
                setStatusText(QString::fromStdString(warmed.error().message));
                return;
            }
            m_directVoiceEngine =
                useCharacterModel ? DirectVoiceEngine::CharacterRvc : DirectVoiceEngine::RvcSidecar;
            m_directVoiceId = targetId;
        }
    }

    if (!ensureCaptureRunning()) {
        return;
    }

    m_localRvcCancelFlag = std::make_shared<std::atomic_bool>(false);
    m_pendingLocalRvcChunks.clear();
    m_recordedLocalRvcPcm.clear();
    m_localRvcSeconds = 0.0;
    m_directHasSpeech = false;
    if (m_directVoiceEngine != DirectVoiceEngine::NativeRvc) {
        m_localRvcOutputSampleRate = kLocalRvcSampleRate;
    }
    m_localRvcOutputChannels = kLocalRvcChannels;
    m_droppedDirectChunks = 0;
    m_costLabel->setText(mirrorMode ? QStringLiteral("Character audio: 0.0 s")
                                    : QStringLiteral("RVC audio: 0.0 s"));
    m_localRvcActive = true;
    m_localRvcButton->setText(mirrorMode ? QStringLiteral("Stop Performance Mirror")
                                         : QStringLiteral("Stop Imported RVC"));
    m_cancelLocalRvcButton->setEnabled(true);
    const bool liveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
    m_capture.setMonitorEnabled(liveInput);
    setProcessorPassthrough(liveInput, Qt::BlockingQueuedConnection);
    setProcessorLocalRvcBlockMs(m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror ? 360
                                                                                            : 240,
                                Qt::BlockingQueuedConnection);
    setProcessorLocalRvcCapture(true, Qt::BlockingQueuedConnection);
    setStatusText(
        useCharacterModel
            ? QStringLiteral("%1's trained character model is live.").arg(currentVoiceName())
        : mirrorMode
            ? QStringLiteral("Reference Mirror is live. Speak naturally; only the changed result "
                             "is monitored.")
            : QStringLiteral("Imported RVC is live."));
    updateVoiceHud();
    updateTransportState();
}

void LiveMicPanel::installPerformanceMirror() {
    (void)m_performanceMirrorSidecar.start();
    const auto status = m_performanceMirrorSidecar.status();
    if (status.engineInstalled) {
        setStatusText(QStringLiteral("Performance Mirror is already installed."));
        updateVoiceHud();
        return;
    }
    if (!std::filesystem::exists(status.setupPath)) {
        setStatusText(QStringLiteral("Performance Mirror setup could not be found."));
        return;
    }

    const auto setupPath = QString::fromStdWString(status.setupPath.wstring());
    const auto workingDirectory = QString::fromStdWString(status.sidecarRoot.wstring());
    const bool launched =
        QProcess::startDetached(QStringLiteral("powershell.exe"),
                                {QStringLiteral("-NoProfile"), QStringLiteral("-ExecutionPolicy"),
                                 QStringLiteral("Bypass"), QStringLiteral("-File"), setupPath},
                                workingDirectory);
    setStatusText(
        launched
            ? QStringLiteral(
                  "Performance Mirror setup is running. Start the live voice after it finishes.")
            : QStringLiteral("Unable to launch Performance Mirror setup."));
}

void LiveMicPanel::cancelLocalRvcConversion() {
    if (m_localRvcCancelFlag != nullptr) {
        m_localRvcCancelFlag->store(true, std::memory_order_release);
    }
    m_localRvcActive = false;
    m_pendingLocalRvcChunks.clear();
    m_recordedLocalRvcPcm.clear();
    m_cancelLocalRvcButton->setEnabled(false);
    if (m_audioThread.isRunning()) {
        setProcessorLocalRvcCapture(false, Qt::BlockingQueuedConnection);
    }
    if (!m_cloudActive) {
        const bool keepLiveInput = m_liveInputButton != nullptr && m_liveInputButton->isChecked();
        m_capture.setMonitorEnabled(keepLiveInput);
        setProcessorPassthrough(keepLiveInput, Qt::QueuedConnection);
        if (!keepLiveInput) {
            stopAudioProcessor(Qt::BlockingQueuedConnection);
            m_capture.stop();
        }
    }
    m_performanceMirrorSidecar.stop();
    m_rvcSidecar.stop();
    m_nativeRvcEngine.reset();
    m_directVoiceEngine = DirectVoiceEngine::None;
    m_directVoiceId.clear();
    setStatusText(QStringLiteral("Live voice conversion cancelled."));
    updateVoiceHud();
    updateTransportState();
}

void LiveMicPanel::openRvcModelManager() {
    RvcModelManagerDialog dialog{m_project, this};
    connect(&dialog, &RvcModelManagerDialog::rvcAssignmentsChanged, this,
            &LiveMicPanel::refreshRvcModels);
    dialog.exec();
    refreshRvcModels();
}

void LiveMicPanel::browseMonologueCaptureFolder() {
    const auto initialFolder =
        m_captureFolderEdit == nullptr ? QString{} : m_captureFolderEdit->text().trimmed();
    const auto folder =
        QFileDialog::getExistingDirectory(this, QStringLiteral("Choose Monologue Capture Folder"),
                                          QDir::fromNativeSeparators(initialFolder));
    if (folder.isEmpty()) {
        return;
    }
    m_captureFolderEdit->setText(QDir::toNativeSeparators(folder));
    QSettings settings;
    settings.setValue(QStringLiteral("capture/monologue_folder"), folder);
}

void LiveMicPanel::revealMonologueCapture() {
    if (!m_lastMonologueCapturePath.empty() &&
        std::filesystem::exists(m_lastMonologueCapturePath)) {
        const auto path = QString::fromStdWString(m_lastMonologueCapturePath.wstring());
        if (QProcess::startDetached(QStringLiteral("explorer.exe"),
                                    {QStringLiteral("/select,"), QDir::toNativeSeparators(path)})) {
            setStatusText(QStringLiteral("Opened the monologue capture in File Explorer."));
            return;
        }
    }

    const auto folder =
        m_captureFolderEdit == nullptr ? QString{} : m_captureFolderEdit->text().trimmed();
    if (folder.isEmpty() || !QProcess::startDetached(QStringLiteral("explorer.exe"),
                                                     {QDir::toNativeSeparators(folder)})) {
        setStatusText(QStringLiteral("Choose a valid monologue capture folder first."));
        return;
    }
    setStatusText(QStringLiteral("Opened the monologue capture folder."));
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
    if (chunk.isEmpty() ||
        (m_cloudCancelFlag != nullptr && m_cloudCancelFlag->load(std::memory_order_acquire))) {
        return;
    }

    m_cloudSeconds +=
        static_cast<double>(chunk.size()) / static_cast<double>(kCloudInputSampleRate * 2);
    m_pendingCloudChunks.push_back(std::move(chunk));
    m_pendingCloudTranscripts.emplace_back();
    m_costLabel->setText(
        QStringLiteral("%1: %2 s | %3 section(s)")
            .arg(m_cloudLongTake ? QStringLiteral("Monologue") : QStringLiteral("Character audio"))
            .arg(m_cloudSeconds, 0, 'f', 1)
            .arg(static_cast<int>(m_pendingCloudChunks.size())));
    if (!m_cloudLongTake || !m_cloudActive) {
        startNextCloudChunk();
    }
}

void LiveMicPanel::enqueueLocalRvcChunk(QByteArray chunk) {
    if (chunk.isEmpty() || (m_localRvcCancelFlag != nullptr &&
                            m_localRvcCancelFlag->load(std::memory_order_acquire))) {
        return;
    }

    const auto label = m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror
                           ? QStringLiteral("Mirrored audio: %1 s")
                       : m_directVoiceEngine == DirectVoiceEngine::NativeRvc
                           ? QStringLiteral("Native RVC audio: %1 s")
                           : QStringLiteral("RVC audio: %1 s");
    m_pendingLocalRvcChunks.push_back(std::move(chunk));
    constexpr std::size_t kMaximumWaitingBlocks = 2U;
    while (m_pendingLocalRvcChunks.size() > kMaximumWaitingBlocks) {
        m_pendingLocalRvcChunks.pop_front();
        ++m_droppedDirectChunks;
    }
    auto costText = label.arg(m_localRvcSeconds, 0, 'f', 1);
    if (m_droppedDirectChunks > 0) {
        costText += QStringLiteral(" | %1 stale dropped").arg(m_droppedDirectChunks);
    }
    m_costLabel->setText(costText);
    startNextLocalRvcChunk();
}

void LiveMicPanel::finishCloudConversion() {
    const auto result = m_cloudWatcher->result();
    if (m_cloudCancelFlag != nullptr && m_cloudCancelFlag->load(std::memory_order_acquire)) {
        m_cloudPlaybackGuardActive = false;
        setStatusText(QStringLiteral("Cloud conversion cancelled."));
        return;
    }

    const bool staleVoice = result.success && !m_cloudLongTake && !m_cloudVoiceId.empty() &&
                            result.voiceId != m_cloudVoiceId;
    int playbackDurationMs = 0;
    if (!result.success) {
        m_cloudConversionFailed = m_cloudConversionFailed || m_cloudLongTake;
        setStatusText(result.message);
    } else {
        if (staleVoice) {
            setStatusText(
                QStringLiteral("%1 is active. The previous character phrase was discarded.")
                    .arg(currentVoiceName()));
        } else {
            QString playbackWarning;
            const auto playbackTargets =
                m_cloudLongTake ? PlaybackTargets{} : currentPlaybackTargets();
            const auto bytes = std::span<const std::uint8_t>{
                reinterpret_cast<const std::uint8_t*>(result.convertedPcmBytes.constData()),
                static_cast<std::size_t>(result.convertedPcmBytes.size())};
            const auto playbackQueued = queuePcmForTargets(
                playbackTargets, bytes, result.sampleRate, kCloudOutputChannels, playbackWarning);
            if (playbackQueued) {
                const auto outputFrames =
                    result.convertedPcmBytes.size() /
                    (static_cast<qsizetype>(sizeof(std::int16_t)) * kCloudOutputChannels);
                playbackDurationMs =
                    static_cast<int>(std::ceil(((static_cast<double>(outputFrames) * 1000.0) /
                                                static_cast<double>(result.sampleRate)) *
                                               playbackTargets.durationScale));
            }
            if (m_recordTakeCheck->isChecked() || m_cloudLongTake) {
                m_recordedCloudPcm.append(result.convertedPcmBytes);
            }
            m_cloudOutputSampleRate = result.sampleRate;
            setStatusText(
                playbackWarning.isEmpty()
                    ? result.message
                    : QStringLiteral("%1 Playback: %2").arg(result.message, playbackWarning));
        }
    }

    finishCloudPlaybackGuard(result.success && !m_cloudLongTake && !staleVoice ? playbackDurationMs
                                                                               : 0);
}

void LiveMicPanel::finishLocalRvcConversion() {
    const auto result = m_localRvcWatcher->result();
    if (m_localRvcCancelFlag != nullptr && m_localRvcCancelFlag->load(std::memory_order_acquire)) {
        const bool restart = std::exchange(m_restartDirectAfterCancel, false);
        setStatusText(restart ? QStringLiteral("Switching live character model...")
                              : QStringLiteral("Live voice conversion cancelled."));
        if (restart) {
            m_localRvcCancelFlag.reset();
            QTimer::singleShot(0, this, [this]() { toggleLocalRvcConversion(); });
        }
        return;
    }

    const bool staleTarget =
        result.success && !result.targetId.empty() && result.targetId != m_directVoiceId;
    if (!result.success) {
        setStatusText(result.message);
    } else if (staleTarget) {
        setStatusText(QStringLiteral("Character changed; the previous live block was discarded."));
    } else {
        const bool exactSilence = std::ranges::all_of(result.convertedPcmBytes,
                                                      [](const char value) { return value == 0; });
        if (!exactSilence) {
            m_directHasSpeech = true;
        }
        const bool keepForTake = m_directHasSpeech;
        if (keepForTake && result.sampleRate > 0 && result.channels > 0) {
            m_localRvcSeconds += static_cast<double>(result.convertedPcmBytes.size()) /
                                 static_cast<double>(result.sampleRate * result.channels *
                                                     static_cast<int>(sizeof(std::int16_t)));
        }
        QString playbackWarning;
        if (!exactSilence) {
            const auto bytes = std::span<const std::uint8_t>{
                reinterpret_cast<const std::uint8_t*>(result.convertedPcmBytes.constData()),
                static_cast<std::size_t>(result.convertedPcmBytes.size())};
            (void)queuePcmForTargets(currentPlaybackTargets(), bytes, result.sampleRate,
                                     result.channels, playbackWarning);
        }
        if (keepForTake && m_recordTakeCheck->isChecked()) {
            m_recordedLocalRvcPcm.append(result.convertedPcmBytes);
        }
        m_localRvcOutputSampleRate = result.sampleRate;
        m_localRvcOutputChannels = result.channels;
        const auto label = m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror
                               ? QStringLiteral("Mirrored audio: %1 s")
                           : m_directVoiceEngine == DirectVoiceEngine::NativeRvc
                               ? QStringLiteral("Native RVC audio: %1 s")
                               : QStringLiteral("RVC audio: %1 s");
        auto costText = label.arg(m_localRvcSeconds, 0, 'f', 1);
        if (m_droppedDirectChunks > 0) {
            costText += QStringLiteral(" | %1 stale dropped").arg(m_droppedDirectChunks);
        }
        m_costLabel->setText(costText);
        const auto message =
            QStringLiteral("%1 Latency: %2 ms.").arg(result.message).arg(result.latencyMs);
        if (!result.playbackWarning.isEmpty()) {
            if (!playbackWarning.isEmpty()) {
                playbackWarning += QChar{' '};
            }
            playbackWarning += result.playbackWarning;
        }
        if (!exactSilence || m_directHasSpeech) {
            setStatusText(playbackWarning.isEmpty()
                              ? message
                              : QStringLiteral("%1 Playback: %2").arg(message, playbackWarning));
        }
    }

    startNextLocalRvcChunk();
    saveLocalRvcRecordingIfReady();
}

void LiveMicPanel::testLatency() {
    const auto result = m_latencyProbe.estimateSharedModeLatency(m_frameMsSpin->value());
    setStatusText(
        QStringLiteral("Estimated monitor latency: %1 ms (%2 target).")
            .arg(result.latencyMs)
            .arg(result.withinTarget ? QStringLiteral("within") : QStringLiteral("over")));
}

void LiveMicPanel::setStatusText(const QString& text) {
    m_statusLabel->setText(text);
}

void LiveMicPanel::startAudioProcessor() {
    auto* processor = m_audioProcessor;
    auto* capture = &m_capture;
    QMetaObject::invokeMethod(
        processor, [processor, capture]() { processor->start(capture); }, Qt::QueuedConnection);
}

void LiveMicPanel::stopAudioProcessor(const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(processor, [processor]() { processor->stop(); }, connectionType);
}

void LiveMicPanel::setProcessorPassthrough(const bool enabled,
                                           const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor, [processor, enabled]() { processor->setPassthroughEnabled(enabled); },
        connectionType);
}

void LiveMicPanel::setProcessorCloudCapture(const bool enabled,
                                            const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor, [processor, enabled]() { processor->setCloudCaptureEnabled(enabled); },
        connectionType);
}

void LiveMicPanel::setProcessorCloudCapturePaused(const bool paused,
                                                  const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor, [processor, paused]() { processor->setCloudCapturePaused(paused); },
        connectionType);
}

void LiveMicPanel::setProcessorLocalRvcCapture(const bool enabled,
                                               const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor, [processor, enabled]() { processor->setLocalRvcCaptureEnabled(enabled); },
        connectionType);
}

void LiveMicPanel::setProcessorLocalRvcBlockMs(const int blockMs,
                                               const Qt::ConnectionType connectionType) {
    if (m_audioProcessor == nullptr || !m_audioThread.isRunning()) {
        return;
    }

    auto* processor = m_audioProcessor;
    QMetaObject::invokeMethod(
        processor, [processor, blockMs]() { processor->setLocalRvcBlockMs(blockMs); },
        connectionType);
}

void LiveMicPanel::prepareMonologueTranscripts() {
    if (!m_cloudLongTake || m_pendingCloudChunks.empty()) {
        return;
    }
    const auto script =
        m_lineIdEdit == nullptr ? std::string{} : m_lineIdEdit->text().trimmed().toStdString();
    if (script.empty()) {
        return;
    }

    std::vector<std::size_t> weights;
    weights.reserve(m_pendingCloudChunks.size());
    for (const auto& chunk : m_pendingCloudChunks) {
        weights.push_back(static_cast<std::size_t>(chunk.size()));
    }
    auto transcripts = core::splitTextByWeights(script, weights);
    m_pendingCloudTranscripts.assign(std::make_move_iterator(transcripts.begin()),
                                     std::make_move_iterator(transcripts.end()));
}

void LiveMicPanel::startNextCloudChunk() {
    if (m_cloudWatcher->isRunning() || m_cloudPlaybackGuardActive || m_pendingCloudChunks.empty()) {
        saveCloudRecordingIfReady();
        return;
    }
    if (m_cloudCancelFlag != nullptr && m_cloudCancelFlag->load(std::memory_order_acquire)) {
        m_pendingCloudChunks.clear();
        m_pendingCloudTranscripts.clear();
        return;
    }

    const auto voiceId = m_cloudVoiceId.empty() ? currentVoiceId() : m_cloudVoiceId;
    if (voiceId.empty()) {
        m_pendingCloudChunks.clear();
        m_pendingCloudTranscripts.clear();
        setStatusText(QStringLiteral("Select a target voice first."));
        return;
    }

    auto chunk = std::move(m_pendingCloudChunks.front());
    m_pendingCloudChunks.pop_front();
    auto transcript = std::string{};
    if (!m_pendingCloudTranscripts.empty()) {
        transcript = std::move(m_pendingCloudTranscripts.front());
        m_pendingCloudTranscripts.pop_front();
    }
    if (!m_cloudLongTake && transcript.empty() && m_lineIdEdit != nullptr) {
        transcript = m_lineIdEdit->text().trimmed().toStdString();
    }
    m_cloudPlaybackGuardActive = true;
    ++m_cloudPlaybackGeneration;
    if (!m_cloudLongTake && m_cloudActive) {
        setProcessorCloudCapturePaused(true, Qt::BlockingQueuedConnection);
    }
    m_costLabel->setText(QStringLiteral("Character audio: %1 s | %2 waiting")
                             .arg(m_cloudSeconds, 0, 'f', 1)
                             .arg(static_cast<int>(m_pendingCloudChunks.size())));
    auto cancelFlag = m_cloudCancelFlag;
    const auto endpoint = m_voxCpmSidecar.status().endpoint;
    m_cloudWatcher->setFuture(
        QtConcurrent::run([endpoint, voiceId, chunk, transcript, cancelFlag]() {
            return convertCloudChunk(endpoint, voiceId, chunk, transcript, cancelFlag);
        }));
}

void LiveMicPanel::finishCloudPlaybackGuard(const int playbackDurationMs) {
    const auto generation = m_cloudPlaybackGeneration;
    const auto delayMs = playbackDurationMs > 0 ? playbackDurationMs + kCloudPlaybackTailMs : 0;
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
    if (m_localRvcCancelFlag != nullptr && m_localRvcCancelFlag->load(std::memory_order_acquire)) {
        m_pendingLocalRvcChunks.clear();
        return;
    }

    const auto targetId = m_directVoiceId;
    if (targetId.empty()) {
        m_pendingLocalRvcChunks.clear();
        setStatusText(QStringLiteral("Select a live voice target first."));
        return;
    }

    auto chunk = std::move(m_pendingLocalRvcChunks.front());
    m_pendingLocalRvcChunks.pop_front();
    auto cancelFlag = m_localRvcCancelFlag;
    auto nativeEngine = m_nativeRvcEngine;
    if (nativeEngine != nullptr) {
        m_localRvcWatcher->setFuture(
            QtConcurrent::run([nativeEngine, targetId, chunk, cancelFlag]() {
                return convertNativeRvcChunk(nativeEngine, targetId, chunk, cancelFlag);
            }));
        return;
    }

    const bool mirror = m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror;
    const auto endpoint =
        mirror ? m_performanceMirrorSidecar.status().endpoint : m_rvcSidecar.status().endpoint;
    const auto engineName = mirror ? QStringLiteral("Performance Mirror") : QStringLiteral("RVC");
    m_localRvcWatcher->setFuture(
        QtConcurrent::run([endpoint, targetId, engineName, chunk, cancelFlag]() {
            return convertLocalRvcChunk(endpoint, targetId, engineName, chunk, cancelFlag);
        }));
}

void LiveMicPanel::saveCloudRecordingIfReady() {
    if (m_cloudActive || m_cloudWatcher->isRunning() || !m_pendingCloudChunks.empty()) {
        return;
    }
    if (m_recordedCloudPcm.isEmpty()) {
        m_cloudVoiceId.clear();
        m_cloudLongTake = false;
        m_cloudConversionFailed = false;
        return;
    }
    if (m_cloudLongTake && m_cloudConversionFailed) {
        setStatusText(
            QStringLiteral("Monologue rendering failed; no incomplete capture was saved."));
        m_recordedCloudPcm.clear();
        m_cloudVoiceId.clear();
        m_cloudLongTake = false;
        m_cloudConversionFailed = false;
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
        m_cloudVoiceId.clear();
        m_cloudLongTake = false;
        m_cloudConversionFailed = false;
        return;
    }

    if (m_cloudLongTake) {
        QString playbackWarning;
        const auto playbackQueued =
            queuePcmForTargets(currentPlaybackTargets(), bytes, m_cloudOutputSampleRate,
                               kCloudOutputChannels, playbackWarning);
        (void)playbackQueued;
        if (!playbackWarning.isEmpty()) {
            setStatusText(QStringLiteral("Monologue playback: %1").arg(playbackWarning));
        }
    }

    QString captureMessage;
    if (m_cloudLongTake && !saveMonologueCapture(audio.value(), captureMessage)) {
        m_recordedCloudPcm.clear();
        m_cloudVoiceId.clear();
        m_cloudLongTake = false;
        m_cloudConversionFailed = false;
        return;
    }
    if (!m_recordTakeCheck->isChecked()) {
        m_recordedCloudPcm.clear();
        m_cloudVoiceId.clear();
        m_cloudLongTake = false;
        m_cloudConversionFailed = false;
        setStatusText(captureMessage.isEmpty() ? QStringLiteral("VoxCPM2 rendering finished.")
                                               : captureMessage);
        return;
    }
    if (!m_project.has_value() || m_recordingLineId.empty()) {
        setStatusText(captureMessage.isEmpty()
                          ? QStringLiteral("VoxCPM2 finished. The project take was not saved.")
                          : captureMessage);
        m_recordedCloudPcm.clear();
        m_cloudVoiceId.clear();
        m_cloudLongTake = false;
        m_cloudConversionFailed = false;
        return;
    }

    core::TakeManager takeManager;
    auto saved = takeManager.saveVoxCpmTake(m_project->rootPath(), m_recordingLineId,
                                            m_recordingLineVoiceId, audio.value());
    if (!saved) {
        setStatusText(QString::fromStdString(saved.error().message));
        m_recordedCloudPcm.clear();
        m_cloudVoiceId.clear();
        m_cloudLongTake = false;
        m_cloudConversionFailed = false;
        return;
    }

    m_recordedCloudPcm.clear();
    m_cloudVoiceId.clear();
    m_cloudLongTake = false;
    m_cloudConversionFailed = false;
    refreshRecentTakes();
    setStatusText(
        captureMessage.isEmpty()
            ? QStringLiteral("Saved changed-voice take for \"%1\".").arg(m_recordingLineText)
            : QStringLiteral("%1 Project take saved.").arg(captureMessage));
}

bool LiveMicPanel::saveMonologueCapture(const audio::PcmAudioBuffer& audio, QString& message) {
    if (m_activeCaptureFolder.empty()) {
        setStatusText(QStringLiteral("Choose a monologue capture folder first."));
        return false;
    }

    const auto outputPath = uniqueCapturePath(m_activeCaptureFolder, m_activeCaptureName);
    auto written = audio::writeMp3File(outputPath, audio);
    if (!written) {
        setStatusText(QString::fromStdString(written.error().message));
        return false;
    }

    m_lastMonologueCapturePath = outputPath;
    message = QStringLiteral("Saved monologue capture \"%1\".")
                  .arg(QString::fromStdWString(outputPath.filename().wstring()));
    return true;
}

void LiveMicPanel::saveLocalRvcRecordingIfReady() {
    if (m_localRvcActive || m_localRvcWatcher->isRunning() || !m_pendingLocalRvcChunks.empty()) {
        return;
    }
    if (m_recordedLocalRvcPcm.isEmpty()) {
        m_nativeRvcEngine.reset();
        if (!m_localRvcActive) {
            m_directVoiceEngine = DirectVoiceEngine::None;
            m_directVoiceId.clear();
        }
        return;
    }
    if (!m_recordTakeCheck->isChecked()) {
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }
    if (!m_project.has_value() || m_recordingLineId.empty()) {
        setStatusText(QStringLiteral("Live voice finished. Take was not saved."));
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }

    const auto bytes = std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(m_recordedLocalRvcPcm.constData()),
        static_cast<std::size_t>(m_recordedLocalRvcPcm.size())};
    auto audio =
        audio::pcm16LittleEndianToPcm(bytes, m_localRvcOutputSampleRate, m_localRvcOutputChannels);
    if (!audio) {
        setStatusText(QString::fromStdString(audio.error().message));
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }

    core::TakeManager takeManager;
    const bool mirror = m_directVoiceEngine == DirectVoiceEngine::PerformanceMirror;
    auto saved =
        mirror ? takeManager.savePerformanceMirrorTake(m_project->rootPath(), m_recordingLineId,
                                                       m_recordingLineVoiceId, audio.value())
               : takeManager.saveRvcLocalTake(m_project->rootPath(), m_recordingLineId,
                                              m_recordingRvcModelId, audio.value());
    if (!saved) {
        setStatusText(QString::fromStdString(saved.error().message));
        m_recordedLocalRvcPcm.clear();
        m_nativeRvcEngine.reset();
        return;
    }

    m_recordedLocalRvcPcm.clear();
    m_nativeRvcEngine.reset();
    m_directVoiceEngine = DirectVoiceEngine::None;
    m_directVoiceId.clear();
    refreshRecentTakes();
    setStatusText(mirror
                      ? QStringLiteral("Saved mirrored take for \"%1\".").arg(m_recordingLineText)
                      : QStringLiteral("Saved RVC take for \"%1\".").arg(m_recordingLineText));
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
    setStatusText(
        QStringLiteral("Playing %1 take for %2.")
            .arg(QString::fromStdString(take.source), QString::fromStdString(take.characterName)));
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
                                 {QStringLiteral("/select,"), QDir::toNativeSeparators(path)})) {
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
        QMessageBox::question(this, QStringLiteral("Delete Take"),
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
    const bool importedMode = importedRvcMode();
    const auto characterModelId =
        performanceMirrorMode() ? currentCharacterRvcModelId() : std::string{};
    const auto rvcModelId = importedMode ? currentRvcModelId() : characterModelId;
    const bool localMode = importedMode || !characterModelId.empty();
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
        m_recordingLineVoiceId == savedVoiceId && m_recordingRvcModelId == savedRvcModelId) {
        return true;
    }

    auto line = m_scriptRepository.createPerformanceLine(
        m_project->rootPath(), characterName.toStdString(), savedVoiceId, lineText.toStdString());
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
                                comboDeviceIndex(m_outputDeviceCombo), m_frameMsSpin->value(),
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

std::string LiveMicPanel::currentCharacterRvcModelId() const {
    const auto voiceId = currentVoiceId();
    const auto found = m_characterRvcModels.find(voiceId);
    return found == m_characterRvcModels.end() ? std::string{} : found->second;
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

bool LiveMicPanel::performanceMirrorMode() const {
    return m_modeCombo != nullptr &&
           m_modeCombo->currentText() == QStringLiteral("Performance Mirror");
}

bool LiveMicPanel::importedRvcMode() const {
    return m_modeCombo != nullptr && m_modeCombo->currentText() == QStringLiteral("Imported RVC");
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
