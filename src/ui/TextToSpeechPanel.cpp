#include "ui/TextToSpeechPanel.h"

#include "audio/AudioFile.h"
#include "audio/Capture.h"
#include "core/TakeManager.h"
#include "ui/TakeListWidget.h"
#include "voxcpm/VoxCpmClient.h"

#include <QAbstractButton>
#include <QButtonGroup>
#include <QComboBox>
#include <QDir>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QSizePolicy>
#include <QStyle>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

namespace voxstudio::ui {
namespace {

template <typename TWidget, typename... TArgs>
[[nodiscard]] TWidget* addOwnedWidget(QLayout& layout, TArgs&&... args) {
    auto widget = std::make_unique<TWidget>(std::forward<TArgs>(args)...);
    auto* widgetPointer = widget.get();
    layout.addWidget(widget.release());
    return widgetPointer;
}

[[nodiscard]] int defaultDeviceIndex(
    const std::vector<audio::AudioDeviceInfo>& devices) {
    const auto found = std::ranges::find_if(devices, &audio::AudioDeviceInfo::isDefault);
    return found == devices.end()
               ? -1
               : static_cast<int>(std::distance(devices.begin(), found));
}

[[nodiscard]] QString deviceLabel(const audio::AudioDeviceInfo& device) {
    auto label = QString::fromStdString(device.name);
    if (device.isDefault) {
        label += QStringLiteral(" (default)");
    }
    return label;
}

[[nodiscard]] TextSynthesisResult renderText(
    const std::string& endpoint,
    const voxcpm::VoxCpmTextRequest& request) {
    const voxcpm::VoxCpmClient client{endpoint};
    auto rendered = client.renderText(request);
    if (!rendered) {
        return TextSynthesisResult{
            false, QString::fromStdString(rendered.error().message)};
    }

    const auto& value = rendered.value();
    return TextSynthesisResult{
        true,
        QStringLiteral("%1 section(s) rendered in %2 ms.")
            .arg(value.sectionCount)
            .arg(value.latencyMs),
        QByteArray{reinterpret_cast<const char*>(value.pcm16Audio.data()),
                   static_cast<qsizetype>(value.pcm16Audio.size())},
        value.sampleRate,
        value.latencyMs,
        value.sectionCount,
        QString::fromStdString(value.delivery),
        QString::fromStdString(value.pronunciations),
    };
}

} // namespace

TextToSpeechPanel::TextToSpeechPanel(QWidget* parent)
    : QWidget(parent)
    , m_generationWatcher(std::make_unique<QFutureWatcher<TextSynthesisResult>>()) {
    setObjectName(QStringLiteral("TextToSpeechPanel"));
    setStyleSheet(QStringLiteral(
        "#TextToSpeechPanel { background: #111517; color: #f4f7fb; }"
        "#TextToSpeechTitle { font-size: 22px; font-weight: 700; }"
        "#TextToSpeechEditor { background: #181d20; border: 1px solid #4a535a;"
        " border-radius: 6px; padding: 10px; color: #f4f7fb; }"
        "QPushButton[deliveryTag=\"true\"] { min-height: 30px; padding: 3px 10px;"
        " border: 1px solid #4a535a; border-radius: 6px; background: #20262a; }"
        "QPushButton[deliveryTag=\"true\"]:checked { color: #111517;"
        " background: #f0b35c; border-color: #f7cf91; font-weight: 700; }"
        "#TextToSpeechGenerateButton { min-height: 38px; background: #10cfc0;"
        " color: #071211; border: 0; border-radius: 6px; font-weight: 700; }"
        "#TextToSpeechGenerateButton:disabled { background: #46504f; color: #9aa4a3; }"
        "#TextToSpeechStatus { color: #b7c0c7; }"));

    auto rootLayout = std::make_unique<QVBoxLayout>();
    rootLayout->setContentsMargins(18, 16, 18, 16);
    rootLayout->setSpacing(12);

    auto* title =
        addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Text to Speech"));
    title->setObjectName(QStringLiteral("TextToSpeechTitle"));

    auto mainLayout = std::make_unique<QHBoxLayout>();
    mainLayout->setSpacing(18);
    auto editorLayout = std::make_unique<QVBoxLayout>();
    editorLayout->setSpacing(10);

    auto selectors = std::make_unique<QGridLayout>();
    selectors->setColumnStretch(1, 1);
    selectors->addWidget(
        std::make_unique<QLabel>(QStringLiteral("Character")).release(), 0, 0);
    m_voiceCombo = addOwnedWidget<QComboBox>(*selectors);
    m_voiceCombo->setObjectName(QStringLiteral("TextToSpeechVoiceCombo"));
    selectors->addWidget(m_voiceCombo, 0, 1);
    selectors->addWidget(
        std::make_unique<QLabel>(QStringLiteral("Output")).release(), 1, 0);
    m_outputCombo = addOwnedWidget<QComboBox>(*selectors);
    m_outputCombo->setObjectName(QStringLiteral("TextToSpeechOutputCombo"));
    selectors->addWidget(m_outputCombo, 1, 1);
    editorLayout->addLayout(selectors.release());

    editorLayout->addWidget(
        std::make_unique<QLabel>(QStringLiteral("Script")).release());
    m_textEdit = addOwnedWidget<QPlainTextEdit>(*editorLayout);
    m_textEdit->setObjectName(QStringLiteral("TextToSpeechEditor"));
    m_textEdit->setPlaceholderText(QStringLiteral("Type or paste character dialogue"));
    m_textEdit->setMinimumHeight(250);

    editorLayout->addWidget(
        std::make_unique<QLabel>(QStringLiteral("Emotional delivery")).release());
    auto deliveryGrid = std::make_unique<QGridLayout>();
    deliveryGrid->setSpacing(7);
    m_deliveryGroup = new QButtonGroup(this);
    m_deliveryGroup->setExclusive(true);
    static constexpr std::array deliveries{
        std::pair{"Natural", "natural"},
        std::pair{"Calm", "calm"},
        std::pair{"Measured", "measured"},
        std::pair{"Reflective", "reflective"},
        std::pair{"Warm", "warm"},
        std::pair{"Wry", "wry"},
        std::pair{"Guarded", "guarded"},
        std::pair{"Wounded", "wounded"},
        std::pair{"Resolute", "resolute"},
        std::pair{"Urgent", "urgent"},
        std::pair{"Questioning", "questioning"},
        std::pair{"Sarcastic", "sarcastic"},
    };
    for (auto index = 0; index < static_cast<int>(deliveries.size()); ++index) {
        auto button = std::make_unique<QPushButton>(
            QString::fromLatin1(deliveries[static_cast<std::size_t>(index)].first));
        auto* buttonPointer = button.get();
        buttonPointer->setObjectName(
            QStringLiteral("TextToSpeechDelivery_%1")
                .arg(QString::fromLatin1(
                    deliveries[static_cast<std::size_t>(index)].second)));
        buttonPointer->setProperty("deliveryTag", true);
        buttonPointer->setProperty(
            "delivery",
            QString::fromLatin1(deliveries[static_cast<std::size_t>(index)].second));
        buttonPointer->setCheckable(true);
        buttonPointer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        m_deliveryGroup->addButton(buttonPointer, index);
        deliveryGrid->addWidget(button.release(), index / 4, index % 4);
    }
    m_deliveryGroup->button(0)->setChecked(true);
    editorLayout->addLayout(deliveryGrid.release());

    auto commandLayout = std::make_unique<QHBoxLayout>();
    m_generateButton =
        addOwnedWidget<QPushButton>(*commandLayout, QStringLiteral("Generate Speech"));
    m_generateButton->setObjectName(QStringLiteral("TextToSpeechGenerateButton"));
    m_generateButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    m_stopButton = addOwnedWidget<QPushButton>(*commandLayout);
    m_stopButton->setObjectName(QStringLiteral("TextToSpeechStopButton"));
    m_stopButton->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    m_stopButton->setToolTip(QStringLiteral("Stop playback"));
    m_stopButton->setFixedSize(38, 38);
    editorLayout->addLayout(commandLayout.release());

    mainLayout->addLayout(editorLayout.release(), 1);

    auto recentTakes = std::make_unique<TakeListWidget>(this);
    m_takesWidget = recentTakes.get();
    m_takesWidget->setObjectName(QStringLiteral("TextToSpeechTakes"));
    m_takesWidget->setTitle(QStringLiteral("Generated Takes"));
    m_takesWidget->setMinimumWidth(330);
    mainLayout->addWidget(recentTakes.release());
    rootLayout->addLayout(mainLayout.release(), 1);

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setObjectName(QStringLiteral("TextToSpeechStatus"));
    m_statusLabel->setWordWrap(true);
    setLayout(rootLayout.release());

    connect(m_generateButton, &QPushButton::clicked,
            this, &TextToSpeechPanel::generateSpeech);
    connect(m_stopButton, &QPushButton::clicked,
            this, &TextToSpeechPanel::stopPlayback);
    connect(m_outputCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &TextToSpeechPanel::updateOutputDevice);
    connect(m_generationWatcher.get(),
            &QFutureWatcher<TextSynthesisResult>::finished,
            this,
            &TextToSpeechPanel::finishGeneration);
    connect(m_takesWidget, &TakeListWidget::playTakeRequested,
            this, &TextToSpeechPanel::playTake);
    connect(m_takesWidget, &TakeListWidget::starTakeRequested,
            this, &TextToSpeechPanel::starTake);
    connect(m_takesWidget, &TakeListWidget::revealTakeRequested,
            this, &TextToSpeechPanel::revealTake);
    connect(m_takesWidget, &TakeListWidget::deleteTakeRequested,
            this, &TextToSpeechPanel::deleteTake);

    refreshOutputs();
    refreshVoices();
    refreshTakes();
}

TextToSpeechPanel::~TextToSpeechPanel() {
    if (m_generationWatcher->isRunning()) {
        m_generationWatcher->waitForFinished();
    }
}

void TextToSpeechPanel::setProject(std::optional<core::Project> project) {
    m_project = std::move(project);
    refreshVoices();
    refreshTakes();
}

void TextToSpeechPanel::refreshVoices() {
    m_voiceCombo->clear();
    if (!m_project.has_value()) {
        m_voiceCombo->setEnabled(false);
        m_generateButton->setEnabled(false);
        setStatus(QStringLiteral("Open a project to generate character speech."));
        return;
    }

    auto voices = m_voiceRepository.listVoices(m_project->rootPath());
    if (!voices) {
        m_voiceCombo->setEnabled(false);
        m_generateButton->setEnabled(false);
        setStatus(QString::fromStdString(voices.error().message));
        return;
    }

    auto records = std::move(voices).value();
    std::erase_if(records, [](const db::VoiceRecord& voice) {
        return voice.origin == "premade";
    });
    std::ranges::sort(records, {}, &db::VoiceRecord::name);
    for (const auto& voice : records) {
        m_voiceCombo->addItem(QString::fromStdString(voice.name),
                              QString::fromStdString(voice.id));
    }
    const bool hasVoices = m_voiceCombo->count() > 0;
    m_voiceCombo->setEnabled(hasVoices);
    m_generateButton->setEnabled(hasVoices);
    setStatus(hasVoices
                  ? QStringLiteral("Ready for local character synthesis.")
                  : QStringLiteral("Clone or sync a character voice first."));
}

void TextToSpeechPanel::refreshOutputs() {
    m_outputCombo->clear();
    m_outputDevices.clear();
    auto outputs = audio::Capture::listOutputDevices();
    if (!outputs) {
        setStatus(QString::fromStdString(outputs.error().message));
        return;
    }
    m_outputDevices = std::move(outputs).value();
    for (const auto& output : m_outputDevices) {
        m_outputCombo->addItem(deviceLabel(output), output.index);
    }
    const auto defaultIndex = defaultDeviceIndex(m_outputDevices);
    if (defaultIndex >= 0) {
        m_outputCombo->setCurrentIndex(defaultIndex);
    }
}

void TextToSpeechPanel::refreshTakes() {
    if (!m_project.has_value()) {
        m_takesWidget->setTakes({});
        return;
    }
    auto takes = m_takeRepository.listRecentTakes(m_project->rootPath());
    if (!takes) {
        setStatus(QString::fromStdString(takes.error().message));
        return;
    }
    auto records = std::move(takes).value();
    std::erase_if(records, [](const db::TakeRecord& take) {
        return take.source != "voxcpm2_tts";
    });
    m_takesWidget->setTakes(std::move(records));
}

void TextToSpeechPanel::generateSpeech() {
    if (!m_project.has_value()) {
        setStatus(QStringLiteral("Open a project first."));
        return;
    }
    const auto voiceId = selectedVoiceId();
    const auto voiceName = selectedVoiceName();
    const auto text = m_textEdit->toPlainText().trimmed();
    if (voiceId.empty() || voiceName.isEmpty()) {
        setStatus(QStringLiteral("Select a character voice."));
        return;
    }
    if (text.isEmpty()) {
        setStatus(QStringLiteral("Enter dialogue to generate."));
        return;
    }
    if (text.size() > 10000) {
        setStatus(QStringLiteral("Text-to-speech captures are limited to 10,000 characters."));
        return;
    }

    auto line = m_scriptRepository.createPerformanceLine(
        m_project->rootPath(),
        voiceName.toStdString(),
        voiceId,
        text.toStdString());
    if (!line) {
        setStatus(QString::fromStdString(line.error().message));
        return;
    }
    auto sidecar = m_sidecar.start();
    if (!sidecar) {
        setStatus(QString::fromStdString(sidecar.error().message));
        return;
    }

    m_activeLineId = line.value().id;
    m_activeVoiceId = voiceId;
    m_activeDelivery = selectedDelivery();
    m_activeProjectRoot = m_project->rootPath();
    const voxcpm::VoxCpmTextRequest request{
        voiceId, text.toStdString(), m_activeDelivery};
    const auto endpoint = sidecar.value().endpoint;
    setBusy(true);
    setStatus(QStringLiteral("Generating %1 with %2 delivery...")
                  .arg(voiceName, QString::fromStdString(m_activeDelivery)));
    m_generationWatcher->setFuture(
        QtConcurrent::run([endpoint, request]() {
            return renderText(endpoint, request);
        }));
}

void TextToSpeechPanel::finishGeneration() {
    const auto result = m_generationWatcher->result();
    setBusy(false);
    if (!result.success) {
        setStatus(result.message);
        return;
    }

    const auto bytes = std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(result.pcm16Audio.constData()),
        static_cast<std::size_t>(result.pcm16Audio.size())};
    auto audio = audio::pcm16LittleEndianToPcm(bytes, result.sampleRate, 1);
    if (!audio) {
        setStatus(QString::fromStdString(audio.error().message));
        return;
    }
    auto queued = m_audioEngine.queuePcm(audio.value());
    if (!queued) {
        setStatus(QString::fromStdString(queued.error().message));
        return;
    }

    core::TakeManager takeManager;
    auto saved = takeManager.saveVoxCpmTextTake(
        m_activeProjectRoot,
        m_activeLineId,
        m_activeVoiceId,
        audio.value(),
        m_activeDelivery);
    if (!saved) {
        setStatus(QString::fromStdString(saved.error().message));
        return;
    }

    refreshTakes();
    auto message = QStringLiteral(
        "Generated and saved %1 section(s) with %2 delivery in %3 ms.")
                       .arg(result.sectionCount)
                       .arg(result.delivery)
                       .arg(result.latencyMs);
    if (!result.pronunciations.isEmpty()) {
        message += QStringLiteral(" Pronunciation guide applied.");
    }
    setStatus(message);
}

void TextToSpeechPanel::stopPlayback() {
    m_audioEngine.clear();
    setStatus(QStringLiteral("Playback stopped."));
}

void TextToSpeechPanel::updateOutputDevice(int) {
    if (m_outputCombo->currentIndex() < 0) {
        return;
    }
    auto routed = m_audioEngine.setOutputDeviceIndex(m_outputCombo->currentData().toInt());
    if (!routed) {
        setStatus(QString::fromStdString(routed.error().message));
    }
}

void TextToSpeechPanel::playTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }
    const auto path = m_project->rootPath() / std::filesystem::path{take.filePath};
    auto played = m_audioEngine.playFile(path);
    setStatus(played ? QStringLiteral("Playing generated take.")
                     : QString::fromStdString(played.error().message));
}

void TextToSpeechPanel::starTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }
    auto starred =
        m_takeRepository.setActiveTake(m_project->rootPath(), take.lineId, take.id);
    if (!starred) {
        setStatus(QString::fromStdString(starred.error().message));
        return;
    }
    refreshTakes();
    setStatus(QStringLiteral("Active take updated."));
}

void TextToSpeechPanel::revealTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }
    const auto path = m_project->rootPath() / std::filesystem::path{take.filePath};
    if (!std::filesystem::exists(path) ||
        !QProcess::startDetached(
            QStringLiteral("explorer.exe"),
            {QStringLiteral("/select,"),
             QDir::toNativeSeparators(QString::fromStdWString(path.wstring()))})) {
        setStatus(QStringLiteral("Unable to open the generated take in File Explorer."));
        return;
    }
    setStatus(QStringLiteral("Opened the generated take in File Explorer."));
}

void TextToSpeechPanel::deleteTake(db::TakeRecord take) {
    if (!m_project.has_value()) {
        return;
    }
    const auto answer =
        QMessageBox::question(this,
                              QStringLiteral("Delete Take"),
                              QStringLiteral("Delete this generated take and its audio file?"));
    if (answer != QMessageBox::Yes) {
        return;
    }
    auto deleted =
        m_takeRepository.deleteTake(m_project->rootPath(), take.lineId, take.id);
    if (!deleted) {
        setStatus(QString::fromStdString(deleted.error().message));
        return;
    }
    refreshTakes();
    setStatus(QStringLiteral("Generated take deleted."));
}

void TextToSpeechPanel::setBusy(const bool busy) {
    m_generateButton->setEnabled(!busy && m_voiceCombo->count() > 0);
    m_voiceCombo->setEnabled(!busy && m_voiceCombo->count() > 0);
    m_textEdit->setReadOnly(busy);
    for (auto* button : m_deliveryGroup->buttons()) {
        button->setEnabled(!busy);
    }
}

void TextToSpeechPanel::setStatus(const QString& text) {
    if (m_statusLabel != nullptr) {
        m_statusLabel->setText(text);
    }
}

std::string TextToSpeechPanel::selectedVoiceId() const {
    return m_voiceCombo->currentIndex() < 0
               ? std::string{}
               : m_voiceCombo->currentData().toString().toStdString();
}

QString TextToSpeechPanel::selectedVoiceName() const {
    return m_voiceCombo->currentIndex() < 0 ? QString{} : m_voiceCombo->currentText();
}

std::string TextToSpeechPanel::selectedDelivery() const {
    const auto* checked = m_deliveryGroup->checkedButton();
    return checked == nullptr
               ? std::string{"natural"}
               : checked->property("delivery").toString().toStdString();
}

} // namespace voxstudio::ui
