#include "ui/SettingsDialog.h"

#include "net/elevenlabs/Client.h"
#include "rvc/OnnxRvcEngine.h"
#include "rvc/RvcClient.h"
#include "rvc/RvcSidecar.h"
#include "ui/RvcModelManagerDialog.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <memory>
#include <utility>

namespace voxstudio::ui {
namespace {

[[nodiscard]] QString apiErrorText(const net::elevenlabs::ApiError& error) {
    if (error.statusCode > 0) {
        return QStringLiteral("%1 (HTTP %2)")
            .arg(QString::fromStdString(error.message))
            .arg(error.statusCode);
    }

    return QString::fromStdString(error.message);
}

[[nodiscard]] core::Expected<ConnectionTestResult, net::elevenlabs::ApiError>
runConnectionTest(std::string apiKey) {
    net::elevenlabs::Client client{std::move(apiKey)};

    auto user = client.getUser();
    if (!user) {
        return user.error();
    }

    auto subscription = client.getSubscription();
    if (!subscription) {
        return subscription.error();
    }

    auto voices = client.getVoices();
    if (!voices) {
        return voices.error();
    }

    auto models = client.getModels();
    if (!models) {
        return models.error();
    }

    return ConnectionTestResult{user.value().email,
                                subscription.value().tier,
                                subscription.value().characterCount,
                                subscription.value().characterLimit,
                                static_cast<int>(voices.value().voices.size()),
                                static_cast<int>(models.value().models.size())};
}

} // namespace

SettingsDialog::SettingsDialog(QWidget* parent)
    : QDialog(parent)
    , m_connectionWatcher(std::make_unique<QFutureWatcher<
                          core::Expected<ConnectionTestResult, net::elevenlabs::ApiError>>>()) {
    setWindowTitle(QStringLiteral("Settings"));
    setModal(true);

    auto rootLayout = std::make_unique<QVBoxLayout>();

    auto introLabel = std::make_unique<QLabel>();
    m_introLabel = introLabel.get();
    m_introLabel->setWordWrap(true);
    m_introLabel->setVisible(false);
    rootLayout->addWidget(introLabel.release());

    auto apiKeyEdit = std::make_unique<QLineEdit>();
    m_apiKeyEdit = apiKeyEdit.get();
    m_apiKeyEdit->setEchoMode(QLineEdit::Password);
    m_apiKeyEdit->setClearButtonEnabled(true);
    m_apiKeyEdit->setPlaceholderText(QStringLiteral("ElevenLabs API key"));
    rootLayout->addWidget(apiKeyEdit.release());

    auto statusLabel = std::make_unique<QLabel>();
    m_statusLabel = statusLabel.get();
    m_statusLabel->setWordWrap(true);
    rootLayout->addWidget(statusLabel.release());

    auto testButton = std::make_unique<QPushButton>(QStringLiteral("Test Connection"));
    m_testButton = testButton.get();
    connect(m_testButton, &QPushButton::clicked, this, &SettingsDialog::testConnection);
    rootLayout->addWidget(testButton.release());

    auto rvcGroup = std::make_unique<QGroupBox>(QStringLiteral("Local RVC"));
    auto rvcLayout = std::make_unique<QVBoxLayout>();
    auto engineLayout = std::make_unique<QHBoxLayout>();
    engineLayout->addWidget(std::make_unique<QLabel>(QStringLiteral("Engine")).release());
    auto rvcEngineCombo = std::make_unique<QComboBox>();
    m_rvcEngineCombo = rvcEngineCombo.get();
    m_rvcEngineCombo->setObjectName(QStringLiteral("SettingsRvcEngineCombo"));
    m_rvcEngineCombo->addItem(QStringLiteral("Sidecar"), QStringLiteral("sidecar"));
    m_rvcEngineCombo->addItem(QStringLiteral("Native ONNX"), QStringLiteral("native_onnx"));
    const QSettings settings;
    const auto savedMode =
        settings.value(QStringLiteral("rvc/runtime"), QStringLiteral("sidecar")).toString();
    const auto savedIndex = m_rvcEngineCombo->findData(savedMode);
    if (savedIndex >= 0) {
        m_rvcEngineCombo->setCurrentIndex(savedIndex);
    }
    connect(m_rvcEngineCombo, &QComboBox::currentIndexChanged, this,
            &SettingsDialog::saveRvcEngineMode);
    engineLayout->addWidget(rvcEngineCombo.release());
    rvcLayout->addLayout(engineLayout.release());

    auto rvcStatusLabel = std::make_unique<QLabel>();
    m_rvcStatusLabel = rvcStatusLabel.get();
    m_rvcStatusLabel->setObjectName(QStringLiteral("SettingsRvcStatusLabel"));
    m_rvcStatusLabel->setWordWrap(true);
    rvcLayout->addWidget(rvcStatusLabel.release());

    auto rvcButtonsLayout = std::make_unique<QHBoxLayout>();
    auto rvcRefreshButton = std::make_unique<QPushButton>(QStringLiteral("Refresh RVC"));
    m_rvcRefreshButton = rvcRefreshButton.get();
    m_rvcRefreshButton->setObjectName(QStringLiteral("SettingsRvcRefreshButton"));
    connect(m_rvcRefreshButton, &QPushButton::clicked, this,
            &SettingsDialog::refreshRvcDiagnostics);
    rvcButtonsLayout->addWidget(rvcRefreshButton.release());

    auto rvcModelButton = std::make_unique<QPushButton>(QStringLiteral("Manage Models"));
    m_rvcModelButton = rvcModelButton.get();
    m_rvcModelButton->setObjectName(QStringLiteral("SettingsRvcModelButton"));
    connect(m_rvcModelButton, &QPushButton::clicked, this, &SettingsDialog::openRvcModelManager);
    rvcButtonsLayout->addWidget(rvcModelButton.release());
    rvcLayout->addLayout(rvcButtonsLayout.release());
    rvcGroup->setLayout(rvcLayout.release());
    rootLayout->addWidget(rvcGroup.release());

    auto buttons = std::make_unique<QDialogButtonBox>(QDialogButtonBox::Save |
                                                       QDialogButtonBox::Close);
    m_saveButton = buttons->button(QDialogButtonBox::Save);
    connect(m_saveButton, &QPushButton::clicked, this, &SettingsDialog::saveApiKey);
    connect(buttons->button(QDialogButtonBox::Close), &QPushButton::clicked, this,
            &SettingsDialog::reject);
    rootLayout->addWidget(buttons.release());

    connect(m_connectionWatcher.get(),
            &QFutureWatcher<core::Expected<ConnectionTestResult,
                                           net::elevenlabs::ApiError>>::finished,
            this, &SettingsDialog::finishConnectionTest);

    if (m_vault.hasElevenLabsApiKey()) {
        setStatusText(QStringLiteral("Saved ElevenLabs API key found."));
    } else {
        setStatusText(QStringLiteral("No ElevenLabs API key saved."));
    }

    setLayout(rootLayout.release());
    refreshRvcDiagnostics();
    resize(520, 300);
}

void SettingsDialog::setIntroMessage(const QString& message) {
    m_introLabel->setText(message);
    m_introLabel->setVisible(!message.isEmpty());
}

void SettingsDialog::saveApiKey() {
    const auto apiKey = m_apiKeyEdit->text().trimmed().toStdString();
    if (apiKey.empty()) {
        setStatusText(QStringLiteral("Enter an API key before saving."));
        return;
    }

    auto stored = m_vault.storeElevenLabsApiKey(apiKey);
    if (!stored) {
        setStatusText(QString::fromStdString(stored.error().message));
        return;
    }

    m_apiKeyEdit->clear();
    setStatusText(QStringLiteral("ElevenLabs API key saved."));
    emit apiKeySaved();
}

void SettingsDialog::testConnection() {
    if (m_connectionWatcher->isRunning()) {
        return;
    }

    std::string apiKey = m_apiKeyEdit->text().trimmed().toStdString();
    if (apiKey.empty()) {
        auto loaded = m_vault.loadElevenLabsApiKey();
        if (!loaded) {
            setStatusText(QStringLiteral("Enter or save an API key before testing."));
            return;
        }
        apiKey = std::move(loaded).value();
    }

    setBusy(true);
    setStatusText(QStringLiteral("Testing ElevenLabs connection..."));
    m_connectionWatcher->setFuture(QtConcurrent::run(runConnectionTest, std::move(apiKey)));
}

void SettingsDialog::finishConnectionTest() {
    setBusy(false);
    const auto result = m_connectionWatcher->result();
    if (!result) {
        setStatusText(apiErrorText(result.error()));
        return;
    }

    const auto& value = result.value();
    const auto accountText =
        value.userEmail.empty()
            ? QString{}
            : QStringLiteral(" as %1").arg(QString::fromStdString(value.userEmail));
    const auto statusTemplate =
        QStringLiteral("Connected%1. Tier: %2. Characters: %3/%4. Voices: %5. Models: %6.");
    setStatusText(statusTemplate
                      .arg(accountText)
                      .arg(QString::fromStdString(value.subscriptionTier))
                      .arg(value.characterCount)
                      .arg(value.characterLimit)
                      .arg(value.voiceCount)
                      .arg(value.modelCount));
}

void SettingsDialog::saveRvcEngineMode() {
    if (m_rvcEngineCombo == nullptr || m_rvcEngineCombo->currentIndex() < 0) {
        return;
    }

    QSettings settings;
    settings.setValue(QStringLiteral("rvc/runtime"), m_rvcEngineCombo->currentData().toString());
    refreshRvcDiagnostics();
}

void SettingsDialog::refreshRvcDiagnostics() {
    const auto mode =
        m_rvcEngineCombo == nullptr ? QStringLiteral("sidecar")
                                    : m_rvcEngineCombo->currentData().toString();
    if (mode == QStringLiteral("native_onnx")) {
        const rvc::OnnxRvcEngine engine;
        auto runtime = engine.probeRuntime();
        if (!runtime) {
            m_rvcStatusLabel->setText(QString::fromStdString(runtime.error().message));
            return;
        }
        const auto status =
            runtime.value().available
                ? QStringLiteral("Native ONNX: runtime %1 at %2. Models: %3.")
                      .arg(QString::fromStdString(runtime.value().version),
                           QString::fromStdString(runtime.value().runtimeDllPath.string()),
                           QString::fromStdString(
                               rvc::OnnxRvcEngine::defaultNativeModelRoot().string()))
                : QStringLiteral("Native ONNX: %1")
                      .arg(QString::fromStdString(runtime.value().message));
        m_rvcStatusLabel->setText(status);
        return;
    }

    const rvc::RvcSidecar sidecar;
    const auto sidecarStatus = sidecar.status();

    QString status =
        QStringLiteral("Sidecar: %1 at %2. Endpoint: %3.")
            .arg(sidecarStatus.installed ? QStringLiteral("installed")
                                         : QStringLiteral("not installed"))
            .arg(QString::fromStdString(sidecarStatus.sidecarRoot.string()),
                 QString::fromStdString(sidecarStatus.endpoint));
    if (sidecarStatus.running) {
        const rvc::RvcClient client{sidecarStatus.endpoint};
        auto health = client.health();
        if (health) {
            status += QStringLiteral(" GPU: %1. CUDA: %2. Model: %3. Last latency: %4 ms.")
                          .arg(health.value().cudaAvailable ? QStringLiteral("available")
                                                            : QStringLiteral("not reported"))
                          .arg(QString::fromStdString(health.value().cudaVersion))
                          .arg(QString::fromStdString(health.value().loadedModelId))
                          .arg(health.value().lastLatencyMs);
        } else {
            status += QStringLiteral(" Health: %1")
                          .arg(QString::fromStdString(health.error().message));
        }
    } else {
        status += QStringLiteral(" GPU/CUDA/model/latency appear after Local mode starts.");
    }
    m_rvcStatusLabel->setText(status);
}

void SettingsDialog::openRvcModelManager() {
    RvcModelManagerDialog dialog{this};
    dialog.exec();
    refreshRvcDiagnostics();
}

void SettingsDialog::setBusy(const bool isBusy) {
    m_testButton->setEnabled(!isBusy);
    m_saveButton->setEnabled(!isBusy);
    m_apiKeyEdit->setEnabled(!isBusy);
    m_rvcEngineCombo->setEnabled(!isBusy);
    m_rvcRefreshButton->setEnabled(!isBusy);
    m_rvcModelButton->setEnabled(!isBusy);
}

void SettingsDialog::setStatusText(const QString& text) {
    m_statusLabel->setText(text);
}

} // namespace voxstudio::ui
