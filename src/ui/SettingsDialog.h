#pragma once

#include "core/Expected.h"
#include "net/elevenlabs/Models.h"
#include "secrets/DpapiVault.h"

#include <QDialog>
#include <QFutureWatcher>
#include <QString>

#include <memory>
#include <functional>
#include <optional>
#include <string>

class QDialogButtonBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;

namespace voxstudio::ui {

struct ConnectionTestResult final {
    std::string userEmail;
    std::string subscriptionTier;
    int characterCount{0};
    int characterLimit{0};
    int voiceCount{0};
    int modelCount{0};
};

using ConnectionTestOutcome =
    core::Expected<ConnectionTestResult, net::elevenlabs::ApiError>;
using ConnectionTester = std::function<ConnectionTestOutcome(std::string)>;

class SettingsDialog final : public QDialog {
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget* parent = nullptr);
    SettingsDialog(secrets::DpapiVault vault, QWidget* parent = nullptr);
    SettingsDialog(secrets::DpapiVault vault,
                   ConnectionTester connectionTester,
                   QWidget* parent = nullptr);

    void setIntroMessage(const QString& message);
    void requireApiKeyBeforeUse();

signals:
    void apiKeySaved();

private:
    void saveApiKey();
    void testConnection();
    void startConnectionTest(std::string apiKey);
    void finishConnectionTest();
    void saveRvcEngineMode();
    void refreshRvcDiagnostics();
    void openRvcModelManager();
    void setBusy(bool isBusy);
    void setStatusText(const QString& text);

    secrets::DpapiVault m_vault;
    ConnectionTester m_connectionTester;
    QLineEdit* m_apiKeyEdit{nullptr};
    QLabel* m_introLabel{nullptr};
    QLabel* m_statusLabel{nullptr};
    QLabel* m_rvcStatusLabel{nullptr};
    QComboBox* m_rvcEngineCombo{nullptr};
    QPushButton* m_testButton{nullptr};
    QPushButton* m_saveButton{nullptr};
    QPushButton* m_closeButton{nullptr};
    QPushButton* m_rvcRefreshButton{nullptr};
    QPushButton* m_rvcModelButton{nullptr};
    bool m_apiKeyRequired{false};
    std::optional<std::string> m_pendingApiKey;
    std::unique_ptr<QFutureWatcher<ConnectionTestOutcome>> m_connectionWatcher;
};

} // namespace voxstudio::ui
