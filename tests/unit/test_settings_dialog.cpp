#include "secrets/DpapiVault.h"
#include "support/TemporarySecretFile.h"
#include "ui/SettingsDialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QtTest/QtTest>

#include <string>

class SettingsDialogTest final : public QObject {
    Q_OBJECT

private slots:
    void requiredSetupValidatesStoresAndAcceptsTheUsersKey();
    void requiredSetupRejectsAnInvalidKey();
};

void SettingsDialogTest::requiredSetupValidatesStoresAndAcceptsTheUsersKey() {
    const voxstudio::test::TemporarySecretFile secretFile{"voxstudio_settings_test_"};
    const voxstudio::secrets::DpapiVault vault{secretFile.secretPath()};
    voxstudio::ui::SettingsDialog dialog{
        vault,
        [](std::string) {
            return voxstudio::core::Expected<
                voxstudio::ui::ConnectionTestResult,
                voxstudio::net::elevenlabs::ApiError>{
                voxstudio::ui::ConnectionTestResult{
                    "owner@example.test", "creator", 1, 1000, 2, 3}};
        }};
    dialog.requireApiKeyBeforeUse();

    QCOMPARE(dialog.windowTitle(), QStringLiteral("Connect ElevenLabs"));

    auto* introLabel =
        dialog.findChild<QLabel*>(QStringLiteral("SettingsIntroLabel"));
    QVERIFY(introLabel != nullptr);
    QVERIFY(introLabel->text().contains(QStringLiteral("your own ElevenLabs API key")));

    auto* apiKeyEdit =
        dialog.findChild<QLineEdit*>(QStringLiteral("SettingsApiKeyEdit"));
    auto* connectButton =
        dialog.findChild<QPushButton*>(QStringLiteral("SettingsSaveButton"));
    auto* exitButton =
        dialog.findChild<QPushButton*>(QStringLiteral("SettingsCloseButton"));
    QVERIFY(apiKeyEdit != nullptr);
    QVERIFY(connectButton != nullptr);
    QVERIFY(exitButton != nullptr);
    QCOMPARE(connectButton->text(), QStringLiteral("Connect"));
    QCOMPARE(exitButton->text(), QStringLiteral("Exit Vox Studio"));

    apiKeyEdit->setText(QStringLiteral("test_key_owned_by_current_user"));
    QTest::mouseClick(connectButton, Qt::LeftButton);

    QTRY_COMPARE(dialog.result(), static_cast<int>(QDialog::Accepted));
    auto loaded = vault.loadElevenLabsApiKey();
    QVERIFY(loaded.hasValue());
    QCOMPARE(loaded.value(), std::string{"test_key_owned_by_current_user"});
    QVERIFY(vault.hasElevenLabsApiKey());
}

void SettingsDialogTest::requiredSetupRejectsAnInvalidKey() {
    const voxstudio::test::TemporarySecretFile secretFile{"voxstudio_settings_test_"};
    const voxstudio::secrets::DpapiVault vault{secretFile.secretPath()};
    voxstudio::ui::SettingsDialog dialog{
        vault,
        [](std::string) {
            return voxstudio::core::Expected<
                voxstudio::ui::ConnectionTestResult,
                voxstudio::net::elevenlabs::ApiError>{
                voxstudio::net::elevenlabs::ApiError{
                    voxstudio::net::elevenlabs::ApiErrorCode::HttpError,
                    "ElevenLabs rejected this API key.",
                    401}};
        }};
    dialog.requireApiKeyBeforeUse();

    auto* apiKeyEdit =
        dialog.findChild<QLineEdit*>(QStringLiteral("SettingsApiKeyEdit"));
    auto* connectButton =
        dialog.findChild<QPushButton*>(QStringLiteral("SettingsSaveButton"));
    auto* statusLabel =
        dialog.findChild<QLabel*>(QStringLiteral("SettingsStatusLabel"));
    QVERIFY(apiKeyEdit != nullptr);
    QVERIFY(connectButton != nullptr);
    QVERIFY(statusLabel != nullptr);

    apiKeyEdit->setText(QStringLiteral("not_a_valid_key"));
    QTest::mouseClick(connectButton, Qt::LeftButton);

    QTRY_VERIFY(statusLabel->text().contains(QStringLiteral("rejected")));
    QCOMPARE(dialog.result(), static_cast<int>(QDialog::Rejected));
    QVERIFY(!vault.hasElevenLabsApiKey());
}

QTEST_MAIN(SettingsDialogTest)

#include "test_settings_dialog.moc"
