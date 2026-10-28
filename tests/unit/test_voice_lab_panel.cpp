#include "ui/VoiceLabPanel.h"
#include <QComboBox>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QPushButton>
#include <QTreeWidget>
#include <QtTest/QtTest>

class VoiceLabPanelTest final : public QObject {
    Q_OBJECT
private slots:
    void preservesCharacterRecipeAndRequiresPreview();
    void emptyInventoryCannotRender();
    void localInventoryVisualCheck();
};

void VoiceLabPanelTest::preservesCharacterRecipeAndRequiresPreview() {
    voxstudio::ui::VoiceLabPanel panel;
    QJsonObject preset{{"name", "Atris"},        {"backend", "voxcpm"},
                       {"script", "exact.json"}, {"checkpoint", "step_0000400"},
                       {"lines", "E198"},        {"seeds", 1}};
    panel.setInventory(QJsonObject{{"presets", QJsonArray{preset}}});
    auto* script = panel.findChild<QLineEdit*>("VoiceLabScript");
    auto* checkpoint = panel.findChild<QLineEdit*>("VoiceLabCheckpoint");
    auto* preview = panel.findChild<QPushButton*>("VoiceLabPreview");
    auto* render = panel.findChild<QPushButton*>("VoiceLabRender");
    QVERIFY(script && checkpoint && preview && render);
    QCOMPARE(script->text(), QStringLiteral("exact.json"));
    QCOMPARE(checkpoint->text(), QStringLiteral("step_0000400"));
    QVERIFY(preview->isEnabled());
    QVERIFY(!render->isEnabled());
    QVERIFY(!panel.isRunning());
    script->setText(QStringLiteral("different.json"));
    QVERIFY(!render->isEnabled());
    preset.insert("backend", "qwen");
    panel.setInventory(QJsonObject{{"presets", QJsonArray{preset}}});
    QVERIFY(!checkpoint->isEnabled());
    QVERIFY(!render->isEnabled());
}

void VoiceLabPanelTest::emptyInventoryCannotRender() {
    voxstudio::ui::VoiceLabPanel panel;
    panel.setInventory({});
    QVERIFY(!panel.findChild<QPushButton*>("VoiceLabPreview")->isEnabled());
    QVERIFY(!panel.findChild<QPushButton*>("VoiceLabRender")->isEnabled());
}

void VoiceLabPanelTest::localInventoryVisualCheck() {
    const auto path = qEnvironmentVariable("VOXSTUDIO_TEST_LAB_CATALOG");
    if (path.isEmpty())
        QSKIP("Optional local inventory visual check");
    QFile file(path);
    QVERIFY(file.open(QIODevice::ReadOnly));
    const auto inventory = QJsonDocument::fromJson(file.readAll());
    QVERIFY(inventory.isObject());
    voxstudio::ui::VoiceLabPanel panel;
    panel.resize(1280, 850);
    panel.setInventory(inventory.object());
    panel.show();
    QVERIFY(QTest::qWaitForWindowExposed(&panel));
    auto* tree = panel.findChild<QTreeWidget*>("VoiceLabInventory");
    const auto matches =
        tree->findItems(QStringLiteral("Atris"), Qt::MatchExactly | Qt::MatchRecursive);
    QVERIFY(!matches.isEmpty());
    tree->setCurrentItem(matches.front());
    QCoreApplication::processEvents();
    if (qEnvironmentVariable("VOXSTUDIO_TEST_LAB_PREVIEW") == QStringLiteral("1")) {
        panel.findChild<QLineEdit*>("VoiceLabLines")->setText(QStringLiteral("E198"));
        panel.findChild<QPushButton*>("VoiceLabPreview")->click();
        QTRY_VERIFY_WITH_TIMEOUT(!panel.isRunning(), 60000);
        QVERIFY(panel.findChild<QPushButton*>("VoiceLabRender")->isEnabled());
    }
    const auto screenshot = qEnvironmentVariable("VOXSTUDIO_TEST_LAB_SCREENSHOT");
    if (!screenshot.isEmpty())
        QVERIFY(panel.grab().save(screenshot));
}

QTEST_MAIN(VoiceLabPanelTest)
#include "test_voice_lab_panel.moc"
