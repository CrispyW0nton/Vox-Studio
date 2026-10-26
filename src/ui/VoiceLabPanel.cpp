#include "ui/VoiceLabPanel.h"

#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMap>
#include <QPlainTextEdit>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QUuid>
#include <QVBoxLayout>
#include <array>

namespace voxstudio::ui {
namespace {
QString dataRoot() {
    return QDir(qEnvironmentVariable("LOCALAPPDATA")).filePath(QStringLiteral("VoxStudio"));
}
QString workerPath() {
    return QDir(QCoreApplication::applicationDirPath())
        .filePath(QStringLiteral("voice_lab/voice_studio_lab.py"));
}
} // namespace

VoiceLabPanel::VoiceLabPanel(QWidget* parent) : QWidget(parent), m_process(new QProcess(this)) {
    setObjectName(QStringLiteral("VoiceLabPanel"));
    auto* root = new QVBoxLayout(this);
    auto* heading = new QHBoxLayout;
    heading->addWidget(
        new QLabel(QStringLiteral("Voice Lab — characters, models and local engines")));
    heading->addStretch();
    m_refresh = new QPushButton(QStringLiteral("Refresh inventory"));
    heading->addWidget(m_refresh);
    root->addLayout(heading);
    auto* split = new QSplitter(this);
    m_tree = new QTreeWidget(split);
    m_tree->setObjectName(QStringLiteral("VoiceLabInventory"));
    m_tree->setHeaderLabels({QStringLiteral("Library"), QStringLiteral("Files")});
    m_tree->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    auto* right = new QWidget(split);
    auto* detailLayout = new QVBoxLayout(right);
    m_details = new QPlainTextEdit;
    m_details->setReadOnly(true);
    detailLayout->addWidget(m_details);
    m_links = new QListWidget;
    m_links->setToolTip(
        QStringLiteral("Double-click to open a local listening pack, report or folder."));
    detailLayout->addWidget(m_links);
    auto* open = new QPushButton(QStringLiteral("Open selected file or folder"));
    detailLayout->addWidget(open);
    split->setStretchFactor(1, 1);
    root->addWidget(split, 2);

    auto* form = new QFormLayout;
    m_presets = new QComboBox;
    m_presets->setObjectName(QStringLiteral("VoiceLabPreset"));
    form->addRow(QStringLiteral("Render recipe"), m_presets);
    m_note = new QLabel;
    m_note->setWordWrap(true);
    form->addRow(m_note);
    auto* scriptRow = new QHBoxLayout;
    m_script = new QLineEdit;
    m_script->setObjectName(QStringLiteral("VoiceLabScript"));
    m_browse = new QPushButton(QStringLiteral("Choose script…"));
    m_import = new QPushButton(QStringLiteral("Import Qwen / Chatterbox recipe…"));
    scriptRow->addWidget(m_script, 1);
    scriptRow->addWidget(m_browse);
    scriptRow->addWidget(m_import);
    form->addRow(QStringLiteral("Script / audition recipe"), scriptRow);
    m_checkpoint = new QLineEdit;
    m_checkpoint->setObjectName(QStringLiteral("VoiceLabCheckpoint"));
    form->addRow(QStringLiteral("LoRA checkpoint"), m_checkpoint);
    auto* takeRow = new QHBoxLayout;
    m_lines = new QLineEdit;
    m_lines->setObjectName(QStringLiteral("VoiceLabLines"));
    m_lines->setPlaceholderText(
        QStringLiteral("Optional comma-separated line IDs, e.g. E196,E198"));
    takeRow->addWidget(m_lines, 1);
    takeRow->addWidget(new QLabel(QStringLiteral("Seeds per line")));
    m_seeds = new QSpinBox;
    m_seeds->setRange(1, 6);
    takeRow->addWidget(m_seeds);
    form->addRow(QStringLiteral("Takes"), takeRow);
    root->addLayout(form);
    auto* actions = new QHBoxLayout;
    m_preview = new QPushButton(QStringLiteral("Preview render"));
    m_preview->setObjectName(QStringLiteral("VoiceLabPreview"));
    m_render = new QPushButton(QStringLiteral("Render preview"));
    m_render->setObjectName(QStringLiteral("VoiceLabRender"));
    m_stop = new QPushButton(QStringLiteral("Stop"));
    auto* output = new QPushButton(QStringLiteral("Open job outputs"));
    actions->addWidget(m_preview);
    actions->addWidget(m_render);
    actions->addWidget(m_stop);
    actions->addWidget(output);
    actions->addStretch();
    root->addLayout(actions);
    m_log = new QPlainTextEdit;
    m_log->setObjectName(QStringLiteral("VoiceLabLog"));
    m_log->setReadOnly(true);
    m_log->setMaximumBlockCount(2500);
    root->addWidget(m_log, 1);

    connect(m_refresh, &QPushButton::clicked, this, &VoiceLabPanel::refresh);
    connect(m_tree, &QTreeWidget::itemSelectionChanged, this, &VoiceLabPanel::showSelection);
    connect(m_links, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem* item) { openPath(item->text()); });
    connect(open, &QPushButton::clicked, this, [this]() {
        if (m_links->currentItem())
            openPath(m_links->currentItem()->text());
    });
    connect(output, &QPushButton::clicked, this, [this]() {
        if (!m_jobFile.isEmpty())
            openPath(QFileInfo(m_jobFile).absolutePath());
    });
    connect(m_presets, &QComboBox::currentIndexChanged, this, &VoiceLabPanel::selectPreset);
    for (auto* edit : {m_script, m_checkpoint, m_lines}) {
        connect(edit, &QLineEdit::textChanged, this, &VoiceLabPanel::invalidatePreview);
    }
    connect(m_seeds, &QSpinBox::valueChanged, this, &VoiceLabPanel::invalidatePreview);
    connect(m_browse, &QPushButton::clicked, this, [this]() {
        const auto path =
            QFileDialog::getOpenFileName(this, QStringLiteral("Choose a voice script"),
                                         m_script->text(), QStringLiteral("JSON (*.json)"));
        if (!path.isEmpty())
            m_script->setText(path);
    });
    connect(m_import, &QPushButton::clicked, this, [this]() {
        const auto path = QFileDialog::getOpenFileName(
            this, QStringLiteral("Import local narration audition recipe"), {},
            QStringLiteral("JSON (*.json)"));
        if (path.isEmpty())
            return;
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly))
            return;
        const auto recipe = QJsonDocument::fromJson(file.readAll()).object();
        const auto backend = recipe.value(QStringLiteral("backend")).toString();
        if (backend != QStringLiteral("qwen") && backend != QStringLiteral("chatterbox") &&
            backend != QStringLiteral("indextts25")) {
            m_log->setPlainText(
                QStringLiteral("Select an existing Qwen, Chatterbox or IndexTTS-2.5 audition recipe."));
            return;
        }
        QJsonObject preset{
            {QStringLiteral("backend"), backend},
            {QStringLiteral("script"), path},
            {QStringLiteral("note"),
             QStringLiteral("Uses the exact reference, model snapshot, seeds and text in the "
                            "imported recipe. New takes require new take and batch IDs.")}};
        m_presetData.append(preset);
        m_presets->addItem(backend + QStringLiteral(" — ") + QFileInfo(path).fileName());
        m_presets->setCurrentIndex(m_presets->count() - 1);
    });
    connect(m_preview, &QPushButton::clicked, this, [this]() { start(QStringLiteral("preview")); });
    connect(m_render, &QPushButton::clicked, this, [this]() { start(QStringLiteral("render")); });
    connect(m_stop, &QPushButton::clicked, this, [this]() {
        if (m_action == QStringLiteral("render") && !m_jobFile.isEmpty()) {
            QFile stop(QFileInfo(m_jobFile).absolutePath() + QStringLiteral("/STOP"));
            if (stop.open(QIODevice::WriteOnly))
                stop.write("stop");
            m_stop->setEnabled(false);
            m_log->appendPlainText(
                QStringLiteral("Stopping this job; partial outputs will be retained."));
        }
    });
    m_process->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
        const auto chunk = m_process->readAllStandardOutput();
        m_output += chunk;
        if (m_action != QStringLiteral("catalog"))
            m_log->appendPlainText(QString::fromUtf8(chunk));
    });
    connect(m_process, &QProcess::finished, this, &VoiceLabPanel::finished);
    connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) {
            m_log->appendPlainText(m_process->errorString());
            invalidatePreview();
            setBusy(false);
        }
    });
    setBusy(false);
    // Inventory is explicit so tests and other workspaces never spawn a process.
    m_log->setPlainText(
        QStringLiteral("Refresh inventory to discover installed runtimes and character labs. "
                       "Preview validates inputs without loading a synthesis model. Render uses "
                       "one compute GPU and preserves existing takes."));
}

bool VoiceLabPanel::isRunning() const {
    return m_process->state() != QProcess::NotRunning;
}

void VoiceLabPanel::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    if (!m_inventoryLoaded && !isRunning()) {
        QTimer::singleShot(0, this, &VoiceLabPanel::refresh);
    }
}

void VoiceLabPanel::openPath(const QString& path) {
    if (QFileInfo::exists(path))
        QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    else
        m_log->appendPlainText(QStringLiteral("Missing: ") + path);
}

void VoiceLabPanel::invalidatePreview() {
    m_revision.clear();
    m_render->setEnabled(false);
}

void VoiceLabPanel::setBusy(bool busy) {
    const std::array<QWidget*, 8> controls{m_refresh, m_presets, m_script, m_checkpoint,
                                           m_lines,   m_seeds,   m_browse, m_import};
    for (auto* widget : controls)
        widget->setEnabled(!busy);
    const auto backend = m_presetData.at(m_presets->currentIndex())
                             .toObject()
                             .value(QStringLiteral("backend"))
                             .toString();
    const bool audition =
        backend == QStringLiteral("qwen") || backend == QStringLiteral("chatterbox") ||
        backend == QStringLiteral("indextts25");
    m_checkpoint->setEnabled(!busy && backend == QStringLiteral("voxcpm"));
    m_lines->setEnabled(!busy && !audition);
    m_seeds->setEnabled(!busy && !audition);
    m_preview->setEnabled(!busy && m_presets->count() > 0);
    m_render->setEnabled(!busy && !m_revision.isEmpty());
    m_stop->setEnabled(busy && m_action == QStringLiteral("render"));
}

void VoiceLabPanel::refresh() {
    start(QStringLiteral("catalog"));
}

void VoiceLabPanel::showSelection() {
    const auto* item = m_tree->currentItem();
    m_links->clear();
    if (!item)
        return;
    const auto object = item->data(0, Qt::UserRole).toJsonObject();
    m_details->setPlainText(
        object.value(QStringLiteral("name")).toString() + QStringLiteral("\n\n") +
        object.value(QStringLiteral("detail")).toString() + QStringLiteral("\n\n") +
        object.value(QStringLiteral("path")).toString());
    const auto path = object.value(QStringLiteral("path")).toString();
    if (!path.isEmpty())
        m_links->addItem(path);
    for (const auto& link : object.value(QStringLiteral("links")).toArray())
        m_links->addItem(link.toString());
    m_links->setCurrentRow(0);
}

void VoiceLabPanel::selectPreset() {
    invalidatePreview();
    const auto p = m_presetData.at(m_presets->currentIndex()).toObject();
    m_script->setText(p.value(QStringLiteral("script")).toString());
    m_checkpoint->setText(p.value(QStringLiteral("checkpoint")).toString());
    m_lines->setText(p.value(QStringLiteral("lines")).toString());
    m_seeds->setValue(p.value(QStringLiteral("seeds")).toInt(1));
    m_note->setText(p.value(QStringLiteral("note")).toString());
    m_preview->setEnabled(!isRunning() && m_presets->currentIndex() >= 0);
    setBusy(isRunning());
}

void VoiceLabPanel::start(const QString& action) {
    if (isRunning())
        return;
    const auto python =
        QDir(dataRoot()).filePath(QStringLiteral("engines/voxcpm2/.venv/Scripts/python.exe"));
    if (!QFileInfo::exists(python) || !QFileInfo::exists(workerPath())) {
        m_log->setPlainText(
            QStringLiteral("Missing desktop bridge or VoxCPM Python runtime. Rebuild the desktop "
                           "or install the local VoxCPM runtime."));
        return;
    }
    QStringList args{QStringLiteral("-B"), workerPath(), action};
    if (action == QStringLiteral("preview")) {
        if (m_presets->currentIndex() < 0)
            return;
        auto job = m_presetData.at(m_presets->currentIndex()).toObject();
        job.insert(QStringLiteral("script"), m_script->text());
        job.insert(QStringLiteral("checkpoint"), m_checkpoint->text());
        job.insert(QStringLiteral("lines"), m_lines->text());
        job.insert(QStringLiteral("seeds"), m_seeds->value());
        const auto directory = QDir(dataRoot())
                                   .filePath(QStringLiteral("desktop_jobs/") +
                                             QDateTime::currentDateTimeUtc().toString(
                                                 QStringLiteral("yyyyMMdd-HHmmss-")) +
                                             QUuid::createUuid().toString(QUuid::WithoutBraces));
        if (!QDir().mkpath(directory)) {
            m_log->setPlainText(QStringLiteral("Cannot create job directory."));
            return;
        }
        m_jobFile = directory + QStringLiteral("/job.json");
        QFile file(m_jobFile);
        if (!file.open(QIODevice::WriteOnly) || file.write(QJsonDocument(job).toJson()) < 0) {
            m_log->setPlainText(QStringLiteral("Cannot save the render job."));
            return;
        }
        invalidatePreview();
    }
    if (action != QStringLiteral("catalog")) {
        args << QStringLiteral("--job") << m_jobFile;
        if (action == QStringLiteral("render"))
            args << QStringLiteral("--revision") << m_revision;
    }
    m_action = action;
    m_output.clear();
    m_log->setPlainText(action == QStringLiteral("catalog")
                            ? QStringLiteral("Reading local inventory…")
                            : QStringLiteral("Starting ") + action + QStringLiteral("…"));
    auto env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    env.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    m_process->setProcessEnvironment(env);
    setBusy(true);
    m_process->start(python, args);
}

void VoiceLabPanel::finished(int code, QProcess::ExitStatus status) {
    m_output += m_process->readAllStandardOutput();
    const bool ok = code == 0 && status == QProcess::NormalExit;
    if (m_action == QStringLiteral("catalog") && ok) {
        QJsonParseError error;
        const auto document = QJsonDocument::fromJson(m_output, &error);
        if (error.error != QJsonParseError::NoError || !document.isObject()) {
            m_log->setPlainText(QStringLiteral("Invalid inventory response: ") +
                                error.errorString());
            setBusy(false);
            return;
        }
        setInventory(document.object());
    } else {
        if (m_action == QStringLiteral("catalog"))
            m_log->setPlainText(QString::fromUtf8(m_output));
        invalidatePreview();
        if (m_action == QStringLiteral("preview") && ok) {
            QFile receipt(QFileInfo(m_jobFile).absolutePath() + QStringLiteral("/preview.json"));
            if (receipt.open(QIODevice::ReadOnly))
                m_revision = QJsonDocument::fromJson(receipt.readAll())
                                 .object()
                                 .value(QStringLiteral("revision"))
                                 .toString();
        }
        m_log->appendPlainText(
            ok ? QStringLiteral("Completed. Outputs and receipts are available in the job folder.")
               : QStringLiteral(
                     "Failed (exit %1). Nothing was retried. See the worker output above.")
                     .arg(code));
    }
    setBusy(false);
}

void VoiceLabPanel::setInventory(const QJsonObject& inventory) {
    m_inventoryLoaded = true;
    invalidatePreview();
    m_tree->clear();
    QMap<QString, QTreeWidgetItem*> groups;
    for (const auto& value : inventory.value(QStringLiteral("items")).toArray()) {
        const auto item = value.toObject();
        const auto group = item.value(QStringLiteral("group")).toString();
        if (!groups.contains(group))
            groups[group] = new QTreeWidgetItem(m_tree, {group});
        auto* row =
            new QTreeWidgetItem(groups[group], {item.value(QStringLiteral("name")).toString(),
                                                item.value(QStringLiteral("status")).toString()});
        row->setData(0, Qt::UserRole, item);
    }
    m_tree->expandToDepth(0);
    m_presets->blockSignals(true);
    m_presets->clear();
    m_presetData = inventory.value(QStringLiteral("presets")).toArray();
    for (const auto& value : m_presetData)
        m_presets->addItem(value.toObject().value(QStringLiteral("name")).toString());
    m_presets->blockSignals(false);
    if (!m_presetData.isEmpty())
        selectPreset();
    m_log->setPlainText(
        QStringLiteral("Inventory refreshed. Presence checks do not load or qualify models."));
    for (const auto& warning : inventory.value(QStringLiteral("warnings")).toArray())
        m_log->appendPlainText(warning.toString());
    setBusy(isRunning());
}

} // namespace voxstudio::ui
