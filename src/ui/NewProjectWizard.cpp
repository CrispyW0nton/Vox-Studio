#include "ui/NewProjectWizard.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QVBoxLayout>

#include <memory>

namespace voxstudio::ui {
namespace {

[[nodiscard]] QString sanitizedFolderName(QString name) {
    name = name.trimmed();
    name.replace(QRegularExpression(QStringLiteral(R"([<>:"/\\|?*]+)")), QStringLiteral("_"));
    return name;
}

} // namespace

NewProjectWizard::NewProjectWizard(QWidget* parent)
    : QDialog(parent) {
    setWindowTitle(QStringLiteral("New Project"));
    setModal(true);

    auto rootLayout = std::make_unique<QVBoxLayout>();
    auto formLayout = std::make_unique<QFormLayout>();

    auto nameEdit = std::make_unique<QLineEdit>();
    m_nameEdit = nameEdit.get();
    formLayout->addRow(QStringLiteral("Name"), nameEdit.release());

    auto locationEdit = std::make_unique<QLineEdit>(QDir::homePath());
    m_locationEdit = locationEdit.get();

    auto browseButton = std::make_unique<QPushButton>(QStringLiteral("Browse"));
    connect(browseButton.get(), &QPushButton::clicked, this, &NewProjectWizard::browseForLocation);

    auto locationLayout = std::make_unique<QHBoxLayout>();
    locationLayout->addWidget(locationEdit.release(), 1);
    locationLayout->addWidget(browseButton.release());
    formLayout->addRow(QStringLiteral("Location"), locationLayout.release());

    rootLayout->addLayout(formLayout.release());

    auto buttons =
        std::make_unique<QDialogButtonBox>(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons.get(), &QDialogButtonBox::accepted, this, &NewProjectWizard::accept);
    connect(buttons.get(), &QDialogButtonBox::rejected, this, &NewProjectWizard::reject);
    rootLayout->addWidget(buttons.release());

    setLayout(rootLayout.release());
    resize(520, 140);
}

QString NewProjectWizard::projectName() const {
    return m_nameEdit->text().trimmed();
}

std::filesystem::path NewProjectWizard::projectDirectory() const {
    const QString folderName = sanitizedFolderName(projectName());
    const QDir parentDirectory{m_locationEdit->text().trimmed()};
    return std::filesystem::path{parentDirectory.filePath(folderName + QStringLiteral(".vox"))
                                     .toStdString()};
}

void NewProjectWizard::accept() {
    if (projectName().isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("New Project"),
                             QStringLiteral("Enter a project name."));
        return;
    }

    const QDir parentDirectory{m_locationEdit->text().trimmed()};
    if (!parentDirectory.exists()) {
        QMessageBox::warning(this, QStringLiteral("New Project"),
                             QStringLiteral("Choose an existing location."));
        return;
    }

    if (std::filesystem::exists(projectDirectory())) {
        QMessageBox::warning(this, QStringLiteral("New Project"),
                             QStringLiteral("A project folder already exists there."));
        return;
    }

    QDialog::accept();
}

void NewProjectWizard::browseForLocation() {
    const QString directory = QFileDialog::getExistingDirectory(
        this, QStringLiteral("Project Location"), m_locationEdit->text().trimmed());
    if (!directory.isEmpty()) {
        m_locationEdit->setText(directory);
    }
}

} // namespace voxstudio::ui
