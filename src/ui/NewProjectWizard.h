#pragma once

#include <QDialog>
#include <QString>

#include <filesystem>

class QLineEdit;

namespace voxstudio::ui {

class NewProjectWizard final : public QDialog {
    Q_OBJECT

public:
    explicit NewProjectWizard(QWidget* parent = nullptr);

    [[nodiscard]] QString projectName() const;
    [[nodiscard]] std::filesystem::path projectDirectory() const;

protected:
    void accept() override;

private:
    void browseForLocation();

    QLineEdit* m_nameEdit{nullptr};
    QLineEdit* m_locationEdit{nullptr};
};

} // namespace voxstudio::ui

