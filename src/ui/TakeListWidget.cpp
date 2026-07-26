#include "ui/TakeListWidget.h"

#include <QAbstractItemView>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QVBoxLayout>

#include <algorithm>
#include <memory>
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

[[nodiscard]] QString takeText(const db::TakeRecord& take, const int index) {
    const auto seconds = take.durationMs / 1000.0;
    const auto star = take.starred ? QStringLiteral("* ") : QString{};
    const auto character = take.characterName.empty()
                               ? QStringLiteral("Character")
                               : QString::fromStdString(take.characterName);
    auto line = QString::fromStdString(take.lineText).simplified();
    if (line.isEmpty()) {
        line = QStringLiteral("Take %1").arg(index + 1);
    } else if (line.size() > 72) {
        line = line.left(69) + QStringLiteral("...");
    }
    return QStringLiteral("%1%2 | %3\n%4 | %5s")
        .arg(star, character, line, QString::fromStdString(take.source))
        .arg(seconds, 0, 'f', 1);
}

} // namespace

TakeListWidget::TakeListWidget(QWidget* parent)
    : QWidget(parent) {
    auto rootLayout = std::make_unique<QVBoxLayout>();
    rootLayout->setContentsMargins(0, 0, 0, 0);

    m_titleLabel = addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Takes"));
    m_titleLabel->setObjectName(QStringLiteral("TakeListTitle"));

    auto buttons = std::make_unique<QHBoxLayout>();
    m_playButton = addOwnedWidget<QPushButton>(*buttons, QStringLiteral("Play"));
    m_starButton = addOwnedWidget<QPushButton>(*buttons, QStringLiteral("Star"));
    m_revealButton = addOwnedWidget<QPushButton>(*buttons, QStringLiteral("Open in Explorer"));
    m_deleteButton = addOwnedWidget<QPushButton>(*buttons, QStringLiteral("Delete"));
    m_playButton->setObjectName(QStringLiteral("TakePlayButton"));
    m_starButton->setObjectName(QStringLiteral("TakeStarButton"));
    m_revealButton->setObjectName(QStringLiteral("TakeRevealButton"));
    m_deleteButton->setObjectName(QStringLiteral("TakeDeleteButton"));
    rootLayout->addLayout(buttons.release());

    m_takeList = addOwnedWidget<QListWidget>(*rootLayout);
    m_takeList->setObjectName(QStringLiteral("TakeList"));
    m_takeList->setSelectionMode(QAbstractItemView::SingleSelection);

    m_statusLabel = addOwnedWidget<QLabel>(*rootLayout);
    m_statusLabel->setObjectName(QStringLiteral("TakeListStatus"));
    m_statusLabel->setWordWrap(true);

    setLayout(rootLayout.release());

    connect(m_playButton, &QPushButton::clicked, this, &TakeListWidget::playSelectedTake);
    connect(m_starButton, &QPushButton::clicked, this, &TakeListWidget::starSelectedTake);
    connect(m_revealButton, &QPushButton::clicked, this, &TakeListWidget::revealSelectedTake);
    connect(m_deleteButton, &QPushButton::clicked, this, &TakeListWidget::deleteSelectedTake);
    connect(m_takeList, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem*) { playSelectedTake(); });
    refreshList();
}

void TakeListWidget::setTitle(const QString& title) {
    m_titleLabel->setText(title);
}

void TakeListWidget::setTakes(std::vector<db::TakeRecord> takes) {
    m_takes = std::move(takes);
    refreshList();
}

std::vector<db::TakeRecord> TakeListWidget::takes() const {
    return m_takes;
}

void TakeListWidget::playSelectedTake() {
    const auto* take = selectedTake();
    if (take == nullptr) {
        m_statusLabel->setText(QStringLiteral("Select a take to play."));
        return;
    }
    emit playTakeRequested(*take);
}

void TakeListWidget::starSelectedTake() {
    const auto* take = selectedTake();
    if (take == nullptr) {
        m_statusLabel->setText(QStringLiteral("Select a take to star."));
        return;
    }
    emit starTakeRequested(*take);
}

void TakeListWidget::revealSelectedTake() {
    const auto* take = selectedTake();
    if (take == nullptr) {
        m_statusLabel->setText(QStringLiteral("Select a take to open."));
        return;
    }
    emit revealTakeRequested(*take);
}

void TakeListWidget::deleteSelectedTake() {
    const auto* take = selectedTake();
    if (take == nullptr) {
        m_statusLabel->setText(QStringLiteral("Select a take to delete."));
        return;
    }
    emit deleteTakeRequested(*take);
}

db::TakeRecord* TakeListWidget::selectedTake() {
    const auto selectedItems = m_takeList->selectedItems();
    if (selectedItems.isEmpty()) {
        return nullptr;
    }

    const auto takeId = selectedItems.front()->data(Qt::UserRole).toString().toStdString();
    const auto take = std::ranges::find_if(m_takes, [&takeId](const auto& item) {
        return item.id == takeId;
    });
    if (take == m_takes.end()) {
        return nullptr;
    }
    return &(*take);
}

const db::TakeRecord* TakeListWidget::selectedTake() const {
    return const_cast<TakeListWidget*>(this)->selectedTake();
}

void TakeListWidget::refreshList() {
    m_takeList->clear();
    for (int index = 0; index < static_cast<int>(m_takes.size()); ++index) {
        auto item = std::make_unique<QListWidgetItem>(
            takeText(m_takes[static_cast<std::size_t>(index)], index));
        item->setData(Qt::UserRole,
                      QString::fromStdString(m_takes[static_cast<std::size_t>(index)].id));
        item->setToolTip(
            QStringLiteral("%1\n%2")
                .arg(QString::fromStdString(m_takes[static_cast<std::size_t>(index)].createdAt),
                     QString::fromStdString(m_takes[static_cast<std::size_t>(index)].filePath)));
        m_takeList->addItem(item.release());
    }

    const bool hasTakes = !m_takes.empty();
    if (hasTakes) {
        m_takeList->setCurrentRow(0);
    }
    m_playButton->setEnabled(hasTakes);
    m_starButton->setEnabled(hasTakes);
    m_revealButton->setEnabled(hasTakes);
    m_deleteButton->setEnabled(hasTakes);
    m_statusLabel->setText(
        hasTakes ? QStringLiteral("%1 takes.").arg(static_cast<int>(m_takes.size()))
                 : QStringLiteral("No takes yet."));
}

} // namespace voxstudio::ui
