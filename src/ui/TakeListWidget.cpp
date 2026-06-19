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
    return QStringLiteral("%1Take %2  %3s")
        .arg(star)
        .arg(index + 1)
        .arg(seconds, 0, 'f', 1);
}

} // namespace

TakeListWidget::TakeListWidget(QWidget* parent)
    : QWidget(parent) {
    auto rootLayout = std::make_unique<QVBoxLayout>();
    rootLayout->setContentsMargins(0, 0, 0, 0);

    auto* title = addOwnedWidget<QLabel>(*rootLayout, QStringLiteral("Takes"));
    title->setObjectName(QStringLiteral("TakeListTitle"));

    auto buttons = std::make_unique<QHBoxLayout>();
    m_playButton = addOwnedWidget<QPushButton>(*buttons, QStringLiteral("Play"));
    m_starButton = addOwnedWidget<QPushButton>(*buttons, QStringLiteral("Star"));
    m_deleteButton = addOwnedWidget<QPushButton>(*buttons, QStringLiteral("Delete"));
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
    connect(m_deleteButton, &QPushButton::clicked, this, &TakeListWidget::deleteSelectedTake);
    refreshList();
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
        if (m_takes[static_cast<std::size_t>(index)].starred) {
            item->setSelected(true);
        }
        m_takeList->addItem(item.release());
    }

    const bool hasTakes = !m_takes.empty();
    m_playButton->setEnabled(hasTakes);
    m_starButton->setEnabled(hasTakes);
    m_deleteButton->setEnabled(hasTakes);
    m_statusLabel->setText(
        hasTakes ? QStringLiteral("%1 takes.").arg(static_cast<int>(m_takes.size()))
                 : QStringLiteral("No takes for this line yet."));
}

} // namespace voxstudio::ui
