#include "GameWidget.h"
#include "GameItem.h"
#include "Const.h"
GameWidget::~GameWidget() {
    delete keyboardInputLabel;
    delete levelLabel;
    delete levelLCDNumber;
    delete levelSlider;
    delete layout;
    delete split1;
    delete split2;
    delete container;
    delete containerLayout;
    delete scene;
    delete gameView;
    delete heroItem;
}
void GameWidget::init() {
    // Set up keyboard input label
    this->initKeyboardInputLabel();
    // Set up level infos
    this->initLevel();
    // Set up game view
    this->initGameView();
    
    // Set windows infomation
    this->initLayout();
    this->setWindowTitle(WINDOWS_TITLE);
    this->setMinimumSize(WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);
    this->setMaximumSize(WINDOWS_WIDTH_MAX, WINDOWS_LENGTH_MAX);
    this->setObjectName("mainWindow");
    this->setStyleSheet(STYLE);
    this->initSize();
}

void GameWidget::keyPressEvent(QKeyEvent *event) {
    switch (event->key()) {
        case Qt::Key_D:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": D");
            heroItem->moveItem(50, 0);
            break;
        case Qt::Key_S:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": S");
            heroItem->moveItem(0, 50);
            break;
        case Qt::Key_A:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": A");
            heroItem->moveItem(-50, 0);
            break;
        case Qt::Key_W:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": W");
            heroItem->moveItem(0, -50);
            break;
        case Qt::Key_Q:
            close();
            break;
        default:
            break;
    }
}
