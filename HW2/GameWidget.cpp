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
    for (int i = 0; i < 4; i++)
        delete meows[i];
}

void GameWidget::init() {
    // Reset game status flag
    this->isGameOver = false;
    this->isItemInView = true;
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

    // Set up pthread items
    this->initPthread();
}

void GameWidget::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Q) {
        QMessageBox::information(NULL,
          "QUIT Game",
          "Bye~~Bye~~",
          QMessageBox::Yes,
          QMessageBox::Yes);
        close();
    }
    if (this->isGameOver) {
        return;
    }
    switch (event->key()) {
        case Qt::Key_D:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": D");
            this->isGameOver = heroItem->moveItem(100, 0);
            break;
        case Qt::Key_S:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": S");
            this->isGameOver = heroItem->moveItem(0, 100);
            break;
        case Qt::Key_A:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": A");
            this->isGameOver = heroItem->moveItem(-100, 0);
            break;
        case Qt::Key_W:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": W");
            this->isGameOver = heroItem->moveItem(0, -100);
            break;
        default:
            break;
    }
    if (this->isGameOver) {
        QMessageBox::information(NULL,
          "GOAL",
          "You Win !!!",
          QMessageBox::Yes,
          QMessageBox::Yes);
        keyboardInputLabel->setText(tr("GOAL, you win!"));
    }
}
