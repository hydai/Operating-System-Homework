#include "GameWidget.h"
#include "Const.h"
GameWidget::~GameWidget() {
    delete backgroundImage;
    delete keyboardInputLabel;
    delete characterIcon;
    delete levelLabel;
    delete levelLCDNumber;
    delete levelSlider;
    delete layout;
    delete split1;
    delete split2;
    delete container;
    delete containerLayout;
}
void GameWidget::init() {
    // Set up background image
    this->initBackground();
    // Set up keyboard input label
    this->initKeyboardInputLabel();
    // Set up character icon
    this->initCharacterIcon();
    // Set up level infos
    this->initLevel();
    
    // Set windows infomation
    this->initLayout();
    this->setWindowTitle(WINDOWS_TITLE);
    this->setMinimumSize(WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);
    this->setMaximumSize(WINDOWS_WIDTH_MAX, WINDOWS_LENGTH_MAX);
    this->setObjectName("mainWindow");
    this->setStyleSheet("border-image: url(:/res/char-fly.png) 3 10 3 10");
    //setStyleSheet("#mainWindow{border-image:transparent;}");
    //setStyleSheet("#mainWindow{border-image: url(:/res/char-hard.jpg);}");
    //setAttribute(Qt::WA_TranslucentBackground);
    this->initSize();
}

void GameWidget::keyPressEvent(QKeyEvent *event) {
    switch (event->key()) {
        case Qt::Key_D:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": D");
            break;
        case Qt::Key_S:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": S");
            break;
        case Qt::Key_A:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": A");
            break;
        case Qt::Key_W:
            keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": W");
            break;
        case Qt::Key_Q:
            close();
            break;
        default:
            break;
    }
}
