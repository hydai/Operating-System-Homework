#include "GameWidget.h"
#include "Const.h"

void GameWidget::setupLayout() {
}
void GameWidget::initLayout() {
    layout = new QGridLayout;
    layout->addWidget(levelLabel,           0, 0);
    layout->addWidget(levelLCDNumber,       0, 1);
    layout->addWidget(levelSlider,          0, 2);
    layout->addWidget(NULL,                 0, 3);
    layout->addWidget(keyboardInputLabel,   0, 4);
    layout->addWidget(backgroundImage,      1, 0, 4, 10);
    this->setLayout(layout);
}
void GameWidget::initLevel() {
    // Level label
    this->levelLabel = new QLabel("Level: ", this);
    levelLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_LABEL, QFont::Bold));

    // Level LCDNumber
    this->levelLCDNumber = new QLCDNumber(this);
    levelLCDNumber->setSegmentStyle(QLCDNumber::Filled);
    levelLCDNumber->display(1);

    // Level slider bar
    this->levelSlider = new QSlider(Qt::Horizontal, this);
    levelSlider->setRange(1, 10);
    levelSlider->setValue(1);
    QObject::connect(levelSlider, SIGNAL(valueChanged(int)), levelLCDNumber, SLOT(display(int)));
}
void GameWidget::initBackground() {
    // Image label
    this->backgroundImage = new QLabel("", this);
    backgroundImage->setPixmap(QPixmap(BACKGROUND_PATH));
}
void GameWidget::initKeyboardInputLabel() {
    // User keyboard input
    this->keyboardInputLabel = new QLabel(KEYBOARDINPUT, this);
    keyboardInputLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_DEFAULT, QFont::Bold));
}
void GameWidget::initCharacterIcon() {
    // Character icon
    /*
    this->characterIcon = new QLabel("", this);
    characterIcon->setPixmap(QPixmap(BACKGROUND_PATH));
    */
}
void GameWidget::initSize() {
    levelLabel->resize(80, 50);
    levelLCDNumber->resize(100, 40);
    levelSlider->resize(50, 50);
    backgroundImage->resize(WINDOWS_WIDTH_MIN/2, WINDOWS_LENGTH_MIN/2);
    keyboardInputLabel->resize(200, 50);
    this->resize(WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);
}
