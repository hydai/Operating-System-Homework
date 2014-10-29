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
    this->initSize();
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
    this->levelSlider = new QSlider(this);
    levelSlider->setRange(1, 10);
    levelSlider->setValue(1);
    QObject::connect(levelSlider, SIGNAL(valueChanged(int)), levelLCDNumber, SLOT(display(int)));
}
void GameWidget::initBackground() {
    // Image label
    this->backgroundImage = new QLabel("", this);
    backgroundImage->setPixmap(QPixmap(BACKBROUNG_PATH));
}
void GameWidget::initKeyboardInputLabel() {
    // User keyboard input
    this->keyboardInputLabel = new QLabel(KEYBOARDINPUT, this);
    keyboardInputLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_DEFAULT, QFont::Bold));
}
void GameWidget::initCharacterIcon() {
    // Character icon
    this->characterIcon = new QLabel("", this);
    characterIcon->setPixmap(QPixmap(BACKBROUNG_PATH));
}
void GameWidget::initSize() {
    levelLabel->resize(80, 50);
    levelLCDNumber->resize(100, 40);
    levelSlider->resize(50, 50);
    backgroundImage->resize(WINDOWS_WIDTH_MIN/2, WINDOWS_LENGTH_MIN/2);
    keyboardInputLabel->resize(200, 50);
    this->resize(WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);

}
void GameWidget::keyPressEvent(QKeyEvent *event) {
    QString qstring = event->text();
    qstring = qstring.toLower();
    if (qstring.left(1) == "w") {
        keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": " + qstring);
    } else if (qstring.left(1) == "a") {
        keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": " + qstring);
    } else if (qstring.left(1) == "s") {
        keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": " + qstring);
    } else if (qstring.left(1) == "d") {
        keyboardInputLabel->setText(tr(KEYBOARDINPUT) + ": " + qstring);
    } else if (qstring.left(1) == "q") {
        keyboardInputLabel->setText(tr("PUSHEEN") + qstring);
        close();
    }
}
