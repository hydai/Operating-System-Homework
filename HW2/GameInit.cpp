#include "GameWidget.h"
#include "Const.h"
#include <QtGui>

void GameWidget::initLayout() {
    layout = new QVBoxLayout;
    split1 = new QSplitter;
    split2 = new QSplitter;
    container = new QWidget;
    containerLayout = new QVBoxLayout;
    split1->addWidget(levelLabel);
    split1->addWidget(levelLCDNumber);
    split1->addWidget(levelSlider);
    split1->addWidget(keyboardInputLabel);
    containerLayout->addWidget(split1);
    container->setLayout(containerLayout);
    split2->setOrientation(Qt::Vertical);
    split2->addWidget(container);
    split2->addWidget(backgroundImage);
    layout->addWidget(split2);
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
    /*
    backgroundImage->setPixmap(QPixmap(BACKGROUND_PATH));
    */
}
void GameWidget::initKeyboardInputLabel() {
    // User keyboard input
    this->keyboardInputLabel = new QLabel(KEYBOARDINPUT, this);
    keyboardInputLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_LABEL, QFont::Bold));
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
