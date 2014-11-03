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
    //split2->addWidget(gameView);
    layout->addWidget(split2);
    this->setLayout(layout);
}
void GameWidget::initLevel() {
    // Level label
    this->levelLabel = new QLabel("Level: ", this);
    levelLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_LABEL, QFont::Bold));
    levelLabel->setObjectName("levelLabel");

    // Level LCDNumber
    this->levelLCDNumber = new QLCDNumber(this);
    levelLCDNumber->setSegmentStyle(QLCDNumber::Filled);
    levelLCDNumber->display(1);
    levelLCDNumber->setObjectName("levelLCDNumber");

    // Level slider bar
    this->levelSlider = new QSlider(Qt::Horizontal, this);
    levelSlider->setRange(1, 10);
    levelSlider->setValue(1);
    levelSlider->setObjectName("levelSlider");
    QObject::connect(levelSlider, SIGNAL(valueChanged(int)), levelLCDNumber, SLOT(display(int)));
}
void GameWidget::initBackground() {
    // Image label
    this->backgroundImage = new QLabel("", this);
}
void GameWidget::initKeyboardInputLabel() {
    // User keyboard input
    this->keyboardInputLabel = new QLabel(KEYBOARDINPUT, this);
    keyboardInputLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_LABEL, QFont::Bold));
    keyboardInputLabel->setObjectName("keyboardInputLabel");
}
void GameWidget::initGameView() {
    this->scene = new QGraphicsScene;
    this->gameView = new QGraphicsView(scene);
    scene->addRect(QRectF(0, 0, 100, 100));
}
void GameWidget::initCharacterIcon() {
    // Character icon
    /*
    this->characterIcon = new QLabel("", this);
    characterIcon->setPixmap(QPixmap(BACKGROUND_PATH));
    characterIcon->setGeometry(100, 100, 50, 50);
    */
}
void GameWidget::initSize() {
    levelLabel->resize(60, 50);
    levelLCDNumber->resize(100, 40);
    levelSlider->resize(50, 50);
    backgroundImage->resize(WINDOWS_WIDTH_MIN/2, WINDOWS_LENGTH_MIN/2);
    keyboardInputLabel->resize(200, 50);
    this->resize(WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);
}
