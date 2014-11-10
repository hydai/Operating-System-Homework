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
    split2->addWidget(gameView);
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

void GameWidget::initKeyboardInputLabel() {
    // User keyboard input
    this->keyboardInputLabel = new QLabel(KEYBOARDINPUT, this);
    keyboardInputLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_LABEL, QFont::Bold));
    keyboardInputLabel->setObjectName("keyboardInputLabel");
}

void GameWidget::initGameView() {
    this->gameView = new QGraphicsView(this);
    this->scene = new QGraphicsScene(0, 0, WINDOWS_WIDTH_MIN-60, WINDOWS_LENGTH_MIN/2);
    gameView->setScene(scene);
    heroItem = new GameItem();
    goalBanner = new GameGoal();
    scene->addItem(goalBanner);
    goalBanner->setX(-10);
    goalBanner->setY(-130);
    scene->addItem(heroItem);
    heroItem->setX(0);
    heroItem->setY(WINDOWS_LENGTH_MIN/2);
}

void GameWidget::initSize() {
    levelLabel->resize(60, 50);
    levelLCDNumber->resize(100, 40);
    levelSlider->resize(50, 50);
    keyboardInputLabel->resize(200, 50);
    this->resize(WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);
}

void GameWidget::initPthread() {
    pthread_mutex_init(&mutexOfGameStatus, NULL);
    pthread_mutex_init(&mutexOfItemLocation, NULL);
    pthread_create(&pthCheckGameStatus, NULL, this->checkGameStatus, NULL);
    pthread_create(&pthCheckItemLocation, NULL, this->checkItemLocation, NULL);
}
