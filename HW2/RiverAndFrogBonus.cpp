#include <QApplication>
#include <QtGui>
#include <pthread.h>
#include "Const.h"
#include "GameWidget.h"

#define SHOW_TIME
#define BUTTON_LINE


#ifdef SHOW_TIME
void *countUP (void *tin) {
    QLCDNumber *time = (QLCDNumber *)tin;
    for (int i = 0; ; i++) {
        time->setGeometry(200, 550 - 10*i, 80, 50);
        time->display(i);
        usleep(100000);
    }
    pthread_exit(NULL);
}
#endif

int main(int argc, char *argv[])
{
    // Create application
    QApplication app(argc, argv);

    // Application field
    GameWidget *parent = new GameWidget();
    parent->setWindowTitle(WINDOWS_TITLE);
    parent->setMaximumSize(WINDOWS_WIDTH_MAX, WINDOWS_LENGTH_MAX);
    parent->setMinimumSize(WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);

    // image label
    QLabel *imageLabel = new QLabel("", parent);
    imageLabel->setGeometry(0, 0, WINDOWS_WIDTH_MIN, WINDOWS_LENGTH_MIN);
    imageLabel->setPixmap(QPixmap(BACKBROUNG_PATH));

#ifdef BUTTON_LINE
    // Start button
    QPushButton *startButton = new QPushButton("Start", parent);
    startButton->setFont(QFont(FONT_TYPE, FONT_SIZE_DEFAULT, QFont::Bold));
    startButton->setGeometry(0, 0, 100, 50);
    QObject::connect(startButton, SIGNAL(clicked()), &app, SLOT(quit()));

    // Close button
    QPushButton *closeButton = new QPushButton("Close", parent);
    closeButton->setFont(QFont(FONT_TYPE, FONT_SIZE_DEFAULT, QFont::Bold));
    closeButton->setGeometry(100, 0, 100, 50);
    QObject::connect(closeButton, SIGNAL(clicked()), &app, SLOT(quit()));
#endif
    // Level label
    QLabel *levelLabel = new QLabel("Level: ", parent);
    levelLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_LABEL, QFont::Bold));
    levelLabel->setGeometry(5, 530, 80, 50);

    // Level LCDNumber
    QLCDNumber *levelLCDNumber = new QLCDNumber(parent);
    levelLCDNumber->setSegmentStyle(QLCDNumber::Filled);
    levelLCDNumber->setGeometry(90, 530, 100, 40);
    levelLCDNumber->display(1);

#ifdef SHOW_TIME
    // time label
    QLabel *timeLabel = new QLabel("Time: ", parent);
    timeLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_LABEL, QFont::Bold));
    timeLabel->setGeometry(200, 550, 80, 50);

    // Time LCDNumber
    QLCDNumber *timeLCDNumber = new QLCDNumber(parent);
    timeLCDNumber->setGeometry(280, 550, 100, 40);
    timeLCDNumber->display(0);

    pthread_t countThread;
    pthread_create(&countThread, 0, countUP, (void *)timeLCDNumber);
#endif

    // Get user input
    QLabel *inputLabel = new QLabel(KEYBOARDINPUT, parent);
    inputLabel->setFont(QFont(FONT_TYPE, FONT_SIZE_DEFAULT, QFont::Bold));
    inputLabel->setGeometry(600, 0, 200, 50);
    parent->setInputLabel(inputLabel);

    // Level slider bar
    QSlider *levelSlider = new QSlider(Qt::Horizontal, parent);
    levelSlider->setRange(1, 10);
    levelSlider->setValue(1);
    levelSlider->setGeometry(5, 570, 180, 30);
    QObject::connect(levelSlider, SIGNAL(valueChanged(int)), levelLCDNumber, SLOT(display(int)));

    // Display application
    parent->show();

    // Run application
    app.exec();
    return 0;
}
