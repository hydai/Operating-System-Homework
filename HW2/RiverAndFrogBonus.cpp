#include <QApplication>
#include <QPushButton>
#include <QFont>
#include <QWidget>
#include <QSlider>
#include <QLCDNumber>
#include <QLabel>
#include <QKeyEvent>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include "MyWidget.h"
struct ThreadArgs {
    void *(args[10]);
};

void *countUP (void *tin) {
    QLCDNumber *time = (QLCDNumber *)tin;
    for (int i = 0; ; i++) {
        time->display(i);
        sleep(1);
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    // Create application
    QApplication app(argc, argv);

    // Application field
    //QWidget *parent = new QWidget;
    MyWidget *parent = new MyWidget();
    parent->setWindowTitle("River & Frog");
    parent->setMinimumSize(800, 600);
    parent->setMaximumSize(800, 600);

    // Start button
    QPushButton *startButton = new QPushButton("Start", parent);
    startButton->setFont(QFont("Menlo", 18, QFont::Bold));
    startButton->setGeometry(0, 0, 100, 50);
    QObject::connect(startButton, SIGNAL(clicked()), &app, SLOT(quit()));

    // Close button
    QPushButton *closeButton = new QPushButton("Close", parent);
    closeButton->setFont(QFont("Menlo", 18, QFont::Bold));
    closeButton->setGeometry(100, 0, 100, 50);
    QObject::connect(closeButton, SIGNAL(clicked()), &app, SLOT(quit()));

    // Level label
    QLabel *levelLabel = new QLabel("Level: ", parent);
    levelLabel->setFont(QFont("Menlo", 24, QFont::Bold));
    levelLabel->setGeometry(5, 530, 80, 50);

    // Level LCDNumber
    QLCDNumber *levelLCDNumber = new QLCDNumber(parent);
    levelLCDNumber->setSegmentStyle(QLCDNumber::Filled);
    levelLCDNumber->setFont(QFont("Menlo", 18, QFont::Bold));
    levelLCDNumber->setGeometry(90, 530, 100, 40);
    levelLCDNumber->display(1);

#ifdef SHOW_TIME
    // time label
    QLabel *timeLabel = new QLabel("Time: ", parent);
    timeLabel->setFont(QFont("Menlo", 18, QFont::Bold));
    timeLabel->setGeometry(200, 550, 80, 50);

    // Time LCDNumber
    QLCDNumber *timeLCDNumber = new QLCDNumber(parent);
    timeLCDNumber->setFont(QFont("Menlo", 18, QFont::Bold));
    timeLCDNumber->setGeometry(280, 550, 100, 40);
    timeLCDNumber->display(0);

    pthread_t countThread;
    pthread_create(&countThread, 0, countUP, (void *)timeLCDNumber);
#endif

    // Get user input
    QLabel *inputLabel = new QLabel("Keyboard Input", parent);
    inputLabel->setFont(QFont("Menlo", 18, QFont::Bold));
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
