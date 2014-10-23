#include <QApplication>
#include <QPushButton>
#include <QFont>
#include <QWidget>
#include <QSlider>
#include <QLCDNumber>
#include <pthread.h>
#include <unistd.h>
void *countUP (void *tid) {
    QLCDNumber *oao = (QLCDNumber *)tid;
    for (int i = 0; i < 60; i++) {
        oao->display(i);
        sleep(1);
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    // Create application
    QApplication app(argc, argv);

    // Application field
    QWidget *parent = new QWidget;
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

    // Level LCDNumber
    QLCDNumber *levelLCDNumber = new QLCDNumber(parent);
    levelLCDNumber->setGeometry(70, 60, 100, 30);

    // time LCDNumber
    QLCDNumber *timeLCDNumber = new QLCDNumber(parent);
    timeLCDNumber->setGeometry(70, 110, 100, 30);
    timeLCDNumber->display(100);

    pthread_t countThread;
    pthread_create(&countThread, 0, countUP, (void *)timeLCDNumber);

    // Level slider bar
    QSlider *levelSlider = new QSlider(Qt::Horizontal, parent);
    levelSlider->setRange(0, 99);
    levelSlider->setValue(0);
    levelSlider->setGeometry(70, 150, 100, 30);
    QObject::connect(levelSlider, SIGNAL(valueChanged(int)), levelLCDNumber, SLOT(display(int)));

    // Display application
    parent->show();

    // Run application
    app.exec();

    return 0;
}
