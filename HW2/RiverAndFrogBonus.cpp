#include <QApplication>
#include <QtGui>
#include <pthread.h>
#include "Const.h"
#include "GamePthread.h"
#include "GameWidget.h"
#include "GameItem.h"
#include "GameMeow.h"

int main(int argc, char *argv[])
{
    int exitCode = NORMAL_EXIT;
    bool isGameOver = false;
    pthread_t gameStatusHandler;
    ThreadArgs args;

    // Create application
    QApplication app(argc, argv);

    // Application field
    GameWidget *parent = new GameWidget();
    args.args[0] = parent;
    args.args[1] = &exitCode;
    args.args[2] = &isGameOver;
    pthread_create(&gameStatusHandler, NULL, statusHandler, (void *)&args);

    // Initialize Application
    parent->init();
    // Display application
    parent->show();
    // Run application
    app.exec();

    return 0;
}
