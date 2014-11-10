#include <QApplication>
#include <QtGui>
#include "Const.h"
#include "GameWidget.h"

#define BUTTON_LINE

int main(int argc, char *argv[])
{
    // Create application
    QApplication app(argc, argv);

    // Application field
    GameWidget parent;
    // Initialize Application
    parent.init();
    // Display application
    parent.show();
    // Run application
    app.exec();

    return 0;
}
