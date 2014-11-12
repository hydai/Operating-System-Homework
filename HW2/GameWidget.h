#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H
#include <QApplication>
#include <QtGui>
#include <pthread.h>
#include "GameItem.h"
#include "GameGoal.h"

class GameWidget : public QWidget
{
    Q_OBJECT
public:
    void init();
    ~GameWidget();
private:
    // ==================================
    // Initialize method
    void initGameView();
    void initKeyboardInputLabel();
    void initLevel();
    void initLayout();
    void initSize();
    void initPthread();
    // ==================================
    // Key event
    void keyPressEvent(QKeyEvent *event);
    // ==================================
    // Gui layout
    QSplitter       *split1;
    QSplitter       *split2;
    QVBoxLayout     *layout;
    QWidget         *container;
    QVBoxLayout     *containerLayout;
    // ==================================
    // Gui component
    QLabel          *keyboardInputLabel;
    QLabel          *levelLabel;
    QLCDNumber      *levelLCDNumber;
    QSlider         *levelSlider;
    QGraphicsScene  *scene;
    QGraphicsView   *gameView;
    GameGoal        *goalBanner;
    GameItem *heroItem;
    // ==================================
    // Game flag
    bool isGameOver;
    bool isItemInView;
    // ==================================
    // Game variable
    QPointF sPoint;
    // ==================================
    // Pthread item
    pthread_t       pthCheckGameStatus;
    pthread_t       pthCheckItemLocation;
    pthread_attr_t  attrOfPthread;
    pthread_mutex_t mutexOfGameStatus;
    pthread_mutex_t mutexOfItemLocation;
    // ==================================
    // Pthread method
    static void *cGSHelper(void *);
    void *checkGameStatus(void *);
    static void *cILHelper(void *);
    void *checkItemLocation(void *);
};
#endif

