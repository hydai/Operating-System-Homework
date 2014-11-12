#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H
#include <QApplication>
#include <QtGui>
#include "Const.h"
#include "GameItem.h"
#include "GameMeow.h"
#include "GameGoal.h"

class GameWidget : public QWidget
{
    Q_OBJECT
public:
    // ==================================
    // Initialize method
    void init();
    ~GameWidget();
    // ==================================
    // Speed relative method
    void setHeroSpeed(int);
    int getMeowSpeed(int);
    // ==================================
    // items
    GameMeow *meow[MEOW_SIZE];
    GameItem *heroItem;
private:
    // ==================================
    // Initialize method
    void initGameView();
    void initKeyboardInputLabel();
    void initLevel();
    void initLayout();
    void initSize();
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
    // ==================================
    // Game flag
    bool isGameOver;
};
#endif

