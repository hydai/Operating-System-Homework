#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H
#include <QApplication>
#include <QtGui>
#include "GameItem.h"
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
    GameItem        *heroItem;
};
#endif

