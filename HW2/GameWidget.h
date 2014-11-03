#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H
#include <QApplication>
#include <QtGui>
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
    void initCharacterIcon();
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
    QLabel          *characterIcon;
    QLabel          *levelLabel;
    QLCDNumber      *levelLCDNumber;
    QSlider         *levelSlider;
    QGraphicsScene  *scene;
    QGraphicsView   *gameView;
};
#endif

