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
    void initBackground();
    void initKeyboardInputLabel();
    void initCharacterIcon();
    void initLevel();
    void initLayout();
    void keyPressEvent(QKeyEvent *event);
    void initSize();
    QSplitter *split1;
    QSplitter *split2;
    QVBoxLayout *layout;
    QWidget *container;
    QVBoxLayout *containerLayout;
    QLabel *backgroundImage;
    QLabel *keyboardInputLabel;
    QLabel *characterIcon;
    QLabel *levelLabel;
    QLCDNumber *levelLCDNumber;
    QSlider *levelSlider;
};
#endif

