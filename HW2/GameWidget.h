#ifndef GAMEWIDGET_H
#define GAMEWIDGET_H
#include <QApplication>
#include <QtGui>
class GameWidget : public QWidget
{
    Q_OBJECT
public:
    void setInputLabel(QLabel *label);
    void setPusheenLabel(QLabel *label);
    void keyPressEvent(QKeyEvent *event);
private:
    QLabel *inputLabel;
    QLabel *pusheenLabel;
};
#endif

