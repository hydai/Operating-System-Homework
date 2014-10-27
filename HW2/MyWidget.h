#ifndef MYWIDGET_H
#define MYWIDGET_H
#include <QApplication>
#include <QPushButton>
#include <QFont>
#include <QWidget>
#include <QSlider>
#include <QLCDNumber>
#include <QLabel>
#include <QKeyEvent>
class MyWidget : public QWidget
{
    Q_OBJECT
public:
    void setInputLabel(QLabel *label);
    void keyPressEvent(QKeyEvent *event);
private:
    QLabel *inputLabel;
};
#endif

