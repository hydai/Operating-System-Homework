#ifndef GAMEMEOW_H
#define GAMEMEOW_H

#include <QtGui>

class GameMeow : public QGraphicsItem
{
public:
    bool moveItem(int x, int y);

private:
    int id;
    QRectF boundingRect() const;
    void paint(
            QPainter *painter,
            const QStyleOptionGraphicsItem *option, 
            QWidget *widget);
    
};
#endif
