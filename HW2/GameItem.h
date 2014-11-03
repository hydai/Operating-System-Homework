#ifndef GAMEITEM_H
#define GAMEITEM_H

#include <QtGui>

class GameItem : public QGraphicsItem
{
public:
    void moveItem(int x, int y);

private:
    QRectF boundingRect() const;
    void paint(
            QPainter *painter,
            const QStyleOptionGraphicsItem *option, 
            QWidget *widget);
    
};
#endif

