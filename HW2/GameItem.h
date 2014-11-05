#ifndef GAMEITEM_H
#define GAMEITEM_H

#include <QtGui>

class GameItem : public QGraphicsItem
{
public:
    bool moveItem(int x, int y);

private:
    bool isInView(int x, int y);
    bool isInGoalArea();
    QRectF boundingRect() const;
    void paint(
            QPainter *painter,
            const QStyleOptionGraphicsItem *option, 
            QWidget *widget);
    
};
#endif

