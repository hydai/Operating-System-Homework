#ifndef GAMEGOAL_H
#define GAMEGOAL_H

#include <QtGui>

class GameGoal : public QGraphicsItem
{
public:

private:
    QRectF boundingRect() const;
    void paint(
            QPainter *painter,
            const QStyleOptionGraphicsItem *option, 
            QWidget *widget);
    
};
#endif

