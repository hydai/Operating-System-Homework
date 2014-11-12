#ifndef GAMEITEM_H
#define GAMEITEM_H

#include <QtGui>

class GameItem : public QGraphicsItem
{
public:
    bool moveItem(int x, int y);
    void setGameOverFlag(bool & flag);
    void setItemInViewFlag(bool & flag);
    QPointF &getQP();
    double getVX();
    double getVY();

private:
    double vx, vy;
    bool isItemInView;
    bool isGameOver;
    bool isInView(int x, int y);
    bool isInGoalArea();
    QRectF boundingRect() const;
    void paint(
            QPainter *painter,
            const QStyleOptionGraphicsItem *option, 
            QWidget *widget);
    
};
#endif

