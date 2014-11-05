#include "GameGoal.h"
#include "Const.h"

QRectF GameGoal::boundingRect() const {
    return QRectF(0, 0, BANNER_GOAL_WIDTH, BANNER_GOAL_LENGTH);
}

void GameGoal::paint(
                QPainter *painter,
                const QStyleOptionGraphicsItem *option, 
                QWidget *widget) {
    QPixmap image(":/res/goal.png");
    painter->drawPixmap(0, 0, BANNER_GOAL_WIDTH, BANNER_GOAL_LENGTH, image);
}
