#include "GameItem.h"
#include "Const.h"

QRectF GameItem::boundingRect() const {
    return QRectF(0, 0, PAINTER_WIDTH, PAINTER_WIDTH);
}

void GameItem::paint(
                QPainter *painter,
                const QStyleOptionGraphicsItem *option, 
                QWidget *widget) {
    QPixmap image(":/res/char-small.png");
    painter->drawPixmap(0, 0, PAINTER_WIDTH, PAINTER_WIDTH, image);
}

bool GameItem::moveItem(int x, int y) {
    vx = x;
    vy = y;
    if (isInView(x, y))
        moveBy(x, y);
    if (isInGoalArea()) {
        return true;
    }
    return false;
}

bool GameItem::isInGoalArea() {
    QPointF position = this->scenePos();
    if (position.y() < 0)
        return true;
    return false;
}
bool GameItem::isInView(int x, int y) {
    QPointF position = this->scenePos();
    if (position.x()+x < -40
        || position.x()+x > WINDOWS_WIDTH_MIN-120
        || position.y()+y < -120
        || position.y()+y > WINDOWS_LENGTH_MIN/2) {
        return false;
    }

    return true;
}
QPointF &GameItem::getQP() {
    QPointF pos = this->scenePos();
    return pos;
}
void GameItem::setGameOverFlag(bool &flag) {
    this->isGameOver = flag;
}
void GameItem::setItemInViewFlag(bool &flag) {
    this->isItemInView = flag;
}
double GameItem::getVX() {
    return vx;
}
double GameItem::getVY() {
    return vy;
}
