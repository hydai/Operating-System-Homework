#include "GameMeow.h"
#include "Const.h"

QRectF GameMeow::boundingRect() const {
    return QRectF(0, 0, PAINTER_WIDTH, PAINTER_WIDTH);
}

void GameMeow::paint(
                QPainter *painter,
                const QStyleOptionGraphicsItem *option, 
                QWidget *widget) {
    QPixmap image(":/res/char-small.png");
    painter->drawPixmap(0, 0, PAINTER_WIDTH, PAINTER_WIDTH, image);
}

bool GameMeow::moveItem(int x, int y) {
    if (isInView(x, y))
        moveBy(x, y);
    if (isInGoalArea()) {
        return true;
    }
    return false;
}

bool GameMeow::isInGoalArea() {
    QPointF position = this->scenePos();
    if (position.y() < 0)
        return true;
    return false;
}
bool GameMeow::isInView(int x, int y) {
    QPointF position = this->scenePos();
    if (position.x()+x < -40
        || position.x()+x > WINDOWS_WIDTH_MIN-120
        || position.y()+y < -120
        || position.y()+y > WINDOWS_LENGTH_MIN/2) {
        return false;
    }

    return true;
}
