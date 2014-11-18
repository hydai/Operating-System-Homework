#include "GameMeow.h"
#include "Const.h"

QRectF GameMeow::boundingRect() const {
    return QRectF(0, 0, PUSHEEN_WIDTH, PUSHEEN_LENGTH);
}

void GameMeow::paint(
                QPainter *painter,
                const QStyleOptionGraphicsItem *option, 
                QWidget *widget) {
    QPixmap image(":/res/pusheen.png");
    painter->drawPixmap(0, 0, PUSHEEN_WIDTH, PUSHEEN_LENGTH, image);
}

bool GameMeow::moveItem(int x, int y) {
    moveBy(x, y);
    return false;
}
