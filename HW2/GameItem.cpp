#include "GameItem.h"
QRectF GameItem::boundingRect() const {
    return QRectF(0, 0, 200, 200);
}

void GameItem::paint(
                QPainter *painter,
                const QStyleOptionGraphicsItem *option, 
                QWidget *widget) {
    QPixmap image(":/res/char-small.jpg");
    painter->drawPixmap(0, 0, 200, 200, image);
}

void GameItem::moveItem(int x, int y) {
    moveBy(x, y);
}
