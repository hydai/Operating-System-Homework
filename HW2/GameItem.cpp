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

void GameItem::moveItem(int x, int y) {
    moveBy(x, y);
}
