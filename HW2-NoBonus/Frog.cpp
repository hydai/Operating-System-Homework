#include "Frog.h"

Frog::Frog(int _x, int _y) {
    this->x = _x;
    this->y = _y;
}

int Frog::getX() {
    return this->x;
}

int Frog::getY() {
    return this->y;
}

void Frog::setX(int _x) {
    this->x = _x;
}

void Frog::setY(int _y) {
    this->y = _y;
}
