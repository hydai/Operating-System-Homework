#include "GameWidget.h"
void GameWidget::setInputLabel(QLabel *label) {
    this->inputLabel = label;
}
void GameWidget::setPusheenLabel(QLabel *label) {
    this->pusheenLabel = label;
}
void GameWidget::keyPressEvent(QKeyEvent *event) {
    QString qstring = event->text();
    qstring = qstring.toLower();
    if (qstring.left(1) == "w") {
        inputLabel->setText("Keyboard Input: " + qstring);
    } else if (qstring.left(1) == "a") {
        inputLabel->setText("Keyboard Input: " + qstring);
    } else if (qstring.left(1) == "s") {
        inputLabel->setText("Keyboard Input: " + qstring);
    } else if (qstring.left(1) == "d") {
        inputLabel->setText("Keyboard Input: " + qstring);
    } else if (qstring.left(1) == "q") {
        inputLabel->setText("PUSHEEN" + qstring);
        close();
    }
}
