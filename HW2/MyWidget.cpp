#include "MyWidget.h"
void MyWidget::setInputLabel(QLabel *label) {
    this->inputLabel = label;
}
void MyWidget::keyPressEvent(QKeyEvent *event) {
    QString qstring = event->text();
    qstring = qstring.toLower();
    if (qstring.left(1) == "w"
      ||qstring.left(1) == "a"
      ||qstring.left(1) == "s"
      ||qstring.left(1) == "d") {
        inputLabel->setText(qstring);
    } else if (qstring.left(1) == "q") {
        inputLabel->setText("PUSHEEN" + qstring);
        close();
    }
}
