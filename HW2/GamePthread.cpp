#include "GameWidget.h"
#include "Const.h"
#include <pthread.h>

void* GameWidget::cGSHelper(void *pointer) {
    GameWidget *pt = static_cast<GameWidget *> (pointer);
    pt->checkGameStatus(pointer);
    return NULL;
}
void* GameWidget::checkGameStatus(void *doNothing) {
    while(1) {
        usleep(10000);
        pthread_mutex_lock (&mutexOfGameStatus);
        QPointF position = heroItem->getQP();
        if (position.y() < 0)
            isGameOverPt = true;
        else
            isGameOverPt = false;
        pthread_mutex_unlock (&mutexOfGameStatus);
    }
    pthread_exit(NULL);
}

void* GameWidget::cILHelper(void *pointer) {
    GameWidget *pt = static_cast<GameWidget *>(pointer);
    pt->checkItemLocation(pointer);
    return NULL;
}
void* GameWidget::checkItemLocation(void *doNothing) {
    while(1) {
        usleep(10000);
        pthread_mutex_lock (&mutexOfItemLocation);
        QPointF position = heroItem->getQP();
        double x = heroItem->getVX();
        double y = heroItem->getVY();
        if (position.x()+x < -40
          || position.x()+x > WINDOWS_WIDTH_MIN-120
          || position.y()+y < -120
          || position.y()+y > WINDOWS_LENGTH_MIN/2) {
            isItemInViewPt = false;
        } else {
            isItemInViewPt = true;
        }
        pthread_mutex_unlock (&mutexOfItemLocation);
    }
    pthread_exit(NULL);
}
