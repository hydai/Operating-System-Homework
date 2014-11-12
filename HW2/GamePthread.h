#ifndef GAMEPTHREAD_H
#define GAMEPTHREAD_H
#endif

#include <pthread.h>
#include <QtGui>
#include "Const.h"
#include "GameWidget.h"

void *statusHandler(void *inargs) {
    ThreadArgs *args = (ThreadArgs *)inargs;
    GameWidget *gameView = (GameWidget *)(args->args[0]);
    int *exitCode = (int *)(args->args[1]);
    bool *isGameOver = (bool *)(args->args[2]);

	while(isGameOver == false) {
	    int position = gameView->isInView(
	                        gameView->heroItem->getX(),
	                        gameView->heroItem->getY());
		if(position != 0) {
		    gameView->setHeroSpeed(gameView->meow[position]->getSpeed());
		} else {	
			if(gameView->heroItem->getY() > WINDOWS_LENGTH_MIN/2 - 120) {
			    gameView->setHeroSpeed(0);
			}
			else if(gameView->heroItem->isInGoalArea()){
				*isGameOver = true;
				*exitCode = WIN_EXIT;
				gameView->setHeroSpeed(0);
                QMessageBox::information(gameView,
                  "You Win !!!",
                  "Good job !!!",
                  QMessageBox::Yes,
                  QMessageBox::Yes);
			}
			else{
				*isGameOver = true;
				*exitCode = LOSE_EXIT;
				gameView->setHeroSpeed(0);
                QMessageBox::information(gameView,
                  "You Lose !!!",
                  "QAQ Try again !!!",
                  QMessageBox::Yes,
                  QMessageBox::Yes);
			}
		}
	}
	
	gameView->close();
	pthread_exit(NULL);
}
