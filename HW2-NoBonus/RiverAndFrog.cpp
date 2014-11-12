#include "RiverAndFrog.h"
#include <cstdio>
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <cctype>

GameStage::GameStage() {
    // Init game status
    gameStatus = NORMAL;
    // Get time seed random 
    srand(time(NULL));
    // Init pthread component
    pthread_mutex_init(&mutexGraph, NULL);
    // Init graph
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            if (i == 0 || i == ROW-1)
                graph[i][j] = '|';
            else
                graph[i][j] = ' ';
        }
        graph[i][COL] = '\0';
        woodLength[i] = (rand() % 7) + 7;
        woodLocation[i] = i;
        woodSpeed[i] = (rand() % 1007) % 30 + 3;
    }
    
    // Init frog
    frog = new Frog(ROW, COL/2);

    // Init help args
    ThreadArgs args;
    args.self = this;
    // Create thread
    for (int i = 0; i < ROW; i++) {
        args.tid = &i;
        pthread_create(&woodThreads[i], NULL, &GameStage::pthreadHelper, (void *)&args);
    }
}

void GameStage::exec() {
    // Join thread
    for (int i = 0; i < ROW; i++) {
        pthread_join(woodThreads[i], NULL);
    }

    // Show message
    switch (gameStatus) {
        case WIN_EXIT:
            puts("You Win !!!\nGood Job !!!");
            break;
        case LOSE_EXIT:
            puts("You Lose !!!\n~QAQ~ Try Again !!!");
            break;
        case QUIT_EXIT:
            puts("You Quit the Game !!!\nNever Give Up !!!");
            break;
        default:
            break;
    }

    // Destroy thread
    pthread_mutex_destroy(&mutexGraph);
    pthread_exit(NULL);
}

bool GameStage::isInMap(int x, int y) {
    return x >= 0 && x < COL && y >= 0 && y < ROW;
}

void* GameStage::pthreadHelper(void *in) {
    ThreadArgs *args = static_cast<ThreadArgs *>(in);
    args->self->woodHandler(args->tid);
    return NULL;
}
void* GameStage::woodHandler(void *tid) {
    int currentID = *(int *)tid;
    while (gameStatus == NORMAL) {
        pthread_mutex_lock(&mutexGraph);
        // currentID is odd     <-
        // currentID is even    ->
        if (currentID % 2) {
            woodLocation[currentID] = (woodLocation[currentID] - 1 + COL) % COL;
        } else {
            woodLocation[currentID] = (woodLocation[currentID] + 1) % COL;
        }

        // Clean line
        for (int i = 0; i < COL; i++) {
            graph[currentID][i] = ' ';
        }

        // Draw line
        for (int len = 0, loc = woodLocation[currentID];
                len < woodLength[currentID];
                len++, loc++) {
            graph[currentID][loc%COL] = '=';
        }

        // Draw bound
        for (int i = 0; i < COL; i++) {
            graph[0][i] = graph[ROW-1][i] = '|';
        }
        
        // Get user keyboard input
        if (kbhit()) {
            char keyIn = getchar();
            keyIn = tolower(keyIn);
            switch (keyIn) {
                case 'q':
                    gameStatus = QUIT_EXIT;
                    break;
                case 'w':
                    if (frog->getX() > 0)
                        frog->setX(frog->getX()-1);
                    break;
                case 's':
                    if (frog->getX() < ROW-1)
                        frog->setX(frog->getX()+1);
                    break;
                case 'a':
                    if (frog->getY() > 0)
                        frog->setY(frog->getY()-1);
                    break;
                case 'd':
                    if (frog->getY() < COL-1)
                        frog->setY(frog->getY()+1);
                    break;
                default:
                    // Catch others, do nothing
                    break;
            }
        }
        
        // Judge game status
        if (graph[frog->getX()][frog->getY()] == ' '
          || graph[frog->getX()][frog->getY()] == '\0'
          || !isInMap(frog->getX(), frog->getY())) {
            gameStatus = LOSE_EXIT;
        } else if (frog->getX() == 0) {
            gameStatus = WIN_EXIT;
        }

        // Move frog on the wood
        if (gameStatus == NORMAL) {
            if (frog->getX() == currentID
              && graph[frog->getX()][frog->getY()] == '=') {
                if (frog->getX() % 2) {
                    frog->setY(frog->getY()-1);
                } else {
                    frog->setY(frog->getY()+1);
                }
            }
            graph[frog->getX()][frog->getY()] = '^';
            dumpGraph();
        }
        pthread_mutex_unlock(&mutexGraph);
        usleep(woodSpeed[currentID] * SLEEP_TIME);
    }
    pthread_exit(NULL);
}

void GameStage::dumpGraph() {
    for (int i = 0; i < ROW; i++) {
        puts(graph[i]);
    }
}
