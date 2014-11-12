#include <pthread.h>
#include <unistd.h>
#include "Frog.h"

extern int kbhit(void);

// Game graph
const int ROW = 20;
const int COL = 40;

// Game status
const int NORMAL    = 0;
const int WIN_EXIT  = 1;
const int LOSE_EXIT = 2;
const int QUIT_EXIT = 3;

// Game animation
const int SLEEP_TIME = 4000;

class GameStage {
    public:
        GameStage();
        void exec();
    private:
        // ====================
        // Game method
        // Check if the frog is in map or not
        bool isInMap(int, int);
        // For pthread, to handle all woods
        void *woodHandler(void *);
        // For pthread helper
        static void* pthreadHelper(void *);
        // Print graph
        void dumpGraph();
        // ====================
        // Game component
        Frog    *frog;
        char    graph[ROW][COL+1];
        int     woodLength[ROW];
        int     woodLocation[ROW];
        int     woodSpeed[ROW];
        int     gameStatus;
        // ====================
        // pthread component
        pthread_t       woodThreads[ROW];
        pthread_mutex_t mutexGraph;
};

struct ThreadArgs {
    GameStage *self;
    void *tid;
};
