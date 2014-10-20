#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#define NUM_THREADS 3
#define SLEEP_PERIOD 200
int currentNumber = 2;
int lastestFoundPrimeNumber = 2;
int TCOUNT, COUNT_LIMIT;
pthread_mutex_t mutexOfP;
pthread_cond_t countReachLimit;

void *count(void *tid) {
    int threadId = (int)tid + 1;
    if (threadId == 1) {
        // Thread 1
        printf("Starting watch_count(): thread 1\n");
        usleep(SLEEP_PERIOD*2);
        /* V-------Citical Section------- */
        pthread_mutex_lock (&mutexOfP);
        printf("watch_count(): thread 1 p= %d. Going into wait...\n", currentNumber);
        pthread_mutex_unlock (&mutexOfP);
        /* ^-------Citical Section------- */
        int isFinished = 0;
        /* V-------Citical Section------- */
        pthread_mutex_lock (&mutexOfP);
        pthread_cond_wait(&countReachLimit, &mutexOfP);
        currentNumber--;
        printf("watch_count(): thread 1 Condition signal received. p= %d\n", currentNumber);
        printf("watch_count(): thread 1 Updating the value of p...\n");
        printf("the lastest prime found before p = %d\n", lastestFoundPrimeNumber);
        int tmp = currentNumber;
        currentNumber += lastestFoundPrimeNumber;
        lastestFoundPrimeNumber = tmp;
        printf("watch_count(): thread 1 count p now = %d.\n", currentNumber);
        printf("watch_count(): thread 1 Unlocking mutex.\n");
        pthread_mutex_unlock (&mutexOfP);
        /* ^-------Citical Section------- */
    } else {
        // Thread 2, 3
        int localOldPrime = 2;
        int isFinished = 0, isSentSignal = 0;
        while (1) {
            usleep(SLEEP_PERIOD);
            /* V-------Citical Section------- */
            pthread_mutex_lock (&mutexOfP);
            printf("prime_count(): thread %d, p = %d\n", threadId, currentNumber);
            if (currentNumber >= TCOUNT) {
                isFinished = 1;
            } else {
                int testNum, isPrime = 1;
                for (testNum = 2; testNum < currentNumber; testNum++) {
                    if (currentNumber % testNum == 0) isPrime = 0;
                }
                if (isPrime) {
                    lastestFoundPrimeNumber = localOldPrime;
                    localOldPrime = currentNumber;
                    printf("prime_count(): thread %d, find prime = %d !\n", threadId, currentNumber);
                    if (currentNumber >= COUNT_LIMIT && isSentSignal == 0) {
                        printf("prime_count(): thread %d, prime = %d prime reached.\n", threadId, currentNumber);
                        isSentSignal = 1;
                        printf("Just sent signal.\n");
                        pthread_cond_signal(&countReachLimit);
                    }
                }
            }
            if (isFinished != 1) {
                currentNumber += 1;
            }
            pthread_mutex_unlock (&mutexOfP);
            /* ^-------Citical Section------- */
            if (isFinished) {
                break;
            }
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Do not forget these two args TCOUNT COUNT_LIMIT\n");
        exit(-1);
    }
    TCOUNT = atoi(argv[1]);
    COUNT_LIMIT = atoi(argv[2]);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t threads[NUM_THREADS];
    pthread_mutex_init(&mutexOfP, NULL);
    pthread_cond_init (&countReachLimit, NULL);

    int rc, threadId;
    for (threadId = 0; threadId < NUM_THREADS; threadId++) {
        rc = pthread_create(&threads[threadId], &attr, count, (void *)threadId);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    pthread_attr_destroy(&attr);
    for (threadId = 0; threadId < NUM_THREADS; threadId++) {
        int status;
        pthread_join(threads[threadId], (void **)&status);
    }
    printf("Main(): Waited and joined with 3 threads. Final value of count = %d. Done.!\n", currentNumber);
    pthread_exit(NULL);
    return 0;
}
