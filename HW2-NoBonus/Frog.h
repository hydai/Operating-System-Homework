#ifndef FROG_H
#define FROG_H

class Frog {
    public:
        Frog(int, int);
        int getX();
        int getY();
        void setX(int);
        void setY(int);
    private:
        int x;
        int y;
};

#endif
