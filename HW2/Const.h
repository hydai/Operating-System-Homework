#ifndef CONST_H
#define CONST_H
#include <QString>
// For pthread arguments
struct ThreadArgs {
    void *(args[10]);
};

// Game exit code
const int NORMAL_EXIT   = 0;
const int WIN_EXIT      = NORMAL_EXIT + 1;
const int LOSE_EXIT     = WIN_EXIT + 1;

// QT GUI const setting
// QSS setting
const char          STYLE[] = "                                         \
                        QWidget#mainWindow                              \
                        {                                               \
                            background-color: rgba(200,200,200,125);    \
                        }                                               \
                        QLabel, QLCDNumber, QSlider                     \
                        {                                               \
                            background-color: rgba(100,100,100,155);    \
                            border-radius: 25;                          \
                        }                                               \
                        QLabel#levelLabel, #keyboardInputLabel          \
                        {                                               \
                            margin: 0 0 0 10;                           \
                        }                                               \
                        QGraphicsView                                   \
                        {                                               \
                            background-color: rgba(30,30,30,180);       \
                            border-radius: 25;                          \
                        }                                               \
                        GameGoal                                        \
                        {                                               \
                            border-radius: 25;                          \
                        }                                               \
                        ";
// Windows setting
const char          WINDOWS_TITLE[]         = "River & Frog";
const int           WINDOWS_WIDTH_MIN       = 1200;
const int           WINDOWS_LENGTH_MIN      = 800;
const int           WINDOWS_WIDTH_MAX       = 1920;
const int           WINDOWS_LENGTH_MAX      = 1080;

// Font setting
const char          FONT_TYPE[]             = "Menlo";
const int           FONT_SIZE_DEFAULT       = 24;
const int           FONT_SIZE_LABEL         = 36;

// IMAGE RESOURCE PATH
const char          BACKGROUND_PATH[]       = ":/res/bg.png";

// IMAGE size setting
const int           PAINTER_WIDTH           = 100;
const int           BANNER_GOAL_WIDTH       = 1200;
const int           BANNER_GOAL_LENGTH      = 150;

// Const string
const char          KEYBOARDINPUT[]         = "Keyboard Input";

// Game Const
const int MEOW_SIZE = 4;
#endif

