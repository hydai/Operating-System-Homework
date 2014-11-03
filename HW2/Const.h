#ifndef CONST_H
#define CONST_H
#include <QString>
// For pthread arguments
struct ThreadArgs {
    void *(args[10]);
};

// QT GUI const setting
// Windows setting
const char          WINDOWS_TITLE[]         = "River & Frog";
const int           WINDOWS_WIDTH_MIN       = 1440;
const int           WINDOWS_LENGTH_MIN      = 900;
const int           WINDOWS_WIDTH_MAX       = 1920;
const int           WINDOWS_LENGTH_MAX      = 1080;
// Font setting
const char          FONT_TYPE[]             = "Menlo";
const int           FONT_SIZE_DEFAULT       = 24;
const int           FONT_SIZE_LABEL         = 36;

// IMAGE RESOURCE PATH
const char          BACKGROUND_PATH[]       = ":/res/bg.png";

// Const string
const char          KEYBOARDINPUT[]         = "Keyboard Input";
#endif

