// A libnx program built the ordinary way: starquake's second homebrew fixture.
//
// Unlike hello.c it keeps libnx's own start-up: the weak __appInit in
// libnx/nx/source/runtime/init.c @ b156b477 (sm:, set:sys for the firmware version, applet,
// hid, time, fs and the SD card), and, because consoleInit pulls in the default window
// (libnx/nx/source/display/default_window.c), __nx_win_init, which opens vi and a layer before
// main.
//
// main prints one line on the console (consoleInit opens nvdrv for the framebuffer) and
// returns. libnx then runs __appExit, which, for a program started without a homebrew loader,
// ends the process through appletOE's ISelfController::Exit
// (libnx/nx/source/services/applet.c, _appletCleanup).
//
// The console draws the line into a framebuffer, so the line itself is in no trace: the
// reference trace shows only the calls that put it on screen.

#include <stdio.h>

#include <switch.h>

static const char LINE[] = "starquake startup: hello";

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    consoleInit(NULL);
    printf("%s\n", LINE);
    consoleUpdate(NULL);
    consoleExit(NULL);
    return 0;
}
