// The smallest useful libnx program: starquake's milestone 1 homebrew fixture.
//
// It writes two known lines with svcOutputDebugString, asks set:sys for the firmware version,
// writes a line saying whether that worked, and returns from main (libnx then calls
// svcExitProcess). Start-up opens only sm: (docs/specs/services.md, "Milestone 1 services");
// main opens set:sys itself.
//
// The __appInit/__appExit overrides replace the weak defaults in libnx/nx/source/runtime/init.c
// @ b156b477, which also open apm, appletOE, hid, time and fsp-srv. libnx is under the ISC
// licence:
//
//   Copyright 2017-2018 libnx Authors
//
//   Permission to use, copy, modify, and/or distribute this software for any purpose with or
//   without fee is hereby granted, provided that the above copyright notice and this permission
//   notice appear in all copies.
//
//   THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
//   SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
//   THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
//   DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
//   CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
//   OR PERFORMANCE OF THIS SOFTWARE.
//
// The strings below are checked byte for byte by crates/loader/tests/homebrew_fixture.rs, which
// finds them in the NRO's and NSO's rodata. Change them there too.

#include <switch.h>

static const char LINE_ONE[] = "starquake hello: line one";
static const char LINE_TWO[] = "starquake hello: line two";
static const char FIRMWARE_OK[] = "starquake hello: set:sys firmware version ok";
static const char FIRMWARE_FAILED[] = "starquake hello: set:sys firmware version failed";

static void say(const char *text, size_t len) {
    svcOutputDebugString(text, len);
}

// Opens only sm:. The default also asks set:sys for the firmware version to set hosversion;
// this program leaves hosversion at 0, so libnx picks its pre-3.0.0 command ids
// (setsysGetFirmwareVersion uses command 3, not 4).
void __appInit(void) {
    Result rc = smInitialize();
    if (R_FAILED(rc)) {
        diagAbortWithResult(MAKERESULT(Module_Libnx, LibnxError_InitFail_SM));
    }
}

void __appExit(void) {
    smExit();
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    say(LINE_ONE, sizeof(LINE_ONE) - 1);
    say(LINE_TWO, sizeof(LINE_TWO) - 1);

    Result rc = setsysInitialize();
    if (R_SUCCEEDED(rc)) {
        SetSysFirmwareVersion firmware;
        rc = setsysGetFirmwareVersion(&firmware);
        setsysExit();
    }

    if (R_SUCCEEDED(rc)) {
        say(FIRMWARE_OK, sizeof(FIRMWARE_OK) - 1);
    } else {
        say(FIRMWARE_FAILED, sizeof(FIRMWARE_FAILED) - 1);
    }
    return 0;
}
