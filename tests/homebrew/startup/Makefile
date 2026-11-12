# Builds the startup fixture: startup.nro (what `starquake run` loads), then rewrites
# manifest.toml. Needs devkitPro with devkitA64 and libnx (tools/setup-devkitpro.sh installs
# them); the gate does not, it only checks the manifest. The conventions are the hello fixture's
# (../hello/Makefile).
#
#   export DEVKITPRO=/opt/devkitpro
#   make            # rebuild the binary and the manifest
#   make clean      # remove build/ (the committed binary stays)
#
# Commit the sources, the binary and manifest.toml together (tests/homebrew/README.md says why).

ifeq ($(strip $(DEVKITPRO)),)
$(error Set DEVKITPRO to the devkitPro folder, for example: export DEVKITPRO=/opt/devkitpro)
endif

TOOLS := $(DEVKITPRO)/tools/bin
CC := $(DEVKITPRO)/devkitA64/bin/aarch64-none-elf-gcc
READELF := $(DEVKITPRO)/devkitA64/bin/aarch64-none-elf-readelf
LIBNX := $(DEVKITPRO)/libnx

# The flags of libnx's application template (examples/switch/templates/application/Makefile),
# without -g, so no build path ends up in the ELF and its build id.
ARCH := -march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -fPIE
CFLAGS := -Wall -Wextra -Werror -O2 -ffunction-sections $(ARCH) -D__SWITCH__ \
	-I$(LIBNX)/include -ffile-prefix-map=$(CURDIR)=.
LDFLAGS := -specs=$(LIBNX)/switch.specs $(ARCH) -L$(LIBNX)/lib
LIBS := -lnx

SOURCES := startup.c Makefile
BINARIES := startup.nro

.PHONY: all clean
all: manifest.toml

build:
	mkdir -p build

build/startup.elf: startup.c Makefile | build
	$(CC) $(CFLAGS) $(LDFLAGS) startup.c $(LIBS) -o $@

build/startup.nacp: Makefile | build
	$(TOOLS)/nacptool --create "starquake startup" "starquake" "1.0.0" $@

startup.nro: build/startup.elf build/startup.nacp
	$(TOOLS)/elf2nro build/startup.elf $@ --nacp=build/startup.nacp

# Package versions come from devkitPro's pacman (`dkp-pacman` on macOS, `pacman` elsewhere).
PACMAN := $(shell command -v dkp-pacman 2>/dev/null || command -v pacman 2>/dev/null)
version = $(shell $(PACMAN) -Q $(1) 2>/dev/null | cut -d' ' -f2)

manifest.toml: $(SOURCES) $(BINARIES) build/startup.elf
	@{ \
	echo '# Written by `make` in this folder; do not edit by hand.'; \
	echo '# crates/loader/tests/homebrew_fixture.rs fails when a file no longer matches.'; \
	echo; \
	echo '[toolchain]'; \
	echo 'gcc = "$(shell $(CC) -dumpversion)"'; \
	echo 'devkitA64 = "$(call version,devkitA64)"'; \
	echo 'libnx = "$(call version,libnx)"'; \
	echo 'switch-tools = "$(call version,switch-tools)"'; \
	echo; \
	echo '[elf]'; \
	echo "build-id = \"$$($(READELF) -n build/startup.elf | sed -n 's/.*Build ID: //p')\""; \
	echo; \
	echo '[sources]'; \
	for f in $(SOURCES); do echo "\"$$f\" = \"$$(shasum -a 256 $$f | cut -d' ' -f1)\""; done; \
	echo; \
	echo '[binaries]'; \
	for f in $(BINARIES); do echo "\"$$f\" = \"$$(shasum -a 256 $$f | cut -d' ' -f1)\""; done; \
	} > $@

clean:
	rm -rf build
