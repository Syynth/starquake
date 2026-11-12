#!/bin/sh
# Installs devkitPro's package manager and the Switch homebrew toolchain (devkitA64 and libnx) on
# macOS, asking for your password once. Needed to build the homebrew test fixtures; nothing in the
# emulator itself depends on it.
#
#   tools/setup-devkitpro.sh        show what will be installed and ask before continuing
#   tools/setup-devkitpro.sh -y     don't ask
#
# What it does:
#   1. Downloads devkitPro's official installer package from its GitHub release (pinned below),
#      unless it is already in ~/Downloads, and shows its size and signature.
#   2. Runs the installer and installs the `switch-dev` package group into /opt/devkitpro.
#   3. Adds DEVKITPRO and DEVKITA64 to ~/.zshrc if they aren't there.
#   4. Checks that the compiler and libnx are present.
set -eu

RELEASE=v6.0.2
URL="https://github.com/devkitPro/pacman/releases/download/$RELEASE/devkitpro-pacman-installer.pkg"
PKG="$HOME/Downloads/devkitpro-pacman-installer.pkg"
DKP=/opt/devkitpro
ASSUME_YES=no
[ "${1:-}" = "-y" ] && ASSUME_YES=yes

find_pacman() {
    command -v dkp-pacman 2>/dev/null || { [ -x "$DKP/pacman/bin/dkp-pacman" ] && echo "$DKP/pacman/bin/dkp-pacman"; } || true
}

PACMAN="$(find_pacman)"
if [ -z "$PACMAN" ] && [ ! -f "$PKG" ]; then
    echo "Downloading $URL"
    curl --fail --location --progress-bar --output "$PKG.part" "$URL"
    mv "$PKG.part" "$PKG"
fi

if [ -z "$PACMAN" ]; then
    echo
    echo "Installer package: $PKG ($(du -h "$PKG" | cut -f1))"
    pkgutil --check-signature "$PKG" 2>&1 | sed -n '1,6p' || true
    echo
fi
echo "About to install the devkitPro package manager (if missing) and the switch-dev packages"
echo "into $DKP, using sudo."
if [ "$ASSUME_YES" != yes ]; then
    printf 'Continue? [y/N] '
    read -r answer
    case "$answer" in y | Y | yes) ;; *) echo "Stopped; nothing was installed."; exit 1 ;; esac
fi

# Ask for the password once and keep the sudo session alive until the script ends.
sudo -v
(while kill -0 "$$" 2>/dev/null; do sudo -n true 2>/dev/null || exit; sleep 50; done) &
KEEPALIVE=$!
trap 'kill "$KEEPALIVE" 2>/dev/null || true' EXIT

if [ -z "$PACMAN" ]; then
    sudo installer -pkg "$PKG" -target /
    PACMAN="$(find_pacman)"
    if [ -z "$PACMAN" ]; then
        echo "The installer ran, but dkp-pacman was not found. Look in $DKP and report what is there." >&2
        exit 1
    fi
fi

sudo "$PACMAN" -Sy --noconfirm
sudo "$PACMAN" -S --noconfirm --needed switch-dev

for line in "export DEVKITPRO=$DKP" "export DEVKITA64=$DKP/devkitA64"; do
    grep -qxF "$line" "$HOME/.zshrc" 2>/dev/null || echo "$line" >>"$HOME/.zshrc"
done

echo
echo "Installed:"
"$DKP/devkitA64/bin/aarch64-none-elf-gcc" --version | sed -n '1p'
ls "$DKP/libnx/lib/libnx.a"
echo "Open a new terminal window for DEVKITPRO and DEVKITA64 to take effect."
