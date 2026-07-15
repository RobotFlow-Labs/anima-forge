#!/bin/sh
# Install FORGE as an isolated user tool. See README.md and docs/QUICKSTART.md.

set -eu

PACKAGE_NAME="anima-forge"
REPOSITORY_URL="https://github.com/RobotFlow-Labs/anima-forge-distillation-pipeline"
PYTORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
BACKEND=${FORGE_INSTALL_BACKEND:-uv}
DEVICE=${FORGE_INSTALL_DEVICE:-auto}
VERSION=${FORGE_INSTALL_VERSION:-}
FROM_WHEEL=${FORGE_INSTALL_FROM_WHEEL:-}
NO_MODIFY_PATH=${FORGE_INSTALL_NO_MODIFY_PATH:-0}
UNINSTALL=${FORGE_INSTALL_UNINSTALL:-0}
UV_BIN=${FORGE_INSTALL_UV_BIN:-}
TEMP_DOCTOR=""

say() {
    printf '%s\n' "$*"
}

die() {
    say "FORGE installer error: $*" >&2
    say "Manual fallback: pip install anima-forge" >&2
    exit 1
}

cleanup() {
    status=$?
    if [ -n "$TEMP_DOCTOR" ] && [ -f "$TEMP_DOCTOR" ]; then
        rm -f "$TEMP_DOCTOR"
    fi
    if [ "$status" -ne 0 ]; then
        say "FORGE installation did not complete (exit $status)." >&2
        say "Manual fallback: pip install anima-forge" >&2
    fi
}
trap cleanup 0
trap 'exit 130' INT TERM HUP

usage() {
    cat <<'EOF'
Install FORGE as an isolated command-line tool.

Usage:
  install.sh [--cpu|--cuda] [--version X.Y.Z] [--from-wheel PATH|URL]
             [--backend uv|pipx] [--no-modify-path] [--uninstall]

Environment equivalents:
  FORGE_INSTALL_DEVICE=auto|cpu|cuda
  FORGE_INSTALL_VERSION=X.Y.Z
  FORGE_INSTALL_FROM_WHEEL=PATH|URL
  FORGE_INSTALL_BACKEND=uv|pipx
  FORGE_INSTALL_NO_MODIFY_PATH=0|1
  FORGE_INSTALL_UNINSTALL=0|1
  FORGE_INSTALL_UV_BIN=/path/to/uv
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpu)
            DEVICE=cpu
            ;;
        --cuda)
            DEVICE=cuda
            ;;
        --version)
            [ "$#" -ge 2 ] || die "--version requires a value"
            VERSION=$2
            shift
            ;;
        --from-wheel)
            [ "$#" -ge 2 ] || die "--from-wheel requires a path or URL"
            FROM_WHEEL=$2
            shift
            ;;
        --backend)
            [ "$#" -ge 2 ] || die "--backend requires uv or pipx"
            BACKEND=$2
            shift
            ;;
        --no-modify-path)
            NO_MODIFY_PATH=1
            ;;
        --uninstall)
            UNINSTALL=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
    shift
done

case "$BACKEND" in
    uv|pipx) ;;
    *) die "backend must be uv or pipx" ;;
esac

case "$DEVICE" in
    auto|cpu|cuda) ;;
    *) die "device must be auto, cpu, or cuda" ;;
esac

OS=$(uname -s 2>/dev/null || say unknown)
ARCH=$(uname -m 2>/dev/null || say unknown)
case "$OS" in
    Linux|Darwin) ;;
    MINGW*|MSYS*|CYGWIN*) die "Windows requires install.ps1" ;;
    *) die "unsupported operating system: $OS" ;;
esac
case "$ARCH" in
    x86_64|amd64|arm64|aarch64) ;;
    *) die "unsupported architecture: $ARCH" ;;
esac

if [ "$DEVICE" = auto ]; then
    if [ "$OS" = Linux ] && { command -v nvidia-smi >/dev/null 2>&1 || [ -f /proc/driver/nvidia/version ]; }; then
        DEVICE=cuda
    else
        DEVICE=cpu
    fi
fi
if [ "$OS" = Darwin ] && [ "$DEVICE" = cuda ]; then
    die "CUDA installation is supported on Linux only; use --cpu on macOS"
fi

say "FORGE installer"
say "  platform: $OS/$ARCH"
say "  backend:  $BACKEND"
say "  runtime:  $DEVICE"
if [ -n "$VERSION" ]; then
    say "  version:  $VERSION"
fi
if [ -n "$FROM_WHEEL" ]; then
    say "  source:   $FROM_WHEEL"
fi

ensure_uv() {
    if [ -n "$UV_BIN" ]; then
        [ -x "$UV_BIN" ] || die "FORGE_INSTALL_UV_BIN is not executable: $UV_BIN"
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        UV_BIN=$(command -v uv)
        return
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required to install uv"
    say "Installing uv in user space..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            UV_BIN=$candidate
            return
        fi
    done
    die "uv installation finished but the uv executable was not found"
}

ensure_pipx() {
    if command -v pipx >/dev/null 2>&1 && pipx --version >/dev/null 2>&1; then
        return
    fi
    command -v python3 >/dev/null 2>&1 || die "pipx backend requires python3"
    say "Installing pipx in user space..."
    python3 -m pip install --user --upgrade --force-reinstall pipx
    PATH="$HOME/.local/bin:$PATH"
    export PATH
    if ! command -v pipx >/dev/null 2>&1 || ! pipx --version >/dev/null 2>&1; then
        die "pipx installation finished but its executable is unavailable"
    fi
}

if [ "$BACKEND" = uv ]; then
    ensure_uv
else
    ensure_pipx
fi

if [ "$UNINSTALL" = 1 ]; then
    if [ "$BACKEND" = uv ]; then
        "$UV_BIN" tool uninstall "$PACKAGE_NAME"
    else
        pipx uninstall "$PACKAGE_NAME"
    fi
    say "FORGE uninstalled. PATH changes were left intact for other user tools."
    exit 0
fi

wheel_source() {
    case "$FROM_WHEEL" in
        https://*)
            say "$FROM_WHEEL"
            ;;
        http://*)
            die "remote --from-wheel URLs must use HTTPS"
            ;;
        *)
            [ -f "$FROM_WHEEL" ] || die "wheel not found: $FROM_WHEEL"
            wheel_dir=$(CDPATH='' cd -- "$(dirname -- "$FROM_WHEEL")" && pwd)
            say "file://$wheel_dir/$(basename -- "$FROM_WHEEL")"
            ;;
    esac
}

if [ -n "$FROM_WHEEL" ]; then
    SOURCE=$(wheel_source)
    SPEC="$PACKAGE_NAME @ $SOURCE"
elif [ -n "$VERSION" ]; then
    SPEC="$PACKAGE_NAME==$VERSION"
else
    SPEC="$PACKAGE_NAME"
fi

say "Installing $SPEC ..."
if [ "$BACKEND" = uv ]; then
    if [ "$DEVICE" = cpu ] && [ "$OS" = Linux ] && { [ "$ARCH" = x86_64 ] || [ "$ARCH" = amd64 ]; }; then
        # The default Linux PyPI Torch wheel carries CUDA libraries. Resolve
        # Torch/TorchVision from PyTorch's official CPU index while retaining
        # PyPI for the complete mandatory FORGE dependency set.
        "$UV_BIN" tool install --force --python 3.12 \
            --index "$PYTORCH_CPU_INDEX" --index-strategy unsafe-best-match "$SPEC"
    else
        "$UV_BIN" tool install --force --python 3.12 "$SPEC"
    fi
    TOOL_BIN=$($UV_BIN tool dir --bin)
else
    if [ "$DEVICE" = cpu ] && [ "$OS" = Linux ] && { [ "$ARCH" = x86_64 ] || [ "$ARCH" = amd64 ]; }; then
        UV_INDEX_STRATEGY=unsafe-best-match \
            pipx install --force --python 3.12 --pip-args="--extra-index-url $PYTORCH_CPU_INDEX" "$SPEC"
    else
        pipx install --force --python 3.12 "$SPEC"
    fi
    TOOL_BIN=$(pipx environment --value PIPX_BIN_DIR 2>/dev/null || say "$HOME/.local/bin")
fi

case ":$PATH:" in
    *":$TOOL_BIN:"*) PATH_MISSING=0 ;;
    *) PATH_MISSING=1 ;;
esac
PATH="$TOOL_BIN:$PATH"
export PATH

if [ "$PATH_MISSING" -eq 1 ]; then
    if [ "$NO_MODIFY_PATH" = 1 ]; then
        say "PATH was not modified permanently. Add this directory to PATH: $TOOL_BIN"
    elif [ "$BACKEND" = uv ]; then
        "$UV_BIN" tool update-shell
        say "PATH updated. Restart your shell if forge is not immediately available."
    else
        pipx ensurepath
        say "PATH updated. Restart your shell if forge is not immediately available."
    fi
fi

command -v forge >/dev/null 2>&1 || die "forge executable was not installed into $TOOL_BIN"
INSTALLED_VERSION=$(forge --version) || die "forge --version failed"
say "Installed FORGE $INSTALLED_VERSION"

TEMP_DOCTOR=$(mktemp "${TMPDIR:-/tmp}/forge-doctor.XXXXXX")
DOCTOR_STATUS=0
forge doctor --json >"$TEMP_DOCTOR" 2>/dev/null || DOCTOR_STATUS=$?
if [ "$BACKEND" = uv ]; then
    TOOL_ROOT=$($UV_BIN tool dir)
    TOOL_PYTHON="$TOOL_ROOT/$PACKAGE_NAME/bin/python"
else
    TOOL_PYTHON=$(command -v python3)
fi
[ -x "$TOOL_PYTHON" ] || die "could not locate Python for doctor JSON verification"
"$TOOL_PYTHON" -c 'import json, sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$TEMP_DOCTOR" \
    || die "forge doctor --json emitted invalid JSON"
if [ "$DOCTOR_STATUS" -ne 0 ]; then
    say "forge doctor completed with readiness warnings (exit $DOCTOR_STATUS)."
else
    say "forge doctor passed."
fi

say "Next steps:"
say "  forge doctor"
say "  forge quickstart --yes"
say "  $REPOSITORY_URL"
