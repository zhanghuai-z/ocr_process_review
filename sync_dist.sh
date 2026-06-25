#!/usr/bin/env bash
# WSL bridge for sync_dist.bat.
set -e

PROJ_ROOT="$(cd "$(dirname "$0")"; pwd)"
cd "$PROJ_ROOT"

if ! command -v cmd.exe >/dev/null 2>&1; then
    echo "[ERROR] 未检测到 cmd.exe。该脚本只能在 WSL 中桥接 Windows sync_dist.bat。"
    exit 1
fi

args=()
for arg in "$@"; do
    case "$arg" in
        /mnt/*)
            args+=("$(wslpath -w "$arg")")
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

WIN_PROJ_ROOT="$(wslpath -w "$PROJ_ROOT")"
cmd.exe /c "${WIN_PROJ_ROOT}\\sync_dist.bat" "${args[@]}"
