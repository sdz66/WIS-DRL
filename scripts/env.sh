#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export XDG_CACHE_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"
