#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JEV_OBJ_PROJECT_DIR="$project_dir"
source "$project_dir/tools/jev-env.sh"
jev_obj_load_env "$project_dir" || exit $?
jev_obj_validate_distro || exit $?
python_bin="$(jev_obj_find_python bridge)" || exit $?
trace_dir="${JEV_OBJ_TRACE_DIR:-$project_dir/.runtime/jev_bridge}"
[[ "$trace_dir" == /* ]] || trace_dir="$project_dir/$trace_dir"
port="${JEV_OBJ_BRIDGE_PORT:-8766}"

if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: Python environment not found: $python_bin" >&2
  exit 1
fi

if [[ "${1:-}" == '--check' ]]; then
  echo "distro=${WSL_DISTRO_NAME:-native-linux}"
  echo "python=$python_bin"
  "$python_bin" --version
  echo 'provider=openrouter'
  echo 'model=typesafe/jev-1.13'
  echo "port=$port"
  exit 0
fi

pkill -f "[n]av_jev_bridge.server.*--port $port" 2>/dev/null || true
unset OPENROUTER_API_KEY

read -rsp 'OpenRouter API key (hidden): ' OPENROUTER_API_KEY
echo
if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo 'ERROR: no API key entered' >&2
  exit 2
fi
export OPENROUTER_API_KEY

mkdir -p "$trace_dir"
cd "$project_dir"

echo "Starting real Jev bridge: provider=openrouter model=typesafe/jev-1.13 port=$port"
exec "$python_bin" -m nav_jev_bridge.server \
  --provider openrouter \
  --state-profile full \
  --port "$port" \
  --trace-dir "$trace_dir" \
  --jev-timeout 30
