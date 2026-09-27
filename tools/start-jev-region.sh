#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JEV_OBJ_PROJECT_DIR="$project_dir"
source "$project_dir/tools/jev-env.sh"
jev_obj_load_env "$project_dir" || exit $?
jev_obj_validate_distro || exit $?
python_bin="$(jev_obj_find_python bridge)" || exit $?
trace_dir="${JEV_OBJ_TRACE_DIR:-$project_dir/.runtime/jev_region_graph/bridge}"
[[ "$trace_dir" == /* ]] || trace_dir="$project_dir/$trace_dir"
port="${JEV_OBJ_BRIDGE_PORT:-8766}"
key_file="${JEV_OPENROUTER_KEY_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/jev-obj/openrouter.key}"

if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: Python environment not found: $python_bin" >&2
  exit 1
fi
cd "$project_dir"
if ! "$python_bin" -m nav_jev_bridge.server --help | grep -q 'region_graph'; then
  echo 'ERROR: installed bridge does not support --state-profile region_graph' >&2
  exit 4
fi
if [[ "${1:-}" == '--check' ]]; then
  echo "distro=${WSL_DISTRO_NAME:-native-linux}"
  echo "python=$python_bin"
  "$python_bin" --version
  echo 'provider=openrouter'
  echo 'model=typesafe/jev-1.13'
  echo 'state_profile=region_graph'
  echo "port=$port"
  exit 0
fi
if ss -ltn 2>/dev/null | grep -q ":$port "; then
  echo "ERROR: port $port is already listening. Stop and identify the existing provider first." >&2
  exit 5
fi

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  echo 'Using OPENROUTER_API_KEY from the current environment.'
elif [[ -r "$key_file" ]]; then
  IFS= read -r OPENROUTER_API_KEY < "$key_file" || true
  echo "Loaded OpenRouter API key from $key_file"
else
  read -rsp 'OpenRouter API key (hidden): ' OPENROUTER_API_KEY
  echo
fi
if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "ERROR: no API key available. Run $project_dir/tools/setup-jev-key.sh once." >&2
  exit 2
fi
export OPENROUTER_API_KEY
mkdir -p "$trace_dir"

echo "Starting real Jev bridge: provider=openrouter model=typesafe/jev-1.13 profile=region_graph port=$port"
exec "$python_bin" -m nav_jev_bridge.server \
  --provider openrouter \
  --state-profile region_graph \
  --port "$port" \
  --trace-dir "$trace_dir" \
  --jev-timeout 30
