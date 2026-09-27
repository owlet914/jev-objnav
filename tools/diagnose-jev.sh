#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JEV_OBJ_PROJECT_DIR="$project_dir"
source "$project_dir/tools/jev-env.sh"
jev_obj_load_env "$project_dir" || exit $?
jev_obj_validate_distro || exit $?
python_bin="$(jev_obj_find_python bridge)" || exit $?
cd "$project_dir"
exec "$python_bin" tools/diagnose_jev_api.py --prompt-key
