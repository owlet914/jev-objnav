#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JEV_OBJ_PROJECT_DIR="$project_dir"
source "$project_dir/tools/jev-env.sh"
jev_obj_load_env "$project_dir" || exit $?
jev_obj_validate_distro || exit $?
key_file="${JEV_OPENROUTER_KEY_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/jev-obj/openrouter.key}"

umask 077
mkdir -p "$(dirname "$key_file")"
read -rsp 'OpenRouter API key to save (hidden): ' new_key
echo
if [[ -z "$new_key" ]]; then
  echo 'ERROR: empty key; existing configuration was not changed.' >&2
  exit 2
fi

temporary_file="$(mktemp "${key_file}.tmp.XXXXXX")"
cleanup() {
  rm -f -- "$temporary_file"
}
trap cleanup EXIT
printf '%s\n' "$new_key" > "$temporary_file"
chmod 600 "$temporary_file"
mv -f -- "$temporary_file" "$key_file"
trap - EXIT
unset new_key OPENROUTER_API_KEY

echo "Saved OpenRouter API key to $key_file"
echo 'Permissions: owner read/write only (600).'
echo 'The key file is outside the Git repository.'
