#!/usr/bin/env bash

# Shared, non-evaluating configuration loader for the Jev-ObjNav shell tools.
# Only known NAME=VALUE pairs are accepted from .env; shell expressions are
# treated as plain text and are never evaluated.

jev_obj_load_env() {
  local project_dir="$1"
  local config_file="${JEV_OBJ_CONFIG_FILE:-$project_dir/.env}"
  [[ -r "$config_file" ]] || return 0

  local raw key value
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    raw="${raw%$'\r'}"
    [[ -z "$raw" || "$raw" == \#* || "$raw" != *=* ]] && continue
    key="${raw%%=*}"
    value="${raw#*=}"
    case "$key" in
      JEV_OBJ_WSL_DISTRO|JEV_OBJ_BRIDGE_PYTHON|JEV_OBJ_SIM_PYTHON|\
      JEV_OBJ_BRIDGE_PORT|JEV_OBJ_TRACE_DIR|JEV_OPENROUTER_KEY_FILE)
        if [[ -z "${!key:-}" && -n "$value" ]]; then
          printf -v "$key" '%s' "$value"
          export "$key"
        fi
        ;;
    esac
  done < "$config_file"
}

jev_obj_validate_distro() {
  local expected="${JEV_OBJ_WSL_DISTRO:-}"
  [[ -z "$expected" ]] && return 0
  if [[ "${WSL_DISTRO_NAME:-unknown}" != "$expected" ]]; then
    echo "ERROR: wrong WSL distribution: ${WSL_DISTRO_NAME:-unknown}; expected $expected" >&2
    return 3
  fi
}

jev_obj_find_python() {
  local kind="$1"
  local configured=''
  local probe=''
  if [[ "$kind" == bridge ]]; then
    configured="${JEV_OBJ_BRIDGE_PYTHON:-}"
    probe='import nav_jev_bridge, sys; assert sys.version_info >= (3, 10)'
  else
    configured="${JEV_OBJ_SIM_PYTHON:-}"
    probe='import habitat, habitat_sim, numpy; assert tuple(map(int, numpy.__version__.split(".")[:2])) < (2, 0)'
  fi

  local candidates=()
  [[ -n "$configured" ]] && candidates+=("$configured")
  local candidate
  for candidate in \
      /root/miniconda3/envs/*/bin/python \
      "$HOME"/miniconda3/envs/*/bin/python \
      "$HOME"/anaconda3/envs/*/bin/python; do
    [[ -x "$candidate" ]] && candidates+=("$candidate")
  done
  command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")

  for candidate in "${candidates[@]}"; do
    if (cd "${JEV_OBJ_PROJECT_DIR:?}" && PYTHONPATH="$JEV_OBJ_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$candidate" -c "$probe" >/dev/null 2>&1); then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "ERROR: no validated $kind Python environment found; set JEV_OBJ_${kind^^}_PYTHON in .env" >&2
  return 1
}

