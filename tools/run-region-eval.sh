#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JEV_OBJ_PROJECT_DIR="$project_dir"
source "$project_dir/tools/jev-env.sh"
jev_obj_load_env "$project_dir" || exit $?
jev_obj_validate_distro || exit $?
workspace="$project_dir/jev_obj"
python_bin="$(jev_obj_find_python sim)" || exit $?
bridge_python="$(jev_obj_find_python bridge)" || exit $?
bridge_port="${JEV_OBJ_BRIDGE_PORT:-8766}"
python_runner="$project_dir/tools/run_focal_python.py"
mode="${1:-smoke}"
smoke_episode_index="${2:-0}"

if [[ "$mode" != 'smoke' && "$mode" != 'pilot10' && "$mode" != 'pilot24' ]]; then
  echo 'Usage: run-region-eval.sh smoke [episode_index] | pilot10 | pilot24 ACCEPTED_SMOKE_RUN' >&2
  exit 2
fi
if [[ "$mode" == 'smoke' && ! "$smoke_episode_index" =~ ^[0-9]+$ ]]; then
  echo 'ERROR: smoke episode index must be a non-negative integer' >&2
  exit 2
fi
cd "$project_dir"
if [[ "$mode" == "pilot24" ]]; then
  if [[ -z "${2:-}" ]]; then echo "ERROR: pilot24 requires accepted smoke run path as second argument" >&2; exit 6; fi
  "$python_bin" tools/verify_region_run.py "$2" --admit-pilot > /dev/null
fi
"$bridge_python" tools/preflight_jev_region.py --url "http://127.0.0.1:$bridge_port/health" >/dev/null
"$python_bin" tools/check_real_perception_services.py > /tmp/jev-region-perception-preflight.json

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$project_dir/.runtime/jev_region_graph/${mode}_${run_stamp}"
output_dir="$run_dir/habitat_output"
if [[ "$mode" == 'smoke' ]]; then
  hydra_args=(habitat.dataset.split=pilot test_epi_num="$smoke_episode_index" "video_output_path='$output_dir'")
  expected='1'
elif [[ "$mode" == 'pilot10' ]]; then
  hydra_args=(habitat.dataset.split=pilot test_epi_num=-1 eval_episode_limit=10 "video_output_path='$output_dir'")
  expected='10'
else
  hydra_args=(habitat.dataset.split=pilot test_epi_num=-1 eval_episode_limit=-1 "video_output_path='$output_dir'")
  expected='24'
fi
command_text="$python_bin $python_runner habitat_evaluation.py --dataset ovon ${hydra_args[*]} need_video=false"
manifest_args=(--run-dir "$run_dir" --mode "$mode" --command "$command_text")
if [[ "$mode" == 'smoke' ]]; then
  manifest_args+=(--smoke-episode-index "$smoke_episode_index")
fi
"$python_bin" tools/write_region_manifest.py "${manifest_args[@]}"
cp /tmp/jev-region-perception-preflight.json "$run_dir/perception_preflight.json"

set +u
source /opt/ros/noetic/setup.bash
source "$workspace/devel/setup.bash"
set -u
cd "$workspace"
"$python_bin" "$python_runner" "$project_dir/tools/check_eval_imports.py" >/dev/null
if rosnode list 2>/dev/null | grep -q "^/exploration_node$"; then
  echo "ERROR: existing exploration_node; refusing mixed run" >&2; exit 7
fi
export JEV_REGION_RUN_DIR="$run_dir"
roslaunch exploration_manager exploration.launch \
  jev_enabled:=true \
  jev_url:="http://127.0.0.1:$bridge_port/decide" \
  jev_state_profile:=region_graph \
  habitat_config:="$workspace/config/habitat_eval_ovon.yaml" \
  >"$run_dir/roslaunch.log" 2>&1 &
ros_pid=$!
cleanup() {
  if kill -0 "$ros_pid" 2>/dev/null; then
    kill -TERM "$ros_pid" 2>/dev/null || true
    wait "$ros_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if rosnode list 2>/dev/null | grep -q '^/exploration_node$'; then
    break
  fi
  if ! kill -0 "$ros_pid" 2>/dev/null; then
    echo "ERROR: roslaunch exited; inspect $run_dir/roslaunch.log" >&2
    exit 4
  fi
  sleep 1
done
if ! rosnode list 2>/dev/null | grep -q '^/exploration_node$'; then
  echo 'ERROR: exploration_node did not become ready in 60 seconds' >&2
  exit 5
fi

rosparam set /jev_obj/jev/run_id "$(basename "$run_dir")"
if [[ "$(rosparam get /exploration_node/jev/state_profile)" != "region_graph" || "$(rosparam get /exploration_node/jev/url)" != "http://127.0.0.1:$bridge_port/decide" ]]; then
  echo "ERROR: ROS Jev configuration mismatch" >&2; exit 8
fi
set +e
"$python_bin" "$python_runner" habitat_evaluation.py --dataset ovon "${hydra_args[@]}" need_video=false \
  >"$run_dir/habitat.log" 2>&1
eval_status=$?
set -e
printf '%s\n' "$eval_status" > "$run_dir/habitat_exit_code.txt"
printf '%s\n' "$expected" > "$run_dir/expected_episode_count.txt"
if [[ "$eval_status" -ne 0 ]]; then
  echo "ERROR: Habitat evaluation exited $eval_status; inspect $run_dir/habitat.log" >&2
  exit "$eval_status"
fi
cd "$project_dir"
"$python_bin" tools/verify_region_run.py "$run_dir" > "$run_dir/verification.stdout.json"
echo "Verified $mode run: $run_dir"
