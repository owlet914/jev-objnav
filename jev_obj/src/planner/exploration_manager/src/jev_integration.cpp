/**
 * Optional high-level Jev decision over paths already planned by Jev Obj.
 * Jev only returns a candidate ID; it never supplies a coordinate or motion command.
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <exploration_manager/jev_contract.h>
#include <curl/curl.h>
#include <nlohmann_json.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace jev_obj_planner {
namespace {

using json = nlohmann::json;

struct JevCandidate {
  std::string id;
  std::string target_id;
  std::string kind;
  Eigen::Vector2d goal;
  std::vector<Eigen::Vector2d> path;
  double semantic_score;
  double information_gain;
  bool information_gain_available;
  int result;
  std::string source_stage;
  int source_priority;
  std::string safety_mode;
};

struct HttpResult {
  bool transport_ok = false;
  long status = 0;
  CURLcode curl_code = CURLE_OK;
  std::string body;
};

double unitScore(double value)
{
  if (!std::isfinite(value))
    return 0.0;
  return std::max(0.0, std::min(1.0, value));
}

double normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

double observationTurnAngleRad()
{
  double angle_deg = 30.0;
  ros::param::get("/jev_obj/jev/observation_turn_angle_deg", angle_deg);
  if (!std::isfinite(angle_deg) || angle_deg <= 0.0 || angle_deg > 180.0) {
    ROS_WARN("[Jev] Invalid observation turn angle %.3f deg; use 30 deg", angle_deg);
    angle_deg = 30.0;
  }
  return angle_deg * M_PI / 180.0;
}

json finiteOrNull(double value)
{
  return std::isfinite(value) ? json(value) : json(nullptr);
}

const char* searchResultName(int result)
{
  if (result == Astar2D::REACH_END) return "reach_end";
  if (result == Astar2D::TIMEOUT) return "timeout";
  if (result == Astar2D::NODE_POOL_EXHAUSTED) return "node_pool_exhausted";
  return "no_path";
}

json pointJson(const Eigen::Vector2d& point)
{
  return { { "x", point.x() }, { "y", point.y() } };
}

json pointsJson(const std::vector<Eigen::Vector2d>& points)
{
  json output = json::array();
  for (const auto& point : points) output.push_back(pointJson(point));
  return output;
}

double pathLength(const std::vector<Eigen::Vector2d>& path);

json objectPathJson(const ObjectPathEvidence& evidence)
{
  json attempts = json::array();
  for (const auto& attempt : evidence.attempts) {
    attempts.push_back({
      { "success_distance_m", attempt.success_distance },
      { "max_search_time_s", attempt.max_search_time_s },
      { "safety_mode", attempt.safety_mode == Astar2D::SAFETY_MODE::EXTREME ? "extreme" : "normal" },
      { "search_result", searchResultName(attempt.search_result) },
      { "early_terminate_cost", finiteOrNull(attempt.early_terminate_cost) },
      { "reached", attempt.reached }
    });
  }
  return {
    { "geometry_available", evidence.geometry_available },
    { "nearest_object_point", evidence.geometry_available ? pointJson(evidence.nearest_object_point) : json(nullptr) },
    { "approach_point", evidence.path.empty() ? json(nullptr) : pointJson(evidence.approach_point) },
    { "path_points", pointsJson(evidence.path) },
    { "distance_m", evidence.path.empty() ? json(nullptr) : finiteOrNull(pathLength(evidence.path)) },
    { "attempts", attempts },
    { "final_reason", evidence.final_reason }
  };
}

double pathLength(const std::vector<Eigen::Vector2d>& path)
{
  double length = 0.0;
  for (size_t i = 1; i < path.size(); ++i)
    length += (path[i] - path[i - 1]).norm();
  return length;
}

double rounded(double value, int decimals)
{
  if (!std::isfinite(value)) return value;
  const double scale = std::pow(10.0, decimals);
  return std::round(value * scale) / scale;
}

size_t appendHttpBody(char* data, size_t size, size_t count, void* output)
{
  const size_t bytes = size * count;
  static_cast<std::string*>(output)->append(data, bytes);
  return bytes;
}

HttpResult postJson(const std::string& url, const std::string& payload, int timeout_ms)
{
  HttpResult result;
  static const CURLcode curl_init_result = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (curl_init_result != CURLE_OK)
    return result;
  CURL* curl = curl_easy_init();
  if (!curl)
    return result;
  struct curl_slist* headers = curl_slist_append(nullptr, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::min(timeout_ms, 500)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, appendHttpBody);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result.body);
  result.curl_code = curl_easy_perform(curl);
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &result.status);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  result.transport_ok = result.curl_code == CURLE_OK;
  return result;
}

}  // namespace

JEV_PLAN_STATUS ExplorationManager::planWithJev(const Vector3d& pos, double yaw, int& result)
{
  const Vector2d start(pos.x(), pos.y());
  const bool region_graph_profile = jev_state_profile_ == "region_graph";
  std::vector<JevCandidate> candidates;
  std::string episode_id = jev_session_id_;
  std::string map_episode_id;
  int observation_id = 0;
  int map_revision = 0;
  double observation_timestamp_ms = 0.0;
  ros::param::get("/jev_obj/jev/episode_id", episode_id);
  ros::param::get("/jev_obj/jev/map/episode_id", map_episode_id);
  ros::param::get("/jev_obj/jev/map/observation_id", observation_id);
  ros::param::get("/jev_obj/jev/map/revision", map_revision);
  ros::param::get("/jev_obj/jev/map/observation_timestamp_ms", observation_timestamp_ms);
  if (episode_id.empty()) episode_id = jev_session_id_;
  if (map_episode_id != episode_id || observation_id < 0 || map_revision < 0) {
    ROS_ERROR("[Jev] Map context is not synchronized with episode %s (map episode=%s)",
        episode_id.c_str(), map_episode_id.c_str());
    return JEV_PLAN_NONRETRYABLE_FAILURE;
  }
  bool confirmation_pending = false;
  std::string confirmation_candidate_id, confirmation_target_id;
  std::string confirmation_last_outcome = "idle";
  int confirmation_start_observation_id = -1;
  int confirmation_last_observation_id = -1;
  int confirmation_attempts = 0, confirmation_valid_hits = 0;
  int confirmation_required_hits = 2, confirmation_max_attempts = 4;
  std::string blocked_candidate_id, blocked_target_id;
  int blocked_until_observation_id = -1;
  ros::param::get("/jev_obj/jev/confirmation/pending", confirmation_pending);
  ros::param::get("/jev_obj/jev/confirmation/candidate_id", confirmation_candidate_id);
  ros::param::get("/jev_obj/jev/confirmation/target_id", confirmation_target_id);
  ros::param::get("/jev_obj/jev/confirmation/last_outcome", confirmation_last_outcome);
  ros::param::get("/jev_obj/jev/confirmation/start_observation_id",
      confirmation_start_observation_id);
  ros::param::get("/jev_obj/jev/confirmation/last_observation_id",
      confirmation_last_observation_id);
  ros::param::get("/jev_obj/jev/confirmation/attempts", confirmation_attempts);
  ros::param::get("/jev_obj/jev/confirmation/valid_hits", confirmation_valid_hits);
  ros::param::get("/jev_obj/jev/confirmation/required_hits", confirmation_required_hits);
  ros::param::get("/jev_obj/jev/confirmation/max_attempts", confirmation_max_attempts);
  ros::param::get("/jev_obj/jev/confirmation/blocked_candidate_id", blocked_candidate_id);
  ros::param::get("/jev_obj/jev/confirmation/blocked_target_id", blocked_target_id);
  ros::param::get("/jev_obj/jev/confirmation/blocked_until_observation_id",
      blocked_until_observation_id);
  bool confirmation_block_active = !blocked_candidate_id.empty() &&
      observation_id <= blocked_until_observation_id;
  if (!blocked_candidate_id.empty() && !confirmation_block_active) {
    blocked_candidate_id.clear();
    blocked_target_id.clear();
    blocked_until_observation_id = -1;
    ros::param::set("/jev_obj/jev/confirmation/blocked_candidate_id", "");
    ros::param::set("/jev_obj/jev/confirmation/blocked_target_id", "");
    ros::param::set("/jev_obj/jev/confirmation/blocked_until_observation_id", -1);
  }
  std::string run_id;
  ros::param::get("/jev_obj/jev/run_id", run_id);
  const std::string request_id = run_id + ":" + jev_session_id_ + ":" + episode_id + ":obs-" + std::to_string(observation_id) +
      ":map-" + std::to_string(map_revision);
  int navigation_step_count = 0;
  ros::param::get("/jev_obj/jev/navigation/step_count", navigation_step_count);
  const int revision = map_revision;
  static std::string cached_request_id;
  static json cached_snapshot;
  static std::vector<JevCandidate> cached_candidates;
  json snapshot;
  if (cached_request_id == request_id) {
    snapshot = cached_snapshot; candidates = cached_candidates;
  } else {
  snapshot = {
    { "schema_version", "2.0" },
    { "request_id", request_id },
    { "observation_id", observation_id },
    { "timestamp_ms", ros::WallTime::now().toNSec() / 1000000 },
    { "robot_pose", { { "x", pos.x() }, { "y", pos.y() }, { "yaw", yaw } } },
    { "rooms", json::array() },
    { "objects", json::array() },
    { "semantic_matches", json::array() },
    { "frontiers", json::array() },
    { "candidate_evidence", json::array() },
    { "search_branches", json::array() },
    { "observation_actions", json::array() },
    { "candidates", json::array() }
  };

  std::string target_object = "target object";
  std::string room_prior;
  std::vector<std::string> related_objects;
  std::vector<std::string> target_subcategories;
  ros::param::get("/jev_obj/jev/target_object", target_object);
  ros::param::get("/jev_obj/jev/room_prior", room_prior);
  ros::param::get("/jev_obj/jev/related_objects", related_objects);
  ros::param::get("/jev_obj/jev/target_subcategories", target_subcategories);
  if (target_object.empty())
    target_object = "target object";
  snapshot["target_object"] = target_object;
  snapshot["episode_id"] = episode_id;
  snapshot["room_prior"] = room_prior;
  snapshot["related_objects"] = related_objects;
  snapshot["target_subcategories"] = target_subcategories;
  snapshot["context_availability"] = {
    { "room_prior_available", !room_prior.empty() && room_prior != "everywhere" },
    { "related_objects_available", !related_objects.empty() },
    { "observed_rooms_available", false }
  };
  int collision_count = 0;
  int newest_action = -1, last_published_action = -1, init_scan_actions_completed = 0;
  int stucking_action_count = 0, stucking_target_count = 0, escape_action_index = 0;
  int forced_stuck_point_count = 0, current_path_points = 0;
  bool latest_collision = false;
  bool frontier_changed = false, replan_flag = false, escape_active = false;
  bool last_action_completed = false, traveled_path_complete = false;
  double reach_distance_m = 0.2, soft_reach_distance_m = 0.45;
  std::string last_action_origin = "unknown";
  std::vector<double> traveled_path_xy;
  ros::param::get("/jev_obj/jev/navigation/step_count", navigation_step_count);
  ros::param::get("/jev_obj/jev/navigation/latest_collision", latest_collision);
  ros::param::get("/jev_obj/jev/navigation/collision_count", collision_count);
  ros::param::get("/jev_obj/jev/navigation/newest_action", newest_action);
  ros::param::get("/jev_obj/jev/navigation/last_published_action", last_published_action);
  ros::param::get("/jev_obj/jev/navigation/last_action_completed", last_action_completed);
  ros::param::get("/jev_obj/jev/navigation/last_action_origin", last_action_origin);
  ros::param::get("/jev_obj/jev/navigation/init_scan_actions_completed", init_scan_actions_completed);
  ros::param::get("/jev_obj/jev/navigation/stucking_action_count", stucking_action_count);
  ros::param::get("/jev_obj/jev/navigation/stucking_target_count", stucking_target_count);
  ros::param::get("/jev_obj/jev/navigation/escape_active", escape_active);
  ros::param::get("/jev_obj/jev/navigation/escape_action_index", escape_action_index);
  ros::param::get("/jev_obj/jev/navigation/forced_stuck_point_count", forced_stuck_point_count);
  ros::param::get("/jev_obj/jev/navigation/frontier_changed", frontier_changed);
  ros::param::get("/jev_obj/jev/navigation/replan_flag", replan_flag);
  ros::param::get("/jev_obj/jev/navigation/current_path_points", current_path_points);
  ros::param::get("/jev_obj/jev/navigation/reach_distance_m", reach_distance_m);
  ros::param::get("/jev_obj/jev/navigation/soft_reach_distance_m", soft_reach_distance_m);
  ros::param::get("/jev_obj/jev/navigation/traveled_path_xy", traveled_path_xy);
  ros::param::get("/jev_obj/jev/navigation/traveled_path_complete", traveled_path_complete);
  json traveled_path = json::array();
  for (size_t i = 1; i < traveled_path_xy.size(); i += 2)
    traveled_path.push_back({ { "x", traveled_path_xy[i - 1] }, { "y", traveled_path_xy[i] } });
  snapshot["navigation_history"] = {
    { "step_count", navigation_step_count },
    { "latest_collision", latest_collision },
    { "collision_count", collision_count },
    { "last_candidate_id", jev_last_candidate_id_.empty() ? json(nullptr) : json(jev_last_candidate_id_) },
    { "last_entity_id", jev_last_entity_id_.empty() ? json(nullptr) : json(jev_last_entity_id_) },
    { "consecutive_same_entity_selections", jev_repeated_candidate_count_ },
    { "selected_entity_history", jev_selected_entity_history_ },
    { "controller_state", {
        { "newest_planned_action", newest_action },
        { "last_published_action", last_published_action },
        { "last_action_completed", last_action_completed },
        { "last_action_origin", last_action_origin },
        { "initial_scan_actions_completed", init_scan_actions_completed },
        { "initial_scan_action_count", 26 },
        { "stucking_action_count", stucking_action_count },
        { "stucking_same_target_count", stucking_target_count },
        { "escape_active", escape_active },
        { "escape_action_index", escape_action_index },
        { "forced_stuck_point_count", forced_stuck_point_count },
        { "frontier_changed", frontier_changed },
        { "replan_flag", replan_flag },
        { "current_path_points", current_path_points },
        { "reach_distance_m", reach_distance_m },
        { "soft_reach_distance_m", soft_reach_distance_m }
      } },
    { "traveled_path", traveled_path },
    { "traveled_path_complete", traveled_path_complete }
  };
  snapshot["object_confirmation"] = {
    { "status", confirmation_pending ? "pending" :
        (confirmation_block_active ? "rejected_cooldown" : "idle") },
    { "pending_candidate_id", confirmation_candidate_id.empty() ?
        json(nullptr) : json(confirmation_candidate_id) },
    { "pending_target_id", confirmation_target_id.empty() ?
        json(nullptr) : json(confirmation_target_id) },
    { "start_observation_id", confirmation_start_observation_id >= 0 ?
        json(confirmation_start_observation_id) : json(nullptr) },
    { "last_observation_id", confirmation_last_observation_id >= 0 ?
        json(confirmation_last_observation_id) : json(nullptr) },
    { "attempts", confirmation_attempts },
    { "valid_fresh_hits", confirmation_valid_hits },
    { "required_fresh_hits", confirmation_required_hits },
    { "max_attempts", confirmation_max_attempts },
    { "last_outcome", confirmation_last_outcome },
    { "blocked_candidate_id", confirmation_block_active ?
        json(blocked_candidate_id) : json(nullptr) },
    { "blocked_target_id", confirmation_block_active ?
        json(blocked_target_id) : json(nullptr) },
    { "blocked_until_observation_id", confirmation_block_active ?
        json(blocked_until_observation_id) : json(nullptr) },
    { "stop_rule", "approach is not success; STOP requires fresh synchronized target masks and a fresh matching map object in multiple post-arrival observations" }
  };
  if (jev_has_last_goal_) {
    const double current_distance = (jev_last_goal_ - start).norm();
    snapshot["navigation_history"]["last_goal_execution"] = {
      { "initial_distance_m", jev_last_goal_initial_distance_ },
      { "current_distance_m", current_distance },
      { "cumulative_progress_m", jev_last_goal_initial_distance_ - current_distance },
      { "last_step_progress_m", jev_last_goal_previous_distance_ - current_distance },
      { "initial_step", jev_last_goal_initial_step_ },
      { "current_step", navigation_step_count }
    };
  }

  static const char* policy_names[] = { "distance", "semantic", "hybrid", "tsp_distance" };
  const std::string policy_name = ep_->policy_mode_ >= ExplorationParam::DISTANCE &&
          ep_->policy_mode_ <= ExplorationParam::TSP_DIST ?
      policy_names[ep_->policy_mode_] : "unknown";
  snapshot["original_policy"] = {
    { "mode", policy_name },
    { "mode_id", ep_->policy_mode_ },
    { "decision_order", {
        "high_confidence_object", "over_depth_object", "active_frontier_policy",
        "suspicious_object_if_no_active_path", "dormant_frontier_policy",
        "extreme_object_search"
      } },
    { "semantic_thresholds", {
        { "std_dev", ep_->sigma_threshold_ },
        { "max_to_mean", ep_->max_to_mean_threshold_ },
        { "max_to_mean_percentage", ep_->max_to_mean_percentage_ }
      } },
    { "notes", {
        { "frontier_semantic_value", "mode-specific 5x5 neighborhood feature; raw values retained" },
        { "frontier_information_gain", "unknown/null because the original code does not measure gain" },
        { "object_function_score", "int(observation_cloud_sum * fused_confidence)" }
      } }
  };

  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);
  jev_revision_ = revision;
  snapshot["map"] = {
    { "frame_id", jev_frame_id_ }, { "revision", revision },
    { "observation_id", observation_id },
    { "observation_timestamp_ms", observation_timestamp_ms },
    { "resolution_m", sdf_map_->getResolution() },
    { "origin", { { "x", origin.x() }, { "y", origin.y() } } },
    { "size_m", { { "x", size.x() }, { "y", size.y() } } },
    { "frontier_count", ed_->frontier_averages_.size() },
    { "object_count", ed_->object_averages_.size() }
  };

  if (!region_graph_profile) {
    // Bounded local evidence: rows are north-to-south, centered on the robot.
    // This is the live occupancy/value map, not a ground-truth Habitat floor plan.
    const int grid_side = 17;
    const int grid_half = grid_side / 2;
    const double grid_step = 0.5;
    json occupancy_rows = json::array();
    json semantic_rows = json::array();
    int free_cells = 0, occupied_cells = 0, unknown_cells = 0;
    for (int row = 0; row < grid_side; ++row) {
      std::string occupancy_row, semantic_row;
      for (int col = 0; col < grid_side; ++col) {
        const Vector2d point(start.x() + (col - grid_half) * grid_step,
            start.y() + (grid_half - row) * grid_step);
        char occupancy = '?';
        char semantic = '.';
        if (sdf_map_->isInMap(point)) {
          const int state = sdf_map_->getOccupancy(point);
          if (state == SDFMap2D::FREE) {
            occupancy = sdf_map_->getInflateOccupancy(point) == 1 ? 'i' : '.';
            ++free_cells;
          }
          else if (state == SDFMap2D::OCCUPIED) {
            occupancy = '#';
            ++occupied_cells;
          }
          else {
            ++unknown_cells;
          }
          const double confidence = sdf_map_->value_map_->getConfidence(point);
          const double raw_semantic = sdf_map_->value_map_->getValue(point);
          if (std::isfinite(confidence) && confidence > 0.0 && std::isfinite(raw_semantic)) {
            const int level = static_cast<int>(std::round(9.0 * unitScore(raw_semantic)));
            semantic = static_cast<char>('0' + level);
          }
        }
        else {
          ++unknown_cells;
        }
        occupancy_row.push_back(occupancy);
        semantic_row.push_back(semantic);
      }
      occupancy_rows.push_back(occupancy_row);
      semantic_rows.push_back(semantic_row);
    }
    snapshot["map"]["local_preview"] = {
      { "center", { { "x", start.x() }, { "y", start.y() } } },
      { "cell_size_m", grid_step }, { "width", grid_side }, { "height", grid_side },
      { "occupancy_rows", occupancy_rows }, { "semantic_value_rows", semantic_rows },
      { "legend", { { "?", "unknown" }, { ".", "free or no semantic measurement" },
                      { "i", "inflated obstacle" }, { "#", "occupied" } } },
      { "free_cells", free_cells }, { "occupied_cells", occupied_cells },
      { "unknown_cells", unknown_cells }, { "lossy", true },
      { "native_resolution_m", sdf_map_->getResolution() },
      { "semantic_display_transform", "round(9 * clamp(raw_value, 0, 1)); display only" },
      { "not_raw_map", true }
    };
  }

  bool itm_available = false, itm_fallback = false, map_itm_valid = false;
  bool detection_available = false, detection_fallback = false;
  bool target_detection_available = false, related_detection_available = false;
  std::string itm_backend = "unknown", itm_query_text, itm_score_type;
  std::string detector_status_json;
  std::vector<std::string> detection_backends;
  int detection_count = 0, target_match_count = 0, related_match_count = 0;
  int valid_mask_count = 0, ingested_object_count = 0;
  int itm_observation_id = -1, detection_observation_id = -1;
  ros::param::get("/jev_obj/jev/perception/itm_available", itm_available);
  ros::param::get("/jev_obj/jev/perception/itm_fallback", itm_fallback);
  ros::param::get("/jev_obj/jev/perception/itm_backend", itm_backend);
  ros::param::get("/jev_obj/jev/perception/itm_observation_id", itm_observation_id);
  ros::param::get("/jev_obj/jev/perception/itm_query_text", itm_query_text);
  ros::param::get("/jev_obj/jev/perception/itm_score_type", itm_score_type);
  ros::param::get("/jev_obj/jev/map/itm_valid", map_itm_valid);
  ros::param::get(
      "/jev_obj/jev/perception/object_detection_available", detection_available);
  ros::param::get(
      "/jev_obj/jev/perception/object_detection_fallback", detection_fallback);
  ros::param::get(
      "/jev_obj/jev/perception/object_detection_backends", detection_backends);
  ros::param::get(
      "/jev_obj/jev/perception/object_detection_count", detection_count);
  ros::param::get("/jev_obj/jev/perception/detection_observation_id", detection_observation_id);
  ros::param::get("/jev_obj/jev/perception/target_detection_available", target_detection_available);
  ros::param::get("/jev_obj/jev/perception/related_detection_available", related_detection_available);
  ros::param::get("/jev_obj/jev/perception/target_match_count", target_match_count);
  ros::param::get("/jev_obj/jev/perception/related_match_count", related_match_count);
  ros::param::get("/jev_obj/jev/perception/valid_mask_count", valid_mask_count);
  ros::param::get("/jev_obj/jev/perception/successful_object_ingestion_count", ingested_object_count);
  ros::param::get("/jev_obj/jev/perception/detector_status_json", detector_status_json);
  json detector_sources = json::array();
  if (!detector_status_json.empty()) {
    detector_sources = json::parse(detector_status_json, nullptr, false);
    if (detector_sources.is_discarded()) detector_sources = json::array();
  }
  double latest_itm_score = 0.0;
  const bool itm_score_present = ros::param::get("/jev_obj/jev/latest_itm_score", latest_itm_score);
  const bool itm_synchronized = itm_observation_id == observation_id && map_itm_valid;
  const bool itm_valid = itm_available && !itm_fallback && itm_score_present &&
      itm_synchronized && std::isfinite(latest_itm_score);
  if (jev_recent_perception_observation_ids_.empty() ||
      jev_recent_perception_observation_ids_.back() != observation_id) {
    jev_recent_perception_observation_ids_.push_back(observation_id);
    jev_recent_itm_scores_.push_back(itm_valid ? latest_itm_score : 0.0);
    jev_recent_itm_validity_.push_back(itm_valid ? 1 : 0);
    jev_recent_target_match_counts_.push_back(target_match_count);
    jev_recent_valid_mask_counts_.push_back(valid_mask_count);
    constexpr size_t kRecentPerceptionLimit = 12;
    if (jev_recent_perception_observation_ids_.size() > kRecentPerceptionLimit) {
      jev_recent_perception_observation_ids_.erase(
          jev_recent_perception_observation_ids_.begin());
      jev_recent_itm_scores_.erase(jev_recent_itm_scores_.begin());
      jev_recent_itm_validity_.erase(jev_recent_itm_validity_.begin());
      jev_recent_target_match_counts_.erase(jev_recent_target_match_counts_.begin());
      jev_recent_valid_mask_counts_.erase(jev_recent_valid_mask_counts_.begin());
    }
  }
  json recent_perception = json::array();
  for (size_t i = 0; i < jev_recent_perception_observation_ids_.size(); ++i) {
    recent_perception.push_back({
      { "observation_id", jev_recent_perception_observation_ids_[i] },
      { "image_text_match_valid", jev_recent_itm_validity_[i] != 0 },
      { "image_text_match_raw_score", jev_recent_itm_validity_[i] != 0 ?
          finiteOrNull(jev_recent_itm_scores_[i]) : json(nullptr) },
      { "target_match_count", jev_recent_target_match_counts_[i] },
      { "valid_mask_count", jev_recent_valid_mask_counts_[i] }
    });
  }
  snapshot["perception"] = {
    { "object_detection", {
        { "available", detection_available }, { "fallback", detection_fallback },
        { "backends", detection_backends }, { "observation_id", detection_observation_id },
        { "synchronized_with_map", detection_observation_id == observation_id },
        { "target_service_available", target_detection_available },
        { "related_service_available", related_detection_available },
        { "raw_detection_count", detection_count }, { "target_match_count", target_match_count },
        { "related_match_count", related_match_count }, { "valid_mask_count", valid_mask_count },
        { "successful_map_ingestion_count", ingested_object_count }, { "sources", detector_sources }
      } },
    { "image_text_match", {
        { "query_text", itm_query_text }, { "score_type", itm_score_type },
        { "raw_score", itm_valid ? finiteOrNull(latest_itm_score) : json(nullptr) },
        { "valid", itm_valid }, { "available", itm_available }, { "fallback", itm_fallback },
        { "backend", itm_backend }, { "observation_id", itm_observation_id },
        { "synchronized_with_map", itm_synchronized },
        { "invalid_reason", itm_valid ? json(nullptr) : json("missing_fallback_or_unsynchronized") }
      } }
  };
  snapshot["perception"]["recent_observations"] = recent_perception;
  snapshot["current_view"] = {
    { "query_text", itm_query_text }, { "score_type", itm_score_type },
    { "raw_score", itm_valid ? finiteOrNull(latest_itm_score) : json(nullptr) },
    { "valid", itm_valid }, { "backend", itm_backend }, { "observation_id", observation_id }
  };

  snapshot["object_selection_parameters"] = {
    { "min_confidence", object_map2d_->getMinConfidence() },
    { "min_observation_num", object_map2d_->getMinObservationNum() },
    { "fusion_type", object_map2d_->getFusionType() },
    { "use_observation_reduction", object_map2d_->getUseObservation() },
    { "function_score", "int(observation_cloud_sum * fused_confidence)" }
  };

  std::vector<ObjectEvidence> object_evidence;
  object_map2d_->getObjectEvidence(object_evidence);
  std::sort(object_evidence.begin(), object_evidence.end(),
      [](const ObjectEvidence& a, const ObjectEvidence& b) { return a.id < b.id; });
  snapshot["map"]["object_count"] = object_evidence.size();
  std::unordered_map<std::string, ObjectPathEvidence> normal_object_paths;
  json high_branch = json::array(), suspicious_branch = json::array();
  json extreme_original_branch = json::array(), extreme_relaxed_branch = json::array();
  for (const auto& object : object_evidence) {
    const std::string entity_id = "object-" + std::to_string(object.id);
    json class_scores = json::array();
    for (size_t label = 0; label < object.class_confidences.size(); ++label) {
      const std::string class_name = label == 0 ? target_object :
          label - 1 < related_objects.size() ? related_objects[label - 1] :
          "related-class-" + std::to_string(label);
      class_scores.push_back({
        { "class_index", label }, { "label", class_name },
        { "fused_confidence", finiteOrNull(object.class_confidences[label]) },
        { "observation_count", label < object.class_observations.size() ? object.class_observations[label] : 0 },
        { "observation_point_sum", label < object.class_point_sums.size() ? object.class_point_sums[label] : 0 },
        { "integer_function_score", label < object.class_function_scores.size() ? object.class_function_scores[label] : 0 }
      });
    }
    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> normal_cloud;
    object_map2d_->getObjectCloudForEvidence(object, false, normal_cloud);
    ObjectPathEvidence normal_path;
    Vector2d object_goal;
    std::vector<Vector2d> object_path;
    if (object.relaxed_eligible)
      searchObjectPath(pos, normal_cloud, object_goal, object_path, &normal_path);
    normal_object_paths[entity_id] = normal_path;
    std::vector<std::string> fail_reasons;
    if (object.computed_best_label != 0) fail_reasons.push_back("computed_best_label_is_not_target");
    if (object.target_confidence < object_map2d_->getMinConfidence())
      fail_reasons.push_back("target_confidence_below_threshold");
    if (object.target_observations < object_map2d_->getMinObservationNum())
      fail_reasons.push_back("target_observations_below_threshold");
    if (!object.relaxed_eligible)
      fail_reasons.push_back("target_confidence_not_above_relaxed_threshold_0.01");
    if (confirmation_block_active && blocked_target_id == entity_id)
      fail_reasons.push_back("post_arrival_multiview_confirmation_failed_cooldown");
    json object_json = {
      { "id", entity_id }, { "position", pointJson(object.center) },
      { "stored_best_label", object.stored_best_label },
      { "computed_best_label", object.computed_best_label },
      { "target_confidence", finiteOrNull(object.target_confidence) },
      { "target_observations", object.target_observations },
      { "target_point_count", object.target_point_count },
      { "class_scores", class_scores },
      { "geometry", { { "bbox_min", pointJson(object.box_min) },
          { "bbox_max", pointJson(object.box_max) } } },
      { "selection", { { "high_confidence_eligible", object.high_confidence_eligible },
          { "relaxed_eligible", object.relaxed_eligible },
          { "post_arrival_confirmation_blocked",
              confirmation_block_active && blocked_target_id == entity_id },
          { "failure_reasons", fail_reasons } } },
      { "provenance", { { "episode_id", object.last_episode_id },
          { "observation_id", object.last_observation_id },
          { "timestamp_ms", object.last_observation_timestamp_ms },
          { "historical_map_evidence", object.last_observation_id != static_cast<uint64_t>(observation_id) } } }
    };
    if (!region_graph_profile) {
      object_json["geometry"]["cells"] = pointsJson(object.cells);
      object_json["geometry"]["good_cells"] = pointsJson(object.good_cells);
      object_json["geometry"]["good_cell_seen_counts"] = object.good_cell_seen_counts;
      object_json["normal_path_search"] = objectPathJson(normal_path);
    }
    snapshot["objects"].push_back(object_json);
  }

  auto appendBranch = [&](const std::string& name, const std::string& condition,
                          const json& entity_results) {
    snapshot["search_branches"].push_back({
      { "name", name }, { "stage_condition", condition }, { "entity_results", entity_results }
    });
  };
  for (const auto& object : object_evidence) {
    const std::string entity_id = "object-" + std::to_string(object.id);
    const auto& normal_path = normal_object_paths.at(entity_id);
    const bool normal_reachable = !normal_path.path.empty();
    const json normal_result = { { "entity_id", entity_id },
      { "eligible", object.relaxed_eligible }, { "search", objectPathJson(normal_path) } };
    if (object.high_confidence_eligible) high_branch.push_back(normal_result);
    else if (object.relaxed_eligible) suspicious_branch.push_back(normal_result);
    if (normal_reachable) {
      candidates.push_back({ entity_id + "-goal", entity_id, "approach_object",
        normal_path.approach_point, normal_path.path, object.target_confidence, 0.0, false,
        object.high_confidence_eligible ? SEARCH_BEST_OBJECT : SEARCH_SUSPICIOUS_OBJECT,
        object.high_confidence_eligible ? "high_confidence_object" : "suspicious_object_if_no_active_path",
        object.high_confidence_eligible ? 1 : 4, "normal" });
    }
    if (region_graph_profile || !object.relaxed_eligible) continue;
    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> good_cloud, all_cloud;
    object_map2d_->getObjectCloudForEvidence(object, false, good_cloud);
    ObjectPathEvidence extreme_original;
    Vector2d ignored_goal;
    std::vector<Vector2d> ignored_path;
    searchObjectPathExtreme(pos, good_cloud, ignored_goal, ignored_path, &extreme_original);
    extreme_original_branch.push_back({ { "entity_id", entity_id },
      { "search", objectPathJson(extreme_original) }, { "selectable", false },
      { "exclusion_reason", "extreme_mode_bypasses_occupancy_checks" } });
  }
  const bool has_relaxed_objects = !region_graph_profile && std::any_of(object_evidence.begin(), object_evidence.end(),
      [](const ObjectEvidence& object) { return object.relaxed_eligible; });
  if (has_relaxed_objects) {
    // This mirrors getTopConfidenceObjectCloud(false, true): when any target has
    // confidence > 0.01 the second extreme branch repeats the same good-cell clouds.
    for (const auto& object : object_evidence) {
      if (!object.relaxed_eligible) continue;
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> good_cloud;
      object_map2d_->getObjectCloudForEvidence(object, false, good_cloud);
      ObjectPathEvidence extreme_relaxed;
      Vector2d ignored_goal;
      std::vector<Vector2d> ignored_path;
      searchObjectPathExtreme(pos, good_cloud, ignored_goal, ignored_path, &extreme_relaxed);
      extreme_relaxed_branch.push_back({
        { "entity_ids", { "object-" + std::to_string(object.id) } },
        { "geometry_mode", "good_cells_per_object" },
        { "search", objectPathJson(extreme_relaxed) }, { "selectable", false },
        { "exclusion_reason", "collision_safety_unknown" }
      });
    }
  }
  else if (!region_graph_profile && !object_evidence.empty()) {
    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> merged_all_cells(
        new pcl::PointCloud<pcl::PointXYZ>());
    json source_ids = json::array();
    for (const auto& object : object_evidence) {
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> all_cloud;
      object_map2d_->getObjectCloudForEvidence(object, true, all_cloud);
      *merged_all_cells += *all_cloud;
      source_ids.push_back("object-" + std::to_string(object.id));
    }
    ObjectPathEvidence extreme_relaxed;
    Vector2d ignored_goal;
    std::vector<Vector2d> ignored_path;
    searchObjectPathExtreme(
        pos, merged_all_cells, ignored_goal, ignored_path, &extreme_relaxed);
    extreme_relaxed_branch.push_back({ { "entity_ids", source_ids },
      { "geometry_mode", "merged_all_cells_low_confidence_fallback" },
      { "search", objectPathJson(extreme_relaxed) }, { "selectable", false },
      { "exclusion_reason", "collision_safety_unknown" } });
  }
  appendBranch("high_confidence_object_normal", "always checked first", high_branch);
  appendBranch("suspicious_object_normal",
      "only after no active frontier path in original policy", suspicious_branch);

  json over_depth_branch = json::array();
  if (object_map2d_->over_depth_object_cloud_ && !object_map2d_->over_depth_object_cloud_->empty()) {
    *jev_cached_over_depth_cloud_ = *object_map2d_->over_depth_object_cloud_;
    jev_cached_over_depth_observation_id_ = observation_id;
    ObjectPathEvidence evidence;
    Vector2d goal;
    std::vector<Vector2d> path;
    const bool reachable = searchObjectPath(
        pos, object_map2d_->over_depth_object_cloud_, goal, path, &evidence);
    snapshot["objects"].push_back({ { "id", "over-depth-current" },
      { "label", target_object }, { "score_available", false },
      { "geometry_point_count", object_map2d_->over_depth_object_cloud_->size() },
      { "normal_path_search", objectPathJson(evidence) },
      { "provenance", { { "episode_id", episode_id }, { "observation_id", observation_id } } } });
    over_depth_branch.push_back({ { "entity_id", "over-depth-current" },
      { "search", objectPathJson(evidence) } });
    if (reachable) candidates.push_back({ "over-depth-current-goal", "over-depth-current",
      "approach_object", goal, path, 0.0, 0.0, false, SEARCH_OVER_DEPTH_OBJECT,
      "current_over_depth_object", 2, "normal" });
  }
  appendBranch("current_over_depth_object", "after no high-confidence object path",
      over_depth_branch);

  struct FrontierRecord {
    std::string id;
    Vector2d frontier;
    Vector2d goal;
    std::vector<Vector2d> path;
    int search_result = Astar2D::NO_PATH;
    double semantic_mode_value = 0.0;
    double hybrid_mode_value = 0.0;
    double value_map_confidence = 0.0;
    bool semantic_mode_available = false;
    bool hybrid_mode_available = false;
    size_t cluster_size = 0;
    bool dormant = false;
    std::string dormancy_reason;
    std::vector<int> parent_ids;
    std::string lineage_event;
    json neighborhood = json::array();
  };
  std::vector<FrontierRecord> active_records, dormant_records;
  auto collectFrontiers = [&](const std::vector<Vector2d>& averages,
                              const std::vector<std::vector<Vector2d>>& clusters,
                              const std::vector<int>& stable_ids,
                              const std::vector<std::string>& dormancy_reasons,
                              const std::vector<std::vector<int>>& parent_ids,
                              const std::vector<std::string>& lineage_events,
                              bool dormant, std::vector<FrontierRecord>& records) {
    for (size_t index = 0; index < averages.size(); ++index) {
      FrontierRecord record;
      record.id = std::string("frontier-") +
          std::to_string(index < stable_ids.size() ? stable_ids[index] : static_cast<int>(index));
      record.frontier = averages[index];
      record.cluster_size = index < clusters.size() ? clusters[index].size() : 0;
      record.dormant = dormant;
      record.dormancy_reason = index < dormancy_reasons.size() ?
          dormancy_reasons[index] : "unknown";
      record.parent_ids = index < parent_ids.size() ? parent_ids[index] : std::vector<int>();
      record.lineage_event = index < lineage_events.size() ? lineage_events[index] : "unknown";
      searchFrontierPath(start, record.frontier, record.goal, record.path, &record.search_result);
      Vector2i center_index;
      sdf_map_->posToIndex(record.frontier, center_index);
      for (int dx = -2; dx <= 2; ++dx) {
        for (int dy = -2; dy <= 2; ++dy) {
          const Vector2i index2 = center_index + Vector2i(dx, dy);
          if (!sdf_map_->isInMap(index2)) {
            if (!region_graph_profile)
              record.neighborhood.push_back({ { "dx", dx }, { "dy", dy }, { "in_map", false } });
            continue;
          }
          const int occupancy = sdf_map_->getOccupancy(index2);
          const int inflated = sdf_map_->getInflateOccupancy(index2);
          const double value = sdf_map_->value_map_->getValue(index2);
          const double confidence = sdf_map_->value_map_->getConfidence(index2);
          if (!region_graph_profile)
            record.neighborhood.push_back({ { "dx", dx }, { "dy", dy }, { "in_map", true },
              { "occupancy", occupancy }, { "inflated", inflated },
              { "semantic_raw", finiteOrNull(value) }, { "fusion_weight", finiteOrNull(confidence) } });
          if (!std::isfinite(value) || (region_graph_profile && !regionSemanticSupported(value, confidence))) continue;
          if (!record.semantic_mode_available || value > record.semantic_mode_value) {
            record.semantic_mode_value = value;
            record.semantic_mode_available = true;
          }
          // Match getSortedSemanticFrontiers exactly: the center initializes the value even
          // when occupied; only neighboring occupied/inflated cells are skipped.
          const bool hybrid_cell_allowed = (dx == 0 && dy == 0) ||
              (inflated != 1 && occupancy != SDFMap2D::OCCUPIED);
          if (hybrid_cell_allowed &&
              (!record.hybrid_mode_available || value > record.hybrid_mode_value)) {
            record.hybrid_mode_value = value;
            record.value_map_confidence = confidence;
            record.hybrid_mode_available = true;
          }
        }
      }
      records.push_back(record);
    }
  };
  collectFrontiers(ed_->frontier_averages_, ed_->frontiers_, ed_->frontier_ids_,
      ed_->frontier_dormancy_reasons_, ed_->frontier_parent_ids_,
      ed_->frontier_lineage_events_, false, active_records);
  collectFrontiers(ed_->dormant_frontier_averages_, ed_->dormant_frontiers_,
      ed_->dormant_frontier_ids_, ed_->dormant_frontier_dormancy_reasons_,
      ed_->dormant_frontier_parent_ids_, ed_->dormant_frontier_lineage_events_, true,
      dormant_records);

  auto frontierStats = [&](const std::vector<FrontierRecord>& records) {
    std::vector<double> values;
    for (const auto& record : records)
      if (record.search_result == Astar2D::REACH_END && record.hybrid_mode_available &&
          std::isfinite(record.hybrid_mode_value))
        values.push_back(record.hybrid_mode_value);
    double mean = 0.0, std_dev = 0.0, max_to_mean = 1.0, maximum = 0.0;
    if (!values.empty()) {
      maximum = *std::max_element(values.begin(), values.end());
      mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
      double variance = 0.0;
      for (double value : values) variance += (value - mean) * (value - mean);
      std_dev = std::sqrt(variance / values.size());
      max_to_mean = std::abs(mean) > 1e-12 ? maximum / mean : 1.0;
    }
    return json{ { "total_count", records.size() }, { "reachable_count", values.size() },
      { "mean", mean }, { "std_dev", std_dev }, { "maximum", maximum },
      { "max_to_mean", max_to_mean }, { "zero_mean_ratio_definition", 1.0 },
      { "adaptive_max_to_mean_threshold", std::max(ep_->max_to_mean_threshold_,
          ep_->max_to_mean_percentage_ * max_to_mean) } };
  };
  snapshot["original_policy"]["active_frontier_statistics"] = frontierStats(active_records);
  snapshot["original_policy"]["dormant_frontier_statistics"] = frontierStats(dormant_records);
  const double active_mean = snapshot["original_policy"]["active_frontier_statistics"]["mean"];
  const double active_ratio = snapshot["original_policy"]["active_frontier_statistics"]["max_to_mean"];
  const double adaptive_threshold =
      snapshot["original_policy"]["active_frontier_statistics"]["adaptive_max_to_mean_threshold"];
  json high_semantic_subset = json::array();
  for (const auto& record : active_records) {
    const bool selected = record.search_result == Astar2D::REACH_END &&
        record.hybrid_mode_available && std::abs(active_mean) > 1e-12 &&
        record.hybrid_mode_value / active_mean >= adaptive_threshold;
    high_semantic_subset.push_back({ { "frontier_id", record.id }, { "selected", selected },
      { "value_to_mean", record.hybrid_mode_available && std::abs(active_mean) > 1e-12 ?
          finiteOrNull(record.hybrid_mode_value / active_mean) : json(nullptr) } });
  }
  snapshot["original_policy"]["high_semantic_subset"] = high_semantic_subset;
  snapshot["original_policy"]["hybrid_branch"] = ep_->policy_mode_ != ExplorationParam::HYBRID ?
      "not_applicable" :
      (snapshot["original_policy"]["active_frontier_statistics"]["std_dev"].get<double>() >
              ep_->sigma_threshold_ && active_ratio > ep_->max_to_mean_threshold_ ?
          "semantic_tsp" : "closest");

  auto appendFrontierRecords = [&](const std::vector<FrontierRecord>& records) {
    for (const auto& record : records) {
      const bool reachable = record.search_result == Astar2D::REACH_END && !record.path.empty();
      json frontier_json = {
        { "id", record.id }, { "position", pointJson(record.frontier) },
        { "state", record.dormant ? "dormant" : "active" },
        { "cluster_cells", record.cluster_size }, { "dormancy_reason", record.dormancy_reason },
        { "lineage", { { "event", record.lineage_event },
            { "parent_stable_ids", record.parent_ids } } },
        { "semantic_features", {
            { "semantic_policy_5x5_max_including_obstacles", record.semantic_mode_available ?
                finiteOrNull(record.semantic_mode_value) : json(nullptr) },
            { "hybrid_policy_5x5_center_then_max_excluding_occupied_and_inflated_neighbors",
                record.hybrid_mode_available ? finiteOrNull(record.hybrid_mode_value) : json(nullptr) },
            { "value_map_fov_fusion_weight_at_hybrid_max", record.hybrid_mode_available ?
                finiteOrNull(record.value_map_confidence) : json(nullptr) }
          } },
        { "information_gain", nullptr }, { "information_gain_status", "unknown_not_measured" },
        { "planning_status", searchResultName(record.search_result) },
        { "path", { { "reachable", reachable },
            { "distance_m", reachable ? finiteOrNull(pathLength(record.path)) : json(nullptr) },
            { "safety_mode", "normal" } } }
      };
      if (!region_graph_profile) {
        frontier_json["semantic_features"]["raw_neighborhood"] = record.neighborhood;
        frontier_json["path"]["points"] = pointsJson(record.path);
        snapshot["semantic_matches"].push_back({
          { "query", itm_query_text },
          { "query_type", "actual_image_text_query_projected_into_value_map" },
          { "entity_id", record.id }, { "raw_score", record.hybrid_mode_available ?
              finiteOrNull(record.hybrid_mode_value) : json(nullptr) },
          { "fusion_weight", record.hybrid_mode_available ?
              finiteOrNull(record.value_map_confidence) : json(nullptr) }
        });
        snapshot["candidate_evidence"].push_back({ { "target_entity_id", record.id },
          { "stage", record.dormant ? "dormant_frontier_policy" : "active_frontier_policy" },
          { "search_result", searchResultName(record.search_result) },
          { "path_points", pointsJson(record.path) }, { "selectable", reachable },
          { "exclusion_reason", reachable ? json(nullptr) : json(searchResultName(record.search_result)) } });
      }
      snapshot["frontiers"].push_back(frontier_json);
      if (reachable) candidates.push_back({ record.id + "-goal", record.id, "explore_frontier",
        record.goal, record.path, record.hybrid_mode_available ? record.hybrid_mode_value :
            std::numeric_limits<double>::quiet_NaN(), 0.0, false, EXPLORATION,
        record.dormant ? "dormant_frontier_policy" : "active_frontier_policy",
        record.dormant ? 5 : 3, "normal" });
    }
  };
  appendFrontierRecords(active_records);
  appendFrontierRecords(dormant_records);
  json active_ids = json::array(), dormant_ids = json::array();
  for (const auto& record : active_records) active_ids.push_back(record.id);
  for (const auto& record : dormant_records) dormant_ids.push_back(record.id);
  appendBranch("active_frontier_policy", "after object stages",
      json{ { "frontier_ids", active_ids } });
  appendBranch("dormant_frontier_policy",
      "only after no active frontier and no suspicious object path",
      json{ { "frontier_ids", dormant_ids } });
  appendBranch("original_object_extreme", "only after all normal branches fail",
      extreme_original_branch);
  appendBranch("relaxed_all_cells_extreme", "after original extreme branch fails",
      extreme_relaxed_branch);

  json cached_branch = json::array();
  if (jev_cached_over_depth_cloud_ && !jev_cached_over_depth_cloud_->empty()) {
    ObjectPathEvidence cached_extreme;
    Vector2d ignored_goal;
    std::vector<Vector2d> ignored_path;
    searchObjectPathExtreme(pos, jev_cached_over_depth_cloud_, ignored_goal, ignored_path,
        &cached_extreme);
    cached_branch.push_back({ { "entity_id", "over-depth-cache" },
      { "source_observation_id", jev_cached_over_depth_observation_id_ },
      { "search", objectPathJson(cached_extreme) }, { "selectable", false },
      { "exclusion_reason", "extreme_mode_collision_safety_unknown" } });
  }
  appendBranch("cached_over_depth_extreme", "final original-policy search branch", cached_branch);

  auto addPathGraph = [&](const std::string& name, const std::vector<FrontierRecord>& records) {
    std::vector<const FrontierRecord*> nodes;
    nodes.reserve(records.size());
    for (const auto& record : records) nodes.push_back(&record);
    json ids = json::array();
    ids.push_back("robot");
    for (const auto* record : nodes) ids.push_back(record->id);
    json matrix = json::array();
    for (size_t i = 0; i < ids.size(); ++i) {
      json row = json::array();
      for (size_t j = 0; j < ids.size(); ++j) row.push_back(nullptr);
      matrix.push_back(row);
    }
    auto costJson = [&](const Vector2d& from, const Vector2d& to) {
      PathCostEvidence evidence;
      computePathCost(from, to, &evidence);
      return json{ { "solver_cost", evidence.solver_cost },
        { "physical_cost_m", evidence.reachable ?
            finiteOrNull(evidence.physical_cost_m) : json(nullptr) },
        { "reachable", evidence.reachable },
        { "search_result", searchResultName(evidence.search_result) },
        { "solver_penalty", evidence.solver_penalty } };
    };
    for (size_t i = 0; i < ids.size(); ++i) {
      matrix[i][i] = { { "solver_cost", 100000.0 }, { "physical_cost_m", nullptr },
        { "kind", "solver_diagonal_sentinel" } };
      if (i > 0) matrix[i][0] = { { "solver_cost", 0.0 }, { "physical_cost_m", nullptr },
        { "kind", "solver_return_to_start_construct" } };
    }
    for (size_t j = 1; j < ids.size(); ++j)
      matrix[0][j] = costJson(start, nodes[j - 1]->frontier);
    for (size_t i = 1; i < ids.size(); ++i) {
      for (size_t j = i + 1; j < ids.size(); ++j) {
        const json shared_cost = costJson(nodes[i - 1]->frontier, nodes[j - 1]->frontier);
        matrix[i][j] = shared_cost;
        matrix[j][i] = shared_cost;
      }
    }
    snapshot["original_policy"][name] = { { "node_ids", ids }, { "cost_matrix", matrix },
      { "matrix_semantics",
        "ATSP construction: diagonal=100000, return-to-robot=0, unreachable=10000" } };
  };
  if (!region_graph_profile) {
    addPathGraph("active_frontier_path_graph", active_records);
    addPathGraph("dormant_frontier_path_graph", dormant_records);
  }

  if (confirmation_block_active) {
    const size_t before = candidates.size();
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [&](const JevCandidate& candidate) {
          return candidate.id == blocked_candidate_id;
        }), candidates.end());
    if (candidates.size() != before) {
      ROS_WARN("[Jev] Suppressed post-arrival-unconfirmed candidate %s through observation %d",
          blocked_candidate_id.c_str(), blocked_until_observation_id);
    }
  }

  const bool reliable_target_evidence = detection_observation_id == observation_id &&
      target_detection_available &&
      std::any_of(object_evidence.begin(), object_evidence.end(),
          [](const ObjectEvidence& object) { return object.high_confidence_eligible; });
  snapshot["perception"]["reliable_target_evidence_available"] = reliable_target_evidence;
  // In-place observation is an emergency acquisition action, not a peer of
  // executable movement goals. Offering it whenever target evidence is absent
  // lets a high-level model oscillate left/right despite reachable frontiers.
  if (candidates.empty()) {
    const double turn_angle = observationTurnAngleRad();
    const std::vector<Vector2d> observation_path{ start };
    snapshot["observation_actions"].push_back({
      { "id", "observe-left" }, { "direction", "left" },
      { "rotation_rad", turn_angle }, { "expected_yaw", normalizeAngle(yaw + turn_angle) },
      { "position_unchanged", true }, { "information_gain", nullptr },
      { "information_gain_status", "unknown_not_measured" },
      { "purpose", "rotate in place to acquire a new synchronized observation" }
    });
    snapshot["observation_actions"].push_back({
      { "id", "observe-right" }, { "direction", "right" },
      { "rotation_rad", -turn_angle }, { "expected_yaw", normalizeAngle(yaw - turn_angle) },
      { "position_unchanged", true }, { "information_gain", nullptr },
      { "information_gain_status", "unknown_not_measured" },
      { "purpose", "rotate in place to acquire a new synchronized observation" }
    });
    candidates.push_back({ "observe-left-goal", "observe-left", "observe_rotation", start,
      observation_path, std::numeric_limits<double>::quiet_NaN(), 0.0, false,
      JEV_OBSERVE_LEFT, "active_observation", 3, "in_place" });
    candidates.push_back({ "observe-right-goal", "observe-right", "observe_rotation", start,
      observation_path, std::numeric_limits<double>::quiet_NaN(), 0.0, false,
      JEV_OBSERVE_RIGHT, "active_observation", 3, "in_place" });
  }

  if (candidates.empty())
    return JEV_PLAN_NONRETRYABLE_FAILURE;
  if (!region_graph_profile) for (const auto& candidate : candidates) {
    const double length = pathLength(candidate.path);
    if (!std::isfinite(length) || !std::isfinite(candidate.goal.x()) ||
        !std::isfinite(candidate.goal.y()))
      return JEV_PLAN_NONRETRYABLE_FAILURE;
    double goal_yaw = std::atan2(candidate.goal.y() - start.y(), candidate.goal.x() - start.x());
    if (candidate.kind == "observe_rotation") {
      const double direction = candidate.target_id == "observe-left" ? 1.0 : -1.0;
      goal_yaw = normalizeAngle(yaw + direction * observationTurnAngleRad());
    }
    snapshot["candidates"].push_back({
      { "id", candidate.id }, { "kind", candidate.kind },
      { "target_id", candidate.target_id },
      { "goal_pose", { { "x", candidate.goal.x() }, { "y", candidate.goal.y() },
                         { "yaw", goal_yaw } } },
      { "path", { { "reachable", true }, { "collision_free", true },
                    { "distance_m", length }, { "points", pointsJson(candidate.path) } } },
      { "semantic_score", finiteOrNull(candidate.semantic_score) },
      { "information_gain", candidate.information_gain_available ?
          finiteOrNull(candidate.information_gain) : json(nullptr) },
      { "information_gain_status", candidate.information_gain_available ?
          "proxy" : "unknown_not_measured" },
      { "source_stage", candidate.source_stage },
      { "source_priority", candidate.source_priority },
      { "safety_mode", candidate.safety_mode }
    });
  }

  if (!region_graph_profile) {
    snapshot["map"]["decision_query_evidence"] = {
      { "frontier_neighborhoods_embedded", true }, { "object_geometry_embedded", true },
      { "path_points_embedded", true }, { "raw_semantic_values_preserved", true }
    };
    finalizeJevSnapshot(snapshot, { snapshot["objects"].size(), active_records.size(),
        dormant_records.size(), active_records.size() + dormant_records.size(),
        candidates.size() });
  }

  if (region_graph_profile) {
    if (!region_graph_) return JEV_PLAN_NONRETRYABLE_FAILURE;
    if (!jev_region_grid_.valid()) jev_region_grid_ = RegionGraph2D::capture(*sdf_map_);
    else if (!RegionGraph2D::refresh(*sdf_map_, jev_region_grid_))
      return JEV_PLAN_NONRETRYABLE_FAILURE;
    const RegionGridSnapshot& region_grid = jev_region_grid_;
    if (!region_graph_->update(region_grid, revision, navigation_step_count, start) ||
        region_graph_->robotRegionId().empty()) {
      ROS_ERROR("[Jev] Failed to build region graph or assign robot to free space");
      return JEV_PLAN_NONRETRYABLE_FAILURE;
    }
    std::vector<std::string> target_backends;
    for (const auto& source : detector_sources)
      if (source.value("target_coverage", false)) target_backends.push_back(source.value("backend",std::string("unknown")));
    json region_state = {
      { "schema_version", "jev-region-graph/1.1" },
      { "representation", {
          { "kind", "region_graph" }, { "lossy", true },
          { "region_definition", "known-free four-connected component within configured fixed world tiles; not a semantic room" },
          { "parameters", { { "region_tile_size_m", region_graph_->tileSize() },
              { "tile_origin_xy", {0.0,0.0} }, { "native_resolution_m", region_grid.resolution },
              { "connectivity", 4 },
              { "localization_tolerance_m", region_graph_->localizationTolerance() },
              { "history_window", 1 } } },
          { "frame_id", jev_frame_id_ },
          { "units", { { "position", "m" }, { "yaw", "rad" },
              { "score", "raw similarity, not probability" } } },
          { "precision", { { "position_decimals", 3 }, { "score_decimals", 4 },
              { "local_execution_uses_full_precision", true } } }
        } },
      { "identity", { { "episode_id", episode_id }, { "request_id", request_id },
          { "observation_id", observation_id }, { "map_revision", revision },
          { "region_graph_revision", region_graph_->graphRevision() } } },
      { "task", { { "target_object", target_object },
          { "room_prior", room_prior.empty() || room_prior == "everywhere" ? json(nullptr) : json(room_prior) },
          { "related_objects", related_objects },
          { "prior_source", "runtime_task_metadata_not_dataset_ground_truth" } } },
      { "robot", { { "region_id", region_graph_->robotRegionId() },
          { "region_assignment", region_graph_->robotLocalizationUsedTolerance() ?
              "nearest_verified_free_cell_within_tolerance" : "exact_free_cell" },
          { "pose", { { "x", rounded(pos.x(), 3) }, { "y", rounded(pos.y(), 3) },
              { "yaw", rounded(yaw, 3) } } } } },
          { "perception", {
          { "image_text_match", snapshot["perception"]["image_text_match"] },
          { "target_detection", {
              { "available", target_detection_available }, { "fallback", detection_fallback },
              { "backend", target_backends.empty() ? "unknown" : target_backends.front() },
              { "matched_boxes", target_match_count }, { "valid_masks", valid_mask_count },
              { "observation_id", detection_observation_id }
            } },
          { "related_detection", { { "requested", !related_objects.empty() },
              { "available", related_objects.empty() ? json(nullptr) : json(related_detection_available) } } },
          { "recent_observations", recent_perception }
        } },
      { "jev_obj_policy", {
          { "mode", snapshot["original_policy"]["mode"] },
          { "mode_id", snapshot["original_policy"]["mode_id"] },
          { "decision_order", snapshot["original_policy"]["decision_order"] },
          { "semantic_thresholds", snapshot["original_policy"]["semantic_thresholds"] },
          { "active_frontier_statistics",
              snapshot["original_policy"]["active_frontier_statistics"] },
          { "dormant_frontier_statistics",
              snapshot["original_policy"]["dormant_frontier_statistics"] },
          { "high_semantic_subset", snapshot["original_policy"]["high_semantic_subset"] },
          { "hybrid_branch", snapshot["original_policy"]["hybrid_branch"] },
          { "available_stages", json::array() },
          { "earliest_available_stage", nullptr },
          { "earliest_available_priority", nullptr },
          { "interpretation", "The local planner gate is a strong default, not a hard filter. When the earliest active stage is a frontier stage, below-threshold historical object hypotheses are deferred and should be overridden only by fresh synchronized target evidence or when no active-stage option remains." }
        } },
      { "object_confirmation", snapshot["object_confirmation"] },
      { "regions", json::array() }, { "connections", json::array() },
      { "observation_actions", json::array() }, { "options", json::array() },
      { "unassigned_entities", { { "objects", json::array() }, { "frontiers", json::array() } } }
    };
    std::unordered_map<std::string, size_t> region_indices;
    for (const auto& region : region_graph_->regions()) {
      const size_t index = region_state["regions"].size();
      region_indices[region.id] = index;
      region_state["regions"].push_back({
        { "id", region.id },
        { "bbox_xy", { { "min", { rounded(region.bbox_min.x(), 3), rounded(region.bbox_min.y(), 3) } },
            { "max", { rounded(region.bbox_max.x(), 3), rounded(region.bbox_max.y(), 3) } } } },
        { "anchor_xy", { rounded(region.anchor.x(), 3), rounded(region.anchor.y(), 3) } },
        { "free_area_m2", rounded(region.free_area_m2, 3) },
        { "visited", region.visited },
        { "last_robot_visit_step", region.last_robot_visit_step >= 0 ?
            json(region.last_robot_visit_step) : json(nullptr) },
        { "visit_count", region.visit_count },
        { "lineage", { { "parent_ids", region.parent_ids } } },
        { "semantic", {
            { "valid", region.semantic.valid },
            { "mean", region.semantic.valid ? finiteOrNull(rounded(region.semantic.weighted_mean, 4)) : json(nullptr) },
            { "p90", region.semantic.valid ? finiteOrNull(rounded(region.semantic.p90, 4)) : json(nullptr) },
            { "max", region.semantic.valid ? finiteOrNull(rounded(region.semantic.maximum, 4)) : json(nullptr) },
            { "valid_area_fraction", rounded(region.semantic.valid_area_fraction, 4) },
            { "aggregation", "fov_weighted_mean; unweighted_nearest_rank_p90; known_free_noninflated_cells_only" }
          } },
        { "objects", json::array() }, { "frontiers", json::array() }
      });
    }
    for (const auto& edge : region_graph_->edges()) {
      region_state["connections"].push_back({
        { "id", edge.id }, { "from", edge.from }, { "to", edge.to },
        { "bidirectional", edge.bidirectional }, { "status", "verified_reachable" },
        { "anchor_path_length_m", nullptr },
        { "portal_width_estimate_m", nullptr },
        { "min_clearance_m", finiteOrNull(rounded(edge.min_clearance_m, 3)) },
        { "verified_map_revision", edge.verified_map_revision }
      });
    }

    std::unordered_map<std::string, std::string> candidate_regions;
    std::unordered_map<std::string, std::vector<std::string>> candidate_routes;
    std::unordered_map<std::string, std::string> candidate_route_basis;
    std::unordered_map<std::string, std::string> candidate_region_assignment;
    std::unordered_set<std::string> rejected_region_targets;
    auto retainRegionCandidates = [&]() {
      std::vector<JevCandidate> verified;
      candidate_regions.clear();
      candidate_routes.clear();
      candidate_route_basis.clear();
      candidate_region_assignment.clear();
      for (const auto& candidate : candidates) {
        std::string region_id = region_graph_->regionForPosition(region_grid, candidate.goal);
        std::string region_assignment = "exact_goal_cell";
        if (region_id.empty()) {
          for (auto point = candidate.path.rbegin(); point != candidate.path.rend(); ++point) {
            region_id = region_graph_->regionForPosition(region_grid, *point);
            if (!region_id.empty()) {
              region_assignment = "last_represented_local_path_cell";
              break;
            }
          }
        }
        std::vector<std::string> route = region_graph_->routeRegions(region_grid, candidate.path);
        std::string route_basis = "exact_local_path_projection";
        if (route.empty() || route.front() != region_graph_->robotRegionId() ||
            (!region_id.empty() && route.back() != region_id)) {
          route = region_graph_->routeBetweenRegions(region_graph_->robotRegionId(), region_id);
          route_basis = "verified_region_graph_route_for_local_path";
        }
        if (region_id.empty() || route.empty() ||
            route.front() != region_graph_->robotRegionId() ||
            route.back() != region_id) {
          rejected_region_targets.insert(candidate.target_id);
          ROS_WARN("[Jev] Excluding candidate %s from region graph: goal_region=%s "
                   "route_size=%zu robot_region=%s",
              candidate.id.c_str(), region_id.c_str(), route.size(),
              region_graph_->robotRegionId().c_str());
          continue;
        }
        candidate_regions[candidate.id] = region_id;
        candidate_routes[candidate.id] = route;
        candidate_route_basis[candidate.id] = route_basis;
        candidate_region_assignment[candidate.id] = region_assignment;
        verified.push_back(candidate);
      }
      candidates.swap(verified);
    };
    retainRegionCandidates();
    if (candidates.empty()) {
      const double turn_angle = observationTurnAngleRad();
      const std::vector<Vector2d> observation_path{ start };
      if (snapshot["observation_actions"].empty()) {
        snapshot["observation_actions"].push_back({
          { "id", "observe-left" }, { "direction", "left" },
          { "rotation_rad", turn_angle }, { "expected_yaw", normalizeAngle(yaw + turn_angle) },
          { "position_unchanged", true }, { "information_gain", nullptr },
          { "information_gain_status", "unknown_not_measured" },
          { "purpose", "rotate in place after no candidate has a verified region route" }
        });
        snapshot["observation_actions"].push_back({
          { "id", "observe-right" }, { "direction", "right" },
          { "rotation_rad", -turn_angle }, { "expected_yaw", normalizeAngle(yaw - turn_angle) },
          { "position_unchanged", true }, { "information_gain", nullptr },
          { "information_gain_status", "unknown_not_measured" },
          { "purpose", "rotate in place after no candidate has a verified region route" }
        });
      }
      candidates.push_back({ "observe-left-goal", "observe-left", "observe_rotation", start,
        observation_path, std::numeric_limits<double>::quiet_NaN(), 0.0, false,
        JEV_OBSERVE_LEFT, "active_observation", 3, "in_place" });
      candidates.push_back({ "observe-right-goal", "observe-right", "observe_rotation", start,
        observation_path, std::numeric_limits<double>::quiet_NaN(), 0.0, false,
        JEV_OBSERVE_RIGHT, "active_observation", 3, "in_place" });
      retainRegionCandidates();
    }
    if (candidates.empty()) {
      ROS_WARN("[Jev] No candidate has a verified region route; retry with a newer map");
      return JEV_PLAN_RETRYABLE_FAILURE;
    }

    int earliest_stage_priority = std::numeric_limits<int>::max();
    std::vector<std::string> available_stages;
    for (const auto& candidate : candidates) {
      earliest_stage_priority = std::min(earliest_stage_priority, candidate.source_priority);
      if (std::find(available_stages.begin(), available_stages.end(), candidate.source_stage) ==
          available_stages.end())
        available_stages.push_back(candidate.source_stage);
    }
    std::sort(available_stages.begin(), available_stages.end(), [&](const std::string& left,
                                                                    const std::string& right) {
      auto priorityFor = [&](const std::string& stage) {
        int priority = std::numeric_limits<int>::max();
        for (const auto& candidate : candidates)
          if (candidate.source_stage == stage)
            priority = std::min(priority, candidate.source_priority);
        return priority;
      };
      return priorityFor(left) < priorityFor(right);
    });
    region_state["jev_obj_policy"]["available_stages"] = available_stages;
    region_state["jev_obj_policy"]["earliest_available_priority"] = earliest_stage_priority;
    region_state["jev_obj_policy"]["earliest_available_stage"] =
        available_stages.empty() ? json(nullptr) : json(available_stages.front());

    std::unordered_map<std::string, const JevCandidate*> candidate_by_target;
    for (const auto& candidate : candidates)
      if (!candidate_by_target.count(candidate.target_id)) candidate_by_target[candidate.target_id] = &candidate;
    for (const auto& object : snapshot["objects"]) {
      const std::string entity_id = object.value("id", std::string());
      const auto candidate_it = candidate_by_target.find(entity_id);
      std::string assigned_region;
      if (candidate_it != candidate_by_target.end()) {
        const auto region_it = candidate_regions.find(candidate_it->second->id);
        if (region_it != candidate_regions.end()) assigned_region = region_it->second;
      }
      if (assigned_region.empty() && object.contains("position") && object["position"].is_object()) {
        assigned_region = region_graph_->regionForPosition(region_grid,
            Vector2d(object["position"].value("x", -1e9), object["position"].value("y", -1e9)));
      }
      const bool selectable = candidate_it != candidate_by_target.end() && !assigned_region.empty();
      const auto geometry = object.value("geometry", json::object());
      const auto provenance = object.value("provenance", json::object());
      const auto selection = object.value("selection", json::object());
      const bool confirmation_blocked = confirmation_block_active &&
          blocked_target_id == entity_id;
      const bool updated_current = provenance.value("observation_id", -1) == observation_id;
      const bool high_confidence_eligible =
          selection.value("high_confidence_eligible", false);
      const bool current_frame_target_supported = updated_current &&
          detection_observation_id == observation_id && target_detection_available &&
          !detection_fallback && target_match_count > 0 && valid_mask_count > 0;
      const std::string object_selection_status = !selectable ? "not_selectable" :
          (high_confidence_eligible && current_frame_target_supported ?
              "current_frame_high_confidence_target_with_verified_path" :
           high_confidence_eligible ?
              "historical_high_confidence_target_with_verified_path" :
              "deferred_weak_target_hypothesis_with_verified_path");
      const std::string confirmation_status = confirmation_blocked ? "rejected_cooldown" :
          (current_frame_target_supported ? "not_arrived_current_frame_supported" :
              "not_arrived_historical_or_unsynchronized_only");
      json summary = {
        { "id", entity_id },
        { "center_xy", object.contains("position") ?
            json::array({ rounded(object["position"].value("x", 0.0), 3),
                          rounded(object["position"].value("y", 0.0), 3) }) : json(nullptr) },
        { "bbox_xy", geometry.contains("bbox_min") && geometry.contains("bbox_max") ?
            json{ { "min", { rounded(geometry["bbox_min"].value("x", 0.0), 3),
                               rounded(geometry["bbox_min"].value("y", 0.0), 3) } },
                  { "max", { rounded(geometry["bbox_max"].value("x", 0.0), 3),
                               rounded(geometry["bbox_max"].value("y", 0.0), 3) } } } : json(nullptr) },
        { "region_assignment", selectable ? "primary_safe_approach" :
            (!assigned_region.empty() ? "observed_geometry" : "unassigned") },
        { "target_confidence", object.value("target_confidence", json(nullptr)) },
        { "target_observations", object.value("target_observations", json(nullptr)) },
        { "target_point_count", object.value("target_point_count", json(nullptr)) },
        { "stored_best_label", object.value("stored_best_label", json(nullptr)) },
        { "computed_best_label", object.value("computed_best_label", json(nullptr)) },
        { "high_confidence_eligible", high_confidence_eligible },
        { "relaxed_eligible", selection.value("relaxed_eligible", false) },
        { "post_arrival_confirmation_status", confirmation_status },
        { "class_evidence", object.value("class_scores", json::array()) },
        { "selection_status", object_selection_status },
        { "not_selectable_reason", selectable ? json(nullptr) :
            (confirmation_blocked ?
                json::array({ "post_arrival_multiview_confirmation_failed_cooldown" }) :
             rejected_region_targets.count(entity_id) ?
                json::array({ "candidate_path_not_representable_in_region_graph" }) :
                selection.value("failure_reasons", json::array())) },
        { "last_observed_id", provenance.value("observation_id", json(nullptr)) },
        { "observations_since_seen", provenance.contains("observation_id") ?
            json(std::max(0, observation_id - provenance.value("observation_id", observation_id))) : json(nullptr) },
        { "updated_in_current_observation", updated_current },
        { "provisional", entity_id == "over-depth-current" },
        { "selectable", selectable }
      };
      if (!assigned_region.empty() && region_indices.count(assigned_region))
        region_state["regions"][region_indices[assigned_region]]["objects"].push_back(summary);
      else
        region_state["unassigned_entities"]["objects"].push_back(summary);
    }
    for (const auto& frontier : snapshot["frontiers"]) {
      const std::string entity_id = frontier.value("id", std::string());
      const auto candidate_it = candidate_by_target.find(entity_id);
      std::string assigned_region;
      if (candidate_it != candidate_by_target.end()) {
        const auto region_it = candidate_regions.find(candidate_it->second->id);
        if (region_it != candidate_regions.end()) assigned_region = region_it->second;
      }
      if (assigned_region.empty() && frontier.contains("position"))
        assigned_region = region_graph_->regionForPosition(region_grid,
            Vector2d(frontier["position"].value("x", -1e9), frontier["position"].value("y", -1e9)));
      const auto features = frontier.value("semantic_features", json::object());
      const json score = features.value(
          "hybrid_policy_5x5_center_then_max_excluding_occupied_and_inflated_neighbors", json(nullptr));
      const bool semantic_valid = score.is_number() && std::isfinite(score.get<double>());
      const bool selectable = candidate_it != candidate_by_target.end() && !assigned_region.empty();
      json summary = {
        { "id", entity_id }, { "state", frontier.value("state", "unknown") },
        { "position_xy", frontier.contains("position") ?
            json::array({ rounded(frontier["position"].value("x", 0.0), 3),
                          rounded(frontier["position"].value("y", 0.0), 3) }) : json(nullptr) },
        { "approach_region_id", assigned_region.empty() ? json(nullptr) : json(assigned_region) },
        { "semantic_score", semantic_valid ? finiteOrNull(rounded(score.get<double>(), 4)) : json(nullptr) },
        { "semantic_valid", semantic_valid }, { "semantic_method", "hybrid_5x5" },
        { "fusion_weight", features.value("value_map_fov_fusion_weight_at_hybrid_max", json(nullptr)) },
        { "cluster_cells", frontier.value("cluster_cells", 0) },
        { "unknown_boundary_length_m", nullptr }, { "information_gain", nullptr },
        { "reachability", frontier.value("planning_status", "unknown") },
        { "selectable", selectable },
        { "not_selectable_reason", selectable ? json(nullptr) :
            frontier.value("planning_status", json("no_verified_candidate")) }
      };
      if (!assigned_region.empty() && region_indices.count(assigned_region))
        region_state["regions"][region_indices[assigned_region]]["frontiers"].push_back(summary);
      else
        region_state["unassigned_entities"]["frontiers"].push_back(summary);
    }
    for (const auto& action : snapshot["observation_actions"]) {
      region_state["observation_actions"].push_back({
        { "id", action.value("id", std::string()) },
        { "region_id", region_graph_->robotRegionId() },
        { "direction", action.value("direction", std::string()) },
        { "rotation_rad", action.value("rotation_rad", 0.0) },
        { "expected_yaw", action.value("expected_yaw", 0.0) },
        { "selectable", candidate_by_target.count(action.value("id", std::string())) > 0 }
      });
    }
    json local_candidates = json::array();
    auto recentSelectionCount = [&](const std::string& target_id) {
      const size_t window = 12;
      const size_t begin = jev_selected_entity_history_.size() > window ?
          jev_selected_entity_history_.size() - window : 0;
      int count = 0;
      for (size_t index = begin; index < jev_selected_entity_history_.size(); ++index)
        if (jev_selected_entity_history_[index] == target_id) ++count;
      return count;
    };
    for (const auto& candidate : candidates) {
      const std::string& region_id = candidate_regions.at(candidate.id);
      const std::vector<std::string>& route = candidate_routes.at(candidate.id);
      const bool approach = candidate.kind == "approach_object";
      const std::string option_id = (approach ? "approach:" : "explore:") + candidate.id;
      region_state["options"].push_back({
        { "id", option_id }, { "action_type", approach ? "approach_object" : "explore" },
        { "exploration_kind", candidate.kind == "observe_rotation" ? "observe_rotation" : "frontier" },
        { "target_id", candidate.target_id }, { "region_id", region_id },
        { "distance_m", rounded(pathLength(candidate.path), 3) },
        { "route_region_ids", route }, { "candidate_id", candidate.id },
        { "local_path_verified", true },
        { "repeated_selection_count", recentSelectionCount(candidate.target_id) },
        { "source_stage", candidate.source_stage },
        { "source_priority", candidate.source_priority },
        { "original_policy_stage_active",
            candidate.source_priority == earliest_stage_priority },
        { "safety_mode", candidate.safety_mode },
        { "semantic_score", std::isfinite(candidate.semantic_score) ?
            finiteOrNull(rounded(candidate.semantic_score, 4)) : json(nullptr) },
        { "information_gain", candidate.information_gain_available ?
            finiteOrNull(rounded(candidate.information_gain, 4)) : json(nullptr) }
      });
      local_candidates.push_back({
        { "id", candidate.id }, { "kind", candidate.kind }, { "target_id", candidate.target_id },
        { "goal_pose", { { "x", candidate.goal.x() }, { "y", candidate.goal.y() },
            { "yaw", candidate.kind == "observe_rotation" ?
                normalizeAngle(yaw + (candidate.target_id == "observe-left" ? 1.0 : -1.0) *
                    observationTurnAngleRad()) :
                std::atan2(candidate.goal.y() - start.y(), candidate.goal.x() - start.x()) } } },
        { "path", { { "reachable", true }, { "collision_free", true },
            { "distance_m", pathLength(candidate.path) }, { "points", pointsJson(candidate.path) } } }
      });
    }
    region_state["history"] = {
      { "step_count", navigation_step_count },
      { "action_origin", last_action_origin },
      { "last_mode", jev_last_candidate_id_.empty() ? json(nullptr) :
          json(jev_last_candidate_id_.find("object-")==0 || jev_last_candidate_id_.find("over-depth-")==0 ? "approach_object" : "explore") },
      { "last_target_id", jev_last_entity_id_.empty() ? json(nullptr) : json(jev_last_entity_id_) },
      { "low_level_action_completed", last_action_completed },
      { "goal_reached", nullptr },
      { "latest_collision", latest_collision },
      { "window_size", 1 },
      { "omitted_selections", std::max(0, static_cast<int>(jev_request_sequence_)-1) },
      { "cumulative_progress_m", jev_has_last_goal_ ?
          finiteOrNull(jev_last_goal_initial_distance_ - (jev_last_goal_-start).norm()) : json(nullptr) },
      { "collision_count", collision_count },
      { "last_step_progress_m", snapshot["navigation_history"].contains("last_goal_execution") ?
          snapshot["navigation_history"]["last_goal_execution"].value("last_step_progress_m", json(nullptr)) : json(nullptr) }
    };
    const int object_total = snapshot["objects"].size();
    const int frontier_total = snapshot["frontiers"].size();
    region_state["coverage"] = {
      { "raw_map_complete", false }, { "high_level_evidence_complete", false },
      { "decision_evidence_complete", true },
      { "summary_contract_valid", false }, { "source_profile", "local_maps_and_planner" },
      { "regions_total", region_graph_->regions().size() },
      { "regions_exported", region_graph_->regions().size() },
      { "objects_total", object_total }, { "objects_exported", object_total },
      { "frontiers_total", frontier_total }, { "frontiers_exported", frontier_total },
      { "executable_candidates_total", candidates.size() },
      { "executable_candidates_exported", candidates.size() },
      { "omitted_entities", 0 },
      { "detail_policy", "complete JEV-ObjNav high-level decision evidence and all safe entities/options; geometry compressed to region statistics, object boxes and sparse connectivity; no raw grids, point clouds, or path point arrays" },
      { "edge_absence_meaning", "no direct verified edge; not proof that an indirect route is impossible" }
    };
    region_state["selection_parameters"] = {
      { "min_target_confidence", object_map2d_->getMinConfidence() },
      { "min_target_observations", object_map2d_->getMinObservationNum() },
      { "min_independent_viewpoints", 2 },
      { "min_viewpoint_baseline_m", 0.35 },
      { "eligibility_evaluated_locally_at_full_precision", true },
      { "class_function_score_definition", "int(observation_point_sum * fused_confidence)" }
    };
    snapshot = { { "region_graph", region_state }, { "candidates", local_candidates } };
  }

  cached_request_id = request_id; cached_snapshot = snapshot; cached_candidates = candidates;
  } // immutable snapshot for retries of the same observation
  const HttpResult http = postJson(jev_url_, snapshot.dump(), jev_timeout_ms_);
  if (!http.transport_ok) return JEV_PLAN_RETRYABLE_FAILURE;
  const json response = json::parse(http.body, nullptr, false);
  if (response.is_discarded() || !response.is_object())
    return JEV_PLAN_RETRYABLE_FAILURE;
  if (http.status >= 400) {
    const bool retryable = response.value("retryable", http.status >= 500 || http.status == 429);
    ros::param::set("/jev_obj/jev/last_error", response.value("error_code", std::string("unknown")));
    ROS_WARN("[Jev] HTTP %ld error=%s retryable=%d", http.status,
        response.value("error_code", std::string("unknown")).c_str(), retryable);
    return retryable ? JEV_PLAN_RETRYABLE_FAILURE : JEV_PLAN_NONRETRYABLE_FAILURE;
  }

  try {
    if (response.value("status", std::string()) != "GOAL")
      return JEV_PLAN_RETRYABLE_FAILURE;
    if (response.at("episode_id").get<std::string>() != episode_id ||
        response.at("map_revision").get<int>() != revision ||
        response.at("frame_id").get<std::string>() != jev_frame_id_)
      return JEV_PLAN_RETRYABLE_FAILURE;
    if (jev_state_profile_ == "region_graph" &&
        (response.at("request_id").get<std::string>() != request_id ||
         response.at("observation_id").get<int>() != observation_id ||
         response.at("region_graph_revision").get<int>() != region_graph_->graphRevision()))
      return JEV_PLAN_RETRYABLE_FAILURE;
    const std::string chosen_id = response.at("candidate_id").get<std::string>();
    for (const auto& candidate : candidates) {
      if (candidate.id != chosen_id)
        continue;
      ros::param::set("/jev_obj/jev/accepted_request_id", request_id);
      ros::param::set("/jev_obj/jev/accepted_candidate_id", chosen_id);
      ros::param::set("/jev_obj/jev/accepted_target_id", candidate.target_id);
      ros::param::set("/jev_obj/jev/accepted_source_stage", candidate.source_stage);
      ros::param::set("/jev_obj/jev/accepted_observation_id", observation_id);
      ros::param::set("/jev_obj/jev/accepted_mode", candidate.kind == "approach_object" ? "approach_object" : "explore");
      ros::param::set("/jev_obj/jev/last_error", "");
      ed_->next_pos_ = candidate.goal;
      ed_->next_best_path_ = candidate.path;
      ++jev_request_sequence_;
      const bool same_entity = jev_last_entity_id_ == candidate.target_id;
      jev_repeated_candidate_count_ = same_entity ? jev_repeated_candidate_count_ + 1 : 1;
      if (!same_entity) {
        jev_last_goal_initial_distance_ = (candidate.goal - start).norm();
        jev_last_goal_initial_step_ = navigation_step_count;
      }
      jev_selected_entity_history_.push_back(candidate.target_id);
      if (jev_selected_entity_history_.size() > 64)
        jev_selected_entity_history_.erase(jev_selected_entity_history_.begin(),
            jev_selected_entity_history_.begin() +
            (jev_selected_entity_history_.size() - 64));
      jev_last_candidate_id_ = chosen_id;
      jev_last_entity_id_ = candidate.target_id;
      jev_last_goal_ = candidate.goal;
      jev_last_goal_previous_distance_ = (candidate.goal - start).norm();
      jev_has_last_goal_ = true;
      result = candidate.result;
      ROS_INFO("[Jev] Selected %s (%s), path %.2f m", chosen_id.c_str(),
          candidate.kind.c_str(), pathLength(candidate.path));
      return JEV_PLAN_SUCCESS;
    }
  }
  catch (const json::exception& error) {
    ROS_WARN("[Jev] Invalid decision payload: %s", error.what());
  }
  return JEV_PLAN_RETRYABLE_FAILURE;
}

}  // namespace jev_obj_planner
