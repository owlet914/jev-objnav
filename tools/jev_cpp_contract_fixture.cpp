#include <exploration_manager/jev_contract.h>
#include <nlohmann_json.hpp>

#include <iostream>
#include <string>

using nlohmann::json;
using jev_obj_planner::finalizeJevSnapshot;

namespace {

json emptyPathSearch()
{
  return { { "geometry_available", true }, { "nearest_object_point", nullptr },
    { "approach_point", nullptr }, { "path_points", json::array() },
    { "distance_m", nullptr }, { "attempts", json::array() },
    { "final_reason", "synthetic_fixture" } };
}

json neighborhood()
{
  json rows = json::array();
  for (int dx = -2; dx <= 2; ++dx)
    for (int dy = -2; dy <= 2; ++dy)
      rows.push_back({ { "dx", dx }, { "dy", dy }, { "in_map", true },
        { "occupancy", 0 }, { "inflated", 0 }, { "semantic_raw", 0.0 },
        { "fusion_weight", 1.0 } });
  return rows;
}

json pathGraph(const std::string& prefix, int start, int count)
{
  json ids = json::array({ "robot" });
  for (int i = start; i < start + count; ++i) ids.push_back(prefix + std::to_string(i));
  json matrix = json::array();
  for (int i = 0; i <= count; ++i) {
    json row = json::array();
    for (int j = 0; j <= count; ++j) row.push_back(nullptr);
    matrix.push_back(row);
  }
  return { { "node_ids", ids }, { "cost_matrix", matrix } };
}

}  // namespace

int main()
{
  json snapshot = {
    { "schema_version", "2.0" }, { "request_id", "cpp-fixture:obs-7:req-1" },
    { "episode_id", "cpp-fixture" }, { "observation_id", 7 }, { "timestamp_ms", 1 },
    { "target_object", "mug" },
    { "robot_pose", { { "x", 0.0 }, { "y", 0.0 }, { "yaw", 3.10 } } },
    { "map", { { "frame_id", "world" }, { "revision", 7 }, { "observation_id", 7 },
        { "decision_query_evidence", { { "raw_semantic_values_preserved", true } } } } },
    { "rooms", json::array() }, { "objects", json::array() },
    { "semantic_matches", json::array() }, { "frontiers", json::array() },
    { "candidate_evidence", json::array() }, { "search_branches", json::array() },
    { "observation_actions", json::array() }, { "candidates", json::array() },
    { "perception", { { "object_detection", json::object() },
        { "image_text_match", json::object() } } },
    { "navigation_history", json::object() },
    { "original_policy", {
        { "active_frontier_path_graph", pathGraph("frontier-", 0, 8) },
        { "dormant_frontier_path_graph", pathGraph("frontier-", 8, 7) }
      } }
  };
  for (int i = 0; i < 70; ++i) {
    snapshot["objects"].push_back({ { "id", "object-" + std::to_string(i) },
      { "position", { { "x", i * 0.1 }, { "y", 1.0 } } },
      { "target_confidence", 0.0 }, { "target_observations", 1 },
      { "selection", { { "relaxed_eligible", false } } },
      { "normal_path_search", emptyPathSearch() } });
  }
  for (int i = 0; i < 15; ++i) {
    const bool dormant = i >= 8;
    const std::string id = "frontier-" + std::to_string(i);
    snapshot["frontiers"].push_back({ { "id", id },
      { "position", { { "x", 1.0 + i }, { "y", 2.0 } } },
      { "state", dormant ? "dormant" : "active" },
      { "planning_status", "reach_end" },
      { "semantic_features", { { "raw_neighborhood", neighborhood() } } },
      { "path", { { "reachable", true }, { "distance_m", 2.3 },
          { "points", json::array() }, { "safety_mode", "normal" } } },
      { "information_gain", nullptr },
      { "information_gain_status", "unknown_not_measured" } });
    snapshot["candidate_evidence"].push_back({ { "target_entity_id", id },
      { "stage", dormant ? "dormant_frontier_policy" : "active_frontier_policy" },
      { "search_result", "reach_end" } });
  }
  const char* empty_branches[] = { "high_confidence_object_normal",
    "current_over_depth_object", "suspicious_object_normal", "active_frontier_policy",
    "dormant_frontier_policy", "original_object_extreme", "cached_over_depth_extreme" };
  for (const auto* name : empty_branches)
    snapshot["search_branches"].push_back(
        { { "name", name }, { "stage_condition", "fixture" }, { "entity_results", json::array() } });
  snapshot["search_branches"].push_back({ { "name", "relaxed_all_cells_extreme" },
    { "stage_condition", "fixture" },
    { "entity_results", json::array({ { { "entity_ids", json::array() },
        { "geometry_mode", "merged_all_cells_low_confidence_fallback" },
        { "search", emptyPathSearch() }, { "selectable", false } } }) } });
  snapshot["candidates"].push_back({
    { "id", "frontier-0-goal" }, { "kind", "explore_frontier" },
    { "target_id", "frontier-0" },
    { "goal_pose", { { "x", 1.0 }, { "y", 2.0 }, { "yaw", 1.0 } } },
    { "path", { { "reachable", true }, { "collision_free", true },
        { "distance_m", 2.3 } } },
    { "semantic_score", -0.17 }, { "information_gain", nullptr }
  });

  const jev_obj_planner::JevCoverageCounts counts{ 70, 8, 7, 15, 1 };
  json incomplete = snapshot;
  incomplete["frontiers"][0]["semantic_features"].erase("raw_neighborhood");
  finalizeJevSnapshot(incomplete, counts);
  if (incomplete["coverage"]["high_level_evidence_complete"].get<bool>()) {
    std::cerr << "coverage must be incomplete when a required feature is missing" << std::endl;
    return 2;
  }
  finalizeJevSnapshot(snapshot, counts);
  if (!snapshot["coverage"]["high_level_evidence_complete"].get<bool>()) {
    std::cerr << "complete production contract fixture was rejected" << std::endl;
    return 3;
  }
  std::cout << snapshot.dump(2) << std::endl;
  return 0;
}
