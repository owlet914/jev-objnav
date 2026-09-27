#ifndef _JEV_CONTRACT_H_
#define _JEV_CONTRACT_H_

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <nlohmann_json.hpp>

namespace jev_obj_planner {

struct JevCoverageCounts {
  std::size_t objects = 0;
  std::size_t active_frontiers = 0;
  std::size_t dormant_frontiers = 0;
  std::size_t candidate_evidence = 0;
  std::size_t candidates = 0;
};

struct JevCoverageFeatures {
  bool entity_enumeration = false;
  bool normal_path_searches = false;
  bool extreme_search_evidence = false;
  bool frontier_path_graphs = false;
  bool raw_decision_neighborhoods = false;
};

inline nlohmann::json completeCount(std::size_t count)
{
  return { { "total", count }, { "exported", count }, { "omitted", 0 } };
}

inline const char* coverageStatus(bool complete)
{
  return complete ? "complete" : "missing";
}

inline const nlohmann::json* findSearchBranch(
    const nlohmann::json& snapshot, const std::string& name)
{
  if (!snapshot.contains("search_branches") || !snapshot["search_branches"].is_array())
    return nullptr;
  for (const auto& branch : snapshot["search_branches"])
    if (branch.is_object() && branch.value("name", std::string()) == name)
      return &branch;
  return nullptr;
}

inline bool pathGraphComplete(const nlohmann::json& policy, const char* name,
    const std::unordered_set<std::string>& expected_frontier_ids)
{
  if (!policy.contains(name) || !policy[name].is_object()) return false;
  const auto& graph = policy[name];
  if (!graph.contains("node_ids") || !graph["node_ids"].is_array() ||
      !graph.contains("cost_matrix") || !graph["cost_matrix"].is_array())
    return false;
  const std::size_t size = graph["node_ids"].size();
  if (size != expected_frontier_ids.size() + 1) return false;
  std::unordered_set<std::string> actual_ids;
  for (const auto& id : graph["node_ids"]) {
    if (!id.is_string()) return false;
    actual_ids.insert(id.get<std::string>());
  }
  if (actual_ids.size() != size || actual_ids.count("robot") != 1) return false;
  actual_ids.erase("robot");
  if (actual_ids != expected_frontier_ids) return false;
  if (graph["cost_matrix"].size() != size) return false;
  for (const auto& row : graph["cost_matrix"])
    if (!row.is_array() || row.size() != size) return false;
  return true;
}

inline JevCoverageFeatures inspectJevSnapshot(
    const nlohmann::json& snapshot, const JevCoverageCounts& counts)
{
  JevCoverageFeatures features;
  if (!snapshot.contains("objects") || !snapshot["objects"].is_array() ||
      !snapshot.contains("frontiers") || !snapshot["frontiers"].is_array() ||
      !snapshot.contains("candidate_evidence") || !snapshot["candidate_evidence"].is_array() ||
      !snapshot.contains("candidates") || !snapshot["candidates"].is_array())
    return features;

  std::size_t active_frontiers = 0, dormant_frontiers = 0;
  bool unique_frontier_ids = true;
  std::unordered_set<std::string> frontier_ids, active_ids, dormant_ids;
  for (const auto& frontier : snapshot["frontiers"]) {
    if (!frontier.is_object()) continue;
    if (!frontier.contains("id") || !frontier["id"].is_string()) {
      unique_frontier_ids = false;
      continue;
    }
    const std::string id = frontier["id"].get<std::string>();
    if (!frontier_ids.insert(id).second) unique_frontier_ids = false;
    const std::string state = frontier.value("state", std::string());
    if (state == "active") {
      ++active_frontiers;
      active_ids.insert(id);
    }
    else if (state == "dormant") {
      ++dormant_frontiers;
      dormant_ids.insert(id);
    }
  }
  bool unique_object_ids = true;
  std::unordered_set<std::string> object_ids;
  for (const auto& object : snapshot["objects"]) {
    if (!object.is_object() || !object.contains("id") || !object["id"].is_string() ||
        !object_ids.insert(object["id"].get<std::string>()).second)
      unique_object_ids = false;
  }
  std::unordered_set<std::string> evidence_ids;
  bool valid_evidence_ids = true;
  for (const auto& evidence : snapshot["candidate_evidence"]) {
    if (!evidence.is_object() || !evidence.contains("target_entity_id") ||
        !evidence["target_entity_id"].is_string() ||
        !evidence_ids.insert(evidence["target_entity_id"].get<std::string>()).second)
      valid_evidence_ids = false;
  }
  const bool identity_consistent = snapshot.contains("map") && snapshot["map"].is_object() &&
      snapshot["map"].contains("observation_id") && snapshot.contains("observation_id") &&
      snapshot["map"]["observation_id"] == snapshot["observation_id"];
  features.entity_enumeration = identity_consistent && unique_frontier_ids &&
      unique_object_ids && valid_evidence_ids && evidence_ids == frontier_ids &&
      snapshot.contains("perception") && snapshot["perception"].is_object() &&
      snapshot.contains("navigation_history") && snapshot["navigation_history"].is_object() &&
      snapshot.contains("original_policy") && snapshot["original_policy"].is_object() &&
      snapshot["objects"].size() == counts.objects &&
      active_frontiers == counts.active_frontiers &&
      dormant_frontiers == counts.dormant_frontiers &&
      snapshot["candidate_evidence"].size() == counts.candidate_evidence &&
      snapshot["candidates"].size() == counts.candidates;

  features.normal_path_searches = std::all_of(snapshot["objects"].begin(),
      snapshot["objects"].end(), [](const nlohmann::json& object) {
        return object.is_object() && object.contains("normal_path_search") &&
            object["normal_path_search"].is_object();
      });

  std::size_t relaxed_count = 0, mapped_object_count = 0;
  for (const auto& object : snapshot["objects"]) {
    if (!object.is_object() || !object.contains("selection")) continue;
    ++mapped_object_count;
    if (object["selection"].value("relaxed_eligible", false)) ++relaxed_count;
  }
  const std::size_t expected_relaxed_count = relaxed_count > 0 ? relaxed_count :
      (mapped_object_count > 0 ? 1 : 0);
  const auto* original_extreme = findSearchBranch(snapshot, "original_object_extreme");
  const auto* relaxed_extreme = findSearchBranch(snapshot, "relaxed_all_cells_extreme");
  const auto* cached_extreme = findSearchBranch(snapshot, "cached_over_depth_extreme");
  features.extreme_search_evidence = original_extreme && relaxed_extreme && cached_extreme &&
      original_extreme->contains("entity_results") &&
      (*original_extreme)["entity_results"].is_array() &&
      (*original_extreme)["entity_results"].size() == relaxed_count &&
      relaxed_extreme->contains("entity_results") &&
      (*relaxed_extreme)["entity_results"].is_array() &&
      (*relaxed_extreme)["entity_results"].size() == expected_relaxed_count &&
      cached_extreme->contains("entity_results") &&
      (*cached_extreme)["entity_results"].is_array();

  const auto& policy = snapshot["original_policy"];
  features.frontier_path_graphs =
      pathGraphComplete(policy, "active_frontier_path_graph", active_ids) &&
      pathGraphComplete(policy, "dormant_frontier_path_graph", dormant_ids);
  features.raw_decision_neighborhoods = std::all_of(snapshot["frontiers"].begin(),
      snapshot["frontiers"].end(), [](const nlohmann::json& frontier) {
        return frontier.is_object() && frontier.contains("semantic_features") &&
            frontier["semantic_features"].is_object() &&
            frontier["semantic_features"].contains("raw_neighborhood") &&
            frontier["semantic_features"]["raw_neighborhood"].is_array() &&
            frontier["semantic_features"]["raw_neighborhood"].size() == 25 &&
            frontier.contains("planning_status") && frontier.contains("path");
      });
  return features;
}

inline void finalizeJevSnapshot(nlohmann::json& snapshot, const JevCoverageCounts& counts)
{
  const JevCoverageFeatures features = inspectJevSnapshot(snapshot, counts);
  const bool high_level_complete = features.entity_enumeration &&
      features.normal_path_searches && features.extreme_search_evidence &&
      features.frontier_path_graphs && features.raw_decision_neighborhoods;
  snapshot["coverage"] = {
    { "scope", "all high-level entities, policy features, branch searches and safe actions" },
    { "entity_counts", {
        { "objects", completeCount(counts.objects) },
        { "active_frontiers", completeCount(counts.active_frontiers) },
        { "dormant_frontiers", completeCount(counts.dormant_frontiers) },
        { "candidate_evidence", completeCount(counts.candidate_evidence) },
        { "candidates", completeCount(counts.candidates) }
      } },
    { "required_feature_status", {
        { "entity_enumeration", coverageStatus(features.entity_enumeration) },
        { "normal_path_searches", coverageStatus(features.normal_path_searches) },
        { "extreme_search_evidence", features.extreme_search_evidence ?
            "complete_but_not_selectable_for_safety" : "missing" },
        { "frontier_path_graphs", features.frontier_path_graphs ?
            "complete_for_normally_reachable_frontiers" : "missing" },
        { "raw_decision_neighborhoods", coverageStatus(features.raw_decision_neighborhoods) },
        { "entire_raw_map", "not_exported" }
      } },
    { "raw_map_complete", false },
    { "high_level_evidence_complete", high_level_complete },
    { "limitations", {
        "entire native map is not exported; exact cells are included for every decision query",
        "extreme paths are audit evidence only because occupancy checking is bypassed"
      } }
  };
}

}  // namespace jev_obj_planner

#endif
