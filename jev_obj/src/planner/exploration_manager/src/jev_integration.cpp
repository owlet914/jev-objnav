/**
 * Optional high-level Jev decision over paths already planned by Jev Obj.
 * Jev only returns a candidate ID; it never supplies a coordinate or motion command.
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <curl/curl.h>
#include <nlohmann_json.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
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
  int result;
};

double unitScore(double value)
{
  if (!std::isfinite(value))
    return 0.0;
  return std::max(0.0, std::min(1.0, value));
}

double pathLength(const std::vector<Eigen::Vector2d>& path)
{
  double length = 0.0;
  for (size_t i = 1; i < path.size(); ++i)
    length += (path[i] - path[i - 1]).norm();
  return length;
}

size_t appendHttpBody(char* data, size_t size, size_t count, void* output)
{
  const size_t bytes = size * count;
  static_cast<std::string*>(output)->append(data, bytes);
  return bytes;
}

bool postJson(const std::string& url, const std::string& payload, int timeout_ms,
    std::string& response)
{
  static const CURLcode curl_init_result = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (curl_init_result != CURLE_OK)
    return false;
  CURL* curl = curl_easy_init();
  if (!curl)
    return false;
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
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  const CURLcode code = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  return code == CURLE_OK && status == 200;
}

}  // namespace

bool ExplorationManager::planWithJev(const Vector3d& pos, double yaw, int& result)
{
  const Vector2d start(pos.x(), pos.y());
  std::vector<JevCandidate> candidates;
  json snapshot = {
    { "schema_version", "1.0" },
    { "timestamp_ms", ros::WallTime::now().toNSec() / 1000000 },
    { "robot_pose", { { "x", pos.x() }, { "y", pos.y() }, { "yaw", yaw } } },
    { "rooms", json::array() },
    { "objects", json::array() },
    { "semantic_matches", json::array() },
    { "frontiers", json::array() },
    { "candidates", json::array() }
  };

  std::string target_object = "target object";
  std::string room_prior;
  std::string episode_id = jev_session_id_;
  std::vector<std::string> related_objects;
  std::vector<std::string> target_subcategories;
  ros::param::get("/jev_obj/jev/target_object", target_object);
  ros::param::get("/jev_obj/jev/room_prior", room_prior);
  ros::param::get("/jev_obj/jev/related_objects", related_objects);
  ros::param::get("/jev_obj/jev/target_subcategories", target_subcategories);
  ros::param::get("/jev_obj/jev/episode_id", episode_id);
  if (target_object.empty())
    target_object = "target object";
  if (episode_id.empty())
    episode_id = jev_session_id_;
  snapshot["target_object"] = target_object;
  snapshot["episode_id"] = episode_id;
  snapshot["room_prior"] = room_prior;
  snapshot["related_objects"] = related_objects;
  snapshot["target_subcategories"] = target_subcategories;

  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);
  const int revision = ++jev_revision_;
  snapshot["map"] = {
    { "frame_id", jev_frame_id_ }, { "revision", revision },
    { "resolution_m", sdf_map_->getResolution() },
    { "origin", { { "x", origin.x() }, { "y", origin.y() } } },
    { "size_m", { { "x", size.x() }, { "y", size.y() } } },
    { "frontier_count", ed_->frontier_averages_.size() },
    { "object_count", ed_->object_averages_.size() }
  };

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
        if (std::isfinite(confidence) && confidence > 0.0) {
          const int level = static_cast<int>(std::round(9.0 *
              unitScore(sdf_map_->value_map_->getValue(point))));
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
  snapshot["map"]["local_grid"] = {
    { "center", { { "x", start.x() }, { "y", start.y() } } },
    { "cell_size_m", grid_step }, { "width", grid_side }, { "height", grid_side },
    { "occupancy_rows", occupancy_rows },
    { "semantic_value_rows", semantic_rows },
    { "legend", { { "?", "unknown" }, { ".", "free or no semantic measurement" },
                    { "i", "inflated obstacle" }, { "#", "occupied" } } },
    { "free_cells", free_cells }, { "occupied_cells", occupied_cells },
    { "unknown_cells", unknown_cells }
  };

  double latest_itm_score = -1.0;
  if (ros::param::get("/jev_obj/jev/latest_itm_score", latest_itm_score) &&
      std::isfinite(latest_itm_score) && latest_itm_score >= 0.0)
    snapshot["current_view"] = { { "image_text_match_score", unitScore(latest_itm_score) } };

  // ObjectMap2D exposes fused target confidence and observation counts.
  // The sorted candidate clouds below have a different order from these clusters.
  std::vector<ObjectEvidence> object_evidence;
  object_map2d_->getObjectEvidence(object_evidence);
  std::sort(object_evidence.begin(), object_evidence.end(),
      [&](const ObjectEvidence& a, const ObjectEvidence& b) {
        if (std::abs(unitScore(a.target_confidence) - unitScore(b.target_confidence)) > 1e-6)
          return unitScore(a.target_confidence) > unitScore(b.target_confidence);
        return (a.center - start).squaredNorm() < (b.center - start).squaredNorm();
      });
  snapshot["map"]["object_count"] = object_evidence.size();
  snapshot["map"]["object_evidence_limit"] = 64;
  for (size_t i = 0; i < object_evidence.size() && i < 64; ++i) {
    const auto& object = object_evidence[i];
    json class_scores = json::array();
    for (size_t label = 0; label < object.class_confidences.size(); ++label) {
      const int observations = label < object.class_observations.size() ?
          object.class_observations[label] : 0;
      if (observations == 0 && object.class_confidences[label] <= 0.0)
        continue;
      const std::string class_name = label == 0 ? target_object :
          label - 1 < related_objects.size() ? related_objects[label - 1] :
          "related-class-" + std::to_string(label);
      class_scores.push_back({
        { "class_index", label }, { "label", class_name },
        { "confidence", unitScore(object.class_confidences[label]) },
        { "observations", observations }
      });
    }
    snapshot["objects"].push_back({
      { "id", "observed-" + std::to_string(object.id) },
      { "class_id", object.best_label },
      { "position", { { "x", object.center.x() }, { "y", object.center.y() } } },
      { "target_confidence", unitScore(object.target_confidence) },
      { "target_observations", object.target_observations },
      { "target_point_count", object.target_point_count },
      { "class_scores", class_scores },
      { "bbox_min", { { "x", object.box_min.x() }, { "y", object.box_min.y() } } },
      { "bbox_max", { { "x", object.box_max.x() }, { "y", object.box_max.y() } } },
      { "source", "Jev Obj object map" }
    });
  }

  auto addObject = [&](const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                       const std::string& id, double target_confidence,
                       int target_observations, int object_result,
                       const std::string& confidence_tier) {
    if (!cloud || cloud->points.empty())
      return;
    Eigen::Vector2d goal;
    std::vector<Eigen::Vector2d> path;
    if (!searchObjectPath(pos, cloud, goal, path) || path.empty())
      return;
    double x = 0.0, y = 0.0;
    for (const auto& point : cloud->points) {
      x += point.x;
      y += point.y;
    }
    x /= cloud->points.size();
    y /= cloud->points.size();
    snapshot["objects"].push_back({
      { "id", id }, { "label", target_object },
      { "position", { { "x", x }, { "y", y } } },
      { "target_confidence", unitScore(target_confidence) },
      { "target_observations", target_observations },
      { "score_available", confidence_tier != "over_depth" },
      { "confidence_tier", confidence_tier },
      { "source", "Jev Obj target object map" }
    });
    candidates.push_back({ id + "-goal", id, "approach_object", goal, path,
      unitScore(target_confidence), 0.0, object_result });
  };

  std::vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  std::vector<double> object_confidences;
  std::vector<int> object_observations;
  object_map2d_->getTopConfidenceObjectCloud(object_clouds, true, false,
      &object_confidences, &object_observations);
  for (size_t i = 0; i < object_clouds.size() && i < static_cast<size_t>(jev_max_objects_); ++i)
    addObject(object_clouds[i], "target-high-" + std::to_string(i),
      object_confidences[i], object_observations[i], SEARCH_BEST_OBJECT, "high");

  if (object_map2d_->over_depth_object_cloud_ &&
      !object_map2d_->over_depth_object_cloud_->points.empty())
    addObject(object_map2d_->over_depth_object_cloud_, "target-over-depth",
      0.0, 0, SEARCH_OVER_DEPTH_OBJECT, "over_depth");

  if (object_clouds.empty()) {
    object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, false,
        &object_confidences, &object_observations);
    for (size_t i = 0; i < object_clouds.size() && i < static_cast<size_t>(jev_max_objects_); ++i)
      addObject(object_clouds[i], "target-suspicious-" + std::to_string(i),
        object_confidences[i], object_observations[i], SEARCH_SUSPICIOUS_OBJECT, "suspicious");
  }

  std::vector<size_t> frontier_indices(ed_->frontier_averages_.size());
  std::iota(frontier_indices.begin(), frontier_indices.end(), 0);
  std::sort(frontier_indices.begin(), frontier_indices.end(), [&](size_t a, size_t b) {
    const double av = unitScore(sdf_map_->value_map_->getValue(ed_->frontier_averages_[a]));
    const double bv = unitScore(sdf_map_->value_map_->getValue(ed_->frontier_averages_[b]));
    if (std::abs(av - bv) > 1e-6)
      return av > bv;
    return (ed_->frontier_averages_[a] - start).squaredNorm() <
           (ed_->frontier_averages_[b] - start).squaredNorm();
  });

  for (size_t rank = 0; rank < frontier_indices.size() &&
                        rank < static_cast<size_t>(jev_max_frontiers_); ++rank) {
    const size_t index = frontier_indices[rank];
    const Vector2d& frontier = ed_->frontier_averages_[index];
    Vector2d goal;
    std::vector<Vector2d> path;
    if (!searchFrontierPath(start, frontier, goal, path) || path.empty())
      continue;
    const std::string id = "frontier-" + std::to_string(index);
    const double semantic = unitScore(sdf_map_->value_map_->getValue(frontier));
    const double semantic_confidence = unitScore(sdf_map_->value_map_->getConfidence(frontier));
    const size_t cluster_size = index < ed_->frontiers_.size() ? ed_->frontiers_[index].size() : 0;
    const double information_gain = unitScore(static_cast<double>(cluster_size) / 100.0);
    snapshot["frontiers"].push_back({
      { "id", id }, { "position", { { "x", frontier.x() }, { "y", frontier.y() } } },
      { "cluster_cells", cluster_size }, { "information_gain", information_gain },
      { "semantic_value", semantic }, { "semantic_confidence", semantic_confidence }
    });
    snapshot["semantic_matches"].push_back({
      { "query", target_object }, { "entity_id", id },
      { "score", semantic }, { "confidence", semantic_confidence },
      { "source", "Jev Obj value map" }
    });
    candidates.push_back({ id + "-goal", id, "explore_frontier", goal, path,
      semantic, information_gain, EXPLORATION });
  }

  if (candidates.empty())
    return false;
  for (const auto& candidate : candidates) {
    const double length = pathLength(candidate.path);
    if (!std::isfinite(length) || !std::isfinite(candidate.goal.x()) ||
        !std::isfinite(candidate.goal.y()))
      return false;
    snapshot["candidates"].push_back({
      { "id", candidate.id }, { "kind", candidate.kind },
      { "target_id", candidate.target_id },
      { "goal_pose", { { "x", candidate.goal.x() }, { "y", candidate.goal.y() },
                         { "yaw", std::atan2(candidate.goal.y() - start.y(),
                                             candidate.goal.x() - start.x()) } } },
      { "path", { { "reachable", true }, { "collision_free", true },
                    { "distance_m", length } } },
      { "semantic_score", candidate.semantic_score },
      { "information_gain", candidate.information_gain }
    });
  }

  std::string response_body;
  if (!postJson(jev_url_, snapshot.dump(), jev_timeout_ms_, response_body))
    return false;
  const json response = json::parse(response_body, nullptr, false);
  if (response.is_discarded() || !response.is_object())
    return false;

  try {
    if (response.value("status", std::string()) != "GOAL")
      return false;
    if (response.at("episode_id").get<std::string>() != episode_id ||
        response.at("map_revision").get<int>() != revision ||
        response.at("frame_id").get<std::string>() != jev_frame_id_)
      return false;
    const std::string chosen_id = response.at("candidate_id").get<std::string>();
    for (const auto& candidate : candidates) {
      if (candidate.id != chosen_id)
        continue;
      ed_->next_pos_ = candidate.goal;
      ed_->next_best_path_ = candidate.path;
      result = candidate.result;
      ROS_INFO("[Jev] Selected %s (%s), path %.2f m", chosen_id.c_str(),
          candidate.kind.c_str(), pathLength(candidate.path));
      return true;
    }
  }
  catch (const json::exception& error) {
    ROS_WARN("[Jev] Invalid decision payload: %s", error.what());
  }
  return false;
}

}  // namespace jev_obj_planner
