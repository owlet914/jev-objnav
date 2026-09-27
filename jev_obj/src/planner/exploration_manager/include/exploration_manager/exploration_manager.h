#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

// Third-party libraries
#include <Eigen/Eigen>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Standard C++ libraries
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ROS core
#include <ros/ros.h>

// Plan environment
#include <plan_env/frontier_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/region_graph2d.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>

// Path searching
#include <path_searching/astar2d.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace jev_obj_planner {
class SDFMap2D;
class FrontierMap2D;
class Gcopter;
class KinoAstar;
struct ExplorationParam;
struct ExplorationData;

struct PathAttemptEvidence {
  double success_distance = 0.0;
  double max_search_time_s = 0.0;
  int safety_mode = Astar2D::SAFETY_MODE::NORMAL;
  int search_result = Astar2D::NO_PATH;
  double early_terminate_cost = 0.0;
  bool reached = false;
};

struct ObjectPathEvidence {
  bool geometry_available = false;
  Vector2d nearest_object_point = Vector2d::Zero();
  Vector2d approach_point = Vector2d::Zero();
  vector<Vector2d> path;
  vector<PathAttemptEvidence> attempts;
  std::string final_reason = "not_attempted";
};

struct PathCostEvidence {
  double physical_cost_m = 0.0;
  double solver_cost = 10000.0;
  bool reachable = false;
  int search_result = Astar2D::NO_PATH;
  bool solver_penalty = true;
};

enum JEV_PLAN_STATUS { JEV_PLAN_SUCCESS, JEV_PLAN_RETRYABLE_FAILURE, JEV_PLAN_NONRETRYABLE_FAILURE };

struct SemanticFrontier {
  Vector2d position;      ///< 2D position of the frontier
  double semantic_value;  ///< Semantic value at the frontier location
  double path_length;     ///< Path length to reach this frontier
  vector<Vector2d> path;  ///< Complete path to the frontier

  bool operator<(const SemanticFrontier& other) const
  {
    if (fabs(semantic_value - other.semantic_value) < 1e-4) {
      // If semantic values are equal, sort by path length (ascending)
      return path_length < other.path_length;
    }
    // Otherwise, sort by semantic value (descending)
    return semantic_value > other.semantic_value;
  }
};

enum EXPL_RESULT {
  EXPLORATION,               ///< Normal exploration mode
  SEARCH_BEST_OBJECT,        ///< Found high-confidence object
  SEARCH_OVER_DEPTH_OBJECT,  ///< Searching over-depth object
  SEARCH_SUSPICIOUS_OBJECT,  ///< Investigating suspicious object
  NO_PASSABLE_FRONTIER,      ///< No reachable frontiers available
  NO_COVERABLE_FRONTIER,     ///< No coverable frontiers found
  SEARCH_EXTREME,            ///< Extreme search mode activated
  JEV_DECISION_RETRY,        ///< Waiting for a valid Jev choice; publish no action
  JEV_DECISION_FAILURE,      ///< Fifteen consecutive unsuccessful Jev decisions
  JEV_OBSERVE_LEFT,          ///< Rotate left to acquire a new observation
  JEV_OBSERVE_RIGHT          ///< Rotate right to acquire a new observation
};

class ExplorationManager {
public:
  ExplorationManager() = default;
  ~ExplorationManager();  // Explicit destructor declaration for shared_ptr with forward declaration

  void initialize(ros::NodeHandle& nh);

  int planNextBestPoint(const Vector3d& pos, const double& yaw);
  bool jevEnabled() const { return jev_enabled_; }
  bool planTrajectory(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl);
  void getSortedSemanticFrontiers(const Vector2d& cur_pos, const vector<Vector2d>& frontiers,
      vector<SemanticFrontier>& sem_frontiers);
  void calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers, double& std_dev,
      double& max_to_mean, double& mean, bool if_print = false);

  shared_ptr<ExplorationData> ed_;            ///< Exploration data container
  shared_ptr<ExplorationParam> ep_;           ///< Exploration parameters
  unique_ptr<Astar2D> path_finder_;           ///< A* path finding algorithm
  shared_ptr<FrontierMap2D> frontier_map2d_;  ///< 2D frontier map
  shared_ptr<ObjectMap2D> object_map2d_;      ///< 2D object map
  shared_ptr<SDFMap2D> sdf_map_;              ///< Signed distance field map
  unique_ptr<RegionGraph2D> region_graph_;    ///< Lossy Jev model-view graph
  RegionGridSnapshot jev_region_grid_;
  shared_ptr<Gcopter> gcopter_;               ///< Trajectory optimizer (for real-world)
  shared_ptr<KinoAstar> kinoastar_;           ///< Kinodynamic A* planner (for real-world)

  typedef shared_ptr<ExplorationManager> Ptr;

private:
  // Optional bounded Jev choice over Jev Obj-planned, reachable goals.
  JEV_PLAN_STATUS planWithJev(const Vector3d& pos, double yaw, int& result);

  // Exploration Policy
  void chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findHighestSemanticsFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);
  void findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);

  // Path Search Utils
  bool searchObjectPath(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
      ObjectPathEvidence* evidence = nullptr);
  bool searchObjectPathExtreme(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
      ObjectPathEvidence* evidence = nullptr);
  bool searchFrontierPath(const Vector2d& start, const Vector2d& end, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path, int* search_result = nullptr);
  void shortenPath(vector<Vector2d>& path);

  // Helper functions for object path searching
  Vector2d findNearestObjectPoint(
      const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud);
  bool trySearchObjectPathWithDistance(const Vector2d& start2d, const Vector2d& object_pose,
      double distance, double max_search_time, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path, const std::string& debug_msg,
      PathAttemptEvidence* attempt = nullptr);

  // TSP Optimization Methods
  void computeATSPTour(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices);
  void computeATSPCostMatrix(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& cost_matrix);
  double computePathCost(const Vector2d& pos1, const Vector2d& pos2,
      PathCostEvidence* evidence = nullptr);
  vector<Vector2i> allNeighbors(const Eigen::Vector2i& idx, int grid_radius);

  ros::ServiceClient tsp_client_;         ///< ROS service client for TSP solver
  unique_ptr<RayCaster2D> ray_caster2d_;  ///< Ray casting for collision checking
  bool jev_enabled_ = false;
  std::string jev_url_;
  std::string jev_frame_id_;
  std::string jev_state_profile_ = "full";
  std::string jev_session_id_;
  int jev_timeout_ms_ = 45000;
  int jev_revision_ = 0;
  uint64_t jev_request_sequence_ = 0;
  int jev_consecutive_failures_ = 0;
  ros::WallTime jev_next_retry_at_;
  std::string jev_episode_id_;
  std::string jev_last_candidate_id_;
  std::string jev_last_entity_id_;
  int jev_repeated_candidate_count_ = 0;
  int jev_last_goal_initial_step_ = 0;
  double jev_last_goal_previous_distance_ = 0.0;
  Eigen::Vector2d jev_last_goal_ = Eigen::Vector2d::Zero();
  double jev_last_goal_initial_distance_ = 0.0;
  bool jev_has_last_goal_ = false;
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> jev_cached_over_depth_cloud_;
  uint64_t jev_cached_over_depth_observation_id_ = 0;
  std::vector<std::string> jev_selected_entity_history_;
  std::vector<int> jev_recent_perception_observation_ids_;
  std::vector<double> jev_recent_itm_scores_;
  std::vector<int> jev_recent_itm_validity_;
  std::vector<int> jev_recent_target_match_counts_;
  std::vector<int> jev_recent_valid_mask_counts_;
};

inline bool ExplorationManager::searchFrontierPath(const Vector2d& start, const Vector2d& end,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path, int* search_result)
{
  path_finder_->reset();
  const int status = path_finder_->astarSearch(start, end, 0.25, 0.01);
  if (search_result) *search_result = status;
  if (status == Astar2D::REACH_END) {
    refined_pos = end;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

inline bool ExplorationManager::searchObjectPathExtreme(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    ObjectPathEvidence* evidence)
{
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (evidence) {
    evidence->geometry_available = object_pose.x() >= -999.0;
    evidence->nearest_object_point = object_pose;
    evidence->attempts.clear();
    evidence->path.clear();
  }
  if (object_pose.x() < -999.0) {
    if (evidence) evidence->final_reason = "empty_or_invalid_object_geometry";
    return false;  // Error finding nearest point
  }

  Vector2d start2d = Vector2d(start(0), start(1));
  path_finder_->reset();
  const int search_result = path_finder_->astarSearch(
      start2d, object_pose, 0.25, 0.2, Astar2D::SAFETY_MODE::EXTREME);
  if (evidence) {
    PathAttemptEvidence attempt;
    attempt.success_distance = 0.25;
    attempt.max_search_time_s = 0.2;
    attempt.safety_mode = Astar2D::SAFETY_MODE::EXTREME;
    attempt.search_result = search_result;
    attempt.early_terminate_cost = path_finder_->getEarlyTerminateCost();
    attempt.reached = search_result == Astar2D::REACH_END;
    evidence->attempts.push_back(attempt);
  }
  if (search_result == Astar2D::REACH_END) {
    refined_pos = object_pose;
    refined_path = path_finder_->getPath();
    if (evidence) {
      evidence->approach_point = refined_pos;
      evidence->path = refined_path;
      evidence->final_reason = "reached_with_extreme_occupancy_bypass";
    }
    return true;
  }
  if (evidence)
    evidence->final_reason = search_result == Astar2D::TIMEOUT ? "search_timeout" :
        search_result == Astar2D::NODE_POOL_EXHAUSTED ? "node_pool_exhausted" : "no_path";
  return false;
}

inline void ExplorationManager::shortenPath(vector<Vector2d>& path)
{
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }

  // Shorten the path by keeping only critical intermediate points
  const double dist_thresh = 3.0;  // Minimum distance threshold for waypoint retention
  vector<Vector2d> short_tour = { path.front() };

  for (int i = 1; i < (int)path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints only when necessary to avoid collision
      ray_caster2d_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector2i idx;
      while (ray_caster2d_->nextId(idx) && ros::ok()) {
        if (sdf_map_->getInflateOccupancy(idx) == 1 ||
            sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }

  // Always include the final destination
  if ((path.back() - short_tour.back()).norm() > 1e-3)
    short_tour.push_back(path.back());

  // Ensure minimum path complexity (at least three points)
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));

  path = short_tour;
}

inline vector<Eigen::Vector2i> ExplorationManager::allNeighbors(
    const Eigen::Vector2i& idx, int grid_radius)
{
  vector<Eigen::Vector2i> neighbors;

  for (int x = -grid_radius; x <= grid_radius; ++x) {
    for (int y = -grid_radius; y <= grid_radius; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center point
      Eigen::Vector2i offset(x, y);
      neighbors.push_back(idx + offset);
    }
  }
  return neighbors;
}

}  // namespace jev_obj_planner

#endif
