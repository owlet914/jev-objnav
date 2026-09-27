#ifndef _REGION_GRAPH2D_H_
#define _REGION_GRAPH2D_H_

#include <Eigen/Eigen>
#include <cstdint>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace jev_obj_planner {

class SDFMap2D;

inline bool regionSemanticSupported(double value, double weight) {
  return std::isfinite(value) && std::isfinite(weight) && weight > 0.0;
}

struct RegionGridSnapshot {
  int width = 0;
  int height = 0;
  double resolution = 0.0;
  Eigen::Vector2d origin = Eigen::Vector2d::Zero();
  std::vector<int> occupancy;
  std::vector<int> inflated;
  std::vector<double> semantic_value;
  std::vector<double> semantic_weight;
  std::vector<double> clearance;
  std::set<int> traversable_addresses;

  int address(int x, int y) const { return x * height + y; }
  bool valid() const;
};

struct RegionSemanticSummary {
  bool valid = false;
  double weighted_mean = 0.0;
  double p90 = 0.0;
  double maximum = 0.0;
  double valid_area_fraction = 0.0;
  int valid_cell_count = 0;
};

struct RegionNode2D {
  std::string id;
  int tile_x = 0;
  int tile_y = 0;
  Eigen::Vector2d bbox_min = Eigen::Vector2d::Zero();
  Eigen::Vector2d bbox_max = Eigen::Vector2d::Zero();
  Eigen::Vector2d anchor = Eigen::Vector2d::Zero();
  double free_area_m2 = 0.0;
  RegionSemanticSummary semantic;
  bool visited = false;
  int visit_count = 0;
  int last_robot_visit_step = -1;
  std::vector<std::string> parent_ids;
  std::vector<int> cell_addresses;
};

struct RegionEdge2D {
  std::string id;
  std::string from;
  std::string to;
  bool bidirectional = true;
  int portal_contact_count = 0;
  double portal_width_estimate_m = 0.0;
  double min_clearance_m = 0.0;
  int verified_map_revision = 0;
};

class RegionGraph2D {
public:
  explicit RegionGraph2D(double tile_size_m = 3.0,
      double localization_tolerance_m = 0.35);

  static RegionGridSnapshot capture(SDFMap2D& map);
  static bool refresh(SDFMap2D& map, RegionGridSnapshot& grid);
  bool update(const RegionGridSnapshot& grid, int map_revision, int robot_step,
      const Eigen::Vector2d& robot_position);
  void resetEpisode();

  const std::vector<RegionNode2D>& regions() const { return regions_; }
  const std::vector<RegionEdge2D>& edges() const { return edges_; }
  double tileSize() const { return tile_size_m_; }
  double localizationTolerance() const { return localization_tolerance_m_; }
  int graphRevision() const { return graph_revision_; }
  int lastMapRevision() const { return last_map_revision_; }
  bool lastUpdateWasCacheHit() const { return last_update_cache_hit_; }
  const std::string& robotRegionId() const { return robot_region_id_; }
  bool robotLocalizationUsedTolerance() const { return robot_localization_used_tolerance_; }
  std::string regionForAddress(int address) const;
  std::string regionForPosition(const RegionGridSnapshot& grid,
      const Eigen::Vector2d& position) const;
  std::vector<std::string> routeRegions(
      const RegionGridSnapshot& grid, const std::vector<Eigen::Vector2d>& path) const;
  std::vector<std::string> routeBetweenRegions(
      const std::string& from, const std::string& to) const;

private:
  std::string nextId();
  double tile_size_m_ = 3.0;
  double localization_tolerance_m_ = 0.35;
  int next_region_id_ = 1;
  int graph_revision_ = 0;
  int last_map_revision_ = -1;
  bool last_update_cache_hit_ = false;
  bool robot_localization_used_tolerance_ = false;
  std::string robot_region_id_;
  std::string last_robot_region_id_;
  std::vector<RegionNode2D> regions_;
  std::vector<RegionEdge2D> edges_;
  std::unordered_map<int, std::string> cell_region_ids_;
  std::unordered_map<int, std::set<int>> cell_visits_;
};

}  // namespace jev_obj_planner

#endif
