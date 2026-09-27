#include <plan_env/region_graph2d.h>

#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_set>

namespace jev_obj_planner {
namespace {

struct Component {
  int tile_x = 0;
  int tile_y = 0;
  std::vector<int> cells;
};

Eigen::Vector2d cellCenter(const RegionGridSnapshot& grid, int address)
{
  const int x = address / grid.height;
  const int y = address % grid.height;
  return grid.origin + grid.resolution * Eigen::Vector2d(x + 0.5, y + 0.5);
}

std::pair<int, int> tileFor(const RegionGridSnapshot& grid, int address, double tile_size)
{
  const Eigen::Vector2d position = cellCenter(grid, address);
  return { static_cast<int>(std::floor(position.x() / tile_size)),
    static_cast<int>(std::floor(position.y() / tile_size)) };
}

bool traversable(const RegionGridSnapshot& grid, int address)
{
  return grid.occupancy[address] == SDFMap2D::FREE && grid.inflated[address] != 1;
}

}  // namespace

bool RegionGridSnapshot::valid() const
{
  const int count = width * height;
  return width > 0 && height > 0 && resolution > 0.0 &&
      static_cast<int>(occupancy.size()) == count &&
      static_cast<int>(inflated.size()) == count &&
      static_cast<int>(semantic_value.size()) == count &&
      static_cast<int>(semantic_weight.size()) == count &&
      static_cast<int>(clearance.size()) == count;
}

RegionGraph2D::RegionGraph2D(double tile_size_m, double localization_tolerance_m)
  : tile_size_m_(tile_size_m), localization_tolerance_m_(localization_tolerance_m)
{
  if (!std::isfinite(tile_size_m_) || tile_size_m_ <= 0.0) tile_size_m_ = 3.0;
  if (!std::isfinite(localization_tolerance_m_) || localization_tolerance_m_ < 0.0)
    localization_tolerance_m_ = 0.35;
}

RegionGridSnapshot RegionGraph2D::capture(SDFMap2D& map)
{
  RegionGridSnapshot grid;
  const Eigen::Vector2i dimensions = map.getVoxelDimensions();
  Eigen::Vector2d size;
  map.getRegion(grid.origin, size);
  grid.width = dimensions.x();
  grid.height = dimensions.y();
  grid.resolution = map.getResolution();
  const int count = grid.width * grid.height;
  grid.occupancy.resize(count);
  grid.inflated.resize(count);
  grid.semantic_value.resize(count);
  grid.semantic_weight.resize(count);
  grid.clearance.resize(count);
  for (int x = 0; x < grid.width; ++x) {
    for (int y = 0; y < grid.height; ++y) {
      const Eigen::Vector2i index(x, y);
      const int address = grid.address(x, y);
      grid.occupancy[address] = map.getOccupancy(index);
      grid.inflated[address] = map.getInflateOccupancy(index);
      grid.semantic_value[address] = map.value_map_->getValue(index);
      grid.semantic_weight[address] = map.value_map_->getConfidence(index);
      grid.clearance[address] = map.getDistance(index);
      if (grid.occupancy[address] == SDFMap2D::FREE && grid.inflated[address] != 1)
        grid.traversable_addresses.insert(address);
    }
  }
  Eigen::Vector2i consumed_min, consumed_max;
  map.consumeRegionDirtyBox(consumed_min, consumed_max);
  return grid;
}

bool RegionGraph2D::refresh(SDFMap2D& map, RegionGridSnapshot& grid)
{
  const Eigen::Vector2i dimensions = map.getVoxelDimensions();
  Eigen::Vector2d origin, size;
  map.getRegion(origin, size);
  if (!grid.valid() || dimensions.x() != grid.width || dimensions.y() != grid.height ||
      std::abs(map.getResolution() - grid.resolution) > 1e-12 ||
      (origin - grid.origin).norm() > 1e-12) {
    grid = capture(map);
    return grid.valid();
  }
  Eigen::Vector2i min_index, max_index;
  if (!map.consumeRegionDirtyBox(min_index, max_index)) return true;
  for (int x = min_index.x(); x <= max_index.x(); ++x) {
    for (int y = min_index.y(); y <= max_index.y(); ++y) {
      const Eigen::Vector2i index(x, y);
      const int address = grid.address(x, y);
      grid.occupancy[address] = map.getOccupancy(index);
      grid.inflated[address] = map.getInflateOccupancy(index);
      grid.semantic_value[address] = map.value_map_->getValue(index);
      grid.semantic_weight[address] = map.value_map_->getConfidence(index);
      grid.clearance[address] = map.getDistance(index);
      if (traversable(grid, address)) grid.traversable_addresses.insert(address);
      else grid.traversable_addresses.erase(address);
    }
  }
  return true;
}

std::string RegionGraph2D::nextId()
{
  std::ostringstream output;
  output << 'R' << std::setw(4) << std::setfill('0') << next_region_id_++;
  return output.str();
}

void RegionGraph2D::resetEpisode()
{
  next_region_id_ = 1;
  graph_revision_ = 0;
  last_map_revision_ = -1;
  last_update_cache_hit_ = false;
  robot_localization_used_tolerance_ = false;
  robot_region_id_.clear();
  last_robot_region_id_.clear();
  regions_.clear();
  edges_.clear();
  cell_region_ids_.clear();
  cell_visits_.clear();
}

bool RegionGraph2D::update(const RegionGridSnapshot& grid, int map_revision, int robot_step,
    const Eigen::Vector2d& robot_position)
{
  if (!grid.valid() || map_revision < 0 || map_revision < last_map_revision_ || !robot_position.allFinite()) return false;
  const int rx = static_cast<int>(std::floor((robot_position.x()-grid.origin.x())/grid.resolution));
  const int ry = static_cast<int>(std::floor((robot_position.y()-grid.origin.y())/grid.resolution));
  if (rx>=0 && ry>=0 && rx<grid.width && ry<grid.height && traversable(grid,grid.address(rx,ry)))
    cell_visits_[grid.address(rx,ry)].insert(robot_step);
  if (map_revision == last_map_revision_) {
    last_update_cache_hit_ = true;
    robot_region_id_ = regionForPosition(grid, robot_position);
    const int exact_x = static_cast<int>(
        std::floor((robot_position.x() - grid.origin.x()) / grid.resolution));
    const int exact_y = static_cast<int>(
        std::floor((robot_position.y() - grid.origin.y()) / grid.resolution));
    const std::string exact_region = exact_x >= 0 && exact_y >= 0 &&
        exact_x < grid.width && exact_y < grid.height ?
        regionForAddress(grid.address(exact_x, exact_y)) : std::string();
    robot_localization_used_tolerance_ = exact_region.empty() && !robot_region_id_.empty();
    if (!robot_region_id_.empty()) {
      for (auto& region : regions_) {
        if (region.id != robot_region_id_) continue;
        region.visited = true;
        region.last_robot_visit_step = robot_step;
        if (robot_region_id_ != last_robot_region_id_) ++region.visit_count;
        break;
      }
      last_robot_region_id_ = robot_region_id_;
    }
    return true;
  }
  last_update_cache_hit_ = false;

  std::unordered_map<std::string, RegionNode2D> previous;
  for (const auto& region : regions_) previous[region.id] = region;
  const auto previous_cell_ids = cell_region_ids_;

  std::unordered_set<int> seen;
  std::vector<Component> components;
  const int dx[4] = { 1, -1, 0, 0 };
  const int dy[4] = { 0, 0, 1, -1 };
  for (int start : grid.traversable_addresses) {
      if (seen.count(start) || !traversable(grid, start)) continue;
      const auto tile = tileFor(grid, start, tile_size_m_);
      Component component;
      component.tile_x = tile.first;
      component.tile_y = tile.second;
      std::queue<int> pending;
      pending.push(start);
      seen.insert(start);
      while (!pending.empty()) {
        const int current = pending.front();
        pending.pop();
        component.cells.push_back(current);
        const int cx = current / grid.height;
        const int cy = current % grid.height;
        for (int direction = 0; direction < 4; ++direction) {
          const int nx = cx + dx[direction];
          const int ny = cy + dy[direction];
          if (nx < 0 || ny < 0 || nx >= grid.width || ny >= grid.height) continue;
          const int neighbor = grid.address(nx, ny);
          if (seen.count(neighbor) || !traversable(grid, neighbor) ||
              tileFor(grid, neighbor, tile_size_m_) != tile)
            continue;
          seen.insert(neighbor);
          pending.push(neighbor);
        }
      }
      std::sort(component.cells.begin(), component.cells.end());
      components.push_back(component);
  }
  std::sort(components.begin(), components.end(), [](const Component& left, const Component& right) {
    if (left.tile_x != right.tile_x) return left.tile_x < right.tile_x;
    if (left.tile_y != right.tile_y) return left.tile_y < right.tile_y;
    return left.cells.front() < right.cells.front();
  });

  // Match globally by overlap, so a split's main child inherits the old ID.
  struct Match { int overlap; std::string id; size_t component; };
  std::vector<Match> matches;
  for (size_t i = 0; i < components.size(); ++i) {
    std::map<std::string,int> counts;
    for (int cell : components[i].cells) {
      auto old = previous_cell_ids.find(cell);
      if (old != previous_cell_ids.end()) ++counts[old->second];
    }
    for (const auto& item : counts) matches.push_back({item.second, item.first, i});
  }
  std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
    if (a.overlap != b.overlap) return a.overlap > b.overlap;
    if (a.id != b.id) return a.id < b.id;
    return a.component < b.component;
  });
  std::map<size_t,std::string> inheritance;
  std::set<std::string> inherited;
  for (const auto& match : matches)
    if (!inheritance.count(match.component) && !inherited.count(match.id)) {
      inheritance[match.component] = match.id; inherited.insert(match.id);
    }
  size_t component_index = 0;
  regions_.clear();
  cell_region_ids_.clear();
  std::unordered_set<std::string> reused_ids;
  for (const auto& component : components) {
    std::map<std::string, int> overlaps;
    for (int address : component.cells) {
      const auto old = previous_cell_ids.find(address);
      if (old != previous_cell_ids.end()) ++overlaps[old->second];
    }
    std::vector<std::pair<int, std::string>> ranked;
    for (const auto& overlap : overlaps) ranked.push_back({ overlap.second, overlap.first });
    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
      return left.first != right.first ? left.first > right.first : left.second < right.second;
    });
    RegionNode2D region;
    if (inheritance.count(component_index)) region.id = inheritance.at(component_index);
    ++component_index;
    if (region.id.empty()) region.id = nextId();
    for (const auto& match : ranked) region.parent_ids.push_back(match.second);
    region.tile_x = component.tile_x;
    region.tile_y = component.tile_y;
    region.cell_addresses = component.cells;
    region.free_area_m2 = component.cells.size() * grid.resolution * grid.resolution;
    region.bbox_min = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    region.bbox_max = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    std::vector<double> values;
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (int address : component.cells) {
      const Eigen::Vector2d center = cellCenter(grid, address);
      centroid += center;
      region.bbox_min = region.bbox_min.cwiseMin(center - Eigen::Vector2d::Constant(0.5 * grid.resolution));
      region.bbox_max = region.bbox_max.cwiseMax(center + Eigen::Vector2d::Constant(0.5 * grid.resolution));
      const double value = grid.semantic_value[address];
      const double weight = grid.semantic_weight[address];
      if (regionSemanticSupported(value, weight)) {
        values.push_back(value);
        weighted_sum += value * weight;
        weight_sum += weight;
      }
    }
    centroid /= std::max<std::size_t>(1, component.cells.size());
    region.anchor = cellCenter(grid, component.cells.front());
    double best_anchor_distance = std::numeric_limits<double>::infinity();
    for (int address : component.cells) {
      const Eigen::Vector2d center = cellCenter(grid, address);
      const double distance = (center - centroid).squaredNorm();
      if (distance < best_anchor_distance) {
        best_anchor_distance = distance;
        region.anchor = center;
      }
    }
    region.semantic.valid_cell_count = values.size();
    region.semantic.valid_area_fraction = static_cast<double>(values.size()) / component.cells.size();
    if (!values.empty() && weight_sum > 0.0) {
      std::sort(values.begin(), values.end());
      const std::size_t p90_index = std::max<std::size_t>(1,
          static_cast<std::size_t>(std::ceil(0.9 * values.size()))) - 1;
      region.semantic.valid = true;
      region.semantic.weighted_mean = weighted_sum / weight_sum;
      region.semantic.p90 = values[p90_index];
      region.semantic.maximum = values.back();
    }
    std::set<int> visit_steps;
    for (int address : component.cells) {
      auto visits = cell_visits_.find(address);
      if (visits != cell_visits_.end()) visit_steps.insert(visits->second.begin(), visits->second.end());
    }
    region.visited = !visit_steps.empty();
    int previous_step = -2;
    for (int step : visit_steps) {
      if (step != previous_step + 1) ++region.visit_count;
      previous_step = step;
      region.last_robot_visit_step = step;
    }
    for (int address : component.cells) cell_region_ids_[address] = region.id;
    regions_.push_back(region);
  }

  robot_region_id_ = regionForPosition(grid, robot_position);
  const int exact_x = static_cast<int>(
      std::floor((robot_position.x() - grid.origin.x()) / grid.resolution));
  const int exact_y = static_cast<int>(
      std::floor((robot_position.y() - grid.origin.y()) / grid.resolution));
  const std::string exact_region = exact_x >= 0 && exact_y >= 0 &&
      exact_x < grid.width && exact_y < grid.height ?
      regionForAddress(grid.address(exact_x, exact_y)) : std::string();
  robot_localization_used_tolerance_ = exact_region.empty() && !robot_region_id_.empty();
  if (!robot_region_id_.empty()) {
    for (auto& region : regions_) {
      if (region.id != robot_region_id_) continue;
      region.visited = true;
      region.last_robot_visit_step = robot_step;
      break;
    }
    last_robot_region_id_ = robot_region_id_;
  }

  struct EdgeAggregate { int contacts = 0; double min_clearance = std::numeric_limits<double>::infinity(); };
  std::map<std::pair<std::string, std::string>, EdgeAggregate> aggregates;
  for (const auto& cell : cell_region_ids_) {
    const int address = cell.first;
    const int x = address / grid.height;
    const int y = address % grid.height;
    for (int direction : { 0, 2 }) {
      const int nx = x + dx[direction];
      const int ny = y + dy[direction];
      if (nx >= grid.width || ny >= grid.height) continue;
      const int neighbor = grid.address(nx, ny);
      const auto other = cell_region_ids_.find(neighbor);
      if (other == cell_region_ids_.end() || other->second == cell.second) continue;
      std::pair<std::string, std::string> key = cell.second < other->second ?
          std::make_pair(cell.second, other->second) : std::make_pair(other->second, cell.second);
      auto& aggregate = aggregates[key];
      ++aggregate.contacts;
      const double clearance = std::min(grid.clearance[address], grid.clearance[neighbor]);
      if (std::isfinite(clearance) && clearance >= 0.0)
        aggregate.min_clearance = std::min(aggregate.min_clearance, clearance);
    }
  }
  edges_.clear();
  int edge_index = 1;
  for (const auto& item : aggregates) {
    RegionEdge2D edge;
    edge.id = "E" + std::to_string(edge_index++);
    edge.from = item.first.first;
    edge.to = item.first.second;
    edge.portal_contact_count = item.second.contacts;
    edge.portal_width_estimate_m = item.second.contacts * grid.resolution;
    edge.min_clearance_m = std::isfinite(item.second.min_clearance) ?
        item.second.min_clearance : std::numeric_limits<double>::quiet_NaN();
    edge.verified_map_revision = map_revision;
    edges_.push_back(edge);
  }
  last_map_revision_ = map_revision;
  ++graph_revision_;
  return true;
}

std::string RegionGraph2D::regionForAddress(int address) const
{
  const auto found = cell_region_ids_.find(address);
  return found == cell_region_ids_.end() ? std::string() : found->second;
}

std::string RegionGraph2D::regionForPosition(
    const RegionGridSnapshot& grid, const Eigen::Vector2d& position) const
{
  if (!position.allFinite() || !grid.valid()) return {};
  const int x = static_cast<int>(std::floor((position.x() - grid.origin.x()) / grid.resolution));
  const int y = static_cast<int>(std::floor((position.y() - grid.origin.y()) / grid.resolution));
  if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) return {};
  const std::string exact = regionForAddress(grid.address(x, y));
  if (!exact.empty() || localization_tolerance_m_ <= 0.0) return exact;

  // Collision recovery can leave the robot in a cell newly marked inflated.
  // Snap only to the closest verified free region within the configured bound.
  const int radius = static_cast<int>(
      std::ceil(localization_tolerance_m_ / grid.resolution));
  double best_distance = std::numeric_limits<double>::infinity();
  int best_address = -1;
  for (int nx = std::max(0, x - radius); nx <= std::min(grid.width - 1, x + radius); ++nx) {
    for (int ny = std::max(0, y - radius); ny <= std::min(grid.height - 1, y + radius); ++ny) {
      const int address = grid.address(nx, ny);
      if (cell_region_ids_.find(address) == cell_region_ids_.end()) continue;
      const double distance = (cellCenter(grid, address) - position).norm();
      if (distance > localization_tolerance_m_ + 1e-9) continue;
      if (distance < best_distance - 1e-12 ||
          (std::abs(distance - best_distance) <= 1e-12 && address < best_address)) {
        best_distance = distance;
        best_address = address;
      }
    }
  }
  return best_address < 0 ? std::string() : regionForAddress(best_address);
}

std::vector<std::string> RegionGraph2D::routeRegions(
    const RegionGridSnapshot& grid, const std::vector<Eigen::Vector2d>& path) const
{
  std::vector<std::string> route;
  if (!grid.valid() || path.empty()) return route;
  auto append = [&](const Eigen::Vector2d& point) {
    if (!point.allFinite()) return false;
    const std::string id = regionForPosition(grid, point);
    if (id.empty()) return false;
    if (!route.empty() && route.back() != id) {
      bool linked = false;
      for (const auto& edge : edges_)
        if ((edge.from==route.back() && edge.to==id) ||
            (edge.bidirectional && edge.to==route.back() && edge.from==id)) linked=true;
      if (!linked) return false;
    }
    if (route.empty() || route.back()!=id) route.push_back(id);
    return true;
  };
  if (!append(path.front())) return {};
  for (size_t i=1;i<path.size();++i) {
    const Eigen::Vector2d from=path[i-1], delta=path[i]-from;
    if (!from.allFinite() || !path[i].allFinite()) return {};
    // Exact segment/grid intersections; sample each open cell interval.
    std::vector<double> cuts{0.0,1.0};
    for (int axis=0;axis<2;++axis) {
      if (std::abs(delta[axis])<1e-12) continue;
      double lo=std::min(from[axis],path[i][axis]), hi=std::max(from[axis],path[i][axis]);
      int first=static_cast<int>(std::floor((lo-grid.origin[axis])/grid.resolution))+1;
      int last=static_cast<int>(std::floor((hi-grid.origin[axis])/grid.resolution));
      for (int k=first;k<=last;++k) {
        const double t=(grid.origin[axis]+k*grid.resolution-from[axis])/delta[axis];
        if (t>0 && t<1) cuts.push_back(t);
      }
    }
    std::sort(cuts.begin(),cuts.end());
    cuts.erase(std::unique(cuts.begin(),cuts.end(),[](double x,double y){return std::abs(x-y)<1e-10;}),cuts.end());
    for(size_t j=1;j<cuts.size();++j)
      if (!append(from+delta*((cuts[j-1]+cuts[j])*0.5))) return {};
    if (!append(path[i])) return {};
  }
  return route;
}

std::vector<std::string> RegionGraph2D::routeBetweenRegions(
    const std::string& from, const std::string& to) const
{
  if (from.empty() || to.empty()) return {};
  if (from == to) return { from };
  std::queue<std::string> pending;
  std::map<std::string, std::string> parent;
  pending.push(from);
  parent[from] = std::string();
  while (!pending.empty() && !parent.count(to)) {
    const std::string current = pending.front();
    pending.pop();
    for (const auto& edge : edges_) {
      std::string neighbor;
      if (edge.from == current) neighbor = edge.to;
      else if (edge.bidirectional && edge.to == current) neighbor = edge.from;
      if (neighbor.empty() || parent.count(neighbor)) continue;
      parent[neighbor] = current;
      pending.push(neighbor);
    }
  }
  if (!parent.count(to)) return {};
  std::vector<std::string> route;
  for (std::string current = to; !current.empty(); current = parent.at(current))
    route.push_back(current);
  std::reverse(route.begin(), route.end());
  return route;
}

}  // namespace jev_obj_planner
