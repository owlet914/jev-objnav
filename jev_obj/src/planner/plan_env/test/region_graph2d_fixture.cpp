#include <plan_env/region_graph2d.h>
#include <plan_env/sdf_map2d.h>

#include <cmath>
#include <iostream>
#include <set>

using jev_obj_planner::RegionGraph2D;
using jev_obj_planner::RegionGridSnapshot;

namespace {

RegionGridSnapshot grid(int width, int height, double resolution = 1.0)
{
  RegionGridSnapshot result;
  result.width = width;
  result.height = height;
  result.resolution = resolution;
  const int count = width * height;
  result.occupancy.assign(count, jev_obj_planner::SDFMap2D::OCCUPIED);
  result.inflated.assign(count, 1);
  result.semantic_value.assign(count, 0.0);
  result.semantic_weight.assign(count, 0.0);
  result.clearance.assign(count, 0.5);
  return result;
}

void freeCell(RegionGridSnapshot& grid, int x, int y, double value = 0.0, double weight = 0.0)
{
  const int address = grid.address(x, y);
  grid.occupancy[address] = jev_obj_planner::SDFMap2D::FREE;
  grid.inflated[address] = 0;
  grid.semantic_value[address] = value;
  grid.semantic_weight[address] = weight;
  grid.traversable_addresses.insert(address);
}

bool expect(bool condition, const char* message)
{
  if (!condition) std::cerr << "FAIL: " << message << std::endl;
  return condition;
}

}  // namespace

int main()
{
  bool ok = true;
  ok &= expect(!jev_obj_planner::regionSemanticSupported(0.0,0.0), "unobserved zero must be unknown");
  ok &= expect(jev_obj_planner::regionSemanticSupported(0.0,1.0), "observed zero must be valid");
  ok &= expect(jev_obj_planner::regionSemanticSupported(-0.2,1.0), "observed negative must be valid");
  ok &= expect(!jev_obj_planner::regionSemanticSupported(std::nan(""),1.0), "NaN must be invalid");
  {
    RegionGridSnapshot map = grid(3, 3);
    for (int y = 0; y < 3; ++y) {
      freeCell(map, 0, y);
      freeCell(map, 2, y);
    }
    RegionGraph2D graph(3.0);
    ok &= expect(graph.update(map, 1, 1, Eigen::Vector2d(0.5, 0.5)), "wall map update");
    ok &= expect(graph.regions().size() == 2, "same tile wall must produce two regions");
    ok &= expect(graph.edges().empty(), "wall-separated regions must not have an edge");
    ok &= expect(graph.routeBetweenRegions(graph.regions()[0].id,
        graph.regions()[1].id).empty(), "disconnected regions must not have a graph route");
  }
  {
    RegionGridSnapshot map = grid(6, 2);
    for (int x = 0; x < 6; ++x) freeCell(map, x, 0, x == 0 ? -0.2 : x == 1 ? 0.0 : 0.4, 1.0);
    RegionGraph2D graph(3.0);
    ok &= expect(graph.update(map, 3, 4, Eigen::Vector2d(0.5, 0.5)), "door map update");
    ok &= expect(graph.regions().size() == 2, "tile boundary must preserve two regions");
    ok &= expect(graph.edges().size() == 1, "free tile portal must create one sparse edge");
    ok &= expect(graph.regions()[0].semantic.valid, "weighted semantic summary must be valid");
    ok &= expect(graph.regions()[0].semantic.maximum >= 0.0, "valid zero must not become missing");
    const int revision = graph.graphRevision();
    const std::string first_id = graph.regions()[0].id;
    ok &= expect(graph.update(map, 3, 5, Eigen::Vector2d(3.5, 0.5)), "cached update");
    ok &= expect(graph.lastUpdateWasCacheHit(), "same revision must hit topology cache");
    ok &= expect(graph.graphRevision() == revision, "cache hit must not advance graph revision");
    ok &= expect(graph.regions()[0].id == first_id, "cache hit must preserve stable id");
    ok &= expect(graph.robotRegionId() == graph.regions()[1].id, "cache hit must refresh robot region");
    const auto route = graph.routeRegions(map, { Eigen::Vector2d(0.5, 0.5),
      Eigen::Vector2d(2.5, 0.5), Eigen::Vector2d(3.5, 0.5), Eigen::Vector2d(5.5, 0.5) });
    ok &= expect(route.size() == 2, "path route must retain transit region sequence");
    const auto graph_route = graph.routeBetweenRegions(
        graph.regions()[0].id, graph.regions()[1].id);
    ok &= expect(graph_route.size() == 2, "connected regions must have a graph route");
    ok &= expect(graph.routeBetweenRegions(graph.regions()[0].id,
        graph.regions()[0].id).size() == 1, "same-region route must contain one region");
  }
  {
    RegionGridSnapshot map = grid(2, 2);
    freeCell(map, 0, 0);
    freeCell(map, 1, 1);
    RegionGraph2D graph(3.0);
    ok &= expect(graph.update(map, 1, 1, Eigen::Vector2d(0.5, 0.5)), "diagonal map update");
    ok &= expect(graph.regions().size() == 2, "diagonal contact must not merge regions");
    ok &= expect(graph.edges().empty(), "diagonal contact must not create an edge");
  }
  {
    RegionGridSnapshot map = grid(3, 1);
    freeCell(map, 0, 0);
    freeCell(map, 2, 0);
    RegionGraph2D graph(3.0, 1.1);
    ok &= expect(graph.update(map, 1, 1, Eigen::Vector2d(1.5, 0.5)),
        "tolerant robot localization update");
    ok &= expect(!graph.robotRegionId().empty(),
        "robot in a newly inflated cell must bind to a nearby verified free region");
    ok &= expect(graph.robotLocalizationUsedTolerance(),
        "tolerant assignment must be disclosed to the model contract");
    ok &= expect(graph.robotRegionId() == graph.regionForAddress(map.address(0, 0)),
        "equal-distance localization must be deterministic");
    RegionGraph2D strict_graph(3.0, 0.4);
    ok &= expect(strict_graph.update(map, 1, 1, Eigen::Vector2d(1.5, 0.5)),
        "strict robot localization update");
    ok &= expect(strict_graph.robotRegionId().empty(),
        "localization must not cross a gap beyond the configured tolerance");
  }
  {
    RegionGridSnapshot map = grid(6, 1);
    freeCell(map, 0, 0);
    freeCell(map, 1, 0);
    freeCell(map, 4, 0);
    freeCell(map, 5, 0);
    RegionGraph2D graph(3.0);
    ok &= expect(graph.update(map, 1, 1, Eigen::Vector2d(0.5, 0.5)), "inflated gap update");
    ok &= expect(graph.regions().size() == 2, "inflated boundary gap must preserve two regions");
    ok &= expect(graph.edges().empty(), "inflated boundary gap must not create an edge");
  }
  {
    RegionGridSnapshot map = grid(10, 1);
    for (int x = 0; x < 10; ++x) freeCell(map, x, 0, x == 9 ? 1.0 : 0.0, 1.0);
    RegionGraph2D graph(20.0);
    ok &= expect(graph.update(map, 1, 1, Eigen::Vector2d(0.5, 0.5)), "hotspot update");
    const auto& semantic = graph.regions()[0].semantic;
    ok &= expect(std::abs(semantic.weighted_mean - 0.1) < 1e-9, "hotspot mean must remain low");
    ok &= expect(std::abs(semantic.p90) < 1e-9, "nearest-rank p90 must preserve sparse hotspot contrast");
    ok &= expect(std::abs(semantic.maximum - 1.0) < 1e-9, "region max must retain hotspot");
  }
  {
    RegionGridSnapshot map = grid(9, 1);
    for (int x = 0; x < 9; ++x) freeCell(map, x, 0);
    RegionGraph2D graph(3.0);
    ok &= expect(graph.update(map, 1, 1, Eigen::Vector2d(0.5, 0.5)), "three tile route update");
    const auto route = graph.routeRegions(map, { Eigen::Vector2d(0.5, 0.5),
      Eigen::Vector2d(3.5, 0.5), Eigen::Vector2d(6.5, 0.5), Eigen::Vector2d(8.5, 0.5) });
    ok &= expect(route.size() == 3, "route must retain an intermediate region");
    ok &= expect(graph.edges().size() == 2, "three tile route must have two sparse edges");
  }
  {
    RegionGridSnapshot map = grid(3, 2);
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 2; ++y) freeCell(map, x, y);
    RegionGraph2D graph(3.0);
    ok &= expect(graph.update(map, 1, 1, Eigen::Vector2d(0.5, 0.5)), "initial merge map");
    const std::string parent = graph.regions()[0].id;
    for (int y = 0; y < 2; ++y) {
      const int address = map.address(1, y);
      map.occupancy[address] = jev_obj_planner::SDFMap2D::OCCUPIED;
      map.inflated[address] = 1;
      map.traversable_addresses.erase(address);
    }
    ok &= expect(graph.update(map, 2, 2, Eigen::Vector2d(0.5, 0.5)), "split update");
    ok &= expect(graph.regions().size() == 2, "wall insertion must split a region");
    int inherited = 0;
    for (const auto& region : graph.regions()) inherited += region.id == parent;
    ok &= expect(inherited == 1, "exactly one split child must inherit parent id");
    const std::set<std::string> split_ids{ graph.regions()[0].id, graph.regions()[1].id };
    for (int y = 0; y < 2; ++y) freeCell(map, 1, y);
    ok &= expect(graph.update(map, 3, 3, Eigen::Vector2d(0.5, 0.5)), "merge update");
    ok &= expect(graph.regions().size() == 1, "wall removal must merge regions");
    const std::set<std::string> parents(graph.regions()[0].parent_ids.begin(),
        graph.regions()[0].parent_ids.end());
    ok &= expect(parents == split_ids, "merged region must record both parent ids");
  }
  if (!ok) return 1;
  std::cout << "region_graph2d_fixture: PASS" << std::endl;
  return 0;
}
