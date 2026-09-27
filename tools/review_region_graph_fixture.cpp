// Independent probes exercising production RegionGraph2D. No navigation/API.
#include <plan_env/region_graph2d.h>
#include <plan_env/sdf_map2d.h>
#include <cmath>
#include <iostream>
using namespace jev_obj_planner;
RegionGridSnapshot makeGrid(int w, int h, double r) {
  RegionGridSnapshot g; g.width=w; g.height=h; g.resolution=r;
  g.occupancy.assign(w*h,SDFMap2D::FREE); g.inflated.assign(w*h,0);
  g.semantic_value.assign(w*h,0); g.semantic_weight.assign(w*h,0); g.clearance.assign(w*h,1);
  for(int i=0;i<w*h;i++) g.traversable_addresses.insert(i);
  return g;
}
int main() {
  bool ok=true;
  auto g=makeGrid(120,120,0.05); RegionGraph2D graph(3);
  Eigen::Vector2d a(2.925,2.975), b=a+Eigen::Vector2d(0.25*std::cos(M_PI/6),0.125);
  graph.update(g,1,1,a);
  auto sparse=graph.routeRegions(g,{a,b});
  std::vector<Eigen::Vector2d> dense;
  for(int i=0;i<=100;i++) dense.push_back(a+(b-a)*(i/100.0));
  auto exact=graph.routeRegions(g,dense);
  bool direct=false;
  for(const auto& e:graph.edges()) if((e.from==sparse.front()&&e.to==sparse.back())||(e.to==sparse.front()&&e.from==sparse.back())) direct=true;
  std::cout<<"{\"sparse_route_regions\":"<<sparse.size()<<",\"dense_route_regions\":"<<exact.size()<<",\"sparse_hop_has_edge\":"<<(direct?"true":"false");
  ok &= sparse==exact;
  graph.update(g,9,9,a); bool stale=graph.update(g,2,2,a);
  std::cout<<",\"stale_revision_accepted\":"<<(stale?"true":"false")<<",\"last_map_revision\":"<<graph.lastMapRevision();
  ok &= !stale && graph.lastMapRevision()==9;
  auto split=makeGrid(10,1,0.1); RegionGraph2D splitGraph(3); splitGraph.update(split,1,1,Eigen::Vector2d(.05,.05));
  auto original=splitGraph.regions()[0].id; split.occupancy[1]=SDFMap2D::OCCUPIED; split.inflated[1]=1; split.traversable_addresses.erase(1);
  splitGraph.update(split,2,2,Eigen::Vector2d(.05,.05));
  size_t inherited=0; for(const auto& r:splitGraph.regions()) if(r.id==original) inherited=r.cell_addresses.size();
  ok &= inherited==8;
  std::cout<<",\"split_inherited_child_cells\":"<<inherited<<",\"split_largest_child_cells\":8}\n";
  return ok ? 0 : 1;
}
