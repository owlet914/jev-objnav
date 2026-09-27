#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>
#include <plan_env/map_ros.h>
#include <plan_env/raycast2d.h>
#include <plan_env/region_graph2d.h>
#include <iostream>
namespace jev_obj_planner {
struct RegionGraphFixtureAccess {
  static void initialize(SDFMap2D& m, ros::NodeHandle& nh) {
    m.mp_.reset(new MapParam2D); m.md_.reset(new MapData2D);
    m.mp_->resolution_=0.05; m.mp_->resolution_inv_=20;
    m.mp_->map_origin_=Eigen::Vector2d::Zero(); m.mp_->map_size_=Eigen::Vector2d(3,3);
    m.mp_->map_min_boundary_=m.mp_->map_origin_; m.mp_->map_max_boundary_=m.mp_->map_size_;
    m.mp_->map_voxel_num_=Eigen::Vector2i(60,60); m.mp_->obstacles_inflation_=0.18;
    m.mp_->clamp_min_log_=-2; m.mp_->clamp_max_log_=3; m.mp_->min_occupancy_log_=1;
    m.md_->occupancy_buffer_.assign(3600,-2); m.md_->occupancy_buffer_inflate_.assign(3600,0);
    m.md_->distance_buffer_.assign(3600,0.5);
    m.value_map_.reset(new ValueMap(&m,nh));
  }
  static void obstacle(SDFMap2D& m,int x,int y) {
    Eigen::Vector2i idx(x,y); m.md_->occupancy_buffer_[m.toAddress(idx)]=3;
    m.md_->local_update_min_=m.md_->local_update_max_=idx;
    m.indexToPos(idx,m.md_->local_update_mind_); m.md_->local_update_maxd_=m.md_->local_update_mind_;
    m.clearAndInflateLocalMap();
  }
};
}
int main(int argc,char** argv) {
  ros::init(argc,argv,"region_dirty_fixture",ros::init_options::AnonymousName);
  ros::NodeHandle nh; jev_obj_planner::SDFMap2D map;
  jev_obj_planner::RegionGraphFixtureAccess::initialize(map,nh);
  auto cached=jev_obj_planner::RegionGraph2D::capture(map);
  jev_obj_planner::RegionGraphFixtureAccess::obstacle(map,20,20);
  jev_obj_planner::RegionGraphFixtureAccess::obstacle(map,40,40);
  if(!jev_obj_planner::RegionGraph2D::refresh(map,cached)) return 1;
  auto full=jev_obj_planner::RegionGraph2D::capture(map);
  if(cached.inflated!=full.inflated || cached.occupancy!=full.occupancy || cached.traversable_addresses!=full.traversable_addresses) return 2;
  if(cached.inflated[cached.address(23,20)]!=1 || cached.inflated[cached.address(43,40)]!=1) return 3;
  Eigen::Vector2i lo,hi;
  if(map.consumeRegionDirtyBox(lo,hi)) return 4;
  std::cout<<"region_graph_dirty_fixture: PASS (two updates, inflation beyond one cell, cached equals full capture)"<<std::endl;
}
