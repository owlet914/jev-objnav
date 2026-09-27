#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
cd '/mnt/d/Desktop/jev导航porject'
mkdir -p .runtime/jev_region_graph/independent_review
# Compile production source directly: no stale object file or copied algorithm.
c++ -std=c++14 -O1 -Ijev_obj/src/planner/plan_env/include -I/usr/include/eigen3 \
  -I/opt/ros/noetic/include -I/usr/include/pcl-1.10 \
  tools/review_region_graph_fixture.cpp jev_obj/src/planner/plan_env/src/region_graph2d.cpp \
  -o .runtime/jev_region_graph/independent_review/region_probe \
  -Ljev_obj/devel/lib -lplan_env -L/opt/ros/noetic/lib -lroscpp -lrostime -lrosconsole \
  -Wl,-rpath,'/mnt/d/Desktop/jev导航porject/jev_obj/devel/lib'
.runtime/jev_region_graph/independent_review/region_probe > docs/JEV_REGION_GRAPH_CPP_REVIEW_EVIDENCE_AFTER_FIX.json
cat docs/JEV_REGION_GRAPH_CPP_REVIEW_EVIDENCE_AFTER_FIX.json
./jev_obj/devel/lib/plan_env/region_graph2d_fixture
