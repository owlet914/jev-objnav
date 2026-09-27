std_msgs/Header header
string episode_id
uint64 observation_id

# Point clouds produced from valid masks in this observation.
sensor_msgs/PointCloud2[] point_clouds

# Corresponding confidence scores
float32[] confidence_scores

# List of label indices
int32[] label_indices

# Per-detector provenance. requested_classes uses a JSON array string per detector
# because ROS1 messages do not support nested string arrays.
string[] detector_names
string[] detector_backends
bool[] detector_available
bool[] detector_fallback
bool[] detector_target_coverage
string[] detector_requested_classes
int32[] raw_detection_counts
int32[] target_match_counts
int32[] related_match_counts
int32[] valid_mask_counts

# Per semantic label (0=target, 1..N=related objects), true only when a
# non-fallback detector covered that label for this exact observation.
bool[] label_detection_valid
