Header header

# Class id as produced by the network, and its human-readable name
# (resolved from the ~class_names ROS param; "unknown" if out of range).
int32 label_id
string label_name

# Detection confidence in [0, 1].
float32 score

# Center position of the 3D box, in the frame of header.frame_id.
float64 x
float64 y
float64 z

# Box dimensions in meters.
float64 length
float64 width
float64 height

# Heading around the Z axis, in radians.
float64 yaw

# Euclidean distance from the sensor origin (0,0,0 in header.frame_id)
# to the box center, in meters.
float64 distance
