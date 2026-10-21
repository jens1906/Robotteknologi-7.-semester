# Timing and count of one person-detector run, for plotting (PlotJuggler) and rosbags.
# header: stamp and frame_id of the camera image they belong to.
std_msgs/Header header
float32 detector_time_ms      # YOLO pose + BoT-SORT time for this image
int32 people_count            # tracked people in this image
