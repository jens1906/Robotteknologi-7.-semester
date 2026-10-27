# Per-frame cost of perception, for plotting (PlotJuggler) and rosbags.
# header: stamp and frame_id of the camera image they belong to.
std_msgs/Header header
float32 detector_time_ms      # YOLO pose + BoT-SORT time for this image
float32 reid_time_ms          # time spent embedding the crops the leader logic asked for, 0 when none
int32 reid_crop_count         # crops embedded in this image
int32 people_count            # tracked people in this image
