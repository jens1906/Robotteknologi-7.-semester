# Load of the computer the evaluation node runs on (the Orin), once per second. For rosbags and PlotJuggler.
# NaN = not available on this computer (the tegrastats values exist only on a Jetson) or the node is not running.

std_msgs/Header header

# CPU and memory
float32 cpu_percent                        # all cores, mean
float32[] cpu_core_percent                 # per core
float32 ram_used_mb
float32 ram_total_mb
float32 swap_used_mb

# GPU, temperature, power (tegrastats)
float32 gpu_percent
float32 cpu_temperature                    # deg C
float32 gpu_temperature                    # deg C
float32 junction_temperature               # deg C, hottest point of the chip
float32 power_total_mw                     # VDD_IN, the whole module

# disk and network, since the previous message
float32 disk_read_mbps                     # MB/s
float32 disk_write_mbps                    # MB/s
float32 disk_free_gb                       # root file system
float32 network_receive_mbps               # MB/s, all interfaces except loopback
float32 network_send_mbps                  # MB/s, all interfaces except loopback
float32 loopback_mbps                      # MB/s, traffic between programs on this computer

# pipeline nodes: CPU in % of one core (100 = one full core), resident memory in MB
float32 perception_cpu_percent
float32 perception_memory_mb
float32 lidar_cpu_percent
float32 lidar_memory_mb
float32 navigation_cpu_percent
float32 navigation_memory_mb
float32 search_cpu_percent
float32 search_memory_mb
float32 camera_driver_cpu_percent
float32 camera_driver_memory_mb
float32 bag_recorder_cpu_percent
float32 bag_recorder_memory_mb
