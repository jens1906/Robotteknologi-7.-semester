# One obstacle: a person other than the leader. Frame: base_link (x forward, y left), z = 0. Units: m, m/s.

geometry_msgs/Point closest_point          # point of the obstacle nearest the robot
geometry_msgs/Point line_of_sight_point    # point of the obstacle nearest the line camera -> leader; NaN when there is no leader location
geometry_msgs/Vector3 velocity             # 0 when standing still (constant velocity assumed)
float32 position_uncertainty               # m, 1 sigma
float32 velocity_uncertainty               # m/s, 1 sigma; large = velocity unknown
