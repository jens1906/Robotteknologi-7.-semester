# One tracked detection in one camera frame.
int32 id               # tracker id, the same person keeps it while the tracker follows them
string class_name      # e.g. person
float32 confidence     # detector score, 0..1

# bounding box in image pixels
float32 x_min
float32 y_min
float32 x_max
float32 y_max

# A person who is hidden for now (behind someone or something). The tracker still remembers them:
# the box is where they are predicted to be, confidence is 0, and they get the same id back when
# they are seen again.
bool occluded
float32 seconds_hidden   # how long they have not been seen, 0 for visible people
