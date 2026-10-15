# Layout measured from recorded robot data (differs from Unitree's repo definition,
# which has three video arrays). The robot sends one message per stream, alternating
# between 360 and 720; `data` holds one H.264 Annex-B access unit.
uint64 time_frame
uint32 height
uint8[] data
