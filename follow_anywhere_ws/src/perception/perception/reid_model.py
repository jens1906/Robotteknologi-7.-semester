"""Describes how a person looks (re-identification). Plain python, no ROS. Starts with OSNet.

To include:
- Input: person crops (RGB) cut from the colour image by their box, optionally keypoints.
  Output: embeddings (N, parts, D) and visibility (N, parts). OSNet has 1 part (the whole person);
  KPR, if swapped in later, has 9. Same interface for both, so leader_manager does not change.
- Distance between two people: mean euclidean distance over the parts visible in both
  (0 = same, 2 = opposite); NaN when they share too few parts (= unknown, never "different").
- OSNet: ONNX model run with onnxruntime, downloaded once to ~/.cache. GPU (CUDA provider) when
  available, else CPU with a warning. Measured: ~14 ms per crop on a laptop CPU with 2 threads.
- Only computes what it is asked for: the decision "is this the leader" is in leader_manager.
- Known OSNet weakness: two people overlapping in one box give a mixed description, so the leader is
  not compared while someone overlaps them.
"""
