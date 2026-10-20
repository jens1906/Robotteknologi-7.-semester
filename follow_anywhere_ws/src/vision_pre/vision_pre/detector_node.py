#Replay buffer
#When you are very confident this is the leader save a vector of it to get good definition of what leader is. 


import colorsys
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vision_interfaces.msg import Track, TrackArray

from vision_pre import leader_store
from vision_pre.appearance import Appearance, colour_hist, inner_crop
from vision_pre.embedder import make_embedder
from vision_pre.leader_tracker import Candidate, LeaderTracker, State
from vision_pre.relink import Relinker
from vision_pre.track_metrics import TrackStats
from vision_pre.video_source_node import FrameSource, resize_to_width

GREEN = (0, 220, 0)
ORANGE = (0, 165, 255)
RED = (60, 60, 255)
MIN_BOX = 24   # px, ignore smaller detections
MAX_DEPTH_AGE = 0.3   # s, older depth frames are not used for positions
MIN_GHOST_LEN = 3     # a track needs this many matched frames before it is remembered when hidden


def id_colour(track_id):
    """Vibrant BGR colour for a track id, the same every frame. Green is kept for the leader."""
    deg = (track_id * 0.61803398875 % 1.0) * 240      # golden-ratio steps spread neighbouring ids
    if deg > 60:
        deg += 120                                    # skip the green band, 60..180 degrees
    r, g, b = colorsys.hsv_to_rgb(deg / 360, 1.0, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def put_label(frame, text, x, y, colour, scale, thickness, ink=(0, 0, 0)):
    """Text on a filled chip, so it reads on any background."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = max(0, y - h - base - 4)
    cv2.rectangle(frame, (x, top), (x + w + 6, top + h + base + 4), colour, -1)
    cv2.putText(frame, text, (x + 3, top + h + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                ink, thickness)


def default_tracker_cfg():
    """The tracker config shipped with this package, or ByteTrack if it is not installed."""
    try:
        path = os.path.join(get_package_share_directory('vision_pre'), 'config', 'botsort_tuned.yaml')
    except PackageNotFoundError:
        return 'bytetrack.yaml'
    return path if os.path.isfile(path) else 'bytetrack.yaml'


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector')
        p = self.declare_parameter
        p('yolo_model', 'yolov8n.pt')
        p('target_class', 0)              # YOLO class id of the leader, -1 = any class
        p('imgsz', 640)
        p('tracker_cfg', default_tracker_cfg())   # tracker yaml: bytetrack.yaml or a path
        p('reid_model', 'auto')           # tracker appearance: auto (YOLO), osnet, .onnx path
        p('occlusion_keep_s', 10.0)       # s a hidden person is remembered and published
        p('lost_halflife_s', 1.0)         # s, a hidden person's predicted speed halves
        p('ghost_filter', True)           # only list hidden people that are plausible (see below)
        p('ghost_min_len', 8)             # matched frames a track needs before it is remembered
        p('ghost_dup_iou', 0.5)           # a hidden box this much on a visible one is a duplicate
        p('ghost_cover_iou', 0.05)        # a hidden person needs a visible one this much on them
        p('relink', False)                # give a returning hidden person their old id (OSNet)
        p('relink_sim', 0.7)              # cosine similarity needed to relink
        p('relink_margin', 0.05)          # lead needed over the next best hidden person
        # input: the RealSense topics, or a video played in real time instead
        p('use_dataset', False)           # True: play video_path as the camera (no RealSense)
        p('video_path', '')               # video file, or a folder of images (a MOT17 sequence)
        p('video_loop', False)            # repeat the video; otherwise the node ends with it
        p('video_width', 640)             # px, the video is resized to this width, 0 = keep
        p('video_fps', 0.0)               # 0 = frame rate of the file
        p('report_tracks', False)         # ID-stability numbers on exit (always with use_dataset)
        p('report_file', '')              # MOT txt of the tracks, empty = ~/.ros/follow_anywhere/
        p('gt_file', '')                  # ground truth to score, empty = <video_path>/gt/gt.txt
        p('gt_min_height', 40)            # px (at video_width): smaller people are not scored
        p('process_hz', 50.0)             # upper limit, real rate is set by the model speed
        p('embedder', 'osnet')            # osnet (person re-identification) or dino (generic)
        p('osnet_path', '')               # empty = download once to ~/.cache/follow_anywhere
        # matching: scores are 0..1, the overlay shows them as %, tune from there
        p('sim_threshold', 0.75)          # score needed to accept a candidate as the leader
        p('margin', 0.05)                 # lead needed over the next best candidate
        p('other_margin', 0.02)           # lead needed over known bystanders
        p('drop_floor', 0.5)              # below this a continuous track is doubted
        p('w_emb', 0.8)                   # weight of the neural embedding in the score
        p('w_col', 0.2)                   # weight of clothing colour (0 = off)
        # sticky lock
        p('uncertain_on_overlap', True)   # crossing someone at the same distance -> UNCERTAIN
        p('depth_gap', 0.8)               # m, people further apart in depth are not a crossing
        p('uncertain_timeout', 5.0)       # s, UNCERTAIN turns into LOST after this
        p('motion_gate', True)            # reject candidates the leader could not have reached
        p('max_speed', 1.5)               # m/s
        p('gate_radius', 0.6)             # m
        # memory
        p('calib_frames', 30)
        p('lose_after', 3)                # very low checks in a row before UNCERTAIN
        p('rolling_size', 30)             # views learned while tracking (calibration is kept)
        p('update_max_sim', 0.97)         # views more similar than this are not stored again
        p('leader_file', '~/.ros/follow_anywhere/leader.npz')
        # display
        p('show', True)                   # open the OpenCV window (keys: c = calibrate, r = reset)
        p('window_width', 1280)           # px, drawn this wide; drag the window corner to scale
        p('show_similarity', True)        # leader match % on every detected target
        p('diag', True)                   # score breakdown under each box
        p('sim_refresh', 1.0)             # s, how often each person's match % is refreshed
        p('max_embeds_per_frame', 2)      # cap on extra embedder runs for the match % labels
        p('color_topic', '/camera/camera/color/image_raw')
        p('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        p('info_topic', '/camera/camera/color/camera_info')
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        self.target_class = g('target_class')
        self.imgsz = g('imgsz')
        self.tracker_cfg = self.tracker_config(os.path.expanduser(g('tracker_cfg')),
                                               g('reid_model'), g('osnet_path'))
        self.occlusion_keep_s = g('occlusion_keep_s')
        self.lost_halflife_s = g('lost_halflife_s')
        self.hidden = []                  # (id, xyxy, seconds hidden, class) of remembered people
        self.relinker = (Relinker(keep_s=g('occlusion_keep_s'), sim_threshold=g('relink_sim'),
                                  margin=g('relink_margin')) if g('relink') else None)
        self.last_seen = {}               # track id -> time it was last visible
        self.hooks_warned = False
        self.ghost_filter = g('ghost_filter')
        self.ghost_min_len = g('ghost_min_len') if self.ghost_filter else MIN_GHOST_LEN
        self.ghost_dup_iou = g('ghost_dup_iou')
        self.ghost_cover_iou = g('ghost_cover_iou')
        self.ghost_rows = []              # hidden people per processed frame, for the report
        self.dt = 1 / 15                  # s between the last two processed frames
        self.last_stamp = None
        self.use_dataset = g('use_dataset')
        self.video_loop = g('video_loop')
        self.video_width = g('video_width')
        self.video_fps = g('video_fps')
        self.report = self.use_dataset or g('report_tracks')
        clip = 'live'
        if g('video_path'):
            clip = os.path.splitext(os.path.basename(os.path.normpath(g('video_path'))))[0]
        tracker_name = os.path.splitext(os.path.basename(self.tracker_cfg))[0]
        self.report_file = os.path.expanduser(g('report_file')) or os.path.expanduser(
            time.strftime(f'~/.ros/follow_anywhere/tracks_{clip}_{tracker_name}_%Y%m%d_%H%M%S.txt')
        )
        gt_file = os.path.expanduser(g('gt_file'))
        if not gt_file and os.path.isdir(os.path.expanduser(g('video_path'))):
            gt_file = os.path.join(os.path.expanduser(g('video_path')), 'gt', 'gt.txt')
        self.gt_file = gt_file if self.use_dataset and os.path.isfile(gt_file) else ''
        self.gt_min_height = g('gt_min_height')
        self.frame_scale = 1.0               # original image pixels per processed pixel
        self.stats = TrackStats()
        self.infer_ms_sum = 0.0
        self.hidden_sum = 0               # remembered-but-hidden people, summed over frames
        self.done = False                 # set when the video is over, ends main()
        self.video_done = False
        self.stop_video = False
        self.source = None
        if self.use_dataset:
            if not g('video_path'):
                raise ValueError('use_dataset is true: set video_path to a video or image folder')
            self.source = FrameSource(g('video_path'))     # fails now, before the slow model load
        self.show = g('show')
        self.window_width = g('window_width')
        self.window_open = False
        self.show_similarity = g('show_similarity')
        self.diag = g('diag')
        self.sim_refresh = g('sim_refresh')
        self.max_embeds = g('max_embeds_per_frame')
        self.leader_file = g('leader_file')
        self.sims = {}                # track_id -> (time, Score) against the leader

        self.bridge = CvBridge()
        self.latest = None            # newest colour frame only, older ones are dropped
        self.depth_msg = None
        self.intr = None              # fx, fy, cx, cy
        self.lock = threading.Lock()

        # fps bookkeeping: frames received / frames processed per 1 s window
        self.n_in = self.n_out = 0
        self.fps_in = self.fps_out = 0.0
        self.t_window = time.perf_counter()

        self.get_logger().info(
            f'Loading YOLO and the re-identification model (tracker: {self.tracker_cfg})...')
        # Ultralytics would pip-install missing packages (it pulled numpy 2 into the venv once,
        # which broke everything), so keep it from touching the environment.
        os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
        from ultralytics import YOLO
        self.yolo = YOLO(g('yolo_model'))
        try:
            self.embedder = make_embedder(g('embedder'), g('osnet_path'))
        except ImportError as e:
            self.get_logger().error(
                f'Missing package for embedder "{g("embedder")}": {e}. For osnet run: '
                'pip install --user --break-system-packages onnxruntime')
            raise
        self.tracker = LeaderTracker(
            target_class=self.target_class, sim_threshold=g('sim_threshold'),
            margin=g('margin'), other_margin=g('other_margin'), drop_floor=g('drop_floor'),
            w_emb=g('w_emb'), w_col=g('w_col'), calib_frames=g('calib_frames'),
            lose_after=g('lose_after'), rolling_size=g('rolling_size'),
            update_max_sim=g('update_max_sim'), uncertain_on_overlap=g('uncertain_on_overlap'),
            depth_gap=g('depth_gap'), uncertain_timeout=g('uncertain_timeout'),
            motion_gate=g('motion_gate'), max_speed=g('max_speed'),
            gate_radius=g('gate_radius'))

        saved = leader_store.load(self.leader_file)
        if saved is not None:
            gallery, cls, col, made_with = saved
            if made_with == self.embedder.name:
                self.tracker.load(gallery, cls, col)
                self.get_logger().info(f'Loaded leader from {self.leader_file}, searching for it')
            else:
                self.get_logger().warn(
                    f'Saved leader was made with "{made_with or "unknown"}" but the embedder '
                    f'is "{self.embedder.name}": ignored, calibrate again (press c)')

        if not self.use_dataset:
            self.create_subscription(Image, g('color_topic'), self.on_image,
                                     qos_profile_sensor_data)
            self.create_subscription(Image, g('depth_topic'), self.on_depth,
                                     qos_profile_sensor_data)
            self.create_subscription(CameraInfo, g('info_topic'), self.on_info,
                                     qos_profile_sensor_data)
        self.state_pub = self.create_publisher(String, '/vision/leader_state', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/vision/leader_pose', 1)
        self.tracks_pub = self.create_publisher(TrackArray, '/vision/tracks', 10)
        self.create_service(Trigger, '/vision/calibrate_leader', self.on_calibrate)

        # Inference runs at its own rate, decoupled from the camera fps
        self.create_timer(1 / g('process_hz'), self.process)

        if self.use_dataset:
            self.get_logger().info(
                f'Playing {g("video_path")} as the camera at '
                f'{self.video_fps or self.source.fps:.1f} fps (real time, frames are dropped '
                'when inference is slower)')
            threading.Thread(target=self.play_video, daemon=True).start()

    # callbacks -----------------------------------------------------------
    def on_image(self, msg: Image):
        # Keep the callback cheap: just store the newest message
        with self.lock:
            self.latest = msg
            self.n_in += 1

    def play_video(self):
        """Feed the video in as if it were the camera: real time, newest frame wins."""
        period = 1.0 / (self.video_fps or self.source.fps)
        t_next = time.monotonic()
        while not self.stop_video and rclpy.ok():
            frame = self.source.read()
            if frame is None:
                if self.video_loop:
                    self.source.rewind()
                    continue
                break
            msg = self.bridge.cv2_to_imgmsg(resize_to_width(frame, self.video_width), 'bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f'dataset:{self.source.index}'   # frame number, for scoring
            self.on_image(msg)
            t_next += period
            wait = t_next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                t_next = time.monotonic()             # decoding fell behind: do not burst
        self.video_done = True

    def tracker_config(self, cfg, reid_model, osnet_path):
        """The tracker yaml to use; with a ReID model other than 'auto' a copy that sets it."""
        if reid_model == 'auto':
            return cfg
        os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
        from ultralytics.utils.checks import check_yaml
        from vision_pre.embedder import OSNET_URL, _ensure_osnet
        model = (_ensure_osnet(osnet_path, OSNET_URL) if reid_model == 'osnet'
                 else os.path.expanduser(reid_model))
        with open(check_yaml(cfg)) as f:
            data = yaml.safe_load(f)
        if data.get('tracker_type') != 'botsort':
            self.get_logger().warn(f'reid_model needs a botsort tracker, ignored for {cfg}')
            return cfg
        data.update(with_reid=True, model=model)
        out = os.path.expanduser('~/.ros/follow_anywhere/tracker_runtime.yaml')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            yaml.safe_dump(data, f)
        self.get_logger().info(f'Tracker appearance model: {model}')
        return out

    def dataset_frame(self, msg, frame):
        """(video frame number, factor to original pixels) of a played frame, else (None, 1)."""
        tag = msg.header.frame_id
        if self.source is None or not tag.startswith('dataset:'):
            return None, 1.0
        return int(tag.split(':')[1]), self.source.orig_size[0] / frame.shape[1]

    def on_depth(self, msg: Image):
        self.depth_msg = msg

    def on_info(self, msg: CameraInfo):
        self.intr = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_calibrate(self, request, response):
        self.tracker.start_calibration(time.monotonic())
        response.success = True
        response.message = 'Calibrating: keep the leader near the centre of the image'
        self.get_logger().info(response.message)
        return response

    # pipeline ------------------------------------------------------------
    def detect(self, frame):
        """Run YOLO tracking, return confirmed tracks as Candidates and the class names."""
        if self.target_class >= 0:
            classes = [self.target_class]
        elif self.tracker.state in (State.TRACKING, State.UNCERTAIN, State.LOST):
            classes = [self.tracker.leader_cls]
        else:
            classes = None
        res = self.yolo.track(frame, persist=True, tracker=self.tracker_cfg,
                              classes=classes, imgsz=self.imgsz, verbose=False)[0]
        cands = []
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            for tid, cls, xyxy, conf in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(),
                                            boxes.xyxy.tolist(), boxes.conf.tolist()):
                if xyxy[2] - xyxy[0] >= MIN_BOX and xyxy[3] - xyxy[1] >= MIN_BOX:
                    cands.append(Candidate(tid, cls, tuple(xyxy), conf))
        self.update_hidden(cands, frame.shape)
        return cands, res.names

    def update_hidden(self, cands, shape):
        """Occlusion memory: remember hidden people for occlusion_keep_s and list them.

        This reaches into the Ultralytics tracker (predictor.trackers[0], lost_stracks,
        max_frames_lost). That is not a public API, so it is guarded: if the layout is different
        the tracker keeps its own defaults and nothing is listed.
        """
        self.hidden = []
        now = self.last_stamp
        for c in cands:
            self.last_seen[c.track_id] = now
        trackers = getattr(getattr(self.yolo, 'predictor', None), 'trackers', None)
        trk = trackers[0] if trackers else None
        if trk is None or not hasattr(trk, 'lost_stracks') or not hasattr(trk, 'max_frames_lost'):
            if not self.hooks_warned:
                self.get_logger().warn('Tracker internals not found, occlusion memory is off')
                self.hooks_warned = True
            return
        # the tracker counts processed frames, so convert seconds with the real processing rate
        trk.max_frames_lost = max(30, math.ceil(self.occlusion_keep_s * (self.fps_out or 15.0)))
        damp = 0.5 ** (self.dt / self.lost_halflife_s) if self.lost_halflife_s > 0 else 1.0
        for st in trk.lost_stracks:
            if st.mean is None:
                continue
            st.mean[4:8] *= damp              # a hidden person's predicted box slows down
            hidden_s = now - self.last_seen.get(st.track_id, now)
            if hidden_s <= self.occlusion_keep_s and st.tracklet_len >= self.ghost_min_len:
                x, y, w, h = st.tlwh
                box = (x, y, x + w, y + h)
                if self.ghost_filter and not self.plausible_ghost(box, cands, shape):
                    continue
                self.hidden.append((st.track_id, box, hidden_s, int(st.cls)))
        if len(self.last_seen) > 1000:
            old = now - 2 * self.occlusion_keep_s
            self.last_seen = {i: t for i, t in self.last_seen.items() if t >= old}

    def plausible_ghost(self, box, cands, shape):
        """A remembered person is only listed while it makes sense that they are just hidden.

        Not when the predicted spot is outside the picture (they walked out), and not when it is
        practically the box of somebody visible (the person came back under another id). With
        ghost_cover_iou, only people who have a visible person in front of them count.
        """
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (0 <= cx < shape[1] and 0 <= cy < shape[0]):
            return False
        overlaps = [LeaderTracker._iou(box, c.box) for c in cands]
        if any(o >= self.ghost_dup_iou for o in overlaps):
            return False
        # somebody hidden behind another person is partly under that person's box
        return self.ghost_cover_iou <= 0 or any(o >= self.ghost_cover_iou for o in overlaps)

    def pid(self, tracker_id):
        """The id shown and published for a tracker id (the old one if it was relinked)."""
        return self.relinker.published(tracker_id) if self.relinker is not None else tracker_id

    def embed_box(self, frame, box):
        crop = inner_crop(frame, box)
        return self.embedder.embed(crop) if crop.size else None

    def appearance(self, frame, cand):
        """Neural embedding and clothing colour of one detection."""
        crop = inner_crop(frame, cand.box)
        return Appearance(self.embedder.embed(crop), colour_hist(crop))

    def embed_and_score(self, frame, cand, now):
        """Describe a candidate and remember how well it matches the leader."""
        app = self.appearance(frame, cand)
        tracker = self.tracker
        if tracker.state != State.CALIBRATING and tracker.gallery is not None:
            self.sims[cand.track_id] = (now, tracker.score(app, cand, now))
            if tracker.state == State.TRACKING and cand.track_id != tracker.locked_id:
                tracker.observe_other(cand.track_id, app, now)   # teaches the bystander memory
        return app

    def refresh_similarities(self, frame, cands, now):
        """Keep a match % for every target of the leader's class (a few per frame)."""
        tracker = self.tracker
        if tracker.state == State.CALIBRATING or tracker.gallery is None:
            return
        self.sims = {i: v for i, v in self.sims.items()
                     if any(c.track_id == i for c in cands)}
        stale = sorted(
            (c for c in cands
             if c.cls == tracker.leader_cls
             # the tracker already re-checks the locked leader itself
             and not (tracker.state == State.TRACKING and c.track_id == tracker.locked_id)
             and now - self.sims.get(c.track_id, (-1e9, None))[0] >= self.sim_refresh),
            key=lambda c: self.sims.get(c.track_id, (-1e9, None))[0])
        for c in stale[:self.max_embeds]:
            self.embed_and_score(frame, c, now)

    def current_depth(self, color_msg):
        """The aligned depth image if it belongs to (about) the same moment, else None."""
        d = self.depth_msg
        if d is None or self.intr is None:
            return None
        dt = abs((color_msg.header.stamp.sec - d.header.stamp.sec)
                 + (color_msg.header.stamp.nanosec - d.header.stamp.nanosec) * 1e-9)
        if dt > MAX_DEPTH_AGE:
            return None
        return self.bridge.imgmsg_to_cv2(d, 'passthrough')

    def box_position(self, cand, depth, color_msg):
        """3D position (m) of a detection in the colour camera frame, or None."""
        sx = depth.shape[1] / color_msg.width
        sy = depth.shape[0] / color_msg.height
        u, v = cand.center
        half_w = (cand.box[2] - cand.box[0]) / 6     # central third of the box
        half_h = (cand.box[3] - cand.box[1]) / 6
        patch = depth[int(max(0, (v - half_h) * sy)):int((v + half_h) * sy) + 1,
                      int(max(0, (u - half_w) * sx)):int((u + half_w) * sx) + 1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        z = float(np.median(valid))
        if depth.dtype == np.uint16:
            z /= 1000.0                                   # mm -> m
        fx, fy, cx, cy = self.intr
        return (u - cx) * z / fx, (v - cy) * z / fy, z

    def publish_tracks(self, header, cands, names):
        """Every tracked detection of this frame (also an empty list), id = tracker's track id."""
        out = TrackArray(header=header)
        for c in cands:
            out.tracks.append(Track(id=self.pid(c.track_id), class_name=names[c.cls],
                                    confidence=float(c.conf),
                                    x_min=float(c.box[0]), y_min=float(c.box[1]),
                                    x_max=float(c.box[2]), y_max=float(c.box[3])))
        for tid, box, hidden_s, cls in self.hidden:           # remembered, not visible now
            out.tracks.append(Track(id=self.pid(tid), class_name=names.get(cls, str(cls)),
                                    confidence=0.0,
                                    x_min=float(box[0]), y_min=float(box[1]),
                                    x_max=float(box[2]), y_max=float(box[3]),
                                    occluded=True, seconds_hidden=float(hidden_s)))
        self.tracks_pub.publish(out)

    def print_report(self):
        """ID-stability numbers of this run (see track_metrics) and the tracks as a MOT file."""
        if not self.report:
            return
        s = self.stats.summary()
        if not s.frames:
            print('detector: no frames were processed, nothing to report', flush=True)
            return
        self.stats.write_mot(self.report_file)
        fps = s.frames / s.seconds if s.seconds > 0 else 0.0
        print('\n=== ID stability ===\n' + s.text()
              + f'\nprocessed {fps:.1f} fps | inference mean {self.infer_ms_sum / s.frames:.0f} ms'
              + f' | hidden people remembered {self.hidden_sum / s.frames:.1f} per frame'
              + (f' | ids given back {self.relinker.relinks}' if self.relinker else '')
              + f'\nMOT tracks written to {self.report_file}', flush=True)
        if self.gt_file:
            for line in self.ground_truth_text():
                print(line, flush=True)

    def ground_truth_text(self):
        """Lines comparing this run with the ground truth: the tracks and the hidden people."""
        try:
            from vision_pre import mot_eval
            gt = mot_eval.load_gt(self.gt_file)
            kw = dict(min_height=self.gt_min_height * self.frame_scale)
            sc = mot_eval.score(gt, self.stats.rows, self.stats.frame_numbers, **kw)
            # hidden people are exactly the badly visible ones, so they all count here
            ghost = mot_eval.ghost_quality(gt, self.ghost_rows, self.stats.rows,
                                           self.stats.frame_numbers, min_vis=0.0, **kw)
        except ImportError:
            return ['Ground-truth scoring needs motmetrics: .venv/bin/pip install motmetrics']
        return [sc.text() if sc else 'No ground truth to score in the processed frames']\
            + ([ghost.text()] if ghost else [])

    def handle_event(self, res):
        ev, why = res.event, f' ({res.reason})' if res.reason else ''
        if ev == 'calibrated':
            # only the calibration views are saved, the rolling memory is per session
            leader_store.save(self.leader_file, self.tracker.anchors, self.tracker.leader_cls,
                              self.tracker.anchor_cols, self.embedder.name)
            self.get_logger().info(
                f'Leader calibrated ({len(self.tracker.anchors)} samples), '
                f'saved to {self.leader_file}')
        elif ev == 'calibration_failed':
            self.get_logger().warn('Calibration failed: no target seen long enough')
        elif ev == 'lost':
            self.get_logger().warn(f'Leader lost, searching{why}')
        elif ev == 'uncertain':
            self.get_logger().warn(f'Not sure who the leader is{why}, waiting to verify')
        elif ev == 'verified':
            self.get_logger().info(f'Leader verified again (score {res.sim:.2f})')
        elif ev == 'switched':
            self.get_logger().warn(
                f'Leader is now track #{res.leader.track_id} (score {res.sim:.2f}), '
                f'lock changes so far: {self.tracker.lock_changes}')
        elif ev == 'reacquired':
            self.get_logger().info(f'Leader found again (score {res.sim:.2f})')

    def process(self):
        video_over = self.video_done          # read before taking the frame: the last one is in
        with self.lock:
            msg, self.latest = self.latest, None
        if msg is None:
            if video_over:
                self.done = True
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None:
            self.dt = min(1.0, max(0.01, stamp - self.last_stamp))
        self.last_stamp = stamp
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')   # numpy HxWx3, OpenCV order
        t0 = time.perf_counter()
        cands, names = self.detect(frame)
        if self.relinker is not None:
            self.relinker.update(
                stamp, [(c.track_id, c.box) for c in cands],
                [(tid, box, secs) for tid, box, secs, _ in self.hidden],
                lambda box: self.embed_box(frame, box), (frame.shape[1], frame.shape[0]))
        depth = self.current_depth(msg)
        if depth is not None:
            for c in cands:
                c.pos = self.box_position(c, depth, msg)

        now_m = time.monotonic()
        res = self.tracker.update(cands, lambda c: self.embed_and_score(frame, c, now_m),
                                  (frame.shape[1], frame.shape[0]), now_m)
        if self.show and self.show_similarity:
            self.refresh_similarities(frame, cands, now_m)
        infer_ms = (time.perf_counter() - t0) * 1000
        self.n_out += 1
        self.handle_event(res)

        self.state_pub.publish(String(data=res.state.value))
        self.publish_tracks(msg.header, cands, names)
        if self.report:
            self.infer_ms_sum += infer_ms
            self.hidden_sum += len(self.hidden)
            number, self.frame_scale = self.dataset_frame(msg, frame)
            self.ghost_rows.extend(
                (number, self.pid(tid), box[0] * self.frame_scale, box[1] * self.frame_scale,
                 (box[2] - box[0]) * self.frame_scale, (box[3] - box[1]) * self.frame_scale, secs)
                for tid, box, secs, _ in self.hidden)
            self.stats.add(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                           [(self.pid(c.track_id), c.box, c.conf) for c in cands], number,
                           self.frame_scale)
        # the pose is only handed on while the identity is certain
        if res.state == State.TRACKING and res.leader is not None and res.leader.pos is not None:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = res.leader.pos
            pose.pose.orientation.w = 1.0
            self.pose_pub.publish(pose)

        # Update the fps numbers once per second
        now = time.perf_counter()
        if now - self.t_window >= 1.0:
            dt = now - self.t_window
            with self.lock:
                self.fps_in, self.n_in = self.n_in / dt, 0
            self.fps_out, self.n_out = self.n_out / dt, 0
            self.t_window = now

        if self.show:
            self.draw(frame, cands, names, res, infer_ms)

    def draw(self, frame, cands, names, res, infer_ms):
        # draw on an enlarged copy, so lines and text stay sharp in a big window
        k = self.window_width / frame.shape[1] if self.window_width else 1.0
        if k != 1.0:
            frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        line = max(2, round(2 * k))
        fs, ft = 0.42 * k, max(1, round(0.75 * k))
        leader_id = res.leader.track_id if res.leader is not None else None
        calibrating = res.state == State.CALIBRATING
        for c in cands:
            is_leader = c.track_id == leader_id
            x1, y1, x2, y2 = (int(v * k) for v in c.box)
            colour = id_colour(self.pid(c.track_id))
            label = f'#{self.pid(c.track_id)}'
            if names[c.cls] != 'person':
                label = f'{names[c.cls]} {label}'
            score = self.sims.get(c.track_id, (0, None))[1]
            if is_leader and calibrating:
                label, colour = 'CALIBRATING', GREEN
            else:
                if is_leader:
                    label, colour = f'LEADER {label}', GREEN
                pct = score.fused if score is not None else (res.sim if is_leader else None)
                if pct is not None:
                    label += f' {max(0.0, pct) * 100:.0f}%'
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, line * 2 if is_leader else line)
            put_label(frame, label, x1, y1, colour, fs, ft)
            if self.diag and score is not None and not calibrating:
                parts = f'emb {score.emb * 100:.0f}'
                if score.col is not None:
                    parts += f' col {score.col * 100:.0f}'
                if score.moved:
                    parts += ' OUT-OF-REACH'
                cv2.putText(frame, parts, (x1, min(frame.shape[0] - 5, y2 + int(18 * k))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, colour, ft)

        for tid, box, hidden_s, _ in self.hidden:                 # remembered, drawn thin and dim
            tid = self.pid(tid)
            colour = id_colour(tid)
            x1, y1, x2, y2 = (int(v * k) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(int(v * 0.7) for v in colour), ft + 1)
            put_label(frame, f'#{tid} ? {hidden_s:.0f}s', x1, y1,
                      tuple(int(v * 0.55) for v in colour), fs * 0.9, ft, (255, 255, 255))

        state_colour = {State.TRACKING: GREEN, State.UNCERTAIN: ORANGE,
                        State.LOST: RED}.get(res.state, GREEN)
        speed = f'in {self.fps_in:.1f} fps | out {self.fps_out:.1f} fps | infer {infer_ms:.0f} ms'
        status = res.state.value
        if self.tracker.gallery is not None:
            status += (f'  accept>={self.tracker.sim_threshold * 100:.0f}%'
                       f'  memory {len(self.tracker.anchors)}+{self.tracker.rolling_count}'
                       f'  lock changes {self.tracker.lock_changes}')
        hs, ht, black = 0.6 * k, max(1, round(k)), (0, 0, 0)     # status lines on dark chips
        put_label(frame, speed, 0, int(30 * k), black, hs, ht, GREEN)
        put_label(frame, status, 0, int(58 * k), black, hs, ht, state_colour)
        put_label(frame, '[c] calibrate   [r] reset', 0, int(84 * k), black, 0.85 * hs, ht, GREEN)
        if not self.window_open:
            cv2.namedWindow('feed', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # resizable
            cv2.resizeWindow('feed', frame.shape[1], frame.shape[0])
            self.window_open = True
        cv2.imshow('feed', frame)
        key = cv2.waitKey(1) & 0xFF   # required, or the window never refreshes
        if key == ord('c'):
            self.tracker.start_calibration(time.monotonic())
            self.get_logger().info('Calibrating: keep the leader near the centre of the image')
        elif key == ord('r'):
            self.tracker.reset()
            self.get_logger().info(f'Leader cleared (saved file {self.leader_file} is kept)')


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.stop_video = True
    node.print_report()
    if node.show:
        cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():                    # the launch file may have shut the context down already
        rclpy.shutdown()
