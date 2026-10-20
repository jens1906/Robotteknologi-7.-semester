"""Watch the tracking on a video or an image folder, matchers side by side (OSNet | KPR).

    PYTHONPATH=src/vision_pre:src/kpr .venv/bin/python -m vision_pre.view data/mot17_09
    ... -m vision_pre.view data/mot17_09 --embedders kpr          # one panel
    ... -m vision_pre.view data/mot17_09 --no-show --out out.mp4 --click 60,540,600   # headless

Click a person to make them the leader in every panel. The leader is saved (~/.ros/follow_anywhere) and
loaded again at the next start; --fresh ignores it. --loop plays the clip again and again. Keys: c person
nearest the centre, r forget the leader (the file is kept), space pause, q or Esc quit. Every box shows its track id; under it the distance to the leader's stored views
(lower = more alike) after each comparison. Green = identified leader.
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')     # OpenCV's Qt has no wayland plugin

import argparse  # noqa: E402
import colorsys  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision_pre.frames import FrameSource, TopicSource  # noqa: E402
from vision_pre.leader import ENROLL_VIEWS, Leader, State  # noqa: E402
from vision_pre.people import PersonTracker  # noqa: E402
from vision_pre.pipeline import Pipeline  # noqa: E402
from vision_pre.reid import THRESHOLDS, make_embedder  # noqa: E402

WINDOW = 'tracking'
TITLES = {'kpr': 'KPR', 'osnet': 'OSNet'}
GREEN, ORANGE, YELLOW, WHITE, BLACK, RED = (0, 220, 0), (0, 165, 255), (0, 220, 255), (255, 255, 255), (0, 0, 0), (60, 60, 255)
STATE_COLOUR = {State.FOLLOWING: GREEN, State.SEARCHING: ORANGE, State.ENROLLING: YELLOW,
                State.IDLE: WHITE}


def id_colour(track_id):
    """Vibrant BGR colour per track id, the same every frame. Green is kept for the leader."""
    deg = (track_id * 0.61803398875 % 1.0) * 240          # golden-ratio steps spread neighbours
    if deg > 60:
        deg += 120                                        # skip the green band, 60..180 degrees
    r, g, b = colorsys.hsv_to_rgb(deg / 360, 1.0, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def label(img, text, x, y, colour, scale=0.5, thickness=1, ink=BLACK):
    """Text on a filled chip whose bottom-left corner is (x, y), readable on any background."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = max(0, y - h - base - 4)
    cv2.rectangle(img, (x, top), (x + w + 6, top + h + base + 4), colour, -1)
    cv2.putText(img, text, (x + 3, top + h + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thickness)


def panel(frame, people, result, leader, name, event, now, k, failed=None, note=None, mark=None):
    """One matcher's view: the frame scaled by k with boxes, ids, distances and the leader state."""
    img = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    for p in people:
        x1, y1, x2, y2 = (int(v * k) for v in p.box)
        is_leader = result.leader_id == p.id
        colour = GREEN if is_leader else id_colour(p.id)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 4 if is_leader else 2)
        text = f'LEADER #{p.id}' if is_leader else f'#{p.id}'
        seen, dist = leader.scores.get(p.id, (-1e9, None))
        if dist is not None and now - seen < 2.0:
            text += f' {dist:.2f}' if dist == dist else ' ?'
        label(img, text, x1, y1, colour)
    state = result.state
    title = f'{TITLES.get(name, name)}  {state.value}'
    if state == State.ENROLLING:
        title += f' {leader.n_views}/{ENROLL_VIEWS}'
    elif state == State.FOLLOWING and result.leader_id is None:
        title += ' (out of view)'
    label(img, title, 0, 30, BLACK, 0.8, 2, STATE_COLOUR[state])
    if state != State.IDLE:
        detail = f'views {leader.n_views}   accept<{leader.accept:.2f}   drop>{leader.drop:.2f}'
    elif failed:
        detail = f'enrolment failed: {failed} - click a person again'
    else:
        detail = 'no leader yet: click a person (or press c)'
    label(img, detail, 0, 54, BLACK, 0.5, 1, ORANGE if failed and state == State.IDLE else WHITE)
    if event and now - event[0] < 4.0:
        label(img, event[1], 0, img.shape[0] - 34, BLACK, 0.7, 2, ORANGE)
    if note:
        label(img, note[0], 0, img.shape[0] - 66, BLACK, 0.7, 2, note[1])
    if mark:
        centre = (int(mark[0] * k), int(mark[1] * k))
        cv2.circle(img, centre, 22, BLACK, 5)
        cv2.circle(img, centre, 22, YELLOW, 2)
    return img


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', nargs='?', help='video file, or image folder (a MOT17 sequence)')
    ap.add_argument('--camera', nargs='?', const='/camera/camera/color/image_raw', metavar='TOPIC',
                    help='live: use a ROS image topic instead of a file (default: the RealSense colour image). '
                    'Only under ros2 run')
    ap.add_argument('--embedders', default='osnet,kpr', help='panels, left to right: osnet, kpr')
    ap.add_argument('--pose', default='yolov8n-pose.pt', help='YOLO pose weights (downloaded on first use)')
    ap.add_argument('--imgsz', type=int, default=960, help='detector input size (1280 finds more far-away people)')
    ap.add_argument('--weights', default='', help='KPR weights, default models/kpr_occ_pt_IN_*.pth.tar')
    ap.add_argument('--device', default='auto', help='auto, cpu or cuda (auto warns when it has to use the CPU)')
    ap.add_argument('--no-prompt', action='store_true', help='KPR without keypoint prompts')
    ap.add_argument('--accept', type=float, help='distance below which a person can be the leader (default per embedder)')
    ap.add_argument('--margin', type=float, default=0.05, help='lead needed over the second best candidate')
    ap.add_argument('--drop', type=float, help='distance above which the locked person looks wrong (default per embedder)')
    ap.add_argument('--window', type=int, default=1600, help='width of the whole window in px')
    ap.add_argument('--realtime', action='store_true', help='play at video speed and drop frames when slow')
    ap.add_argument('--loop', action='store_true', help='start again at --start when the clip ends (new tracker, leader kept)')
    ap.add_argument('--fresh', action='store_true', help='ignore the saved leader')
    ap.add_argument('--leader-dir', default='~/.ros/follow_anywhere', help='where the leader is saved between runs')
    ap.add_argument('--start', type=int, default=0, help='first frame')
    ap.add_argument('--limit', type=int, default=10 ** 9, help='stop after this many processed frames')
    ap.add_argument('--out', default='', help='also write the window to this video file')
    ap.add_argument('--no-show', action='store_true', help='no window, run to the end')
    ap.add_argument('--click', default='', help='FRAME,X,Y: click this image point at that frame')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    names = [n.strip() for n in args.embedders.split(',') if n.strip()]
    live = args.camera is not None
    if not live and not args.path:
        raise SystemExit('give a video file / image folder, or --camera for a live ROS image topic')
    src = TopicSource(args.camera) if live else FrameSource(args.path)
    print((f'live from {args.camera}' if live else f'{args.path}: {src.count} frames at {src.fps:g} fps')
          + '. Loading models...', flush=True)
    matchers = {n: (make_embedder(n, args.weights, args.device, not args.no_prompt),
                    Leader(**{**THRESHOLDS[n], 'margin': args.margin,
                              **{k: v for k, v in (('accept', args.accept), ('drop', args.drop)) if v}}))
                for n in names}
    pipe = Pipeline(PersonTracker(args.pose, args.imgsz,
                                  device=None if args.device == 'auto' else args.device), matchers)
    scripted = tuple(int(float(v)) for v in args.click.split(',')) if args.click else None
    leader_file = {n: os.path.join(os.path.expanduser(args.leader_dir), f'leader_{n}.npz') for n in names}
    for n in names:
        if not args.fresh and matchers[n][1].load(leader_file[n], matchers[n][0].tag):
            print(f'{TITLES.get(n, n)}: saved leader loaded ({matchers[n][1].n_views} views), searching for them. '
                  'Use --fresh to start without.', flush=True)

    clicks = []
    if not args.no_show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WINDOW, lambda e, x, y, f, p: clicks.append((x, y))
                             if e == cv2.EVENT_LBUTTONDOWN else None)
    index, paused, pause_t, writer = (0 if live else args.start), False, 0.0, None
    wall0 = time.monotonic() - index / src.fps
    shown, ms_avg, events, processed, k, loops = None, None, {}, 0, None, 0
    last, failed, note, mark = None, {}, None, None     # last frame's data, enrolment failures, click feedback
    silent_warned = False

    def render():
        frame, people, results, now, number = last
        t = time.monotonic()
        panels = [panel(frame, people, results[n], matchers[n][1], n, events.get(n), now, k, failed.get(n),
                        note if note and note[2] > t else None, mark if mark and mark[2] > t else None)
                  for n in names]
        img = np.hstack(panels)
        label(img, (f'live, image {number + 1}' if live else f'frame {number + 1}/{src.count}')
              + (f' (loop {loops + 1})' if loops else '')
              + f'   {1000 / ms_avg:.1f} fps   {ms_avg:.0f} ms   [click] leader  '
              '[c] centre person  [r] reset  [space] pause  [q] quit', 0, img.shape[0], BLACK, 0.6, 1, WHITE)
        return img

    def enrolled(person, what):
        nonlocal note
        text = f'{what}: enrolling #{person.id}' if person else f'{what}: nobody there - click inside a box'
        note = (text, GREEN if person else RED, time.monotonic() + 3.0)
        print(text, flush=True)

    while True:
        frame = None
        if not paused and live:
            got = src.next(0.2)                                     # newest image, None while the camera is silent
            if processed >= args.limit:
                break
            if got:
                frame, now = got
                index = processed
            elif shown is None and time.monotonic() - wall0 > 10 and not silent_warned:
                silent_warned = True
                print(f'no image on {args.camera} yet: is the camera driver running? (ros2 topic list)', flush=True)
        elif not paused:
            if args.realtime:                                    # frames that went by while busy are dropped
                lag = index / src.fps - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
                else:
                    index = int((time.monotonic() - wall0) * src.fps)
            frame = src.frame(index)
            if frame is None and args.loop and processed:           # end of the clip: play it again
                index, loops = args.start, loops + 1
                wall0 = time.monotonic() - index / src.fps
                pipe.new_scene()
                print(f'--- loop {loops + 1}: new tracker, the leader is searched again', flush=True)
                continue
            if frame is None or processed >= args.limit:
                break
            now = (loops * src.count + index) / src.fps             # keeps counting across loops
        if frame is not None:
            t0 = time.perf_counter()
            people, results = pipe.process(frame, now)
            ms = (time.perf_counter() - t0) * 1000
            if scripted and index >= scripted[0]:
                person = pipe.enroll_at(scripted[1], scripted[2], now)
                print(f'[{now:6.1f}s] scripted click: ' + (f'enrolling #{person.id}' if person else 'nobody there'),
                      flush=True)
                scripted = None
            ms_avg, processed = ms if ms_avg is None else 0.9 * ms_avg + 0.1 * ms, processed + 1
            for name, res in results.items():
                if res.event:
                    events[name] = (now, res.event + (f': {res.reason}' if res.reason else ''))
                    print(f'[{now:6.1f}s] {TITLES.get(name, name)}: {events[name][1]}', flush=True)
                if res.event == 'enrolled':
                    matchers[name][1].save(leader_file[name], matchers[name][0].tag)
                    print(f'{TITLES.get(name, name)}: leader saved to {leader_file[name]}', flush=True)
                if res.event == 'enroll_failed':
                    failed[name] = res.reason
                elif res.state != State.IDLE:
                    failed.pop(name, None)
            k = args.window / len(names) / frame.shape[1]
            last = (frame, people, results, now, index)
            if args.out or not args.no_show:
                shown = render()
            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), src.fps,
                                             (shown.shape[1], shown.shape[0]))
                writer.write(shown)
            if not live:
                index += 1
        if not args.no_show:
            if shown is not None:
                cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(30 if paused or shown is None else 1) & 0xFF
            if key in (ord('q'), 27) or (shown is not None and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                break
            if key == ord(' '):
                paused = not paused
                if paused:
                    pause_t = time.monotonic()
                else:
                    wall0 += time.monotonic() - pause_t
            if key == ord('r') and last:
                pipe.reset()
                failed.clear()
                note = ('leader cleared', WHITE, time.monotonic() + 3.0)
                print('leader cleared', flush=True)
            if key == ord('c') and last:
                enrolled(pipe.enroll_centre(last[0].shape[1], last[0].shape[0], last[3]), 'key c')
                shown = render()
        while clicks and k and last:
            cx, cy = clicks.pop(0)
            x, y = (cx % (shown.shape[1] // len(names))) / k, cy / k       # window -> image pixels
            enrolled(pipe.enroll_at(x, y, last[3]), f'click at ({x:.0f}, {y:.0f})')
            mark = (x, y, time.monotonic() + 1.5)
            shown = render()
    if writer:
        writer.release()
    if live:
        src.close()
    if not args.no_show:
        cv2.destroyAllWindows()
    print(f'{processed} frames processed' + (f', {1000 / ms_avg:.1f} fps' if ms_avg else ''))


if __name__ == '__main__':
    main()
