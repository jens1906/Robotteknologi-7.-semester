'use strict';

// Storage
const PERCEPTION_VIEW_STORE = 'web_viewer.perception';

// Leader states (PerceptionDebugFrame.leader_state, PerceptionLeader.status)
const STATE_NAMES = ['IDLE', 'ENROLLING', 'FOLLOWING', 'SEARCHING', 'LOST'];
const ENROLLING_STATE = 1;
const FOLLOWING_STATE = 2;
const SEARCHING_STATE = 3;
const LOST_STATE = 4;
const NO_TRACK_ID = -1;
const CENTER_PERSON_TRACK_ID = -1;
const NO_LEADER_STATUS = 0;
const VISIBLE_LEADER_STATUS = 1;

// Leader certainty (REID_ACCEPT and REID_DROP are reidAccept and reidDrop in perception_params.yaml)
const REID_ACCEPT = 0.45;
const REID_DROP = 0.82;
const CERTAINTY_BAR_MAX_DISTANCE = 1.0;
const CERTAINTY_BAR_HEIGHT_PIXELS = 7;
const CERTAINTY_BAR_MIN_WIDTH_PIXELS = 70;

// Colors
const LEADER_COLOR = '#00dc00';
const NOTICE_COLOR = '#ffa500';
const LABEL_TEXT_COLOR = '#000';
const CLICK_MARKER_COLOR = '#ffdc00';
const DYNAMIC_OBSTACLE_COLOR = '#ff8c3a';
const STATIC_OBSTACLE_COLOR = '#4fa3ff';
const COMBINED_OBSTACLE_COLOR = '#c58cff';
const STATE_COLORS = ['#ffffff', '#ffdc00', '#00dc00', '#ffa500', '#ff3c3c'];
const CERTAIN_COLOR = '#3fb950';
const UNSURE_COLOR = '#e3b341';
const WRONG_COLOR = '#f85149';
const BAR_BACKGROUND_COLOR = 'rgba(0, 0, 0, 0.65)';
const BAR_TICK_COLOR = '#ffffff';
const MAP_BACKGROUND_COLOR = '#0b0e12';
const MAP_RING_COLOR = 'rgba(255, 255, 255, 0.14)';
const MAP_TEXT_COLOR = '#8592a3';
const MAP_ROBOT_COLOR = '#ffffff';
const GOLDEN_RATIO = 0.61803398875;

// Camera panel
const LABEL_FONT = '14px system-ui, sans-serif';
const BANNER_FONT = '16px system-ui, sans-serif';
const PLACEHOLDER_WIDTH = 640;
const PLACEHOLDER_HEIGHT = 480;
const NOTICE_SECONDS = 3;
const CLICK_MARKER_SECONDS = 1.5;
const CLICK_MARKER_RADIUS_PIXELS = 22;
const SILENCE_SECONDS = 3;
const RATE_WINDOW_SECONDS = 2;
const STATUS_PERIOD_MILLISECONDS = 500;

// Obstacle map panel
const MAP_FONT = '12px system-ui, sans-serif';
const MAP_RANGE_METERS = 6;
const MAP_MIN_RANGE_METERS = 2;
const MAP_MAX_RANGE_METERS = 20;
const MAP_ROBOT_HEIGHT_FRACTION = 0.8;
const MAP_ARROW_SECONDS = 1;
const MAP_ARROW_HEAD_PIXELS = 8;
const MAP_DOT_RADIUS_PIXELS = 6;
const MAP_MIN_SPEED = 0.05;
const MAP_STALE_SECONDS = 1;
const MAP_STALE_OPACITY = 0.35;

// Perception view state
let perceptionView = null;

class PerceptionView {
  constructor() {
    this.subscribedTopics = new Map();
    this.missingTopics = new Set();
    this.isExactMode = true;
    this.cameraPanel = new CameraPanel(this);
    this.combinedObstacleMapPanel = new ObstacleMapPanel('Obstacle map: combined', COMBINED_OBSTACLE_COLOR);
    this.staticObstacleMapPanel = new ObstacleMapPanel('Obstacle map: static', STATIC_OBSTACLE_COLOR);
    this.dynamicObstacleMapPanel = new ObstacleMapPanel('Obstacle map: dynamic', DYNAMIC_OBSTACLE_COLOR);
    this.obstacleMapPanels = [this.dynamicObstacleMapPanel, this.staticObstacleMapPanel, this.combinedObstacleMapPanel];
    openView(this);

    // Input topics
    this.Subscribe('/perception/debug_frame', { hz: 0 },
      (debugFrame) => this.cameraPanel.HandleDebugFrame(debugFrame));
    this.Subscribe('/perception/obstacles', { hz: 0 },
      (obstacleArray) => this.dynamicObstacleMapPanel.HandleObstacles(obstacleArray));
    this.Subscribe('/perception/obstacles_static', { hz: 0 },
      (obstacleArray) => this.staticObstacleMapPanel.HandleObstacles(obstacleArray));
    this.Subscribe('/perception/obstacles_combined', { hz: 0 },
      (obstacleArray) => this.combinedObstacleMapPanel.HandleObstacles(obstacleArray));
    this.Subscribe('/perception/leader', { hz: 0 }, (leader) => this.HandleLeader(leader));
    this.SubscribeCameraImage();
  }

  HandleLeader(leader) {
    for (const obstacleMapPanel of this.obstacleMapPanels) {
      obstacleMapPanel.HandleLeader(leader);
    }
  }

  HandleTopicList(topicNames) {
    for (const topicName of topicNames) {
      if (this.missingTopics.has(topicName)) {
        this.missingTopics.delete(topicName);
        const subscription = this.subscribedTopics.get(topicName);
        send({ op: 'sub', topic: topicName, ...subscription.subscribeOptions });
      }
    }
  }

  HandleTopicError(topicName, errorText) {
    if (errorText === 'topic not found') {
      this.missingTopics.add(topicName);
    }
    this.cameraPanel.SetNotice(`${topicName}: ${errorText}`, NOTICE_COLOR);
  }

  HandleServiceResult(result) {
    this.cameraPanel.SetNotice(result.message, result.success ? LEADER_COLOR : NOTICE_COLOR);
  }

  Subscribe(topicName, subscribeOptions, HandleMessage) {
    this.subscribedTopics.set(topicName, { subscribeOptions, HandleMessage });
    send({ op: 'sub', topic: topicName, ...subscribeOptions });
  }

  SubscribeCameraImage() {
    const pairOptions = { pair: this.isExactMode ? '/perception/debug_frame' : '' };
    this.Subscribe('/camera/camera/color/image_raw', pairOptions,
      (imageBlob, imageStamp) => this.cameraPanel.HandleImage(imageBlob, imageStamp));
  }

  ToggleMode() {
    this.isExactMode = !this.isExactMode;
    this.SubscribeCameraImage();
  }

  CallCalibrateLeader(trackId) {
    if (!send({ op: 'calibrate', track_id: trackId })) {
      this.cameraPanel.SetNotice('not connected to web_viewer', NOTICE_COLOR);
    }
  }

  CallClearLeader() {
    if (!send({ op: 'clear_leader' })) {
      this.cameraPanel.SetNotice('not connected to web_viewer', NOTICE_COLOR);
    }
  }

  Close() {
    closeView(this);
    this.cameraPanel.Destroy();
    for (const obstacleMapPanel of this.obstacleMapPanels) {
      obstacleMapPanel.Destroy();
    }
  }
}

class CameraPanel {
  constructor(parentView) {
    this.parentView = parentView;
    this.latestBitmap = null;
    this.latestBitmapStamp = '';
    this.pendingImage = null;
    this.isDecoding = false;
    this.latestDebugFrame = null;
    this.latestDebugFrameStamp = '';
    this.latestDebugFrameArrivalSeconds = 0;
    this.shownBitmap = null;
    this.shownDebugFrame = null;
    this.shownPeople = [];
    this.shownFrameSeconds = [];
    this.lastShownSeconds = NowSeconds();
    this.isDrawScheduled = false;
    this.noticeText = '';
    this.noticeColor = NOTICE_COLOR;
    this.noticeExpirySeconds = 0;
    this.clickMarkerPoint = null;
    this.clickMarkerExpirySeconds = 0;

    const panelParts = CreatePanelElement('Perception camera', true);
    this.element = panelParts.element;
    this.meta = panelParts.meta;
    this.modeButton = CreateButton(panelParts.actions, 'exact', 'exact: only images with a result, live: every image',
      () => this.HandleModeClick());
    CreateButton(panelParts.actions, 'calibrate centre', 'make the person nearest the image centre the leader',
      () => this.parentView.CallCalibrateLeader(CENTER_PERSON_TRACK_ID));
    CreateButton(panelParts.actions, 'clear leader', 'forget the leader', () => this.parentView.CallClearLeader());
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'camera';
    this.canvas.width = PLACEHOLDER_WIDTH;
    this.canvas.height = PLACEHOLDER_HEIGHT;
    this.canvas.title = 'click a person to make them the leader';
    this.canvas.onclick = (clickEvent) => this.HandleCanvasClick(clickEvent);
    panelParts.body.appendChild(this.canvas);
    this.context = this.canvas.getContext('2d');
    this.statusTimer = setInterval(() => this.HandleStatusTimer(), STATUS_PERIOD_MILLISECONDS);
    this.ScheduleDraw();
  }

  HandleImage(imageBlob, imageStamp) {
    this.pendingImage = { imageBlob, imageStamp };
    if (!this.isDecoding) {
      this.DecodePendingImages();
    }
  }

  HandleDebugFrame(debugFrame) {
    this.latestDebugFrame = debugFrame;
    this.latestDebugFrameStamp = FormatStamp(debugFrame.header.stamp);
    this.latestDebugFrameArrivalSeconds = NowSeconds();
    if (this.parentView.isExactMode) {
      this.ShowMatchingFrame();
    }
  }

  HandleModeClick() {
    this.parentView.ToggleMode();
    this.modeButton.textContent = this.parentView.isExactMode ? 'exact' : 'live';
  }

  HandleCanvasClick(clickEvent) {
    const canvasRectangle = this.canvas.getBoundingClientRect();
    const imageX = (clickEvent.clientX - canvasRectangle.left) * this.canvas.width / canvasRectangle.width;
    const imageY = (clickEvent.clientY - canvasRectangle.top) * this.canvas.height / canvasRectangle.height;
    this.clickMarkerPoint = { x: imageX, y: imageY };
    this.clickMarkerExpirySeconds = NowSeconds() + CLICK_MARKER_SECONDS;
    const clickedPerson = FindPersonAtPoint(this.shownPeople, imageX, imageY);
    if (clickedPerson === null) {
      this.SetNotice('nobody there: click inside a box', NOTICE_COLOR);
      return;
    }
    this.parentView.CallCalibrateLeader(clickedPerson.id);
  }

  HandleStatusTimer() {
    const nowSeconds = NowSeconds();
    while (this.shownFrameSeconds.length && nowSeconds - this.shownFrameSeconds[0] > RATE_WINDOW_SECONDS) {
      this.shownFrameSeconds.shift();
    }
    const framesPerSecond = this.shownFrameSeconds.length / RATE_WINDOW_SECONDS;
    const modeName = this.parentView.isExactMode ? 'exact' : 'live';
    let metaText = `${modeName} · ${framesPerSecond.toFixed(1)} fps · people ${this.shownPeople.length}`;
    if (this.shownDebugFrame !== null) {
      metaText += ` · detector ${FormatNumber(this.shownDebugFrame.detector_time_ms, 0)} ms`;
      metaText += ` · reid ${FormatNumber(this.shownDebugFrame.reid_time_ms, 0)} ms`;
    }
    this.meta.textContent = metaText;
    const silentSeconds = nowSeconds - this.lastShownSeconds;
    if (silentSeconds > SILENCE_SECONDS) {
      this.SetNotice(`nothing to show for ${silentSeconds.toFixed(0)} s: are the camera and perception running?`,
        NOTICE_COLOR);
    }
    this.ScheduleDraw();
  }

  async DecodePendingImages() {
    this.isDecoding = true;
    while (this.pendingImage !== null) {
      const { imageBlob, imageStamp } = this.pendingImage;
      this.pendingImage = null;
      try {
        const decodedBitmap = await createImageBitmap(imageBlob);
        this.ReplaceLatestBitmap(decodedBitmap, imageStamp);
      } catch (decodeError) {
        this.SetNotice(`could not decode the image: ${decodeError.message}`, NOTICE_COLOR);
      }
    }
    this.isDecoding = false;
  }

  ReplaceLatestBitmap(decodedBitmap, imageStamp) {
    const oldBitmap = this.latestBitmap;
    this.latestBitmap = decodedBitmap;
    this.latestBitmapStamp = imageStamp;
    if (oldBitmap !== null && oldBitmap !== this.shownBitmap) {
      oldBitmap.close();
    }
    if (this.parentView.isExactMode) {
      this.ShowMatchingFrame();
      return;
    }
    const isDebugFrameFresh = NowSeconds() - this.latestDebugFrameArrivalSeconds < SILENCE_SECONDS;
    this.ShowFrame(this.latestBitmap, isDebugFrameFresh ? this.latestDebugFrame : null);
  }

  ShowMatchingFrame() {
    const isMatching = this.latestBitmap !== null && this.latestBitmapStamp === this.latestDebugFrameStamp;
    const isAlreadyShown = this.shownBitmap === this.latestBitmap && this.shownDebugFrame === this.latestDebugFrame;
    if (isMatching && !isAlreadyShown) {
      this.ShowFrame(this.latestBitmap, this.latestDebugFrame);
    }
  }

  ShowFrame(bitmap, debugFrame) {
    const oldBitmap = this.shownBitmap;
    this.shownBitmap = bitmap;
    this.shownDebugFrame = debugFrame;
    this.shownPeople = debugFrame !== null ? KeepMessageObjects(debugFrame.people) : [];
    if (oldBitmap !== null && oldBitmap !== bitmap && oldBitmap !== this.latestBitmap) {
      oldBitmap.close();
    }
    const nowSeconds = NowSeconds();
    this.lastShownSeconds = nowSeconds;
    this.shownFrameSeconds.push(nowSeconds);
    this.ScheduleDraw();
  }

  SetNotice(text, color) {
    this.noticeText = text;
    this.noticeColor = color;
    this.noticeExpirySeconds = NowSeconds() + NOTICE_SECONDS;
    this.ScheduleDraw();
  }

  ScheduleDraw() {
    if (this.isDrawScheduled) {
      return;
    }
    this.isDrawScheduled = true;
    requestAnimationFrame(() => this.Draw());
  }

  Draw() {
    this.isDrawScheduled = false;
    const context = this.context;
    if (this.shownBitmap !== null) {
      if (this.canvas.width !== this.shownBitmap.width || this.canvas.height !== this.shownBitmap.height) {
        this.canvas.width = this.shownBitmap.width;
        this.canvas.height = this.shownBitmap.height;
      }
      context.drawImage(this.shownBitmap, 0, 0);
    } else {
      context.fillStyle = '#000';
      context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    for (const trackedPerson of this.shownPeople) {
      DrawTrackedPerson(context, trackedPerson);
    }
    DrawLeaderBanner(context, this.shownDebugFrame, this.shownPeople);
    const nowSeconds = NowSeconds();
    if (this.clickMarkerPoint !== null && nowSeconds < this.clickMarkerExpirySeconds) {
      DrawClickMarker(context, this.clickMarkerPoint);
    }
    if (nowSeconds < this.noticeExpirySeconds) {
      DrawLabel(context, this.noticeText, 0, this.canvas.height - 6, this.noticeColor, BANNER_FONT);
    }
  }

  Destroy() {
    clearInterval(this.statusTimer);
    this.latestBitmap?.close();
    this.shownBitmap?.close();
    this.element.remove();
  }
}

class ObstacleMapPanel {
  constructor(title, obstacleColor) {
    this.obstacleColor = obstacleColor;
    this.obstacles = [];
    this.leader = null;
    this.obstaclesArrivalSeconds = 0;
    this.leaderArrivalSeconds = 0;
    this.mapRangeMeters = MAP_RANGE_METERS;
    this.isDrawScheduled = false;

    const panelParts = CreatePanelElement(title, false);
    this.element = panelParts.element;
    this.meta = panelParts.meta;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'obstacle-map';
    this.canvas.title = 'wheel to zoom, double-click to reset';
    this.canvas.onwheel = (wheelEvent) => this.HandleWheel(wheelEvent);
    this.canvas.ondblclick = () => this.HandleDoubleClick();
    panelParts.body.appendChild(this.canvas);
    this.context = this.canvas.getContext('2d');
    this.resizeObserver = new ResizeObserver(() => this.ScheduleDraw());
    this.resizeObserver.observe(this.canvas);
    this.statusTimer = setInterval(() => this.HandleStatusTimer(), STATUS_PERIOD_MILLISECONDS);
  }

  HandleObstacles(obstacleArray) {
    this.obstacles = KeepMessageObjects(obstacleArray.obstacles);
    this.obstaclesArrivalSeconds = NowSeconds();
    this.ScheduleDraw();
  }

  HandleLeader(leader) {
    this.leader = leader;
    this.leaderArrivalSeconds = NowSeconds();
    this.ScheduleDraw();
  }

  HandleWheel(wheelEvent) {
    wheelEvent.preventDefault();
    const zoomedRangeMeters = this.mapRangeMeters * Math.exp(wheelEvent.deltaY * 0.001);
    this.mapRangeMeters = Math.max(MAP_MIN_RANGE_METERS, Math.min(MAP_MAX_RANGE_METERS, zoomedRangeMeters));
    this.ScheduleDraw();
  }

  HandleDoubleClick() {
    this.mapRangeMeters = MAP_RANGE_METERS;
    this.ScheduleDraw();
  }

  HandleStatusTimer() {
    const nowSeconds = NowSeconds();
    const leaderStatusName = this.leader === null ? 'none' : ['NO LEADER', 'VISIBLE', 'LOST'][this.leader.status];
    this.meta.textContent = `obstacles ${this.obstacles.length} · leader ${leaderStatusName}`
      + ` · ${DescribeAge(nowSeconds, this.obstaclesArrivalSeconds)}`;
    this.ScheduleDraw();
  }

  ScheduleDraw() {
    if (this.isDrawScheduled) {
      return;
    }
    this.isDrawScheduled = true;
    requestAnimationFrame(() => this.Draw());
  }

  Draw() {
    this.isDrawScheduled = false;
    const pixelRatio = window.devicePixelRatio || 1;
    const mapWidth = this.canvas.clientWidth;
    const mapHeight = this.canvas.clientHeight;
    if (!mapWidth || !mapHeight) {
      return;
    }
    if (this.canvas.width !== Math.round(mapWidth * pixelRatio)) {
      this.canvas.width = Math.round(mapWidth * pixelRatio);
      this.canvas.height = Math.round(mapHeight * pixelRatio);
    }
    const context = this.context;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = MAP_BACKGROUND_COLOR;
    context.fillRect(0, 0, mapWidth, mapHeight);
    const robotY = mapHeight * MAP_ROBOT_HEIGHT_FRACTION;
    const mapFrame = { robotX: mapWidth / 2, robotY, pixelsPerMeter: robotY / this.mapRangeMeters };
    DrawMapRings(context, mapFrame, mapWidth, mapHeight);
    const nowSeconds = NowSeconds();
    context.globalAlpha = nowSeconds - this.obstaclesArrivalSeconds > MAP_STALE_SECONDS ? MAP_STALE_OPACITY : 1;
    for (const obstacle of this.obstacles) {
      DrawObstacle(context, mapFrame, obstacle, this.obstacleColor);
    }
    context.globalAlpha = nowSeconds - this.leaderArrivalSeconds > MAP_STALE_SECONDS ? MAP_STALE_OPACITY : 1;
    if (this.leader !== null) {
      DrawMapLeader(context, mapFrame, this.leader);
    }
    context.globalAlpha = 1;
    DrawMapRobot(context, mapFrame);
  }

  Destroy() {
    clearInterval(this.statusTimer);
    this.resizeObserver.disconnect();
    this.element.remove();
  }
}

function CreatePanelElement(title, isWide) {
  const element = document.createElement('section');
  element.className = isWide ? 'panel wide perception' : 'panel perception';
  element.innerHTML = '<header><span class="title"></span><span class="meta"></span><span class="actions"></span></header>'
    + '<div class="body"></div>';
  element.querySelector('.title').textContent = title;
  $('#panels').prepend(element);
  return {
    element,
    meta: element.querySelector('.meta'),
    actions: element.querySelector('.actions'),
    body: element.querySelector('.body'),
  };
}

function CreateButton(parent, text, tooltip, HandleClick) {
  const button = document.createElement('button');
  button.textContent = text;
  button.title = tooltip;
  button.onclick = HandleClick;
  parent.appendChild(button);
  return button;
}

function DrawTrackedPerson(context, trackedPerson) {
  const boxWidth = trackedPerson.x_max - trackedPerson.x_min;
  const boxHeight = trackedPerson.y_max - trackedPerson.y_min;
  const boxColor = trackedPerson.is_leader ? LEADER_COLOR : PickColorForTrackId(trackedPerson.id);
  context.lineWidth = trackedPerson.is_leader ? 4 : 2;
  context.strokeStyle = boxColor;
  context.strokeRect(trackedPerson.x_min, trackedPerson.y_min, boxWidth, boxHeight);
  let labelText = trackedPerson.is_leader ? `LEADER #${trackedPerson.id}` : `#${trackedPerson.id}`;
  if (trackedPerson.reid_distance !== null) {
    labelText += `  reid ${trackedPerson.reid_distance.toFixed(2)}`;
  }
  if (trackedPerson.position_valid) {
    labelText += `  ${Math.hypot(trackedPerson.position.x, trackedPerson.position.y).toFixed(1)} m`;
  }
  DrawLabel(context, labelText, trackedPerson.x_min, trackedPerson.y_min, boxColor, LABEL_FONT);
  DrawCertaintyBar(context, trackedPerson.reid_distance, trackedPerson.x_min, trackedPerson.y_min + 2, boxWidth);
}

function DrawCertaintyBar(context, reidDistance, left, top, boxWidth) {
  const barWidth = Math.max(boxWidth, CERTAINTY_BAR_MIN_WIDTH_PIXELS);
  context.fillStyle = BAR_BACKGROUND_COLOR;
  context.fillRect(left, top, barWidth, CERTAINTY_BAR_HEIGHT_PIXELS);
  if (reidDistance === null) {
    return;
  }
  context.fillStyle = PickCertaintyColor(reidDistance);
  context.fillRect(left, top, barWidth * CalculateCertaintyFraction(reidDistance), CERTAINTY_BAR_HEIGHT_PIXELS);
  context.fillStyle = BAR_TICK_COLOR;
  for (const thresholdDistance of [REID_ACCEPT, REID_DROP]) {
    context.fillRect(left + barWidth * CalculateCertaintyFraction(thresholdDistance) - 1, top, 2,
      CERTAINTY_BAR_HEIGHT_PIXELS);
  }
}

function CalculateCertaintyFraction(reidDistance) {
  return Math.max(0, Math.min(1, 1 - reidDistance / CERTAINTY_BAR_MAX_DISTANCE));
}

function PickCertaintyColor(reidDistance) {
  if (reidDistance < REID_ACCEPT) {
    return CERTAIN_COLOR;
  }
  return reidDistance < REID_DROP ? UNSURE_COLOR : WRONG_COLOR;
}

function DrawLeaderBanner(context, debugFrame, trackedPeople) {
  let bannerText;
  let bannerColor;
  if (debugFrame === null) {
    bannerText = 'LEADER: no debug_frame yet';
    bannerColor = STATE_COLORS[0];
  } else if ([FOLLOWING_STATE, SEARCHING_STATE, LOST_STATE].includes(debugFrame.leader_state)
             && debugFrame.leader_track_id === NO_TRACK_ID) {
    bannerText = DescribeLeaderOutOfFrame(debugFrame);
    bannerColor = NOTICE_COLOR;
  } else if (debugFrame.leader_state === FOLLOWING_STATE) {
    bannerText = DescribeFollowedLeader(debugFrame, trackedPeople);
    bannerColor = LEADER_COLOR;
  } else if (debugFrame.leader_state === ENROLLING_STATE) {
    bannerText = `LEADER: ENROLLING, turn around slowly  ${DescribeStoredViews(debugFrame)}`;
    bannerColor = STATE_COLORS[ENROLLING_STATE];
  } else {
    bannerText = `LEADER: ${STATE_NAMES[debugFrame.leader_state]}`;
    bannerColor = STATE_COLORS[debugFrame.leader_state];
  }
  DrawLabel(context, bannerText, 0, 26, bannerColor, BANNER_FONT);
}

function DescribeLeaderOutOfFrame(debugFrame) {
  if (debugFrame.seconds_since_seen === null) {
    return `LEADER: ${STATE_NAMES[debugFrame.leader_state]}, looking for the saved leader`;
  }
  const statusText = debugFrame.leader_state === LOST_STATE ? 'LEADER LOST' : 'LEADER NOT IN FRAME';
  let description = `${statusText}: last seen ${debugFrame.seconds_since_seen.toFixed(0)} s ago`;
  if (debugFrame.last_seen_position.x !== null) {
    description += ` at x ${debugFrame.last_seen_position.x.toFixed(1)} y ${debugFrame.last_seen_position.y.toFixed(1)} m`;
  }
  return description;
}

function DescribeFollowedLeader(debugFrame, trackedPeople) {
  let description = `LEADER: FOLLOWING #${debugFrame.leader_track_id}`;
  if (debugFrame.leader_distance !== null) {
    description += `  ${debugFrame.leader_distance.toFixed(1)} m`;
  }
  const leaderPerson = trackedPeople.find((trackedPerson) => trackedPerson.is_leader);
  if (leaderPerson !== undefined && leaderPerson.reid_distance !== null) {
    description += `  reid ${leaderPerson.reid_distance.toFixed(2)}`;
  }
  return `${description}  views ${DescribeStoredViews(debugFrame)}`;
}

function DescribeStoredViews(debugFrame) {
  return `f${debugFrame.front_views} b${debugFrame.back_views} l${debugFrame.side_left_views} `
    + `r${debugFrame.side_right_views} h${debugFrame.partly_hidden_views}`;
}

function DrawLabel(context, text, left, bottom, backgroundColor, font) {
  context.font = font;
  const textMetrics = context.measureText(text);
  const textHeight = textMetrics.actualBoundingBoxAscent + textMetrics.actualBoundingBoxDescent;
  const top = Math.max(0, bottom - textHeight - 8);
  context.fillStyle = backgroundColor;
  context.fillRect(left, top, textMetrics.width + 8, textHeight + 8);
  context.fillStyle = LABEL_TEXT_COLOR;
  context.fillText(text, left + 4, top + 4 + textMetrics.actualBoundingBoxAscent);
}

function DrawClickMarker(context, markerPoint) {
  context.beginPath();
  context.arc(markerPoint.x, markerPoint.y, CLICK_MARKER_RADIUS_PIXELS, 0, 2 * Math.PI);
  context.lineWidth = 5;
  context.strokeStyle = '#000';
  context.stroke();
  context.lineWidth = 2;
  context.strokeStyle = CLICK_MARKER_COLOR;
  context.stroke();
}

function FindPersonAtPoint(trackedPeople, pointX, pointY) {
  const peopleUnderPoint = trackedPeople.filter((trackedPerson) => trackedPerson.x_min <= pointX
    && pointX <= trackedPerson.x_max && trackedPerson.y_min <= pointY && pointY <= trackedPerson.y_max);
  if (!peopleUnderPoint.length) {
    return null;
  }
  return peopleUnderPoint.reduce((smallest, trackedPerson) =>
    CalculateBoxArea(trackedPerson) < CalculateBoxArea(smallest) ? trackedPerson : smallest);
}

function CalculateBoxArea(trackedPerson) {
  return (trackedPerson.x_max - trackedPerson.x_min) * (trackedPerson.y_max - trackedPerson.y_min);
}

function PickColorForTrackId(trackId) {
  let hueDegrees = (trackId * GOLDEN_RATIO % 1) * 240;
  if (hueDegrees > 60) {
    hueDegrees += 120;
  }
  return `hsl(${hueDegrees}, 100%, 50%)`;
}

function DrawMapRings(context, mapFrame, mapWidth, mapHeight) {
  const farthestMeters = Math.hypot(mapWidth, mapHeight) / mapFrame.pixelsPerMeter;
  const ringStepMeters = mapFrame.pixelsPerMeter < 25 ? 2 : 1;
  context.lineWidth = 1;
  context.strokeStyle = MAP_RING_COLOR;
  context.fillStyle = MAP_TEXT_COLOR;
  context.font = MAP_FONT;
  for (let ringMeters = ringStepMeters; ringMeters < farthestMeters; ringMeters += ringStepMeters) {
    const ringRadius = ringMeters * mapFrame.pixelsPerMeter;
    context.beginPath();
    context.arc(mapFrame.robotX, mapFrame.robotY, ringRadius, 0, 2 * Math.PI);
    context.stroke();
    context.fillText(`${ringMeters} m`, mapFrame.robotX + 4, mapFrame.robotY - ringRadius - 3);
  }
  context.beginPath();
  context.moveTo(mapFrame.robotX, 0);
  context.lineTo(mapFrame.robotX, mapHeight);
  context.moveTo(0, mapFrame.robotY);
  context.lineTo(mapWidth, mapFrame.robotY);
  context.stroke();
}

function DrawMapRobot(context, mapFrame) {
  context.fillStyle = MAP_ROBOT_COLOR;
  context.beginPath();
  context.moveTo(mapFrame.robotX, mapFrame.robotY - 10);
  context.lineTo(mapFrame.robotX - 7, mapFrame.robotY + 6);
  context.lineTo(mapFrame.robotX + 7, mapFrame.robotY + 6);
  context.closePath();
  context.fill();
}

function DrawObstacle(context, mapFrame, obstacle, obstacleColor) {
  if (!IsValidPoint(obstacle.closest_point)) {
    return;
  }
  const obstaclePixels = ConvertToMapPixels(mapFrame, obstacle.closest_point);
  DrawUncertaintyCircle(context, mapFrame, obstaclePixels, obstacle.position_uncertainty, obstacleColor);
  DrawMapDot(context, obstaclePixels, obstacleColor, true);
  DrawVelocityArrow(context, mapFrame, obstaclePixels, obstacle.velocity, obstacleColor);
  if (IsValidPoint(obstacle.line_of_sight_point)) {
    const lineOfSightPixels = ConvertToMapPixels(mapFrame, obstacle.line_of_sight_point);
    context.lineWidth = 2;
    context.strokeStyle = obstacleColor;
    context.beginPath();
    context.moveTo(lineOfSightPixels.x - 4, lineOfSightPixels.y - 4);
    context.lineTo(lineOfSightPixels.x + 4, lineOfSightPixels.y + 4);
    context.moveTo(lineOfSightPixels.x + 4, lineOfSightPixels.y - 4);
    context.lineTo(lineOfSightPixels.x - 4, lineOfSightPixels.y + 4);
    context.stroke();
  }
}

function DrawMapLeader(context, mapFrame, leader) {
  if (leader.status === NO_LEADER_STATUS) {
    return;
  }
  const isVisible = leader.status === VISIBLE_LEADER_STATUS;
  let leaderPoint = isVisible && IsValidPoint(leader.closest_point) ? leader.closest_point : leader.last_known_point;
  if (!IsValidPoint(leaderPoint)) {
    return;
  }
  const lastKnownPixels = ConvertToMapPixels(mapFrame, leaderPoint);
  if (!isVisible && leader.lost_duration !== null) {
    leaderPoint = {
      x: leaderPoint.x + leader.velocity.x * leader.lost_duration,
      y: leaderPoint.y + leader.velocity.y * leader.lost_duration,
    };
  }
  const leaderPixels = ConvertToMapPixels(mapFrame, leaderPoint);
  context.lineWidth = 1.5;
  context.strokeStyle = LEADER_COLOR;
  context.setLineDash([6, 5]);
  context.beginPath();
  context.moveTo(mapFrame.robotX, mapFrame.robotY);
  context.lineTo(leaderPixels.x, leaderPixels.y);
  if (!isVisible) {
    context.moveTo(lastKnownPixels.x, lastKnownPixels.y);
    context.lineTo(leaderPixels.x, leaderPixels.y);
  }
  context.stroke();
  context.setLineDash([]);
  DrawUncertaintyCircle(context, mapFrame, leaderPixels, leader.position_uncertainty, LEADER_COLOR);
  DrawMapDot(context, leaderPixels, LEADER_COLOR, isVisible);
  DrawVelocityArrow(context, mapFrame, leaderPixels, leader.velocity, LEADER_COLOR);
  const distanceMeters = Math.hypot(leaderPoint.x, leaderPoint.y);
  let labelText = `LEADER ${distanceMeters.toFixed(1)} m`;
  if (!isVisible && leader.lost_duration !== null) {
    labelText += `, lost ${leader.lost_duration.toFixed(0)} s`;
  }
  context.fillStyle = LEADER_COLOR;
  context.font = MAP_FONT;
  context.fillText(labelText, leaderPixels.x + 10, leaderPixels.y - 8);
}

function DrawUncertaintyCircle(context, mapFrame, centerPixels, uncertaintyMeters, color) {
  if (uncertaintyMeters === null || uncertaintyMeters <= 0) {
    return;
  }
  context.beginPath();
  context.arc(centerPixels.x, centerPixels.y, uncertaintyMeters * mapFrame.pixelsPerMeter, 0, 2 * Math.PI);
  context.globalAlpha *= 0.18;
  context.fillStyle = color;
  context.fill();
  context.globalAlpha /= 0.18;
  context.lineWidth = 1;
  context.strokeStyle = color;
  context.stroke();
}

function DrawMapDot(context, centerPixels, color, isFilled) {
  context.beginPath();
  context.arc(centerPixels.x, centerPixels.y, MAP_DOT_RADIUS_PIXELS, 0, 2 * Math.PI);
  context.lineWidth = 2;
  context.strokeStyle = color;
  context.fillStyle = color;
  if (isFilled) {
    context.fill();
  } else {
    context.stroke();
  }
}

function DrawVelocityArrow(context, mapFrame, startPixels, velocity, color) {
  if (velocity.x === null || Math.hypot(velocity.x, velocity.y) < MAP_MIN_SPEED) {
    return;
  }
  const endPixels = {
    x: startPixels.x - velocity.y * MAP_ARROW_SECONDS * mapFrame.pixelsPerMeter,
    y: startPixels.y - velocity.x * MAP_ARROW_SECONDS * mapFrame.pixelsPerMeter,
  };
  const arrowAngle = Math.atan2(endPixels.y - startPixels.y, endPixels.x - startPixels.x);
  context.lineWidth = 2;
  context.strokeStyle = color;
  context.beginPath();
  context.moveTo(startPixels.x, startPixels.y);
  context.lineTo(endPixels.x, endPixels.y);
  for (const headSide of [-1, 1]) {
    const headAngle = arrowAngle + Math.PI + headSide * 0.45;
    context.moveTo(endPixels.x, endPixels.y);
    context.lineTo(endPixels.x + MAP_ARROW_HEAD_PIXELS * Math.cos(headAngle),
      endPixels.y + MAP_ARROW_HEAD_PIXELS * Math.sin(headAngle));
  }
  context.stroke();
}

function ConvertToMapPixels(mapFrame, point) {
  return {
    x: mapFrame.robotX - point.y * mapFrame.pixelsPerMeter,
    y: mapFrame.robotY - point.x * mapFrame.pixelsPerMeter,
  };
}

function IsValidPoint(point) {
  return point !== null && point !== undefined && point.x !== null && point.y !== null;
}

function KeepMessageObjects(messageArray) {
  return messageArray.filter((element) => element !== null && typeof element === 'object');
}

function FormatStamp(stamp) {
  return `${stamp.sec}.${stamp.nanosec}`;
}

function FormatNumber(value, digits) {
  return value === null ? '-' : value.toFixed(digits);
}

function DescribeAge(nowSeconds, arrivalSeconds) {
  return arrivalSeconds === 0 ? 'no message yet' : `${(nowSeconds - arrivalSeconds).toFixed(1)} s ago`;
}

function NowSeconds() {
  return performance.now() / 1000;
}

function TogglePerceptionView() {
  if (perceptionView === null) {
    perceptionView = new PerceptionView();
  } else {
    perceptionView.Close();
    perceptionView = null;
  }
  localStorage.setItem(PERCEPTION_VIEW_STORE, perceptionView === null ? 'closed' : 'open');
  $('#perception-toggle').classList.toggle('open', perceptionView !== null);
}

$('#perception-toggle').onclick = TogglePerceptionView;
if (localStorage.getItem(PERCEPTION_VIEW_STORE) !== 'closed') {
  TogglePerceptionView();
}
