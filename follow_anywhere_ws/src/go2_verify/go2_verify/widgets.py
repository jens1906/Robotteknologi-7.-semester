# SPDX-License-Identifier: BSD-3-Clause

"""Qt widgets for go2_verify: camera views, lidar view, charts and the rate table."""

import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder='row-major', antialias=False, background='#16181d',
                    foreground='#c8ccd4')

STALE_S = 1.0
BEV_RANGE = 8.0
BEV_RES = 0.05
TITLE_PERIOD_S = 0.25
TURBO = pg.colormap.get('turbo')
SERIES_COLORS = ['#4fc3f7', '#ffb74d', '#81c784', '#e57373']
OK, WARN, BAD = QtGui.QColor('#2e7d32'), QtGui.QColor('#a68a00'), QtGui.QColor('#b71c1c')


class FpsMeter:

    def __init__(self):
        self._times = []

    def tick(self) -> None:
        now = time.monotonic()
        self._times.append(now)
        while self._times and now - self._times[0] > 2.0:
            self._times.pop(0)

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


class TitleThrottle:
    """
    Only touch a label when its text changes, at most every TITLE_PERIOD_S.

    Setting a label re-lays out the whole graphics layout, so doing it every
    frame costs more than drawing the data.
    """

    def __init__(self, set_text):
        self._set_text = set_text
        self._text = None
        self._next = 0.0

    def __call__(self, make_text) -> None:
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + TITLE_PERIOD_S
        text = make_text()
        if text != self._text:
            self._text = text
            self._set_text(text)


class ImageView(pg.GraphicsLayoutWidget):
    """Shows an RGB image or a depth map (metres, colourised), with fps/age in the title."""

    def __init__(self, title: str, depth_max: float = None):
        super().__init__()
        self._title = title
        self._depth_max = depth_max
        self._label = self.addLabel(title, row=0, col=0)
        self._title_update = TitleThrottle(self._label.setText)
        self._view = self.addViewBox(row=1, col=0, lockAspect=True, invertY=True,
                                     enableMouse=False)
        self._img = pg.ImageItem()
        self._view.addItem(self._img)
        if depth_max is not None:
            self._img.setLookupTable(TURBO.getLookupTable(nPts=256))
        self._seq = -1
        self._shape = None
        self._fps = FpsMeter()
        self.setMinimumSize(240, 160)

    def update_from(self, latest) -> None:
        if latest.seq != self._seq and latest.value is not None:
            self._seq = latest.seq
            self._fps.tick()
            img = latest.value
            if self._depth_max is None:
                self._img.setImage(img, autoLevels=False, levels=(0, 255))
            else:
                self._img.setImage(np.nan_to_num(img), autoLevels=False,
                                   levels=(0.0, self._depth_max))
            if img.shape[:2] != self._shape:
                self._shape = img.shape[:2]
                h, w = self._shape
                self._view.setRange(xRange=(0, w), yRange=(0, h), padding=0)
        self._title_update(lambda: self._title_text(latest))

    def _title_text(self, latest) -> str:
        age = time.monotonic() - latest.stamp if latest.seq else None
        if age is None:
            return f'{self._title} — <span style="color:#e57373">NO DATA</span>'
        if age > STALE_S:
            return f'{self._title} — <span style="color:#e57373">STALE {age:.1f} s</span>'
        h, w = latest.value.shape[:2]
        return f'{self._title} — {w}x{h} @ {self._fps.fps():.1f} fps'


class LidarView(QtWidgets.QWidget):
    """Point cloud coloured by height: 3D (OpenGL) or 2D top-down."""

    def __init__(self, title: str, use_3d: bool = True, z_range=(-0.5, 1.5)):
        super().__init__()
        self._title = title
        self._z_range = z_range
        self._seq = -1
        self._fps = FpsMeter()
        self._label = QtWidgets.QLabel(title)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self._title_update = TitleThrottle(self._label.setText)
        self._gl = None
        if use_3d:
            try:
                import pyqtgraph.opengl as gl
                view = gl.GLViewWidget()
                view.setBackgroundColor('#16181d')
                view.setCameraPosition(distance=8.0, elevation=35, azimuth=-135)
                grid = gl.GLGridItem()
                grid.setSize(16, 16)
                grid.setSpacing(1, 1)
                view.addItem(grid)
                axis = gl.GLAxisItem()
                axis.setSize(0.5, 0.5, 0.5)
                view.addItem(axis)
                self._scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2.0,
                                                     pxMode=True)
                view.addItem(self._scatter)
                self._gl = view
                layout.addWidget(view, 1)
            except Exception as exc:  # noqa: BLE001 - any GL failure -> 2D fallback
                print(f'go2_verify: OpenGL unavailable ({exc}), using 2D lidar view')
        if self._gl is None:
            # Top-down raster, forward up, left to the left. Cell = BEV_RES m.
            plot = pg.PlotWidget()
            plot.setAspectLocked(True)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setLabel('bottom', 'y (left +) [m]')
            plot.setLabel('left', 'x (forward) [m]')
            plot.invertX(True)
            plot.setRange(xRange=(-BEV_RANGE, BEV_RANGE), yRange=(-BEV_RANGE, BEV_RANGE))
            self._bev = pg.ImageItem()
            plot.addItem(self._bev)
            plot.plot([0], [0], symbol='t1', symbolSize=14, symbolBrush='#ffffff')
            layout.addWidget(plot, 1)
        self.setMinimumSize(320, 240)

    def _colors(self, z: np.ndarray) -> np.ndarray:
        lo, hi = self._z_range
        return TURBO.map(np.clip((z - lo) / (hi - lo), 0.0, 1.0), mode='float')

    @staticmethod
    def _raster(xyz: np.ndarray, colors: np.ndarray) -> np.ndarray:
        n = int(2 * BEV_RANGE / BEV_RES)
        img = np.zeros((n, n, 4), np.uint8)
        # Image row = y (rect bottom..top), column = x; highest point wins per cell.
        col = ((xyz[:, 1] + BEV_RANGE) / BEV_RES).astype(int)
        row = ((xyz[:, 0] + BEV_RANGE) / BEV_RES).astype(int)
        keep = (col >= 0) & (col < n) & (row >= 0) & (row < n)
        order = np.argsort(xyz[keep, 2])
        img[row[keep][order], col[keep][order]] = colors[keep][order]
        return img

    def update_from(self, latest) -> None:
        if latest.seq != self._seq and latest.value is not None:
            self._seq = latest.seq
            self._fps.tick()
            xyz = latest.value
            if len(xyz):
                colors = self._colors(xyz[:, 2])
                if self._gl is None:
                    colors = (colors * 255).astype(np.uint8)
                if self._gl is not None:
                    self._scatter.setData(pos=xyz, color=colors)
                else:
                    self._bev.setImage(self._raster(xyz, colors), autoLevels=False)
                    # setRect maps image pixels to metres; it needs the image set first.
                    self._bev.setRect(QtCore.QRectF(-BEV_RANGE, -BEV_RANGE,
                                                    2 * BEV_RANGE, 2 * BEV_RANGE))
        self._title_update(lambda: self._title_text(latest))

    def _title_text(self, latest) -> str:
        age = time.monotonic() - latest.stamp if latest.seq else None
        if age is None:
            return f'{self._title} — NO DATA'
        if age > STALE_S:
            return f'{self._title} — STALE {age:.1f} s'
        return f'{self._title} — {len(latest.value)} pts @ {self._fps.fps():.1f} Hz'


class Chart:
    """
    Rolling time plot of a Series (a PlotItem in a shared ChartPanel).

    The x axis is seconds before now, so it never changes; the y range only moves
    when the data leaves it or uses less than a third of it. Both axes are cached
    as pixmaps, so a frame redraws just the curves.
    """

    def __init__(self, plot: pg.PlotItem, names, unit: str, history: float):
        self._plot = plot
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel('left', unit)
        plot.getAxis('left').enableAutoSIPrefix(False)
        plot.addLegend(offset=(5, 1), labelTextSize='8pt', colCount=len(names))
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.enableAutoRange(x=False, y=False)
        plot.setXRange(-history, 0, padding=0)
        for axis in ('left', 'bottom'):
            plot.getAxis(axis).setCacheMode(
                QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self._curves = []
        for i, name in enumerate(names):
            curve = plot.plot(name=name, pen=pg.mkPen(SERIES_COLORS[i % 4], width=1))
            curve.setSkipFiniteCheck(True)
            self._curves.append(curve)
        self._last_t = None
        self._y_range = None

    def update_from(self, data: np.ndarray, now: float, history: float) -> None:
        # Redraw only when a new sample arrived; the state topics come at ~18 Hz.
        if not len(data) or data[-1, 0] == self._last_t:
            return
        self._last_t = data[-1, 0]
        data = data[data[:, 0] > now - history]
        t = data[:, 0] - now
        for i, curve in enumerate(self._curves):
            curve.setData(t, data[:, i + 1])
        self._fit_y(data[:, 1:])

    def _fit_y(self, values: np.ndarray) -> None:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if not np.isfinite(lo):
            return
        span = max(hi - lo, 1e-3)
        if self._y_range is not None:
            cur_lo, cur_hi = self._y_range
            inside = cur_lo <= lo and hi <= cur_hi
            if inside and span > (cur_hi - cur_lo) / 3:
                return
        self._y_range = (lo - 0.25 * span, hi + 0.25 * span)
        self._plot.setYRange(*self._y_range, padding=0)


class ChartPanel(pg.GraphicsLayoutWidget):
    """All charts stacked in one graphics view: one paint pass per frame."""

    def __init__(self, specs, history: float):
        """specs: list of (title, series names, unit)."""
        super().__init__()
        self.charts = []
        for row, (title, names, unit) in enumerate(specs):
            plot = self.addPlot(row=row, col=0, title=title)
            self.charts.append(Chart(plot, names, unit, history))
        self.setMinimumWidth(300)


class RateTable(QtWidgets.QTableWidget):
    """Measured vs expected rate per topic, coloured by health."""

    def __init__(self, topics):
        super().__init__(len(topics), 3)
        self._topics = list(topics)
        self._prev = {t: (0, time.monotonic()) for t in self._topics}
        self.setHorizontalHeaderLabels(['topic', 'Hz', 'expect'])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in (1, 2):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for row, topic in enumerate(self._topics):
            for col in range(3):
                self.setItem(row, col, QtWidgets.QTableWidgetItem(topic if col == 0 else ''))
        self.resizeRowsToContents()

    def update_from(self, rates) -> None:
        now = time.monotonic()
        for row, topic in enumerate(self._topics):
            count, last, expected = rates[topic]
            prev_count, prev_t = self._prev[topic]
            hz = (count - prev_count) / max(now - prev_t, 1e-6)
            self._prev[topic] = (count, now)
            if not expected:
                color = QtGui.QColor('#30343c')  # event-driven, no expected rate
            elif count == 0 or now - last > 2.0:
                color = BAD
            elif expected and hz < 0.7 * expected:
                color = WARN
            else:
                color = OK
            self.item(row, 1).setText(f'{hz:.1f}')
            self.item(row, 2).setText(f'{expected:g}' if expected else '-')
            for col in range(3):
                self.item(row, col).setBackground(color)
