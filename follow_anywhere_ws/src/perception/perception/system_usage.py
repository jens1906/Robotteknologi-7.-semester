"""Load of this computer for the evaluation node. Plain python, no ROS.

To include:
- psutil: CPU (mean and per core), RAM, swap, disk read/write and free space, network (loopback apart), and the CPU
  (% of one core) and resident memory of the pipeline's nodes, found by their executable name.
- tegrastats (Jetson only, read in a background thread): GPU load, CPU / GPU / junction temperature, total power.
  Without tegrastats those values are NaN.
- Rates (CPU %, MB/s) cover the time since the previous sample.
"""

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil

# Watched nodes: message name -> executable name and the arguments it must also have
WATCHED_NODES = {
    'perception': ('perception_node', ()),
    'lidar': ('lidar_obstacles_node', ()),
    'navigation': ('follow_leader_cpp', ()),
    'search': ('leader_search_node', ()),
    'camera_driver': ('realsense2_camera_node', ()),
    'bag_recorder': ('ros2', ('bag', 'record')),
}

# tegrastats
TEGRASTATS_COMMAND = 'tegrastats'
TEGRASTATS_INTERVAL_MS = '1000'
GPU_PATTERN = re.compile(r'GR3D_FREQ (\d+)%')
CPU_TEMPERATURE_PATTERN = re.compile(r'\bcpu@([\d.]+)C')
GPU_TEMPERATURE_PATTERN = re.compile(r'\bgpu@([\d.]+)C')
JUNCTION_TEMPERATURE_PATTERN = re.compile(r'\btj@([\d.]+)C')
POWER_PATTERN = re.compile(r'VDD_IN (\d+)mW')

# Units and file systems
BYTES_PER_MB = 1e6
BYTES_PER_GB = 1e9
LOOPBACK_INTERFACE = 'lo'
ROOT_PATH = '/'
NAN = float('nan')


@dataclass
class NodeUsage:
    cpuPercent: float = NAN
    memoryMb: float = NAN


@dataclass
class SystemSample:
    cpuPercent: float
    cpuCorePercent: list
    ramUsedMb: float
    ramTotalMb: float
    swapUsedMb: float
    gpuPercent: float
    cpuTemperature: float
    gpuTemperature: float
    junctionTemperature: float
    powerTotalMw: float
    diskReadMbps: float
    diskWriteMbps: float
    diskFreeGb: float
    networkReceiveMbps: float
    networkSendMbps: float
    loopbackMbps: float
    nodes: dict = field(default_factory=dict)


class SystemUsageSampler:
    def __init__(self):
        self.tegrastatsLine = ''
        self.tegrastatsProcess = None
        self.processesByNode = {}
        self.previousSeconds = time.monotonic()
        self.previousDisk = psutil.disk_io_counters()
        self.previousNetwork = psutil.net_io_counters(pernic=True)
        psutil.cpu_percent(percpu=True)
        if shutil.which(TEGRASTATS_COMMAND):
            self.tegrastatsProcess = subprocess.Popen([TEGRASTATS_COMMAND, '--interval', TEGRASTATS_INTERVAL_MS],
                                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self.ReadTegrastats, daemon=True).start()

    def ReadTegrastats(self):
        for line in self.tegrastatsProcess.stdout:
            self.tegrastatsLine = line

    def Stop(self):
        if self.tegrastatsProcess is not None:
            self.tegrastatsProcess.terminate()
            self.tegrastatsProcess.wait(timeout=2.0)

    def Sample(self):
        nowSeconds = time.monotonic()
        elapsedSeconds = max(nowSeconds - self.previousSeconds, 1e-3)
        self.previousSeconds = nowSeconds
        corePercents = psutil.cpu_percent(percpu=True)
        memory = psutil.virtual_memory()
        diskReadMbps, diskWriteMbps = self.MeasureDisk(elapsedSeconds)
        networkReceiveMbps, networkSendMbps, loopbackMbps = self.MeasureNetwork(elapsedSeconds)
        line = self.tegrastatsLine
        return SystemSample(
            cpuPercent=sum(corePercents) / len(corePercents),
            cpuCorePercent=corePercents,
            ramUsedMb=(memory.total - memory.available) / BYTES_PER_MB,
            ramTotalMb=memory.total / BYTES_PER_MB,
            swapUsedMb=psutil.swap_memory().used / BYTES_PER_MB,
            gpuPercent=ReadNumber(GPU_PATTERN, line),
            cpuTemperature=ReadNumber(CPU_TEMPERATURE_PATTERN, line),
            gpuTemperature=ReadNumber(GPU_TEMPERATURE_PATTERN, line),
            junctionTemperature=ReadNumber(JUNCTION_TEMPERATURE_PATTERN, line),
            powerTotalMw=ReadNumber(POWER_PATTERN, line),
            diskReadMbps=diskReadMbps,
            diskWriteMbps=diskWriteMbps,
            diskFreeGb=psutil.disk_usage(ROOT_PATH).free / BYTES_PER_GB,
            networkReceiveMbps=networkReceiveMbps,
            networkSendMbps=networkSendMbps,
            loopbackMbps=loopbackMbps,
            nodes={nodeName: self.MeasureNode(nodeName) for nodeName in WATCHED_NODES})

    def MeasureDisk(self, elapsedSeconds):
        disk = psutil.disk_io_counters()
        previousDisk, self.previousDisk = self.previousDisk, disk
        if disk is None or previousDisk is None:
            return NAN, NAN
        return ((disk.read_bytes - previousDisk.read_bytes) / BYTES_PER_MB / elapsedSeconds,
                (disk.write_bytes - previousDisk.write_bytes) / BYTES_PER_MB / elapsedSeconds)

    def MeasureNetwork(self, elapsedSeconds):
        network = psutil.net_io_counters(pernic=True)
        previousNetwork, self.previousNetwork = self.previousNetwork, network
        receivedBytes = sentBytes = loopbackBytes = 0
        for interfaceName, counters in network.items():
            previousCounters = previousNetwork.get(interfaceName)
            if previousCounters is None:
                continue
            if interfaceName == LOOPBACK_INTERFACE:
                loopbackBytes = counters.bytes_sent - previousCounters.bytes_sent
                continue
            receivedBytes += counters.bytes_recv - previousCounters.bytes_recv
            sentBytes += counters.bytes_sent - previousCounters.bytes_sent
        scale = 1.0 / BYTES_PER_MB / elapsedSeconds
        return receivedBytes * scale, sentBytes * scale, loopbackBytes * scale

    def MeasureNode(self, nodeName):
        process = self.processesByNode.get(nodeName)
        if process is None or not process.is_running():
            process = FindNodeProcess(*WATCHED_NODES[nodeName])
            self.processesByNode[nodeName] = process
            if process is None:
                return NodeUsage()
        try:
            return NodeUsage(cpuPercent=process.cpu_percent(None),
                             memoryMb=process.memory_info().rss / BYTES_PER_MB)
        except psutil.Error:
            self.processesByNode[nodeName] = None
            return NodeUsage()


def FindNodeProcess(executableName, requiredArguments):
    for process in psutil.process_iter(['cmdline']):
        commandLine = process.info['cmdline'] or []
        executableNames = [os.path.basename(argument) for argument in commandLine[:2]]
        if executableName in executableNames and all(argument in commandLine for argument in requiredArguments):
            process.cpu_percent(None)
            return process
    return None


def ReadNumber(pattern, text):
    match = pattern.search(text)
    return float(match.group(1)) if match else NAN
