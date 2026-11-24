import math
import os
from types import SimpleNamespace

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from follow_anywhere_interfaces.msg import (EvaluationMetrics, EvaluationSystemUsage, PerceptionDebugFrame,
                                            PerceptionLeader, PerceptionObstacleArray)
from follow_anywhere_interfaces.srv import EvaluationFinishRun, EvaluationMarkEvent
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from unitree_api.msg import Request

from perception.evaluation_metrics import EVALUATION_SETTING_DEFAULTS, ConvertStampToSeconds, RunMetrics
from perception.evaluation_report import FormatReportJson, FormatReportMarkdown
from perception.system_usage import SystemUsageSampler

# Parameter defaults (set in the evaluation_node section of config/perception_params.yaml, the run settings have their
# defaults in evaluation_metrics.py)
METRICS_RATE = 5.0
SYSTEM_USAGE_RATE = 1.0
PARAMS_FILE_NAME = 'perception_params.yaml'
PARAMS_SECTION = 'evaluation_node'


class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')

        fileSettings = LoadFileSettings()
        self.declare_parameter('metricsRate', fileSettings.get('metricsRate', METRICS_RATE))
        self.declare_parameter('systemUsageRate', fileSettings.get('systemUsageRate', SYSTEM_USAGE_RATE))
        for settingName, defaultValue in EVALUATION_SETTING_DEFAULTS.items():
            self.declare_parameter(settingName, fileSettings.get(settingName, defaultValue))
        self.metricsRate = self.get_parameter('metricsRate').value
        self.systemUsageRate = self.get_parameter('systemUsageRate').value
        self.runSettings = SimpleNamespace(
            **{settingName: self.get_parameter(settingName).value for settingName in EVALUATION_SETTING_DEFAULTS})

        self.runMetrics = RunMetrics(self.runSettings)
        self.runActive = False
        self.lastDebugMessage = ''
        self.newestCameraStampSeconds = math.nan
        self.systemUsageSampler = SystemUsageSampler()

        # Output topics
        self.metricsPublisher = self.create_publisher(EvaluationMetrics, '/evaluation/metrics', 10)
        self.systemUsagePublisher = self.create_publisher(EvaluationSystemUsage, '/evaluation/system_usage', 10)
        self.debugPublisher = self.create_publisher(String, '/evaluation/debug', 10)

        # Services
        self.create_service(Trigger, '/evaluation/start_run', self.HandleStartRun)
        self.create_service(EvaluationFinishRun, '/evaluation/finish_run', self.HandleFinishRun)
        self.create_service(EvaluationMarkEvent, '/evaluation/mark_event', self.HandleMarkEvent)

        # Input topics
        self.create_subscription(PerceptionLeader, '/perception/leader', self.HandleLeader, 10)
        self.create_subscription(PerceptionLeader, '/perception/camera/leader', self.HandleCameraLeader, 10)
        self.create_subscription(PerceptionDebugFrame, '/perception/debug_frame', self.HandleDebugFrame, 10)
        self.create_subscription(PerceptionObstacleArray, '/perception/obstacles_combined', self.HandleObstacles, 10)
        self.create_subscription(Request, '/api/sport/request', self.HandleNavigationCommand, 10)

        self.create_timer(1.0 / self.metricsRate, self.PublishMetrics)
        self.create_timer(1.0 / self.systemUsageRate, self.PublishSystemUsage)

    def HandleLeader(self, leaderMessage):
        if not self.runActive:
            return
        stampSeconds = ConvertStampToSeconds(leaderMessage.header.stamp)
        self.runMetrics.UpdateLeader(stampSeconds, leaderMessage.status, leaderMessage.closest_point,
                                     leaderMessage.last_known_point, self.ReadReceiveSeconds())
        self.runMetrics.UpdateCameraAgeAtFusion((stampSeconds - self.newestCameraStampSeconds) * 1000.0)

    def HandleCameraLeader(self, leaderMessage):
        self.newestCameraStampSeconds = ConvertStampToSeconds(leaderMessage.header.stamp)

    def HandleDebugFrame(self, debugFrameMessage):
        if not self.runActive:
            return
        self.runMetrics.UpdateDebugFrame(ConvertStampToSeconds(debugFrameMessage.header.stamp),
                                         debugFrameMessage.leader_state, debugFrameMessage.seconds_since_seen,
                                         debugFrameMessage.frame_time_ms, debugFrameMessage.image_age_ms)

    def HandleObstacles(self, obstaclesMessage):
        if not self.runActive:
            return
        self.runMetrics.UpdateObstacles([math.hypot(obstacle.closest_point.x, obstacle.closest_point.y)
                                         for obstacle in obstaclesMessage.obstacles])

    def HandleNavigationCommand(self, requestMessage):
        if not self.runActive:
            return
        self.runMetrics.CountNavigationCommand(self.ReadReceiveSeconds())

    def HandleStartRun(self, request, response):
        if self.runActive:
            self.PublishDebugMessage('EvaluationNode', 'start_run while a run was active, previous run discarded')
        self.runMetrics.Reset()
        self.runActive = True
        self.get_logger().info('EvaluationNode: run started')
        response.success = True
        response.message = 'evaluation run started'
        return response

    def HandleFinishRun(self, request, response):
        if not self.runActive:
            self.PublishDebugMessage('EvaluationNode', 'finish_run without an active run, no report')
            response.success = False
            return response
        self.PublishMetrics()
        report = self.runMetrics.BuildReport(request)
        self.runActive = False
        response.report_json = FormatReportJson(report)
        response.report_markdown = FormatReportMarkdown(report)
        response.success = True
        self.get_logger().info(f'EvaluationNode: run finished after {report["runDuration"]:.1f} s, '
                               f'{report["recovery"]["lossCount"]} losses')
        return response

    def HandleMarkEvent(self, request, response):
        if not self.runActive:
            self.PublishDebugMessage('EvaluationNode', f'{request.event_type} marked without an active run')
            response.success = False
            response.message = 'no evaluation run active'
            return response
        response.success = self.runMetrics.MarkEvent(request.event_type, request.note)
        if not response.success:
            self.PublishDebugMessage('EvaluationNode', f'unknown event type {request.event_type}')
        response.message = f'{request.event_type} marked' if response.success else 'unknown event type'
        return response

    def PublishMetrics(self):
        if not self.runActive or math.isnan(self.runMetrics.runStartSeconds):
            return
        self.metricsPublisher.publish(self.runMetrics.BuildMetrics(self.ReadReceiveSeconds()))

    def PublishSystemUsage(self):
        systemSample = self.systemUsageSampler.Sample()
        if self.runActive:
            self.runMetrics.UpdateSystemUsage(systemSample)
        self.systemUsagePublisher.publish(self.BuildSystemUsageMessage(systemSample))

    def BuildSystemUsageMessage(self, systemSample):
        usageMessage = EvaluationSystemUsage(
            cpu_percent=systemSample.cpuPercent,
            cpu_core_percent=systemSample.cpuCorePercent,
            ram_used_mb=systemSample.ramUsedMb,
            ram_total_mb=systemSample.ramTotalMb,
            swap_used_mb=systemSample.swapUsedMb,
            gpu_percent=systemSample.gpuPercent,
            cpu_temperature=systemSample.cpuTemperature,
            gpu_temperature=systemSample.gpuTemperature,
            junction_temperature=systemSample.junctionTemperature,
            power_total_mw=systemSample.powerTotalMw,
            disk_read_mbps=systemSample.diskReadMbps,
            disk_write_mbps=systemSample.diskWriteMbps,
            disk_free_gb=systemSample.diskFreeGb,
            network_receive_mbps=systemSample.networkReceiveMbps,
            network_send_mbps=systemSample.networkSendMbps,
            loopback_mbps=systemSample.loopbackMbps)
        usageMessage.header.stamp = self.get_clock().now().to_msg()
        for nodeName, nodeUsage in systemSample.nodes.items():
            setattr(usageMessage, f'{nodeName}_cpu_percent', nodeUsage.cpuPercent)
            setattr(usageMessage, f'{nodeName}_memory_mb', nodeUsage.memoryMb)
        return usageMessage

    def ReadReceiveSeconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def PublishDebugMessage(self, area, whatHappened):
        debugMessage = f'{area}: {whatHappened}'
        if debugMessage == self.lastDebugMessage:
            return
        self.lastDebugMessage = debugMessage
        self.get_logger().warning(debugMessage)
        self.debugPublisher.publish(String(data=debugMessage))


def LoadFileSettings():
    paramsPath = os.path.join(get_package_share_directory('perception'), 'config', PARAMS_FILE_NAME)
    with open(paramsPath) as paramsFile:
        fileContent = yaml.safe_load(paramsFile) or {}
    return fileContent.get(PARAMS_SECTION, {}).get('ros__parameters', {})


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.systemUsageSampler.Stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
