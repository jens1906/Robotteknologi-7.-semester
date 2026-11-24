import json
import math

from perception.evaluation_metrics import ANSWER_UNANSWERED

# Formatting
DECIMALS = 3
MISSING_TEXT = '-'

# Report text
PERCEPTION_CAVEAT = ('Distance, losses and recoveries are what perception believes: a re-identification failure counts '
                     'as a loss even when the leader was in sight. The manual answers are the ground truth.')


def FormatReportJson(report):
    return json.dumps(CleanValue(report), indent=2)


def CleanValue(value):
    if isinstance(value, float):
        return round(value, DECIMALS) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: CleanValue(item) for key, item in value.items()}
    if isinstance(value, list):
        return [CleanValue(item) for item in value]
    return value


def FormatNumber(value, unit=''):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return MISSING_TEXT
    return f'{value:.2f}{unit}' if isinstance(value, float) else f'{value}{unit}'


def FormatPercent(value):
    if value is None or not math.isfinite(value):
        return MISSING_TEXT
    return f'{value * 100:.1f} %'


def FormatAnswer(value):
    if value is None:
        return 'unknown'
    return 'yes' if value else 'no'


def FormatManualSource(manualAnswer):
    return 'auto' if manualAnswer == ANSWER_UNANSWERED else 'manual'


def FormatReportMarkdown(report):
    followSuccess = report['followSuccess']
    collision = report['collision']
    distance = report['distance']
    recovery = report['recovery']
    wrongLeader = report['wrongLeader']
    runtime = report['runtime']
    systemUsage = report['systemUsage']
    manual = report['manual']
    metricRows = [
        ('Follow success', FormatAnswer(followSuccess['success']),
         f'semi: leader in sight at end {FormatAnswer(followSuccess["leaderInSightAtEnd"])} '
         f'({FormatManualSource(manual["leaderInSightAtEnd"])}), collision {FormatAnswer(followSuccess["collision"])}'),
        ('Leader loss time ratio', FormatPercent(report['leaderLossTimeRatio']), 'auto'),
        ('Collision', FormatAnswer(collision['collision']),
         f'manual: {manual["collisionMarks"]} marks; hints: minimum clearance '
         f'{FormatNumber(collision["minimumObstacleClearance"], " m")}, {collision["nearCollisionCount"]} near collisions'),
        ('Average distance', FormatNumber(distance['averageDistance'], ' m'),
         f'auto: desired {FormatNumber(distance["desiredDistance"], " m")}, mean error '
         f'{FormatNumber(distance["averageError"], " m")}, mean absolute error '
         f'{FormatNumber(distance["averageAbsoluteError"], " m")}, RMSE {FormatNumber(distance["rmse"], " m")}, '
         f'min {FormatNumber(distance["minimumDistance"], " m")}, max {FormatNumber(distance["maximumDistance"], " m")}'),
        ('Number of LOS losses', FormatNumber(recovery['lossCount']), 'auto'),
        ('Recovery success rate and time',
         f'{recovery["recoveryCount"]}/{recovery["lossCount"]} ({FormatPercent(recovery["recoverySuccessRate"])})',
         f'auto: mean {FormatNumber(recovery["averageRecoveryTime"], " s")}, median '
         f'{FormatNumber(recovery["medianRecoveryTime"], " s")}, max {FormatNumber(recovery["maximumRecoveryTime"], " s")}'),
        ('Wrong leader', FormatAnswer(wrongLeader['wrongLeader']),
         f'manual: {manual["wrongLeaderMarks"]} marks; hints: {wrongLeader["suspectedSwitchCount"]} suspected switches'),
        ('On robot runtime', FormatNumber(runtime['perceptionRateMean'], ' Hz'),
         f'auto: perception worst {FormatNumber(runtime["perceptionRateWorst"], " Hz")}, navigation mean '
         f'{FormatNumber(runtime["navigationRateMean"], " Hz")} (worst {FormatNumber(runtime["navigationRateWorst"], " Hz")}), '
         f'frame time mean {FormatNumber(runtime["frameTimeMsMean"], " ms")}, p95 {FormatNumber(runtime["frameTimeMsP95"], " ms")}'),
        ('Camera delay', FormatNumber(runtime['cameraAgeAtFusionMs']['p50'], ' ms'),
         f'auto: camera age at fusion p50, p95 {FormatNumber(runtime["cameraAgeAtFusionMs"]["p95"], " ms")}; image age '
         f'when perception got it p50 {FormatNumber(runtime["imageAgeMs"]["p50"], " ms")}, '
         f'p95 {FormatNumber(runtime["imageAgeMs"]["p95"], " ms")}'),
        ('Orin load', FormatNumber(systemUsage['cpuPercent']['mean'], ' % CPU'),
         f'auto: CPU p95 {FormatNumber(systemUsage["cpuPercent"]["p95"], " %")}, GPU mean '
         f'{FormatNumber(systemUsage["gpuPercent"]["mean"], " %")} '
         f'(p95 {FormatNumber(systemUsage["gpuPercent"]["p95"], " %")}), '
         f'RAM max {FormatNumber(systemUsage["ramUsedMbMax"], " MB")}, power mean '
         f'{FormatNumber(systemUsage["powerTotalMw"]["mean"], " mW")}, junction max '
         f'{FormatNumber(systemUsage["junctionTemperatureMax"], " C")}, network in/out '
         f'{FormatNumber(systemUsage["networkReceiveMbps"]["mean"], " MB/s")}/'
         f'{FormatNumber(systemUsage["networkSendMbps"]["mean"], " MB/s")}, disk write '
         f'{FormatNumber(systemUsage["diskWriteMbps"]["mean"], " MB/s")}'),
    ]
    for nodeName, nodeUsage in systemUsage['nodes'].items():
        metricRows.append((f'Node {nodeName}', FormatNumber(nodeUsage['cpuPercent']['mean'], ' % of a core'),
                           f'auto: CPU p95 {FormatNumber(nodeUsage["cpuPercent"]["p95"], " %")}, memory max '
                           f'{FormatNumber(nodeUsage["memoryMbMax"], " MB")}'))
    lines = [
        '# Evaluation report',
        '',
        f'Run duration {FormatNumber(report["runDuration"], " s")}.',
        '',
        '| Metric | Value | How / details |',
        '|---|---|---|',
    ]
    for metricName, metricValue, metricDetails in metricRows:
        lines.append(f'| {metricName} | {metricValue} | {metricDetails} |')
    lines += ['', '## Notes', '', manual['notes'] or MISSING_TEXT, '', '## Events', '']
    if report['events']:
        lines += ['| Run time (s) | Event | Description |', '|---|---|---|']
        for event in report['events']:
            lines.append(f'| {event["runTime"]:.2f} | {event["eventType"]} | {event["description"]} |')
    else:
        lines.append('No events.')
    lines += ['', '## Parameters', '']
    for parameterName, parameterValue in report['parameters'].items():
        lines.append(f'- {parameterName}: {parameterValue}')
    lines += ['', '## Caveat', '', PERCEPTION_CAVEAT, '']
    return '\n'.join(lines)

