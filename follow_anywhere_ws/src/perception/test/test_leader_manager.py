import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from perception.body_view import FRONT_GROUP
from perception.evaluation_metrics import EVALUATION_SETTING_DEFAULTS, RunMetrics
from perception.leader_manager import (ENROLLING_STATE, FOLLOWING_STATE, LEADER_SETTING_DEFAULTS, LOST_EVENT,
                                       SEARCHING_STATE, LeaderManager)
from perception.reid_model import CELL_COUNT, OSNET_EMBEDDING_SIZE, PersonDescription

# Test values
MODEL_TAG = 'test_model'
FOLLOWED_TRACK_ID = 7


def MakeLeaderManager():
    return LeaderManager(lambda query, storedStack, minimumSharedArea: (np.zeros(0), np.zeros(0)),
                         SimpleNamespace(**LEADER_SETTING_DEFAULTS))


def MakePerson(trackId):
    return SimpleNamespace(trackId=trackId, left=100.0, top=50.0, right=200.0, bottom=350.0,
                           bodyKeypoints=np.zeros((17, 3)))


def SaveTestLeader(leaderFilePath):
    leaderManager = MakeLeaderManager()
    leaderManager.viewDescriptions[FRONT_GROUP].append(
        PersonDescription(np.ones((CELL_COUNT, OSNET_EMBEDDING_SIZE), np.float32), np.ones(CELL_COUNT, np.float32)))
    leaderManager.anchorCounts[FRONT_GROUP] = 1
    leaderManager.SaveLeader(leaderFilePath, MODEL_TAG)


def test_saved_leader_loads_again(tmp_path):
    leaderFilePath = str(tmp_path / 'leader.npz')
    SaveTestLeader(leaderFilePath)
    loadedLeaderManager = MakeLeaderManager()
    assert loadedLeaderManager.LoadLeader(leaderFilePath, MODEL_TAG)
    assert loadedLeaderManager.CountStoredViews()[FRONT_GROUP] == 1
    assert loadedLeaderManager.state == SEARCHING_STATE
    assert not os.path.exists(leaderFilePath + '.tmp')


@pytest.mark.parametrize('keptShare', [0.0, 0.5])
def test_cut_off_leader_file_is_refused(tmp_path, keptShare):
    leaderFilePath = str(tmp_path / 'leader.npz')
    SaveTestLeader(leaderFilePath)
    with open(leaderFilePath, 'rb') as leaderFile:
        fileContent = leaderFile.read()
    with open(leaderFilePath, 'wb') as leaderFile:
        leaderFile.write(fileContent[:int(len(fileContent) * keptShare)])
    assert not MakeLeaderManager().LoadLeader(leaderFilePath, MODEL_TAG)


def test_followed_track_far_from_prediction_is_dropped():
    leaderManager = MakeLeaderManager()
    leaderManager.state = FOLLOWING_STATE
    leaderManager.followedTrackId = FOLLOWED_TRACK_ID
    gateDistances = {FOLLOWED_TRACK_ID: leaderManager.settings.swapGateSigma + 1.0}
    leaderEvent = leaderManager.Update([MakePerson(FOLLOWED_TRACK_ID)], None, gateDistances, 1.0, 640)
    assert leaderEvent == LOST_EVENT
    assert leaderManager.state == SEARCHING_STATE


def test_old_distances_are_forgotten():
    leaderManager = MakeLeaderManager()
    leaderManager.RememberDistance(FOLLOWED_TRACK_ID, 0.3, 1.0, 0.0)
    leaderManager.Update([], None, {}, 100.0, 640)
    assert FOLLOWED_TRACK_ID not in leaderManager.recentComparisons


def test_loss_and_recovery_time():
    runMetrics = RunMetrics(SimpleNamespace(**EVALUATION_SETTING_DEFAULTS))
    runMetrics.UpdateDebugFrame(0.0, FOLLOWING_STATE, math.nan, 10.0)
    runMetrics.UpdateDebugFrame(2.0, SEARCHING_STATE, 0.5, 10.0)
    runMetrics.UpdateDebugFrame(4.0, FOLLOWING_STATE, 0.0, 10.0)
    assert runMetrics.lossCount == 1
    assert runMetrics.recoveryTimes == [pytest.approx(2.5)]


def test_loss_ended_by_new_calibration_is_not_a_recovery():
    runMetrics = RunMetrics(SimpleNamespace(**EVALUATION_SETTING_DEFAULTS))
    runMetrics.UpdateDebugFrame(0.0, FOLLOWING_STATE, math.nan, 10.0)
    runMetrics.UpdateDebugFrame(2.0, SEARCHING_STATE, 0.5, 10.0)
    runMetrics.UpdateDebugFrame(3.0, ENROLLING_STATE, math.nan, 10.0)
    runMetrics.UpdateDebugFrame(5.0, FOLLOWING_STATE, 0.0, 10.0)
    assert runMetrics.recoveryTimes == []
    assert math.isnan(runMetrics.lossStartSeconds)
