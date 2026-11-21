"""Ridge-regression leader classifier over body bands (the decisionLayer 'ridge' in leader_manager). Plain python.

To include:
- Method of Ye et al., RA-L 2024 (IEEE 10621449), Eq. 3 and 4: one ridge regression per body band (head to feet,
  reid_model.EmbedBands), trained on the leader's views (label 1) and other people (label 0), closed form
  W = X^T (X X^T + lambda I)^-1 y (dual form: few samples, many features).
- A band is used for training and scoring only when at least BAND_VISIBLE_SHARE of its cells show the person.
- Score of a person = the bands' outputs averaged with their visible share as weight (about 1 = leader, 0 = other).
  Too little visible (share of the whole body below minimumSharedArea): NaN, unknown.
"""

import math

import numpy as np

# Bands
BAND_VISIBLE_SHARE = 0.5


def FitBandClassifiers(bandEmbeddings, bandVisibility, labels, ridgeLambda):
    bandWeights = []
    for bandIndex in range(bandEmbeddings.shape[1]):
        usedSamples = bandVisibility[:, bandIndex] >= BAND_VISIBLE_SHARE
        if not labels[usedSamples].any():
            bandWeights.append(None)
            continue
        sampleFeatures = bandEmbeddings[usedSamples, bandIndex]
        sampleGram = sampleFeatures @ sampleFeatures.T + ridgeLambda * np.eye(len(sampleFeatures))
        bandWeights.append(sampleFeatures.T @ np.linalg.solve(sampleGram, labels[usedSamples].astype(np.float64)))
    return bandWeights


def ScoreBands(bandWeights, bandEmbeddings, bandVisibility, minimumSharedArea):
    usedBands = [bandIndex for bandIndex, weights in enumerate(bandWeights)
                 if weights is not None and bandVisibility[bandIndex] >= BAND_VISIBLE_SHARE]
    visibleShare = float(bandVisibility[usedBands].sum() / len(bandVisibility)) if usedBands else 0.0
    if visibleShare < minimumSharedArea:
        return math.nan, visibleShare
    bandScores = np.array([bandEmbeddings[bandIndex] @ bandWeights[bandIndex] for bandIndex in usedBands])
    return float(np.average(bandScores, weights=bandVisibility[usedBands])), visibleShare
