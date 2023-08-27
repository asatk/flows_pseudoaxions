"""
Author: Anthony Atkinson
Modified: 2023.08.24

Data generation.
"""


from ._gaussian import CovStrategy, CovModStrategy, DiagCov, FullCov, NoneMod,\
    SampleCov, sample_gaussian, RepeatStrategy


__all__ = [
    "CovStrategy",
    "CovModStrategy",
    "DiagCov",
    "FullCov",
    "NoneMod",
    "SampleCov",
    "sample_gaussian",
    "RepeatStrategy"
]