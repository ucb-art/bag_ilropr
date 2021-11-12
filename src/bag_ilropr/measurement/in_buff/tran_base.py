# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Transient measurements of the input buffer path. Previously it's own class, but now
refactored to use the same methods as the rotator transient measurement class.
"""

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, List, Dict, cast

from pathlib import Path
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

from pybag.enum import LogLevel

from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.data import SimData, AnalysisType
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.base import TranTB

from ..rotator.tran import PhaseRotatorTranMM


class InBuffTranMM(PhaseRotatorTranMM):
    def __init__(self, meas_specs: Mapping[str, Any], log_file: str, 
                 log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        super().__init__(meas_specs, log_file, log_level, precision)

        # Workaround to allow InBuffTranMM to share methods with PhaseRotatorTranMM
        self._has_code = False
