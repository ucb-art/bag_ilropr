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

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, List, Dict, cast

from enum import Enum
import numpy as np

from bag3_testbenches.measurement.data.tran import EdgeType

class SweepSpecType(Enum):
    BIN = 0
    ONEHOT = 1
    NHOT = 2

# TODO: flush out for binary, 1-hot, and other fmts


def code_to_pin_list(code: Mapping, code_fmt: Mapping) -> Mapping[str, Union[int, str]]:
    pin_name: str = code['name']
    pin_value: int = code['value']
    pin_diff: str = code.get('diff', '')
    ans = {}

    if code_fmt['type'] == 'NHOT':
        num_hot: int = code_fmt['num']
        width: int = code_fmt['width']
        if num_hot > width:
            raise ValueError("num_hot cannot be greater than width")

        # Special case for disable
        if pin_value < 0:
            for idx in range(width):
                ans[f'{pin_name}<{idx}>'] = 0
                if pin_diff:
                    ans[f'{pin_diff}<{idx}>'] = 1
        else:
            for idx in range(width):
                hot = pin_value <= idx < pin_value + num_hot
                # Wrap around:
                if pin_value + num_hot >= width:
                    hot = hot or pin_value <= idx + width < pin_value + num_hot
                ans[f'{pin_name}<{idx}>'] = 1 if hot else 0
                if pin_diff:
                    ans[f'{pin_diff}<{idx}>'] = 0 if hot else 1
    else:
        raise RuntimeError("Code format currently not supported")

    return ans


def get_all_crossings(tvec: np.ndarray, yvec: np.ndarray, threshold: Union[float, np.ndarray],
                    start: Union[float, np.ndarray] = 0,
                    stop: Union[float, np.ndarray] = float('inf'),
                    etype: EdgeType = EdgeType.CROSS, rtol: float = 1e-8, atol: float = 1e-22,
                    shape: Optional[Tuple[int, ...]] = None) -> List[np.ndarray]:
    """Find the first time where waveform crosses a given threshold.

    tvec and yvec can be multi-dimensional, in which case the waveforms are stored in the
    last axis.  The returned numpy array will have the same shape as yvec with the last
    axis removed.  If the waveform never crosses the threshold, positive infinity will be
    returned.
    """
    swp_shape = yvec.shape[:-1]
    if shape is None:
        shape = yvec.shape[:-1]

    try:
        th_vec = np.broadcast_to(np.asarray(threshold), swp_shape)
        start = np.broadcast_to(np.asarray(start), swp_shape)
        stop = np.broadcast_to(np.asarray(stop), swp_shape)
    except ValueError as err:
        raise ValueError('Failed to make threshold/start/stop the same shape as data.  '
                         'Make sure they are either scalar or has the same sweep shape.') from err

    t_shape = tvec.shape
    nlast = t_shape[len(t_shape) - 1]

    yvec = yvec.reshape(-1, nlast)
    tvec = tvec.reshape(-1, nlast)
    th_vec = th_vec.flatten()
    t0_vec = start.flatten()
    t1_vec = stop.flatten()
    n_swp = th_vec.size
    ans = []
    num_tvec = tvec.shape[0]

    for idx in range(n_swp):
        cur_thres = th_vec[idx]
        cur_t0 = t0_vec[idx]
        cur_t1 = t1_vec[idx]
        ans_tmp = get_all_crossings_1d(tvec[idx % num_tvec, :], yvec[idx, :], cur_thres,
                                        cur_t0, cur_t1, etype, rtol, atol)
        ans.append(ans_tmp)
    return ans


def get_all_crossings_1d(tvec: np.ndarray, yvec: np.ndarray, threshold: float,
                      start: float, stop: float, etype: EdgeType, rtol: float,
                      atol: float) -> float:
    """Get all crossings for a 1d input yvec
    Use get_all_crossings for a multi-D sweep, e.g. with corners.
    """
    # eliminate NaN from time vector in cases where simulation time is different between runs.
    mask = ~np.isnan(tvec)
    tvec = tvec[mask]
    yvec = yvec[mask]

    sidx = np.searchsorted(tvec, start)
    eidx = np.searchsorted(tvec, stop)
    if eidx < tvec.size and np.isclose(stop, tvec[eidx], rtol=rtol, atol=atol):
        eidx += 1

    # quantize waveform values, then detect edge.
    dvec = np.diff((yvec[sidx:eidx] >= threshold).astype(int))

    ans = np.array([])
    if EdgeType.RISE in etype:
        idx_list = np.argwhere(np.maximum(dvec, 0)).flatten()
        for idx in idx_list:
            lidx = max(sidx + idx - 2, 0)
            ridx = min(sidx + idx + 3, len(yvec))
            tzc = np.interp(threshold, yvec[lidx:ridx], tvec[lidx:ridx])
            ans = np.append(ans, tzc)
    if EdgeType.FALL in etype:
        idx_list = np.argwhere(np.minimum(dvec, 0)).flatten()
        for idx in idx_list:
            lidx = max(sidx + idx - 2, 0)
            ridx = min(sidx + idx + 3, len(yvec))
            # Flip order. Inputs to np.interp must be increasing
            tzc = np.interp(threshold, yvec[lidx:ridx][::-1], tvec[lidx:ridx][::-1])
            ans = np.append(ans, tzc)
    return ans
