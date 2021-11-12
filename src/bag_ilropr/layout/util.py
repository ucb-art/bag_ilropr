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

from typing import Union, Mapping, List
from bag.layout.routing.grid import RoutingGrid

from pybag.enum import RoundMode

from bag.typing import TrackType
from bag.layout.routing.base import TrackID, WireArray
from bag.io.file import read_yaml


def track_to_track(grid: RoutingGrid, tidx: Union[TrackType, TrackID, WireArray], 
                   lay1: int, lay2: int, mode=RoundMode.NEAREST):
    """Gets nearest track to another track on another layer. Layers must be the same direction"""
    assert grid.is_horizontal(lay1) == grid.is_horizontal(lay2)
    if isinstance(tidx, WireArray):
        tidx = tidx.track_id.base_index
    elif isinstance(tidx, TrackID):
        tidx = tidx.base_index
    coord = grid.track_to_coord(lay1, tidx)
    return grid.coord_to_track(lay2, coord, mode)

def import_params(params: Union[str, Mapping]) -> Mapping:
    if isinstance(params, str):
        return read_yaml(params).get('params')
    return params

def get_closest_warr(grid: RoutingGrid, warr_list: List[WireArray], ref_warr: WireArray):
    lower, upper = ref_warr.lower, ref_warr.upper
    ans = None
    best_dist = 1e6
    for _warr in warr_list:
        # Get dimensions orthogonal to route
        if grid.is_horizontal(_warr.layer_id):
            wl, wh = _warr.bound_box.yl, _warr.bound_box.yh
        else:
            wl, wh = _warr.bound_box.xl, _warr.bound_box.xh
        # Brute force
        if lower <= wl and wh <= upper:
            # full overlap case 1
            metric = -1e6
        elif wl <= lower and upper <= wh:
            # full overlap case 2
            metric = -1e6
        elif wl <= lower <= wh:
            # partial overlap case 1; save as negative
            overlap = wh - lower
            metric = -overlap
        elif wl <= upper <= wh:
            # partial overlap case 1; save as negative
            overlap = upper - wl
            metric = -overlap
        elif wh <= lower:
            # nonoverlap case 1; save as positive
            metric = lower - wh
        elif upper <= wl:
            # nonoverlap case 2; save as positive
            metric = wl - upper
        else:
            raise RuntimeError("Unknown case")
        
        if metric < best_dist:
            ans = _warr
            best_dist = metric
    return ans
