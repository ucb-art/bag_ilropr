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

# -*- coding: utf-8 -*-

from typing import Mapping, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag_ilropr__core_diff_pair(Module):
    """Module for library bag_ilropr cell core_diff_pair.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'core_diff_pair.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            lch="Unit tranisistor length",
            w="Unit transistor width",
            th="Unit transistor intent",
            unit_seg="Unit transistor segment",
            num_aa="Number of always on units",
            num_inj="Number of injection units",
            num_dum="Number of dummy units",
            seg_dict="Segment dictionary, alt way of describing segments",
        )
    
    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(
            unit_seg=0,
            num_aa=0,
            num_inj=0,
            num_dum=0,
            seg_dict={}
        )

    def design(self, lch: int, w: int, th: str, num_aa: int, num_inj: int, 
               num_dum: int, unit_seg: int, seg_dict: Mapping) -> None:
                
        assert num_aa >= num_inj
        
        # All transistors have the same unit cell (seg, lch, w, th). 
        # Design them, then array them
        diff_pair_names = ['XDPAA', 'XDMAA', 'XDPINJB', 'XDMINJB', 'XDPINJ', 'XDMINJ',
                            'XEPINJB', 'XEMINJB', 'XEPINJ', 'XEMINJ', 'XDUMP', 'XDUMM']
        
        if seg_dict:
            for xt_name in diff_pair_names:
                seg = seg_dict['seg_dp'] if 'AA' in xt_name else \
                    seg_dict['seg_dum'] if 'DUM' in xt_name else seg_dict['seg_inj']
                self.design_transistor(xt_name, w=w, lch=lch, seg=seg, intent=th)
        else:
            for xt_name in diff_pair_names:
                self.design_transistor(xt_name, w=w, lch=lch, seg=unit_seg, intent=th)
                
                if 'AA' in xt_name:
                    num_unit = num_aa
                elif 'DUM' in xt_name:
                    num_unit = num_dum
                else:
                    num_unit = num_inj

                if (num_unit * unit_seg) > 1:
                    self.rename_instance(xt_name, xt_name + f'<{num_unit - 1}:0>')