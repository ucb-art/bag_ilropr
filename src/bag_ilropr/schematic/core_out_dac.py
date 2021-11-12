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

from turtle import Terminator
from typing import Mapping, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param
from pybag.enum import TermType


# noinspection PyPep8Naming
class bag_ilropr__core_out_dac(Module):
    """Module for library bag_ilropr cell core_out_dac.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'core_out_dac.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)
        self.num_en = -1

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Mapping[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            core_params='Oscillator core params',
            c2c_params="C2C unit params",
            idac_params="IDAC params.",
        )
    
    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict()

    def design(self, core_params: Mapping, c2c_params: Mapping, idac_params: Mapping) -> None:
        num_stage = core_params['num_stage']
        num_enable = idac_params['num_en']
        num_inj = num_enable // 2
        
        self.instances['XCORE'].design(**core_params)
        self.instances['XC2C'].design(export_mid=False, **c2c_params)
        self.instances['XIDAC0'].design(**idac_params)
        self.instances['XIDAC1'].design(**idac_params)

        # Tails
        for tail in ['tail_aa', 'tail_inj', 'tail_injb', 'taile_inj', 'taile_injb']:
            self.reconnect_instance_terminal('XCORE', f'{tail}<{num_stage-1}:0>', f'{tail}<{num_stage-1}:0>')
        # Note, the IDAC pin selection is done to match the layout for visual clarity.
        # However, to pass LVS, this isn't required because the IDACs are identical.
        for tail in ['tail_aa', 'tail_inj', 'tail_injb', 'taile_inj', 'taile_injb']:
            self.reconnect_instance_terminal('XIDAC0', f'{tail}<{num_stage // 2 - 1}:0>', f'{tail}<3>,{tail}<0>')
            self.reconnect_instance_terminal('XIDAC1', f'{tail}<{num_stage // 2 - 1}:0>', f'{tail}<2:1>')

        # enables
        num_tot = num_enable * num_stage
        for en in ['en', 'enb']:
            en0_str, en1_str = '', ''
            idx = 0
            while idx < num_tot:
                en0_str = f',{en}<{idx + num_inj - 1}:{idx}>' + en0_str
                en1_str = f',{en}<{idx + 2 * num_inj - 1}:{num_inj + idx}>' + en1_str
                en1_str = f',{en}<{idx + 3 * num_inj - 1}:{2 * num_inj + idx}>' + en1_str
                en0_str = f',{en}<{idx + 4 * num_inj - 1}:{3 * num_inj + idx}>' + en0_str
                idx += 4 * num_inj
            en0_str, en1_str = en0_str[1:], en1_str[1:]  # Remove the leading ','
            self.reconnect_instance_terminal('XIDAC0', f'{en}<{num_stage // 2 * num_enable - 1}:0>', en0_str)
            self.reconnect_instance_terminal('XIDAC1', f'{en}<{num_stage // 2 * num_enable - 1}:0>', en1_str)
            self.rename_pin(en, f'{en}<{num_tot-1}:0>')
        self.num_en = num_tot

        # array output buffer
        self.reconnect_instance_terminal('XCORE', f'V<{2*num_stage-1}:0>', f'mid<{2*num_stage-1}:0>')
        inst_term_list = [
            (f"XC2C{idx}", [
               ('inp', f'mid<{idx}>'), ('inm', f'mid<{num_stage + idx}>'),
               ('outm', f'out<{num_stage + idx}>'), ('outp', f'out<{idx}>'),
            ]) for idx in range(num_stage)
        ]
        self.array_instance('XC2C', inst_term_list=inst_term_list)

        if num_stage != 4: 
            self.rename_pin('out<7:0>', f'out<{2*num_stage-1}:0>')
            self.rename_pin('mid<7:0>', f'mid<{2*num_stage-1}:0>')
