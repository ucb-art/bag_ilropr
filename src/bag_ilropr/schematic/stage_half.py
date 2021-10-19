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
class bag_ilropr__stage_half(Module):
    """Module for library bag_ilropr cell stage_half.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'stage_half.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Mapping[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            invt_params="Tristate inverter unit cell params",
            num_aa="Number of always on units",
            num_inj="Number of injection units",
        )

    def design(self, invt_params: Mapping, num_aa: int, num_inj: int) -> None:
        # Design unit cells and add VTOP / VBOT
        for inst_name in ['XAA', 'XINJP', 'XINJM', 'XFWDP', 'XFWDM']:
            self.instances[inst_name].design(**invt_params)
            self.reconnect_instance_terminal(inst_name, 'VTOP', 'VTOP')
            self.reconnect_instance_terminal(inst_name, 'VBOT', 'VBOT')

        # Array instances and reconnect enables
        self.rename_instance('XAA', f'XAA<{num_aa-1}:0>')

        num_inj2 = num_inj // 2
        inst_term_list = [(f'XINJP<{idx}>',
                           [('en',f'en<{idx}>'), ('enb',f'enb<{idx}>')]) for idx in range(num_inj2)]
        self.array_instance('XINJP', inst_term_list=inst_term_list, dy=-300)

        inst_term_list = [(f'XFWDP<{idx}>',
                           [('en', f'enb<{idx}>'), ('enb', f'en<{idx}>')]) for idx in range(num_inj2)]
        self.array_instance('XFWDP', inst_term_list=inst_term_list, dy=-300)

        inst_term_list = [(f'XINJM<{idx}>',
                           [('en', f'en<{num_inj2 + idx}>'), ('enb', f'enb<{num_inj2 + idx}>')]) for idx in range(num_inj2)]
        self.array_instance('XINJM', inst_term_list=inst_term_list, dy=-300)

        inst_term_list = [(f'XFWDM<{idx}>',
                           [('en', f'enb<{num_inj2 + idx}>'), ('enb', f'en<{num_inj2 + idx}>')]) for idx in range(num_inj2)]
        self.array_instance('XFWDM', inst_term_list=inst_term_list, dy=-300)

        # Rename pins
        if num_inj != 4:
            self.rename_pin('en<3:0>', f'en<{num_inj - 1}:0>')
            self.rename_pin('enb<3:0>', f'enb<{num_inj - 1}:0>')



