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
class bag_ilropr__stage(Module):
    """Module for library bag_ilropr cell stage.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'stage.yaml')))

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
            inv_params='Cross coupled inverter unit cell params',
            num_aa="Number of always on units",
            num_inj="Number of injection units",
            num_cc="Number of cross coupled units"
        )

    def design(self, invt_params: Mapping, inv_params: Mapping, num_aa: int, num_inj: int, num_cc: int) -> None:
        # Design instances
        self.instances['XHALFL'].design(invt_params=invt_params, num_aa=num_aa, num_inj=num_inj)
        self.instances['XHALFR'].design(invt_params=invt_params, num_aa=num_aa, num_inj=num_inj)
        for inst_name in ['XCC0', 'XCC1']:
            self.instances[inst_name].design(**inv_params)
            self.reconnect_instance_terminal(inst_name, 'VTOP', 'VTOP')
            self.reconnect_instance_terminal(inst_name, 'VBOT', 'VBOT')

        # Array CC instances
        if num_cc > 1:
            self.rename_instance('XCC0', f'XCC0<{num_cc-1}:0>')
            self.rename_instance('XCC1', f'XCC1<{num_cc-1}:0>')

        # Reconnect and update enable pins
        if num_inj != 4:
            suf = f'<{num_inj - 1}:0>'
            for pre in ['en', 'enb']:
                self.reconnect_instance_terminal('XHALFL', pre + suf, pre + suf)
                self.reconnect_instance_terminal('XHALFR', pre + suf, pre + suf)
                self.rename_pin(pre + '<3:0>', pre + suf)
