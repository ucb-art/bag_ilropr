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

from pybag.enum import TermType

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag_ilropr__in_buff_core(Module):
    """Module for library bag_ilropr cell in_buff_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'in_buff_core.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)
        self._num_stage = -1 

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            hpf_params='High pass filter params',
            bias_params="Replica bias gen params",
            buff_params="Replica buffer params",
        )

    def design(self, hpf_params: Mapping, bias_params: Mapping, buff_params: Mapping) -> None:
        self.instances['XHPF'].design(**hpf_params)
        self.instances['XBIAS'].design(**bias_params)
        self.instances['XBUFF'].design(**buff_params)
    
        # TODO: fix bias
        self.reconnect_instance_terminal('XHPF', 'VDD', 'VDD')

        self.reconnect_instance_terminal('XBIAS', 'out', 'bias_out')

        self._num_stage = buff_params['num_stage']
        if self._num_stage > 1:
            tail = f'tail<{self._num_stage-1}:0>'
            self.reconnect_instance_terminal('XBUFF', tail, tail)
            self.rename_pin('tail', tail)

        # TODO: debug
        self.add_pin('inbp', 'inout')
        self.add_pin('inbm', 'inout')
