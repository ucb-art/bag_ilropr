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
class bag_ilropr__in_buff(Module):
    """Module for library bag_ilropr cell in_buff.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'in_buff.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)
        self._num_stage = -1 

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            core_params='Input buffer core params',
            imirr_params="IMIRR array params",
        )

    def design(self, core_params: Mapping, imirr_params: Mapping) -> None:
        self.instances['XCORE'].design(**core_params)
        self.instances['XDAC'].design(**imirr_params)

        self._num_stage = self.instances['XCORE'].master._num_stage
        
        core_tail = f'tail<{self._num_stage-1}:0>'
        self.reconnect_instance_terminal('XCORE', core_tail, core_tail)

        dac_tail = core_tail + ',bias_tail' # The order doesn't really matter
        self.reconnect_instance_terminal('XDAC', f'tail_aa<{self._num_stage}:0>', dac_tail)
        
        # TODO: debug
        self.reconnect_instance_terminal('XCORE', 'inbp', 'inbp')
        self.reconnect_instance_terminal('XCORE', 'inbm', 'inbm')
        self.add_pin('inbp', 'inout')
        self.add_pin('inbm', 'inout')
