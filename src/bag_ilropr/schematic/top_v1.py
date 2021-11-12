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
from pybag.enum import TermType


# noinspection PyPep8Naming
class bag_ilropr__top_v1(Module):
    """Module for library bag_ilropr cell top_v1.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'top_v1.yaml')))

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
            in_buff_params='Params for InBuff',
            core_out_params="Params for CoreOutDAC",
            debug_pins="True to export pins for debug",
        )
    
    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(debug_pins=True)

    def design(self, in_buff_params: Mapping, core_out_params: Mapping, debug_pins: bool) -> None:
        self.instances['XBUFF'].design(**in_buff_params)
        self.instances['XCORE'].design(**core_out_params)

        self.num_en = self.instances['XCORE'].master.num_en

        suf = f'<{self.num_en - 1}:0>'
        for en in ['en', 'enb']:
            self.reconnect_instance_terminal('XCORE', en + suf, en + suf)
            self.rename_pin(en, en + suf)

        if debug_pins: 
            self.reconnect_instance_terminal('XBUFF', 'inbp', 'inbp')
            self.reconnect_instance_terminal('XBUFF', 'inbm', 'inbm')
            self.add_pin('inbp', 'inout')
            self.add_pin('inbm', 'inout')
        else:
            if 'inbp' in self.instances['XBUFF'].master.pins:
                self.reconnect_instance_terminal('XBUFF', 'inbp', 'NC0')
                self.reconnect_instance_terminal('XBUFF', 'inbm', 'NC1')
