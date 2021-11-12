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
class bag_ilropr__buf_unit(Module):
    """Module for library bag_ilropr cell buf_unit.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'buf_unit.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            core_params='Parameters for the diff pair / Gm core',
            res_params='Parameters for the differential resistor pair',
            is_bias="True if is replica bias",      
        )
    
    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(is_bias=False)

    def design(self, core_params: Mapping, res_params: Mapping, is_bias: bool) -> None:
        self.instances['XDP'].design(**core_params)
        self.instances['XRES'].design(**res_params)
        
        bias_node = res_params['bias_node']
        if is_bias:
            self.reconnect_instance('XRES', [('rout', 'out'), ('rout_b', 'out')])
            self.reconnect_instance('XDP', [('outp', 'out'), ('outm', 'out'),
                                            ('inp', 'out'), ('inm', 'out')])
            
            self.rename_pin('outp', 'out')
            self.remove_pin('outm')
            self.remove_pin('inp')
            self.remove_pin('inm')

            if bias_node in self.instances['XRES'].master.pins:
                self.reconnect_instance_terminal('XRES', bias_node, bias_node)
                self.reconnect_instance_terminal('XRES', bias_node + '_b', bias_node)

