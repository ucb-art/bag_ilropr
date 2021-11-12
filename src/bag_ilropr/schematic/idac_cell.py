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
class bag_ilropr__idac_cell(Module):
    """Module for library bag_ilropr cell idac_cell.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'idac_cell.yaml')))

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
            seg_tail="Unit cell segments",
            cs_stack="Current source stack",
            w_tail="Unit width",
            lch="Unit channel length",
            th_tail="Unit threshold",
            is_diode="True if diode mirror, to remove switches",
            is_ao="True if always on, remove outb"
        )

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(is_diode=False, is_ao=False)

    def design(self, lch: int, w_tail: int, th_tail: str, seg_tail: int, cs_stack: int,
               is_diode: bool, is_ao: bool) -> None:

        # Design current source
        self.design_transistor('XCS', w_tail, lch, seg_tail, th_tail)
        top_name = 'XCS'
        if cs_stack > 1:
            inst_names, inst_term = [], []
            for idx in range(cs_stack):
                inst_names.append(f'XCS{idx}')
                _terms = [('D', 'tail' if idx == 0 else f'm<{idx-1}>'),
                          ('S', 'VSS' if idx == cs_stack-1 else f'm<{idx}>')]
                inst_term.append(_terms)
            itl = list(zip(inst_names, inst_term))
            self.array_instance('XCS', inst_term_list=itl, dy=-200)
            top_name = 'XCS0'
        
        # Design switches
        if is_diode: 
            self.remove_instance('XSW')
            self.remove_instance('XSWB')
            self.remove_pin('en')
            self.remove_pin('enb')
            self.remove_pin('iout')
            self.remove_pin('ioutb')
            self.reconnect_instance_terminal(top_name, 'D', 'NBIAS')
        elif is_ao:
            self.design_transistor('XSW', w_tail, lch, seg_tail, th_tail)
            self.remove_instance('XSWB')
            self.remove_pin('enb')
            self.remove_pin('ioutb')
        else:
            self.design_transistor('XSW', w_tail, lch, seg_tail, th_tail)
            self.design_transistor('XSWB', w_tail, lch, seg_tail, th_tail)
