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
class bag_ilropr__idac_unit(Module):

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'idac_unit.yaml')))

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
            num_en="Number of enable unit columns",
            num_aa="Number of always on unit columns",
            num_diode="Number of diode unit columns",
            num_dum="Number of dummies units, across all columns",
            seg_tail="Unit cell segments",
            cs_stack="Current source stack",
            w_tail="Unit width",
            lch="Unit channel length",
            th_tail="Unit threshold"
        )

    def design(self, lch: int, w_tail: int, th_tail: str, seg_tail: int, cs_stack: int,
               num_en: int, num_aa: int, num_diode: int, num_dum: int) -> None:
        
        num_inj = num_en // 2

        # Design switched sources
        if num_en:
            for idx, name in enumerate(['XCS', 'XCSE']):
                self.instances[name].design(lch=lch, w_tail=w_tail, th_tail=th_tail, 
                                            seg_tail=seg_tail, cs_stack=cs_stack)
                suf = f'<{(idx+1) * num_inj-1}:{idx * num_inj}>'
                self.rename_instance(name, f'XCS{suf}')
                self.reconnect_instance(f'XCS{suf}', [('en', f'en{suf}'), ('enb', f'enb{suf}')])
            if num_en > 1:
                self.rename_pin('en', f'en<{num_en-1}:0>')
                self.rename_pin('enb', f'enb<{num_en-1}:0>')
        else:
            self.remove_instance('XCS')
            self.remove_instance('XCSE')

        if not num_en:
            for not_tail in ['tail_injb', 'tail_inj', 'taile_injb', 'taile_inj']:
                self.remove_pin(not_tail)

        # Design always on
        if num_aa:
            self.instances['XAA'].design(lch=lch, w_tail=w_tail, th_tail=th_tail,
                                        seg_tail=seg_tail, cs_stack=cs_stack, is_ao=True)
            if num_aa > 1:
                self.rename_instance('XAA', f'XAA<{num_aa-1}:0>')
        else:
            self.remove_instance('XAA')
    
        # Design diode
        self.instances['XDIODE'].design(lch=lch, w_tail=w_tail, th_tail=th_tail, 
                                        seg_tail=seg_tail, cs_stack=cs_stack, is_diode=True)
        self.rename_instance('XDIODE', f'XDIODE<{num_diode-1}:0>')

        # Design dummies
        if num_dum:
            self.design_transistor('XDUM', w_tail, lch, seg_tail, th_tail)
            if num_dum > 1:
                self.rename_instance('XDUM', f'XDUM<{num_dum-1}:0>')
        else:
            self.remove_instance('XDUM')

