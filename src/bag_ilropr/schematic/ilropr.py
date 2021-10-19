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
class bag_ilropr__ilropr(Module):
    """Module for library bag_ilropr cell ilropr.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'ilropr.yaml')))

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
            num_stage='Number of stages in the oscillator',
            invt_params="Tristate inverter unit cell params",
            inv_params='Cross coupled inverter unit cell params',
            num_aa="Number of always on units",
            num_inj="Number of injection units",
            num_cc="Number of cross coupled units",
            has_dummy="True if has dummy loads to get FO2"
        )

    def design(self, num_stage: int, invt_params: Mapping, inv_params: Mapping,
               num_aa: int, num_inj: int, num_cc: int, has_dummy: bool) -> None:
        # TODO: add input buffer?
        # Design stage
        _params=dict(invt_params=invt_params, inv_params=inv_params, num_aa=num_aa,
                     num_inj=num_inj, num_cc=num_cc)
        self.instances['XSTG'].design(**_params)
        self.instances['XDUM'].design(**_params)

        # Array instances
        term_list = [[
            ('outp', f'V<{idx}>'),
            ('outm', f'V<{idx+num_stage}>'),
            # ('inm', f'V<{idx - 1 if idx > 0 else 2 * num_stage - 1}>'),
            # ('inp', f'V<{idx + num_stage - 1 if idx > 0 else num_stage - 1}>'),
            ('inp', f'V<{idx - 1 if idx > 0 else 2 * num_stage - 1}>'),
            ('inm', f'V<{idx + num_stage - 1 if idx > 0 else num_stage - 1}>'),
            (f'en<{num_inj-1}:0>', f'en<{(idx + num_stage + 1) * num_inj // 2 - 1}:{(idx+num_stage) * num_inj // 2}>,'
                                  f'en<{(idx + 1) * num_inj // 2 - 1}:{idx * num_inj // 2}>'),
            (f'enb<{num_inj-1}:0>', f'enb<{(idx + num_stage + 1) * num_inj // 2 - 1}:{(idx+num_stage) * num_inj // 2}>,'
                                  f'enb<{(idx + 1) * num_inj // 2 - 1}:{idx * num_inj // 2}>'),
        ] for idx in range(num_stage)]
        self.array_instance('XSTG', dx=300,
                            inst_term_list=[(f'XSTG{idx}', term_list[idx]) for idx in range(num_stage)])

        if has_dummy:
            term_list = [[
                ('inp', f'V<{idx}>'),
                ('inm', f'V<{idx + num_stage}>'),
                ('outp', f'VD<{idx}>'),  # Assign to some dummy name
                ('outm', f'VD<{idx + num_stage}>'),
                (f'en<{num_inj-1}:0>', f'<*{num_inj}>VSS'),
                (f'enb<{num_inj-1}:0>', f'<*{num_inj}>VDD')
            ] for idx in range(num_stage)]
            self.array_instance('XDUM', dx=300,
                                inst_term_list=[(f'XDUM{idx}', term_list[idx]) for idx in range(num_stage)])
        else:
            self.remove_instance('XDUM')
        self.remove_instance('XNC0<3:0>')
        self.remove_instance('XNC1<3:0>')

        if num_stage != 4:
            self.rename_pin('V<7:0>', f'V<{num_stage-1:0}')
        if num_stage * num_inj != 16:
            self.rename_pin('en<15:0>', f'en<{num_stage * num_inj - 1}:0>')
            self.rename_pin('enb<15:0>', f'enb<{num_stage * num_inj - 1}:0>')
