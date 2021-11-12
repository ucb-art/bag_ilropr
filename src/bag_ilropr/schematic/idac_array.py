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

from .idac_unit import bag_ilropr__idac_unit

# noinspection PyPep8Naming
class bag_ilropr__idac_array(Module):
    """Module for library bag_ilropr cell idac_array.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'idac_array.yaml')))

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
            num_stages="Number of units to include in the array",
            **bag_ilropr__idac_unit.get_params_info()
        )

    def design(self, num_stages: int, num_en: int, **kwargs) -> None:
        
        num_inj = num_en // 2

        self.instances['XUNIT'].design(num_en=num_en, **kwargs)
        
        if num_en:
            tails = ['tail_aa', 'tail_injb', 'tail_inj', 'taile_injb', 'taile_inj']
        else:
            tails = ['tail_aa']
            for not_tail in ['tail_injb', 'tail_inj', 'taile_injb', 'taile_inj']:
                self.remove_pin(not_tail)

        inst_names = [f'XUNIT{idx}' for idx in range(num_stages)]
        if num_en:
            term_list = [[
                    (f'en<{num_en - 1}:0>',
                    f'en<{(idx + num_stages + 1) * num_en // 2 - 1}:{(idx + num_stages) * num_en // 2}>,'
                    f'en<{(idx + 1) * num_en // 2 - 1}:{idx * num_en // 2}>'),
                    (f'enb<{num_en - 1}:0>',
                    f'enb<{(idx + num_stages + 1) * num_en // 2 - 1}:{(idx + num_stages) * num_en // 2}>,'
                    f'enb<{(idx + 1) * num_en // 2 - 1}:{idx * num_en // 2}>'),
                ] for idx in range(num_stages)]
        else:
            term_list = [[] for _ in range(num_stages)]
        
        for idx, rec_list in enumerate(term_list):
            for tail in tails:
                rec_list.append((tail, f'{tail}<{idx}>'))
        inst_term_list = list(zip(inst_names, term_list))
        self.array_instance('XUNIT', inst_term_list=inst_term_list)

        if num_en:
            self.rename_pin('en', f'en<{num_stages * num_en - 1}:0>')
            self.rename_pin('enb', f'enb<{num_stages * num_en - 1}:0>')
        else:
            self.remove_pin('en')
            self.remove_pin('enb')

        for tail in tails:
            self.rename_pin(tail, f'{tail}<{num_stages - 1}:0>')
