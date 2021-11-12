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

from .cml2cmos_ac import bag_ilropr__cml2cmos_ac

# noinspection PyPep8Naming
class bag_ilropr__c2c_array(Module):
    """Module for library bag_ilropr cell c2c_array.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'c2c_array.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            num_units="Number of differential C2C units",
            **bag_ilropr__cml2cmos_ac.get_params_info()
        )

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return bag_ilropr__cml2cmos_ac.get_default_param_values()

    def design(self, num_units: int, **kwargs) -> None:
        self.instances['XUNIT'].design(**kwargs)

        if num_units > 1:
            terms = ['inp', 'inm', 'outp', 'outm']
            if kwargs['export_mid']:
                terms += ['midp', 'midm']
            inst_term_list = [(f'XUNIT{idx}', [(term, f'{term}<{idx}>') for term in terms])
                              for idx in range(num_units)]
            self.array_instance('XUNIT', inst_term_list=inst_term_list)
            for term in terms:
                self.rename_pin(term, f'{term}<{num_units - 1}:0>')

        if not kwargs['export_mid']:
            self.remove_pin('midp')
            self.remove_pin('midm')