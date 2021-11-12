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

from typing import Mapping, Any, Dict

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag_ilropr__cml2cmos_ac(Module):
    """Module for library bag_ilropr cell cml2cmos_ac.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'cml2cmos_ac.yaml')))

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
            inv_params="Parameters for inverters",
            res_params="Parameters for resistors",
            cap_params="Parameters for caps",
            has_invout="True to have a 2nd inverter for full swing",
            invout_params="Optional params for 2nd inverter. If not given, use inv_params",
            use_analoglib="Legacy: True to use analogLib caps and res",
            export_mid="True to export midp/m",
        )
    
    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(use_analoglib=False, has_invout=True, invout_params={}, cap_params={}, export_mid=False)

    def design(self, inv_params: Dict, res_params: Dict, cap_params: Dict,
               use_analoglib: bool, has_invout: bool, invout_params: Dict,
               export_mid: bool) -> None:
        
        if use_analoglib:
            print("Deprecate use_analogLib. See class for details")

        # Design and add the 2nd inverter
        if has_invout:
            invout_params = invout_params if invout_params else inv_params
            self.array_instance('XINV', \
                inst_term_list=[('XINVIN', [('out', 'mid2')]),('XINVOUT', [('in', 'mid2')])])
            self.instances['XINVIN'].design(**inv_params)
            self.instances['XINVOUT'].design(**invout_params)
            mid2_name = 'mid2'
        else:
            self.instances['XINV'].design(**inv_params)
            mid2_name = 'out'

        """
        Shunt resistor
        - If use_analogLib exists anywhere, design the analogLib resistor
        - If res_params.res_type == mos, design the stacked mos_res
        - else, should use the PDK resistor. Not recommended due to area
        """
        if use_analoglib or res_params.get('use_analoglib', False):
            self.remove_instance('XRES')
            self.remove_instance('XRESM')
            res_val = res_params['value']
            self.design_sources_and_loads(
                params_list=[
                    dict(type='res', value=res_val, conns=dict(PLUS='mid', MINUS=mid2_name)),
                ],
                default_name='XRESX'
            )
        elif res_params.get('res_type') == 'mos':
            self.remove_instance('XRES')
            self.remove_instance('XRESX')
            self.instances['XRESM'].design(**res_params['unit_params'])
            self.reconnect_instance_terminal('XRESM', 'b', mid2_name)
            if res_params.get('nser', 1) > 1:
                nser = res_params.get('nser')
                _suf = f'<{nser-2}:0>' if nser > 2 else ''
                self.rename_instance('XRESM', f'XRESM<{nser-1}:0>',
                                     [(f'a', f'mid,m{_suf}'),
                                      (f'b', f'm{_suf},{mid2_name}')])
        else:
            self.remove_instance('XRESX')
            self.remove_instance('XRESM')
            # TODO
            raise ValueError("buffer_ac: Real resistors not yet supported")

        """
        Series cap
        - If use_analogLib or we have a schematic value, keep the cap, else remove.
        - Separately, if we have parameters for metal res for caps, design the cap
        """
        if use_analoglib or 'value' in cap_params:
            # Keep this around for sch simulations
            cap_val = cap_params['value']
            self.design_sources_and_loads(
                params_list=[
                    dict(type='cap', value=cap_val, conns=dict(PLUS='in', MINUS='mid')),
                ],
                default_name='XCAC'
            )
        else:
            self.remove_instance('XCAC')

        if 'res_p' in cap_params:
            _cap_params = cap_params.copy(remove=['value'])
            self.instances['XCMOM'].design(**_cap_params)
        else:
            self.remove_instance('XCMOM')
        
        if not export_mid:
            self.remove_pin('mid')
