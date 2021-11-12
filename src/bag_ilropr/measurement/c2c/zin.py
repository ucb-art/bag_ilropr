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

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, List, Dict, cast

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.data import SimData
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult

from bag3_testbenches.measurement.ac.base import ACTB

"""Measure the input impedance"""

class C2C_ZinMM(MeasurementManager):

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        # This is a single code measurement, so no sweeping
        # TODO: add multi-corner support

        in_clk_info: Mapping = self.specs['in_clk_info']
        out_clk_info: List[Mapping] = self.specs['out_clk_info']
        power_pins: List[str] = self.specs['power_pins']
        iref_dict: Mapping = self.specs.get('iref_dict', {})
        tbm_specs: Mapping = self.specs['tbm_specs']
        
        src_list, save_outputs, load_list = self.get_in_outs(in_clk_info, out_clk_info, power_pins, iref_dict)
        if 'save_outputs' in tbm_specs:
            tbm_specs['save_outputs'].extend(save_outputs)
        else:
            tbm_specs['save_outputs'] = save_outputs
        
        tbm_specs = dict(
            **self.specs['tbm_specs'],
            src_list=src_list,
            load_list=load_list
        )
        tbm = cast(ACTB, self.make_tbm(ACTB, tbm_specs))
        self.tbm = tbm
        tbm_name = name
        tb_params = dict()
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                          tb_params=tb_params)

        # Collect data and report results
        data: SimData = sim_results.data

        inp = data['inp']
        inm = data['inm']
        in_ac = inp-inm

        zin_ac = in_ac[0] / 1e-3  # AC current input.
        freq = data['freq']

        zin_ac_16G = np.interp(16e9, freq, zin_ac)
        
        if self.specs.get('plot'):
            plt.figure()
            plt.loglog(freq, np.abs(zin_ac))
            plt.legend()
            plt.grid()
            plt.savefig(sim_dir / 'zin.png')

        ans = dict(
            data=data,
            zin_ac_16G=np.abs(zin_ac_16G),
        )
        return ans

    @classmethod
    def get_in_outs(cls, in_clk_info, out_clk_info, power_pins, iref_dict: Dict) -> Tuple[List, List, List]:

        in_name = in_clk_info['name']
        in_diff = in_clk_info.get('diff', '')
        save_outputs = [in_name]
        if in_diff:
            save_outputs.append(in_diff)

        conns = {'PLUS': in_name, 'MINUS': in_diff} if in_diff else {'PLUS': in_name, 'MINUS': 'VSS'}
        src_list = [
                dict(type='isin', value=dict(mag=1e-3, ampl=1e-3, sinedc=0, freq='f_osc'),conns=conns)
            ]
        
        for k, v in iref_dict.items():
            src_list.append(
                dict(type='idc', value=dict(idc=v), conns={'PLUS': 'VDD', 'MINUS': k})
            )

        load_list = []
        for item in out_clk_info:
            save_outputs.append(item['name'])
            load_list.append(
                dict(type='cap', value='c_load', conns={'PLUS': item['name'], 'MINUS': 'VSS'})
            )
            if item.get('diff'):
                save_outputs.append(item['diff'])
                load_list.append(
                    dict(type='cap', value='c_load', conns={'PLUS': item['diff'], 'MINUS': 'VSS'})
                )

        for ppin in power_pins:
            save_outputs.append('XDUT:' + ppin)

        return src_list, save_outputs, load_list
