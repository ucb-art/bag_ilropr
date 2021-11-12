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

from bag3_testbenches.measurement.dc.base import DCTB

from ..util import code_to_pin_list

class IDAC_DCMM(MeasurementManager):

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        # This is a single code measurement, so no sweeping

        in_pin: str = self.specs['in_pin']
        out_pins: List[str] = self.specs['out_pins']
        code: Mapping = self.specs.get('code', {})
        code_fmt: Mapping = self.specs.get('code_fmt', {})
        manual_code: Mapping = self.specs.get('manual_code', {})

        if not code and not manual_code:
            raise ValueError("Need to add some codes")

        if code:
            pin_values: Mapping[str, int] = code_to_pin_list(code, code_fmt)
        else:
            pin_values: Mapping[str, int] = self.manual_code_to_pin_list(manual_code)
        tbm_specs: Mapping = self.specs['tbm_specs']
        
        src_list, save_outputs, load_list = self.get_in_outs(in_pin, out_pins)
        if 'save_outputs' in tbm_specs:
            tbm_specs['save_outputs'].extend(save_outputs)
        else:
            tbm_specs['save_outputs'] = save_outputs
        if 'pin_values' in tbm_specs:
            tbm_specs['pin_values'].update(pin_values)
        else:
            tbm_specs['pin_values'] = pin_values

        tbm_specs = dict(
            **self.specs['tbm_specs'],
            src_list=src_list,
            load_list=load_list,
            dut_pins=dut.pin_names
        )
        tbm = cast(DCTB, self.make_tbm(DCTB, tbm_specs))
        self.tbm = tbm
        tbm_name = name
        tb_params = dict()
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                          tb_params=tb_params)

        # Collect data and report results
        data: SimData = sim_results.data

        # Get bias point
        ans = dict(data=data)
        ans[in_pin] = data[in_pin]
        
        # Compute currents
        rload = tbm_specs['sim_params']['rload']
        v_VDD = tbm_specs['sup_values']['VDD']

        for item in out_pins:
            ans[item] = (v_VDD - data[item]) / rload

        return ans

    @classmethod
    def get_in_outs(cls, in_pin: str, out_pins: List) -> Tuple[List, List, List]:
        save_outputs = [in_pin]
        src_list = [dict(type='idc', value=dict(idc='iref'), conns={'PLUS': 'VDD', 'MINUS': in_pin})]
        
        load_list = []
        for out_pin in out_pins:
            save_outputs.append(out_pin)
            load_list.append(dict(type='res', value='rload', conns={'PLUS': 'VDD', 'MINUS': out_pin}))

        return src_list, save_outputs, load_list

    def manual_code_to_pin_list(self, code: Mapping) -> Mapping[str, Union[int, str]]:
        ans = {}
        for pin_name, pin_value in code.items():
            pin_base = pin_name.split('<')[0]
            for idx in range(len(pin_value)):
                # Big endian
                ans[f'{pin_base}<{idx}>'] = int(pin_value[idx])
        return ans