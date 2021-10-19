# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional, Mapping, Type, Tuple, List, Dict, Union, cast, Sequence

from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from bag.concurrent.util import GatherHelper
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SweepSpec, swp_spec_from_dict
from bag.simulation.measure import MeasurementManager, MeasInfo

from .tran_base import PhaseRotatorTranMM


class PhaseRotatorFreqMM(MeasurementManager):
    """
    Sweep VDD and determine frequency
    """
    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo,
                     harnesses: Optional[Sequence[DesignInstance]] = None
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise NotImplementedError

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance,
                   harnesses: Optional[Sequence[DesignInstance]] = None) -> Tuple[bool, MeasInfo]:
        raise NotImplementedError

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise NotImplementedError

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        """Skip existing methods and just do everything here"""

        specs: Mapping = self.specs
        swp_info: Mapping = specs['swp_info']
        swp_spec: SweepSpec = swp_spec_from_dict(swp_info)
        swp_values: Sequence = swp_spec.values

        base_params = deepcopy(specs)
        base_params['plot'] = False
        if isinstance(base_params, Dict):
            del base_params['swp_info']

        # TODO: update sup_values or sim_params? paradigm check?
        gatherer: GatherHelper = GatherHelper()
        for val in swp_values:
            mm_params = deepcopy(base_params)
            mm_params['tbm_specs']['sup_values'][swp_info['name']] = val
            mm = self.make_mm(PhaseRotatorTranMM, mm_params)
            sim_name = name + '_' + swp_info['name'] + str(val).replace('.', 'p')
            gatherer.append(sim_db.async_simulate_mm_obj(sim_name, sim_dir / sim_name, dut, mm, harnesses))

        sim_results: Sequence[MeasureResult] = await gatherer.gather_err()
        data_list: Sequence[Mapping[str, Any]] = [mr.data for mr in sim_results]

        avg_period = []
        out_clk_name = specs['out_clk_info'][0]['name']
        for item in data_list:
            out_clk_period_data = item['period_dict'][out_clk_name]
            avg_period.append(np.mean(out_clk_period_data))
        avg_freq = [1/x for x in avg_period]
        # check if sorted
        issorted = np.all(np.diff(avg_freq) > 0)
        if not issorted:
            print("WARNING: measured frequencies vs VDD were not monotonically increasing")

        target_period = specs['tbm_specs']['sim_params']['t_clk_per']
        target_freq = 1 / target_period
        if min(avg_freq) <= target_freq <= max(avg_freq):
            target_bias = np.interp(target_freq, avg_freq, swp_values)
        else:
            target_bias = -1

        ans = dict(
            swp_values=swp_values,
            avg_period=avg_period,
            avg_freq=avg_freq,
            target_bias=target_bias
        )

        # Collect the jitter and current metrics
        metrics = ['jee', 'jc', 'jcc', 'current']
        # Each of these maps string to float
        # We want to assemble to string to list of floats
        for metric in metrics:
            tmp = {}
            metric_slice: List[Dict[str, float]] = [item[metric] for item in data_list]
            for key in metric_slice[0].keys():
                tmp[key] = [metric_slice[idx][key] for idx in range(len(metric_slice))]
            ans[metric] = tmp

        if self.specs.get('plot'):
            plt.subplot(2, 2, 1)
            plt.plot(swp_values, avg_period, label='Measured period')
            plt.plot(swp_values, np.ones(len(swp_values)) * target_period, label='Target Period')
            plt.grid()
            plt.legend()
            plt.xlabel(swp_info['name'] + ' sweep [v]')
            plt.ylabel('average period [s]')
            plt.title('Osc Free Running period vs VDD')

            plt.subplot(2, 2, 2)
            plt.plot(swp_values, avg_freq, label='Measured frequency')
            plt.plot(swp_values, np.ones(len(swp_values)) * target_freq, label='Target frequency')
            plt.grid()
            plt.legend()
            plt.xlabel(swp_info['name'] + ' sweep [v]')
            plt.ylabel('average frequency [Hz]')
            plt.title('Osc Free Running frequency vs VDD')

            plt.subplot(2, 2, 3)
            plt.plot(swp_values, ans['jee'][out_clk_name])
            plt.grid()
            plt.legend()
            plt.xlabel(swp_info['name'] + ' sweep [v]')
            plt.ylabel('Jee [s rms]')
            plt.title('Jee vs VDD')

            plt.subplot(2, 2, 4)
            for ppin in specs['power_pins']:
                plt.plot(swp_values, ans['current'][ppin], label=ppin)
            plt.grid()
            plt.legend()
            plt.xlabel(swp_info['name'] + ' sweep [v]')
            plt.ylabel('average current [A]')
            plt.title('Current vs VDD')
            plt.show()

            breakpoint()

        return ans
