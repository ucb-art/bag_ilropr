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


class PhaseRotatorINLMM(MeasurementManager):
    """
    Sweep codes and get phase offset
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
            mm_params['code']['value'] = int(val)
            mm = self.make_mm(PhaseRotatorTranMM, mm_params)
            sim_name = name + '_' + swp_info['name'] + str(val).replace('.', 'p')
            gatherer.append(sim_db.async_simulate_mm_obj(sim_name, sim_dir / sim_name, dut, mm, harnesses))

        sim_results: Sequence[MeasureResult] = await gatherer.gather_err()
        data_list: Sequence[Mapping[str, Any]] = [mr.data for mr in sim_results]

        # Map code to average offset info
        out_clk_info = specs['out_clk_info']
        avg_offset = np.empty((len(out_clk_info), len(swp_values)))
        in_clk_name = specs['in_clk_info']['name']
        for code_idx, results in zip(range(len(swp_values)), data_list):
            cross_dict = results['cross_dict']
            in_cross = cross_dict[in_clk_name]
            for out_idx, item in enumerate(specs['out_clk_info']):
                out_clk_name = item['name']
                out_cross = cross_dict[out_clk_name]
                mask = np.where(out_cross >= in_cross[0])  # ensure valid cycles
                out_cross = out_cross[mask]
                avg_offset[out_idx, code_idx] = determine_offset(in_cross, out_cross)

        # Shift relative to lowest code
        ref_col = avg_offset[:, 0]
        ref_mat = ref_col.reshape((len(ref_col), 1)).repeat(len(swp_values), axis=1)
        offset_shifted = avg_offset - ref_mat

        # Adjust offset to be all > period
        mask = np.where(offset_shifted < 0)
        t_clk_per = specs['tbm_specs']['sim_params']['t_clk_per']
        offset_shifted[mask] += t_clk_per

        # Compute INL
        step_size = np.diff(offset_shifted, axis=1)
        ref_step = t_clk_per / specs['num_steps']
        dnl = step_size - ref_step
        inl = np.cumsum(dnl, axis=1)

        ans = dict(
            offset=avg_offset,
            offset_shifted=offset_shifted,
            dnl=dnl,
            inl=inl,
        )

        # Collect the jitter and current metrics
        # TODO: share these methods?
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
            for out_idx, item in enumerate(specs['out_clk_info']):
                out_clk_name = item['name']
                offset = offset_shifted[out_idx, :]
                plt.plot(swp_values, offset, label=out_clk_name)
            plt.grid()
            plt.legend()
            plt.xlabel('code')
            plt.ylabel('offset, shifted st code=0 has no offset [s]')
            plt.title('Offset vs Code')

            plt.subplot(2, 2, 2)
            for out_idx, item in enumerate(specs['out_clk_info']):
                out_clk_name = item['name']
                offset = inl[out_idx, :]
                plt.plot(swp_values[1:], offset, label=out_clk_name)
            plt.grid()
            plt.legend()
            plt.xlabel('code')
            plt.ylabel('INL [s]')
            plt.title('INL vs Code')

            plt.subplot(2, 2, 3)
            for out_idx, item in enumerate(specs['out_clk_info']):
                out_clk_name = item['name']
                plt.plot(swp_values, ans['jee'][out_clk_name], label=out_clk_name)
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


def determine_offset(ref: np.ndarray, meas: np.ndarray):
    # Make the longer one the ref
    if len(meas) > len(ref):
        return -determine_offset(meas, ref)

    # determine mask
    diff = len(ref) - len(meas)
    new_ref = ref
    if diff:
        # Assume that all of meas is encompassed by ref
        best_val = 9999
        best_idx = -1
        for idx in range(diff + 1):
            ref_mask = ref[idx:len(ref)-diff+idx]
            norm1 = np.mean(np.abs(ref_mask - meas))
            if norm1 < best_val:
                best_val = norm1
                best_idx = idx
        new_ref = ref[best_idx: len(ref)-diff+best_idx]
    offset = new_ref - meas
    return np.mean(offset[len(offset) // 2:])
