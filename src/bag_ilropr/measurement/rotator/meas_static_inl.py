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

from typing import Any, Optional, Mapping, List, Dict, Sequence

from pprint import pprint
from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from pybag.enum import LogLevel

from bag.concurrent.util import GatherHelper
from bag.simulation.cache import SimulationDB, DesignInstance, MeasureResult
from bag.simulation.data import SweepSpec, swp_spec_from_dict
from bag.simulation.measure import MeasurementManager

from .tran import PhaseRotatorTranMM


class PhaseRotatorINLMM(MeasurementManager):
    """A MM characterizating phase rotator INL against code.
    Characterizes the following
    - INL, DNL
    - Phase offset per code
    Plotting features include summary plots and waveforms. Main interest in the INL plots.
    
    Each simulation call in the sweep uses a PhaseRotatorTranMM. A class should be selected
        as part of the initialization. Use TopTranMM or OscCoreTranMM.

    Subclasses can inherit this class and add plotting, using the `plot` function.

    This MM supports multiple corners through GatherHelper

    Parameters (self.specs): All of the parameters for a PhaseRotatorTranMM are required.
        These are additional parameters:
    - `swp_info`: Mapping
        Sweep info. Gets turned into a `SweepSpec`. `name`, `type`, `start`, `stop`, `num`
    - `num_steps`: int
        Number of steps. May differ from `swp_info` number of steps
    - `plot_idx`: Optional[int]
        Index of sweep to plot. By default, plot the closest to target frequency
    - `plot_list`: Optional[str]
        List of waveform signals to plot
    
    Return:
        If one corner is provided, the return is a dictionary of the above characteristics.
        If multiple corners are provided, the return is a dictionary mapping corner to 
            subdictionaries with the above characteristics

    Note, `get_sim_info`, `initialize`, and `process_output` were deprecated in 2023.

    """

    def __init__(self, meas_specs: Mapping[str, Any], log_file: str, log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        super().__init__(meas_specs, log_file, log_level, precision)
        self.mm_class: PhaseRotatorTranMM = None
        self.swp_values = None

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None
                                        ) -> Mapping[str, Any]:
        """Execute each corner as it's own sub-call."""
        helper = GatherHelper()
        sim_envs: List[str] = self.specs.get('sim_envs', [])
        if not sim_envs:
            sim_envs = self.specs['tbm_specs'].get('sim_envs', [])
            if not sim_envs:
                raise ValueError("Need sim envs")
            
        for sim_env in sim_envs:
            helper.append(self.async_meas_pvt(name, sim_dir / sim_env, sim_db, dut, 
                                              harnesses, sim_env))

        meas_results = await helper.gather_err()
        results = {}
        for idx, sim_env in enumerate(sim_envs):
            results[sim_env] = meas_results[idx]
        
        # If only one corner exists, return just that results
        if len(sim_envs) == 1:
            return results[sim_env]
        return results

    async def async_meas_pvt(self, name: str, sim_dir: Path, sim_db: SimulationDB, 
                             dut: Optional[DesignInstance],
                             harnesses: Optional[Sequence[DesignInstance]], pvt: str
                             ) -> Mapping[str, Any]:
        specs: Mapping = self.specs
        swp_info: Mapping = specs['swp_info']
        swp_spec: SweepSpec = swp_spec_from_dict(swp_info)
        swp_values: Sequence = swp_spec.values
        self.swp_values = swp_values

        base_params = deepcopy(specs)
        base_params['plot'] = False
        if isinstance(base_params, Dict):
            del base_params['swp_info']

        gatherer: GatherHelper = GatherHelper()
        for val in swp_values:
            mm_params = deepcopy(base_params)
            mm_params['sim_envs'] = [pvt]
            mm_params['code']['value'] = int(val)
            mm = self.make_mm(self.mm_class, mm_params)
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

        # Adjust for wrapping
        t_clk_per = specs['tbm_specs']['sim_params']['t_clk_per']
        if isinstance(t_clk_per, str):
            t_clk_per = eval(t_clk_per)
        # Mask: offset is negative and the adjacent two are not both positive
        # This handles the case where, especially in the early codes, you might have very negative INL
        mask = np.where(
            np.logical_and(offset_shifted < 0, np.logical_not(np.logical_and(
                np.roll(offset_shifted, 1) > 0,
                np.roll(offset_shifted, -1) > 0
            ))))
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
        metrics = ['jee', 'jc', 'jcc', 'current']
        # Each of these maps string to float
        # We want to assemble to string to list of floats
        for metric in metrics:
            tmp = {}
            metric_slice: List[Dict[str, float]] = [item[metric] for item in data_list]
            for key in metric_slice[0].keys():
                tmp[key] = [metric_slice[idx][key] for idx in range(len(metric_slice))]
            ans[metric] = tmp

        with open(sim_dir / "ans.txt", 'w') as f:
            pprint(ans, stream=f)

        if self.specs.get('plot'):
            self.plot(ans, sim_dir, data_list)
        return ans

    def plot(self, ans: Mapping, sim_dir: Path, data_list: Sequence):
        # TODO: better selection method for what to plot
        specs = self.specs
        out_clk_info = specs['out_clk_info']
        t_clk_per = specs['tbm_specs']['sim_params']['t_clk_per']
        if isinstance(t_clk_per, str):
            t_clk_per = eval(t_clk_per)

        swp_values = self.swp_values
        avg_offset = ans['offset']
        offset_shifted = ans['offset_shifted']
        inl = ans['inl']
        ref_step = t_clk_per / specs['num_steps']
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        for out_idx, item in enumerate(specs['out_clk_info']):
            out_clk_name = item['name']
            if 'inj' in out_clk_name:
                # Ignore for plots
                continue
            offset = offset_shifted[out_idx, :]
            plt.plot(swp_values, offset * 1e12, label=out_clk_name)
        ref_steps = np.arange(0, ref_step * len(swp_values), ref_step)
        plt.plot(swp_values, ref_steps * 1e12, label='reference')
        plt.grid()
        plt.legend()
        plt.xlabel('code')
        plt.ylabel('Rotation, shifted st code=0 has no offset [ps]')
        plt.title('Rotation vs Code')

        plt.subplot(2, 2, 2)
        plot_list = specs.get('plot_list', [])
        for out_idx, item in enumerate(specs['out_clk_info']):
            out_clk_name = item['name']
            if 'inj' in out_clk_name:
                # Ignore for plots
                continue
            if plot_list and out_clk_name not in plot_list:
                continue
            offset = inl[out_idx, :] * 1e12
            plt.plot(swp_values[1:], offset, 'o-', label=out_clk_name)
        plt.grid()
        plt.legend()
        plt.xlabel('code')
        plt.ylabel('INL [ps]')
        plt.title('INL vs Code')

        ax = plt.subplot(2, 2, 3)
        period = t_clk_per
        for out_idx, item in enumerate(specs['out_clk_info']):
            out_clk_name = item['name']
            if 'out' in out_clk_name:
                # You do see the same shape, but it goes from ~15 mV difference to < 3 mV difference
                continue
            wave_amp_list = []
            for code_idx, results in enumerate(data_list):
                wave = results['data'][out_clk_name][0]
                # Assumption: its settled by the 2nd half
                # TODO: settle check
                start_idx = len(wave) // 2
                wave = wave[start_idx:]
                wave_min, wave_max = np.min(wave), np.max(wave)
                wave_amp = wave_max - wave_min
                wave_amp_list.append(wave_amp)
            ax.plot(swp_values, wave_amp_list, label=out_clk_name)
        plt.grid()
        plt.legend(loc='lower right')
        plt.xlabel('code')
        plt.ylabel('peak to peak amp [V]')
        plt.title('Amplitude vs Code')

        plt.subplot(2, 2, 4)
        for ppin in specs['power_pins']:
            plt.plot(swp_values, [x * 1000 for x in ans['current'][ppin]], label=ppin)
        plt.grid()
        plt.legend()
        plt.xlabel('code')
        plt.ylabel('average current [mA]')
        plt.title('Current vs code')
        plt.tight_layout()
        plt.savefig(sim_dir / "metrics.png")
        plt.close()

        num_codes = len(swp_values)
        num_plot_page = 4
        num_pages = -(-num_codes // num_plot_page)
        print("Plotting waveforms...")
        for page_idx in range(num_pages):
            start_idx = page_idx * num_plot_page
            end_idx = min(start_idx + num_plot_page, num_codes)
            plt.figure(figsize=(10, 8))
            for code_idx in range(start_idx, end_idx):
                results = data_list[code_idx]
                plt.subplot(2, 2, (code_idx % num_plot_page) +1)
                data = results['data']
                start_idx = np.where(data['time'] > data['time'][-1] - 3 * period)[0][0]
                for name in specs['plot_list']:
                    plt.plot(data['time'][start_idx:], data[name][0][start_idx:], label=name)
                plt.title(f'code: {code_idx}')
                plt.grid()
                plt.legend()
                plt.xlabel('time')
                plt.ylabel('signal [v]')
            plt.tight_layout()
            plt.savefig(sim_dir / f"waveforms{page_idx}.png")
            plt.close()  # Save memory

        # TODO: write a "plot list" for this
        # This code is rigid for the 2 circuits of interest in this project
        is_osc_core = True
        for item in out_clk_info:
            if 'mid' in item['name'] or 'out' in item['name']:
                is_osc_core = False
        if is_osc_core:
            pairs = [('V<1>', 'V<0>'), ('V<2>', 'V<1>'), ('V<3>', 'V<2>'), ('V<0>', 'V<3>')]
        else:
            pairs = [('mid<1>', 'mid<0>'), ('mid<2>', 'mid<1>'), ('mid<3>', 'mid<2>'), ('mid<0>', 'mid<3>')]

        fig, ax = plt.subplots(figsize=(10,8))
        # plt.subplot(121)
        bb = []
        for p, m in pairs:
            pidx = np.where(np.array([x['name'] for x in specs['out_clk_info']]) == p)[0]
            midx = np.where(np.array([x['name'] for x in specs['out_clk_info']]) == m)[0]
            off = avg_offset[pidx, :] - avg_offset[midx, :]
            off = off.flatten()
            off += t_clk_per / 8

            mask = np.where(off > t_clk_per/2)[0]
            off[mask] -= t_clk_per
            
            mask = np.where(off > t_clk_per/4)[0]
            if np.any(mask):
                off[mask] -= t_clk_per
                off += t_clk_per / 2

            plt.plot(swp_values, off * 1e12, label=p+m)
            print(off)
            bb.append(off)
        secax_y = ax.secondary_yaxis(
            'right',functions=(lambda x: x * 1e-12/t_clk_per * 360, lambda x: x * t_clk_per / 1e-12 / 360))
        secax_y.set_ylabel('Error [% degrees]')

        plt.title("Phase difference between adjacent phase pairs")
        plt.xlabel('code')
        plt.ylabel('Phase Error [ps]')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(sim_dir / "phase differences.png")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10,8))
        for out_idx, item in enumerate(specs['out_clk_info']):
            out_clk_name = item['name']
            if 'inj' in out_clk_name or 'out' in out_clk_name:
                # Ignore for plots
                continue
            offset = inl[out_idx, :] * 1e12
            plt.plot(swp_values[1:], offset, 'o-', label=out_clk_name)
        secax_y = ax.secondary_yaxis(
            'right',functions=(lambda x: x * 1e-12/t_clk_per * 360, lambda x: x * t_clk_per / 1e-12 / 360))
        secax_y.set_ylabel('Error [% degrees]')
        plt.grid()
        plt.legend()
        plt.xlabel('code')
        plt.ylabel('INL [ps]')
        plt.title('INL vs Code')

        plt.tight_layout()
        plt.savefig(sim_dir / "INL.png")
        plt.close()


def determine_offset(ref: np.ndarray, meas: np.ndarray):
    """Determine phase offset between ref and meas"""
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
    # Improve mean quality by selecting last few
    return np.mean(offset[-10:])
