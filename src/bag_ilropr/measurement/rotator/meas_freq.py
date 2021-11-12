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


class PhaseRotatorFreqMM(MeasurementManager):
    """A MM characterizating oscillator frequency against some bias sweep, either supply or bias.
    Characterizes the following
    - Swing
    - Period and frequency
    - Target bias for the target period
    - duty cycle (dc)
    Plotting features include summary plots and waveforms. Main interest in the frequency plots.
    
    Each simulation call in the sweep uses a PhaseRotatorTranMM. A class should be selected
        as part of the initialization. Use TopTranMM or OscCoreTranMM.

    Subclasses can inherit this class and add plotting, using the `plot` function.

    This MM supports multiple corners through GatherHelper

    Parameters (self.specs): All of the parameters for a PhaseRotatorTranMM are required.
        These are additional parameters:
    - `swp_info`: Mapping
        Sweep info. Gets turned into a `SweepSpec`. `name`, `type`, `start`, `stop`, `num`
    - `plot_idx`: Optional[int]
        Index of sweep to plot. Optional. By default, plot the closest to target frequency
    
    Return:
        If one corner is provided, the return is a dictionary of the above characteristics.
        If multiple corners are provided, the return is a dictionary mapping corner to 
            subdictionaries with the above characteristics

    Note, `get_sim_info`, `initialize`, and `process_output` were deprecated in 2023.

    """

    def __init__(self, meas_specs: Mapping[str, Any], log_file: str, log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        super().__init__(meas_specs, log_file, log_level, precision)
        self.mm_class: PhaseRotatorTranMM = None

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

        base_params = deepcopy(specs)
        base_params['plot'] = False
        if isinstance(base_params, Dict):
            del base_params['swp_info']
            if base_params.get('plot_idx'):
                del base_params['plot_idx']

        gatherer: GatherHelper = GatherHelper()
        for val in swp_values:
            mm_params = deepcopy(base_params)
            mm_params['sim_envs'] = [pvt]
            mm_params['tbm_specs']['sup_values'][swp_info['name']] = val
            mm = self.make_mm(self.mm_class, mm_params)
            sim_name = name + '_' + swp_info['name'] + str(val).replace('.', 'p')
            gatherer.append(sim_db.async_simulate_mm_obj(sim_name, sim_dir / sim_name, dut, mm, harnesses))

        sim_results: Sequence[MeasureResult] = await gatherer.gather_err()
        data_list: Sequence[Mapping[str, Any]] = [mr.data for mr in sim_results]

        # Get a calculator for later
        calc = mm.tbm.get_calculator(sim_results[0].data['data'])
        self.calc = calc

        avg_period = []  # Indexed by swp_values
        out_clk_name = specs['out_clk_info'][0]['name']
        for item in data_list:
            out_clk_period_data = item['period_dict'][out_clk_name]
            avg_period.append(np.mean(out_clk_period_data))
        avg_freq = [1/x for x in avg_period]
        # check if sorted
        issorted = np.all(np.diff(avg_freq) > 0)
        if not issorted:
            print("WARNING: measured frequencies vs bias were not monotonically increasing")

        swing_dict = {}
        for out_idx, item in enumerate(specs['out_clk_info']):
            out_clk_name = item['name']
            wave_amp_list = []
            for code_idx, results in enumerate(data_list):
                data = results['data']
                period = avg_period[code_idx]
                if np.isnan(period):
                    start_idx = len(data['time']) // 10 * 9
                else:
                    start_idx = np.where(data['time'] > data['time'][-1] - 10 * period)[0][0]
                wave = results['data'][out_clk_name][0][start_idx:]
                wave_min, wave_max = np.min(wave), np.max(wave)
                wave_amp = wave_max - wave_min
                wave_amp_list.append(wave_amp)
            swing_dict[out_clk_name] = wave_amp_list

        target_period = specs['tbm_specs']['sim_params']['t_clk_per']
        if isinstance(target_period, str):
            target_period = calc.eval(target_period)
        target_freq = 1 / target_period
        # Swing check
        # Select only the codes that are swinging
        vmin = 0.1
        out_clk_name = specs['out_clk_info'][0]['name']
        wave_amp_list = swing_dict[out_clk_name]
        mask = np.where(np.array(wave_amp_list) > vmin)[0]
        avg_freq_mask = np.array(avg_freq)[mask]
        if mask.any() and min(avg_freq_mask) <= target_freq <= max(avg_freq_mask):
            target_bias = np.interp(target_freq, avg_freq, swp_values)
        else:
            target_bias = -1

        ans = dict(
            swp_values=swp_values,
            swing_dict=swing_dict,
            avg_period=avg_period,
            avg_freq=avg_freq,
            target_bias=target_bias,
            dc=[item['dc_dict'] for item in data_list]
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

    def plot(self, ans: Mapping, sim_dir: Path, data_list: Sequence[Mapping[str, Any]]):
        specs = self.specs
        swp_info: Mapping = specs['swp_info'] 
        target_period = specs['tbm_specs']['sim_params']['t_clk_per']
        if isinstance(target_period, str):
            target_period = self.calc.eval(target_period)
        target_freq = 1 / target_period

        swp_values = ans['swp_values']
        avg_period = ans['avg_period']
        avg_freq = ans['avg_freq']
    
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(swp_values, avg_period, 'o-', label='Measured period')
        plt.plot(swp_values, np.ones(len(swp_values)) * target_period, label='Target Period')
        plt.grid()
        plt.legend()
        plt.xlabel(swp_info['name'] + ' sweep')
        plt.ylabel('average period [s]')
        plt.title('Osc Free Running period vs sweep')

        plt.subplot(2, 2, 2)
        plt.plot(swp_values, avg_freq, 'o-', label='Measured frequency')
        plt.plot(swp_values, np.ones(len(swp_values)) * target_freq, label='Target frequency')
        plt.grid()
        plt.legend()
        plt.xlabel(swp_info['name'] + ' sweep')
        plt.ylabel('average frequency [Hz]')
        plt.title('Osc Free Running frequency vs sweep')

        ax = plt.subplot(2, 2, 3)
        period = target_period
        # Assume the last 10 periods are stable
        
        for out_idx, item in enumerate(specs['out_clk_info']):
            out_clk_name = item['name']
            if 'out' in out_clk_name:
                # ax = ax.twinx()
                # You do see the same shape, but it goes from ~15 mV difference to < 3 mV difference
                continue
            wave_amp_list = []
            for code_idx, results in enumerate(data_list):
                data = results['data']
                start_idx = np.where(data['time'] > data['time'][-1] - 10 * period)[0][0]
                wave = results['data'][out_clk_name][0][start_idx:]
                wave_min, wave_max = np.min(wave), np.max(wave)
                wave_amp = wave_max - wave_min
                wave_amp_list.append(wave_amp)
            ax.plot(swp_values, wave_amp_list, label=out_clk_name)
        # Also plot the bias point
        if 'NBIAS' in results['data'].signals:
            bias_avg_list = []
            for code_idx, results in enumerate(data_list):
                data = results['data']
                start_idx = np.where(data['time'] > data['time'][-1] - 10 * period)[0][0]
                wave = results['data']['NBIAS'][0][start_idx:]
                bias_avg_list.append(np.mean(wave))
            ax.plot(swp_values, bias_avg_list, label='NBIAS')
        
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
        plt.xlabel(swp_info['name'] + ' sweep')
        plt.ylabel('average current [mA]')
        plt.title('Current vs sweep')
        # plt.show()
        plt.savefig(sim_dir / "freq_metrics.png")
        plt.close()

        # Grab the last 3 periods and plot them
        per_diff = [abs(x - target_period) for x in avg_period]
        idx_min = np.argmin(per_diff)  # only works for monotonic. Careful
        if 'plot_idx' in self.specs:
            idx_min = self.specs['plot_idx']
        period = avg_period[idx_min]
        data = data_list[idx_min]['data']
        try:
            start_idx = np.where(data['time'] > data['time'][-1] - 3 * period)[0][0]
        except:
            print("No period found, using all data")
            start_idx = 0

        plt.figure()
        plot_list = specs.get('plot_list', [])
        if not plot_list:
            plot_list.append(specs['in_clk_info']['name'])
            plot_list.append(specs['in_clk_info']['diff'])
            for out_clk_name in specs['out_clk_info']:
                plot_list.append(out_clk_name['name'])
                plot_list.append(out_clk_name['diff'])
        for name in plot_list:
            plt.plot(data['time'][start_idx:], data[name][0][start_idx:], label=name)
        plt.title(f"Waveforms for {swp_info['name']} = {swp_values[idx_min]}")
        plt.grid()
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('signal [v]')
        # plt.show()
        plt.savefig(sim_dir / "freq_waveforms.png")
        plt.close()
