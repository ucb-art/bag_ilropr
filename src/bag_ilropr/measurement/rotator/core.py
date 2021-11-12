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

"""Oscillator core Mixin, Tran and PSS testbenches"""

from typing import Any, Mapping

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pybag.enum import LogLevel

from .tran import PhaseRotatorTranMM
from .pss import PhaseRotatorPSSMM
from .meas_freq import PhaseRotatorFreqMM
from .meas_static_inl import PhaseRotatorINLMM

class OscCoreMixin:
    def setup_code(self, tbm_specs: Mapping) -> Mapping[str, Any]:
        """Set up tail currents based on the code"""
        code: Mapping = self.specs['code']
        code_fmt: Mapping = self.specs['code_fmt']

        # Pin name in code_fmt is irrelevant. We just want the value and formatting
        assert code_fmt['type'] == 'NHOT', 'other formats not supported yet'
        num_hot: int = code_fmt['num']
        width: int = code_fmt['width']
        if num_hot > width:
            raise ValueError("num_hot cannot be greater than width")

        bias_str = 'v_i_nbias'  # TODO: fix this naming scheme
        pin_value: int = code['value']
        num_stages = 4
        num_stages2 = num_stages * 2
        ans = {}

        aa_val = 2
        inj_val = 4
        injb_val = 10000  # "sub threshold" = off

        # Set always on
        for idx in range(num_stages):
            ans[f'tail_aa<{idx}>'] = f'{bias_str} / {aa_val}'

        # Set inj
        order = [(f'tail_injb<{idx}>', f'tail_inj<{idx}>') for idx in range(num_stages)] + \
            [(f'taile_injb<{idx}>', f'taile_inj<{idx}>') for idx in range(num_stages)]
        
        # Special case for disable
        if pin_value < 0:
            for injb_str, inj_str in order:
                ans[injb_str] = f'{bias_str} / {inj_val}'
                ans[inj_str] = f'{bias_str} / {injb_val}'
        else:
            code_dec = pin_value / width
            for inj_idx, (injb_str, inj_str) in enumerate(order):
                cntr = inj_idx / num_stages2
                diff = abs(code_dec - cntr)
                lvlb = min(1 / num_stages2, diff)
                # include wrap around
                diff = abs(code_dec - cntr - 1)
                lvlb = min(lvlb, diff)

                lvlb = lvlb  * num_stages2
                lvl = 1 - lvlb
                
                _injb_val =  max(1 / inj_val * lvlb, 1 / injb_val)
                _inj_val =  max(1 / inj_val * lvl, 1 / injb_val)
                ans[injb_str] = f'{bias_str} * {_injb_val}'
                ans[inj_str] = f'{bias_str} * {_inj_val}'

        src_list = tbm_specs['src_list']
        for k, v in ans.items():
            src_list.append(
                dict(type='idc', value=dict(idc=v), conns={'PLUS': k, 'MINUS': 'VSS'})
            )

        return tbm_specs


class OscCoreTranMM(OscCoreMixin, PhaseRotatorTranMM):
    def plot(self, info: Mapping, sim_dir: Path):
        data = info['data']
        period_dict = info['period_dict']
        cross_dict_rise = info['cross_dict']
        dc_dict = info['dc_dict']
        amp_dict = info['amp']

        avg_period = np.mean(list(period_dict.values())[-1])
        start_idx = np.where(data['time'] > data['time'][-1] - 10 * avg_period)[0][0]

        # Summary plot
        plt.figure(figsize=(12, 10))
        plt.subplot(221)
        if start_idx < 0:
            pil = list(period_dict.values())[0]
            period = np.mean(pil[len(pil)//2:])
            start_idx = np.where(data['time'] > data['time'][-1] - 3 * period)[0][0]
        for name in cross_dict_rise:
            plt.plot(data['time'][start_idx:], data[name][0][start_idx:], label=name)
        plt.legend()
        plt.xlabel("time [s]")
        plt.ylabel("signal [v]")
        plt.title("Signal vs time")
        plt.grid()

        plt.subplot(222)
        for name in cross_dict_rise:
            plt.plot(cross_dict_rise[name][:-1], period_dict[name] * 1e12, 'x-', label=name)
        plt.legend()
        plt.xlabel("time [s]")
        plt.ylabel("period [ps]")
        plt.title("Period vs time")
        plt.grid()

        plt.subplot(223)
        for idx, key in enumerate(dc_dict):
            plt.stem([idx], [dc_dict[key]], label=key)
        plt.legend()
        plt.xlabel("time [s]")
        plt.ylabel("duty cycle [s/s]")
        plt.title("duty cycle")
        plt.grid()

        plt.subplot(224)
        for idx, key in enumerate(amp_dict):
            plt.stem([idx], [amp_dict[key]], label=key)
        plt.legend()
        plt.ylabel("signal [v]")
        plt.title("amp")
        plt.grid()

        plt.tight_layout()
        plt.savefig(sim_dir / 'data.png')
        plt.close()

        # Plot all of time
        plt.figure(figsize=(12, 10))
        for name in cross_dict_rise:
            plt.plot(data['time'], data[name][0], label=name)
        plt.legend()
        plt.xlabel("time [s]")
        plt.ylabel("signal [v]")
        plt.title("Signal vs time")
        plt.grid()
        plt.tight_layout()
        plt.savefig(sim_dir / 'waveforms.png')
        plt.close()

class OscCorePSSMM(OscCoreMixin, PhaseRotatorPSSMM):
    def plot(self, info: Mapping, sim_dir: Path):
        from bag.simulation.data import SimData
        data: SimData = info['data']

        # process tstab
        data.open_group('pss_td')

        plt.figure()
        plt.plot(data['time'], data['injp'][0] - data['injm'][0], label='inj')
        plt.plot(data['time'], data['V_0'][0] - data['V_4'][0], label='mid')
        plt.legend()
        plt.grid()
        plt.savefig(sim_dir / 'pss_td.png')
        plt.close()

        # process tstab
        plt.figure()
        plt.plot(data['time'], data['injp'][0], label='inj')
        plt.plot(data['time'], data['V_0'][0], label='mid')
        plt.legend()
        plt.grid()
        plt.savefig(sim_dir / 'pss_td_se.png')
        plt.close()

        # process frequencies
        data.open_group('pss_fd')

        plt.figure()
        plt.plot(data['freq'], abs(data['injp'][0] - data['injm'][0]), label='inj')
        plt.plot(data['freq'], abs(data['V_0'][0] - data['V_4'][0]), label='mid')
        plt.legend()
        plt.savefig(sim_dir / 'pss_fd.png')
        plt.close()

        # Process pnoise
        data.open_group('pnoise')
        freq_key = data.sweep_params[-1]  # Should be one of "freq", "relative frequency", etc.

        plt.figure()
        freq = data[freq_key]
        noise = data['out'][0]
        noise_dbc_hz = 10 * np.log10(noise ** 2)
        plt.semilogx(freq, noise_dbc_hz)
        # plt.loglog(res['freq'], res['noise'])
        plt.grid()
        plt.savefig(sim_dir / "pnoise.png")
        plt.close()

class OscCoreFreqMM(PhaseRotatorFreqMM):
    def __init__(self, meas_specs: Mapping[str, Any], log_file: str, log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        super().__init__(meas_specs, log_file, log_level, precision)
        self.mm_class = OscCoreTranMM

class OscCoreINLMM(PhaseRotatorINLMM):
    def __init__(self, meas_specs: Mapping[str, Any], log_file: str, log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        super().__init__(meas_specs, log_file, log_level, precision)
        self.mm_class = OscCoreTranMM
