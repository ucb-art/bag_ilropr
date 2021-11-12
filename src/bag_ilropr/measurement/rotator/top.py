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

"""Top Mixin, Tran and PSS testbenches"""

from typing import Any, Mapping

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pybag.enum import LogLevel

from ..util import code_to_pin_list
from .tran import PhaseRotatorTranMM
from .pss import PhaseRotatorPSSMM
from .meas_freq import PhaseRotatorFreqMM
from .meas_static_inl import PhaseRotatorINLMM

class TopMixin:
    def setup_code(self, tbm_specs: Mapping) -> Mapping[str, Any]:
        """Set up digital pins based on the code"""
        code: Mapping = self.specs['code']
        code_fmt: Mapping = self.specs['code_fmt']

        pin_values: Mapping[str, int] = code_to_pin_list(code, code_fmt)
        tbm_specs['pin_values'].update(pin_values)
        return tbm_specs


class TopTranMM(TopMixin, PhaseRotatorTranMM):
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

class TopPSSMM(TopMixin, PhaseRotatorPSSMM):
    def plot(self, info: Mapping, sim_dir: Path):
        data = info['data']

        # process tstab
        data.open_group('pss_td')

        plt.figure()
        plt.plot(data['time'], data['injp'][0] - data['injm'][0], label='inj')
        plt.plot(data['time'], data['injpb'][0] - data['injmb'][0], label='injb')
        plt.plot(data['time'], data['mid_0'][0] - data['mid_4'][0], label='mid')
        plt.plot(data['time'], data['out_0'][0] - data['out_4'][0], label='out')
        plt.legend()
        plt.savefig(sim_dir / 'pss_td.png')

        # process frequencies
        data.open_group('pss_fd')

        plt.figure()
        plt.plot(data['freq'], abs(data['injp'][0] - data['injm'][0]), label='inj')
        plt.plot(data['freq'], abs(data['injpb'][0] - data['injmb'][0]), label='injb')
        plt.plot(data['freq'], abs(data['mid_0'][0] - data['mid_4'][0]), label='mid')
        plt.plot(data['freq'], abs(data['out_0'][0] - data['out_4'][0]), label='out')
        plt.legend()
        plt.savefig(sim_dir / 'pss_fd.png')

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


class TopFreqMM(PhaseRotatorFreqMM):
    def __init__(self, meas_specs: Mapping[str, Any], log_file: str, log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        super().__init__(meas_specs, log_file, log_level, precision)
        self.mm_class = TopTranMM


class TopINLMM(PhaseRotatorINLMM):
    def __init__(self, meas_specs: Mapping[str, Any], log_file: str, log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        super().__init__(meas_specs, log_file, log_level, precision)
        self.mm_class = TopTranMM
