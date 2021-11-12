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

from bag.concurrent.util import GatherHelper
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.data import SimData
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult

from bag3_testbenches.measurement.pnoise.base import PNoiseTB
from bag3_testbenches.measurement.ac.base import ACTB

from .zin import C2C_ZinMM


class SrcLoadMixin:
    """Src Load mixin for the main C2C_AC testbench"""
    @classmethod
    def get_in_outs(cls, in_clk_info, out_clk_info, power_pins, iref_dict: Dict) -> Tuple[List, List, List]:
        in_name = in_clk_info['name']
        in_diff = in_clk_info.get('diff', '')
        save_outputs = [in_name]
        if in_diff:
            save_outputs.append(in_diff)

        # Values from sim_params
        amp = 'v_amp'
        dc = 'v_BIAS'

        src_list = [
            dict(type='vsin', value=dict(mag=amp, ampl=amp, sinedc=dc, freq='f_osc'),
                 conns={'PLUS': in_name, 'MINUS': 'VSS'})
        ]
        if in_diff:
            src_list.append(
                dict(type='vsin', value=dict(mag=f'-{amp}', ampl=f'-{amp}', sinedc=dc, freq='f_osc'),
                     conns={'PLUS': in_diff, 'MINUS': 'VSS'})
            )
        
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


class C2C_ACMM(MeasurementManager, SrcLoadMixin):

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

        outp = data['outp']
        outm = data['outm']
        out_ac = outp-outm

        out_tf = np.abs(out_ac / in_ac)[0]
        freq = data['freq']

        out_tf_16G = np.interp(16e9, freq, out_tf)
        
        ans = dict(
            data=data,
            out_tf_16G=out_tf_16G,
        )
        return ans


class C2C_PSSMM(MeasurementManager, SrcLoadMixin):

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                            dut: Optional[DesignInstance],
                                            harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        # This is a single code measurement, so no sweeping
        # TODO: add multi-corner support
        # TODO: add support for multiple jitter integration ranges
        # TODO: add support for selecting sampled vs time-integrated jitter
        # TODO: sujpport for multiple pnoise probes

        in_clk_info: Mapping = self.specs['in_clk_info']
        out_clk_info: List[Mapping] = self.specs['out_clk_info']
        power_pins: List[str] = self.specs['power_pins']
        tbm_specs: Mapping = self.specs['tbm_specs']
        iref_dict: Mapping = self.specs.get('iref_dict', {})

        src_list, save_outputs, load_list = self.get_in_outs(in_clk_info, out_clk_info, power_pins, iref_dict)
        
        # The way BAG's SimData flow is setup right now doesn't allow for multiple of one type of analysis
        # This means that each pnoise run requires its own pss run. This is not particularly efficient
        # 
        # pnoise_probe = self.setup_pnoise_probe(out_clk_info)
        probe_info_list = [('outp', 'outm')]

        if 'save_outputs' in tbm_specs:
            tbm_specs['save_outputs'].extend(save_outputs)
        else:
            tbm_specs['save_outputs'] = save_outputs
        if 'probe_info_list' in tbm_specs:
            tbm_specs['probe_info_list'].extend(probe_info_list)
        else:
            tbm_specs['probe_info_list'] = probe_info_list

        tbm_specs = dict(
            **self.specs['tbm_specs'],
            src_list=src_list,
            load_list=load_list
        )
        tbm = cast(PNoiseTB, self.make_tbm(PNoiseTB, tbm_specs))
        self.tbm = tbm
        tbm_name = name
        tb_params = dict()
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                        tb_params=tb_params)

        # Collect data and report results
        data: SimData = sim_results.data

        # Process PSS
        data.open_group('pss_td')
        op_freq = tbm.specs['fund']
        
        time = data['time']
        midp = data['midp']
        midm = data['midm']
        outp = data['outp']
        outm = data['outm']

        mid_pss = midp - midm
        out_pss = outp - outm
        mid_p2p = np.max(mid_pss) - np.min(mid_pss)
        out_p2p = np.max(out_pss) - np.min(out_pss)
        
        dout = np.diff(out_pss)
        dt = np.diff(time)
        dout_dt = dout / dt
        dout_dt_peak = np.max(dout_dt)

        i_vdd = data['XDUT:VDD']
        # average 2nd half, when settled
        i_vdd_avg = np.mean(i_vdd[:,len(time) // 2:])

        # Process pnoise
        data.open_group('pnoise')

        from bag.math.interpolate import LinearInterpolator
        # TODO: figure out how to get this always right
        try:
            freq = data['relative frequency']
        except:
            try:
                freq = data['freq']
            except:
                raise RuntimeError("Unknown frequency keyword")
        freq_log = np.log(freq)
        out_log = np.log(data['out'][0] ** 2)  # V/sqrt(Hz) -> V^2/Hz
        freq_log_delta = np.ediff1d(freq_log).mean()
        freq_min, freq_max = np.min(freq_log), np.log(op_freq / 2)
        
        # if self.multiple_points:
        if False:
            noise_loglog_interps = [LinearInterpolator([freq_log], out_log[i], [freq_log_delta]) for i in range(len(data['timeindex']))]
            noise_integrated = np.array([interp.integrate(freq_min, freq_max, logx=True, logy=True) for interp in noise_loglog_interps])
        else:
            noise_loglog = LinearInterpolator([freq_log], out_log, [np.ediff1d(freq_log).mean()])
            noise_integrated = noise_loglog.integrate(freq_min, freq_max, logx=True, logy=True)

        res = dict(
            freq=freq,
            noise=data['out'][0],
            noise_integrated=noise_integrated
        )

        # Jitter computation
        amp = out_p2p
        # x2 for DSB
        jitter = np.sqrt(noise_integrated * 2) / amp / 2 / np.pi / op_freq
        
        if self.specs.get('plot'):
            # process tstab
            data.open_group('pss_td')

            # Time domain plots, Differential
            plt.figure()
            plt.subplot(211)
            plt.plot(data['time'], data['inp'][0] - data['inm'][0], label='in')
            plt.plot(data['time'], data['midp'][0] - data['midm'][0], label='mid')
            plt.plot(data['time'], data['outp'][0] - data['outm'][0], label='out')
            plt.legend()
            plt.xlabel('time [s]')
            plt.ylabel('Signal [V]')
            plt.grid()

            plt.subplot(212)
            plt.plot(data['time'], data['XDUT:VDD'][0] * 1e3, label='Supply Current')
            plt.legend()
            plt.xlabel('time [s]')
            plt.ylabel('Supply Current[mA]')
            plt.grid()
            plt.savefig(sim_dir / 'pss_td.png')

            # Time domain plots, single ended
            plt.figure()
            plt.subplot(211)
            plt.plot(data['time'], data['inp'][0], label='in')
            plt.plot(data['time'], data['midp'][0], label='mid')
            plt.plot(data['time'], data['outm'][0], label='out')
            plt.legend()
            plt.xlabel('time [s]')
            plt.ylabel('Signal [V]')
            plt.grid()

            plt.subplot(212)
            plt.plot(data['time'], data['XDUT:VDD'][0] * 1e3, label='Supply Current')
            plt.legend()
            plt.grid()
            plt.xlabel('time [s]')
            plt.ylabel('Supply Current[mA]')
            plt.savefig(sim_dir / 'pss_td_se.png')

            # Process pnoise
            data.open_group('pnoise')

            plt.figure()
            noise = res['noise']
            noise_dbc_hz = 10 * np.log10(noise ** 2)
            plt.semilogx(res['freq'], noise_dbc_hz)
            plt.grid()
            plt.ylabel("dBc/Hz")
            plt.savefig(sim_dir / "pnoise.png")

        ans = dict(
            data=data,
            mid_p2p=mid_p2p,
            out_p2p=out_p2p,
            dout_dt_peak=dout_dt_peak,
            jee=jitter,
            i_vdd_avg=i_vdd_avg,
            FOM_jee=10*np.log10(jitter**2 * i_vdd_avg / 1e-3)
        )
        return ans

    @classmethod
    def setup_pnoise_probe(cls, out_clk_info: List) -> List[Tuple[str, str, int]]:
        pnoise_probe = []
        for item in out_clk_info:
            # Probe contains PLUS, MINUS, and harmonic
            if item.get('diff'):
                probe = (item['name'], item['diff'], 1)
            else:
                # Assumes VSS exists at the top TB
                probe = (item['name'], 'VSS', 1)
            pnoise_probe.append(probe)
        return pnoise_probe


class C2C_BaseMM(MeasurementManager):
    """Run all and report results"""

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:

        specs: Mapping = self.specs

        gatherer: GatherHelper = GatherHelper()
        tests = dict(
            ac=C2C_ACMM,
            pss=C2C_PSSMM,
            zin=C2C_ZinMM
        )

        for test_name, test_cls in tests.items():
            mm = self.make_mm(test_cls, specs)
            gatherer.append(sim_db.async_simulate_mm_obj(test_name, sim_dir / test_name, dut, mm, harnesses))

        sim_results: Sequence[MeasureResult] = await gatherer.gather_err()
        data_list: Sequence[Mapping[str, Any]] = [mr.data for mr in sim_results]

        # Collect the datas
        ans = {}
        [ans.update(data) for data in data_list]

        return ans

