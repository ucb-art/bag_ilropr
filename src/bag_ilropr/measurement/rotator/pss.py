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

from pprint import pprint
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bag.io.file import write_yaml
from bag.concurrent.util import GatherHelper
from bag.simulation.measure import MeasurementManager
from bag.simulation.data import SimData, AnalysisType
from bag.simulation.cache import SimulationDB, DesignInstance

from bag3_testbenches.measurement.data.tran import EdgeType

from ..pss_base import PSSSampledTB as PSSTB

class PhaseRotatorPSSMM(MeasurementManager):
    """A central MM for setting up PSS and pnoise sims for phase rotator characterizations
    Measures the following characteristics:
    - amplitude
    - period, based on crossing times
    - jitter (jee, jc, and jcc) [Note, noise is not currently enabled]
    - duty cycle (dc)
    - average current
    Plotting features include summary plots and waveforms
    
    Subclasses can inherit this class and add mixins for plotting and test setup

    This MM supports multiple corners through GatherHelper

    Parameters (self.specs):
    - `code`: Mapping
        Rotator code information. `name`, `value`, and `diff` infos
    - `code_fmt`: Mapping
        Info about code mapping. `type`, `num`, and `width`. Only `type`='NHOT' currently supported
    - `in_clk_info`: Mapping
        Info about input reference clock. `in` and `diff`
    - `input_type`: Optional[str]
        Input type, `vsin` or `vpulse`. Defaults to `vsin`
    - `out_clk_info`: List[Mapping]
        Info about output clocks to measure. `in` and `diff`, like in_clk_info
    - `power_pins`: List[str]
        List of power pins to measure current.
    - `iref_dict`: Optional[Mapping]
        # Mapping of current sink names to value. Currents sink to VSS. Defaults to empty
        Mapping of current names to value. Currents source from VDD. Defaults to empty
    - `tbm_specs`: Mapping
        Specs for TBM (TranTB)
    - `harnesses_list`: Optional[List[Mapping]]
        Harness info list. Default to empty
    - `atol`: Optional[float]
        abs tolerance for crossing measurements. Defaults to 1e-22
    - `rtol`: Optional[float]
        rel tolerance for crossing measurements. Defaults to 1e-8
    - `plot`: Optional[bool]
        True to enable plotting. Defaults to False

    Important values for `tbm_specs` are:
    - sim_params: Mapping[str, float]
        Simulation parameters. This should include:
            - `v_amp`: input amplitude (SE)
            - `vdc`: input DC level (SE)
            - `t_clk_per`: input clock period
            - `t_clk_rf`: rise and fall time of input clock (`vpulse` only)
            - `t_d`: Phase offset between clk and clkb. Defaults to 0
            - `c_load`: cap load on each of `out_clk_info`
        
    Return:
        If one corner is provided, the return is a dictionary of the above characteristics.
        If multiple corners are provided, the return is a dictionary mapping corner to 
            subdictionaries with the above characteristics

    Note, `get_sim_info`, `initialize`, and `process_output` were deprecated in 2023.

    TODO: Add support for multiple pnoise probes

    """

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
        """ This is a single code measurement, so no sweeping """
        code: Mapping = self.specs['code']
        code_fmt: Mapping = self.specs['code_fmt']
        in_clk_info: Mapping = self.specs['in_clk_info']
        out_clk_info: List[Mapping] = self.specs['out_clk_info']
        power_pins: List[str] = self.specs['power_pins']
        tbm_specs: Mapping = self.specs['tbm_specs']
        iref_dict: Mapping = self.specs.get('iref_dict', {})
        input_type: str = self.specs.get('input_type', 'vsin')

        # Post process specs, setup TBM specs
        for key in ['save_outputs', 'src_list', 'load_list']:
            if not key in tbm_specs:
                tbm_specs[key] = []
        if not 'pin_values' in tbm_specs:
            tbm_specs['pin_values'] = {}
        
        tmp = self.get_in_outs(in_clk_info, out_clk_info, power_pins, iref_dict, input_type)
        src_list, save_outputs, load_list = tmp

        tbm_specs['save_outputs'].extend(save_outputs)        
        tbm_specs['src_list'].extend(src_list)
        tbm_specs['load_list'].extend(load_list)
        tbm_specs['sim_envs'] = [pvt]

        # Additional setup for the code
        self.logger.info(f"Code: {code['value']}")
        tbm_specs = self.setup_code(tbm_specs)

        # Default value
        if 'td' not in tbm_specs['sim_params']:
            tbm_specs['sim_params']['td'] = 0

        # setup harness
        if harnesses:
            harnesses_list: Sequence[Mapping[str, Any]] = self.specs['harnesses_list']
        else:
            harnesses_list = []
        tbm_specs['harnesses_list'] = harnesses_list
        
        # The way BAG's SimData flow is setup right now doesn't allow for multiple of one type of analysis
        # This means that each pnoise run requires its own pss run. This is not particularly efficient
        # Ideally, we would measure the noise of each output of out_clk_info
        if 'pnoise_probe' not in tbm_specs:
            # Default to the first out_clk_info item
            out_clk = out_clk_info[0]
            if 'diff' in out_clk:
                _pnoise_probe = (out_clk['name'], out_clk['diff'], 1)
            else:
                _pnoise_probe = (out_clk['name'], 'VSS', 1)
            tbm_specs['pnoise_probe'] = [_pnoise_probe]
        pnoise_probe = tbm_specs['pnoise_probe'][0]
        
        # Run TBM
        tbm = cast(PSSTB, self.make_tbm(PSSTB, tbm_specs))
        self.tbm = tbm
        tbm_name = name
        tb_params = dict()
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                          tb_params=tb_params, harnesses=harnesses)

        # Collect data and report results
        data: SimData = sim_results.data

        # swing check
        pport, nport, _harm = pnoise_probe
        data.open_group('pss_td')
        out = data[pport][0] - data[nport][0]
        time_len = len(out)
        # Check if we are swinging
        time_len_10 = time_len // 10
        out_peak = np.max(out[-time_len_10:])
        if out_peak < 50e-3:  # 50 mV heuristic
            raise RuntimeError("Output is not swinging. Results not valid")
        
        # Determine dv/dt at crossing        
        cross_dict: Dict[str, np.ndarray] = {}  # all the crossing times
        if 'diff' in in_clk_info:
            tmp = self.calc_dvdt(data, pport, EdgeType.RISE, nport)
        else:
            tmp = self.calc_dvdt(data, pport, EdgeType.RISE)
        # Trick to improve metrics w/o settling check
        cross_dict[pport] = tmp[len(tmp)*2//3:]
        dvdt = np.mean(cross_dict[pport])

        # Process pnoise
        data.open_group('pnoise')
        op_freq = tbm.specs['fund']

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
        amp = out_peak
        # x2 for DSB
        # Phase noise computation
        jitter = np.sqrt(noise_integrated * 2) / amp / 2 / np.pi / op_freq
        # Sampled jitter computation
        # The results should be similar
        jitter_2 = np.sqrt(noise_integrated * 2) / dvdt
        print(out_peak)
        print("Jitter pnoise:", jitter)
        print("Jitter sampled:", jitter_2)

        ans = dict(
            data=data,
            noise_integrated=noise_integrated,
            jee=jitter_2,
            swing=out_peak,
        )

        if self.specs.get('plot'):
            self.plot(ans, sim_dir)
        with open(sim_dir / "ans.txt", 'w') as f:
            pprint(ans, stream=f)
        write_yaml(sim_dir / 'ans.yaml', ans)
        return ans

    def plot(self, info: Mapping, sim_dir: Path):
        pass

    @classmethod
    def get_in_outs(cls, in_clk_info, out_clk_info, power_pins, iref_dict: Dict, input_type: str) -> Tuple[List, List, List]:
        in_name = in_clk_info['name']
        in_diff = in_clk_info.get('diff', '')
        save_outputs = [in_name]
        if in_diff:
            save_outputs.append(in_diff)

        # Set these in sim_params
        amp = 'v_amp'
        dc = 'v_dc'
        # Select input type
        if input_type == 'vpulse':
            src_list = [
                dict(type='vpulse', value=dict(v1='v_dc + v_amp', v2='v_dc - v_amp', tr='t_clk_rf', tf='t_clk_rf',
                     td='t_clk_per/2', per='t_clk_per', pulse='t_clk_per / 2 - t_clk_rf'),
                     conns={'PLUS': in_name, 'MINUS': 'VSS'})
            ]
            if in_diff:
                src_list.append(
                    dict(type='vpulse', value=dict(v1='v_dc - v_amp', v2='v_dc + v_amp', tr='t_clk_rf', tf='t_clk_rf',
                         td='t_clk_per/2 + td', per='t_clk_per', pw='t_clk_per / 2 - t_clk_rf'),
                         conns={'PLUS': in_diff, 'MINUS': 'VSS'})
                )
        elif input_type == 'vsin':
            src_list = [
                dict(type='vsin', value=dict(ampl=amp, sinedc=dc, delay='t_clk_per/2', freq='1/t_clk_per'),
                    conns={'PLUS': in_name, 'MINUS': 'VSS'})
            ]
            if in_diff:
                src_list.append(
                    dict(type='vsin', value=dict(ampl=f'-{amp}', sinedc=dc,delay='t_clk_per/2  + td', freq='1/t_clk_per'),
                        conns={'PLUS': in_diff, 'MINUS': 'VSS'})
                )
        else:
            raise ValueError("Core pss base: Unsupported input type: " + input_type)

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

        for k, v in iref_dict.items():
            src_list.append(
                # TODO: make configurable
                # dict(type='idc', value=dict(idc=v), conns={'PLUS': k, 'MINUS': 'VSS'})
                dict(type='idc', value=dict(idc=v), conns={'PLUS': 'VDD', 'MINUS': k})
            )

        return src_list, save_outputs, load_list

    def calc_dvdt(self, data: SimData, out_name: str, out_edge: EdgeType,
                   outb_name: str = '',
                   t_start: Union[np.ndarray, float, str] = 0,
                   t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        """
        calculates times of crossing, returns as an np.array. adapted from DigitalTranTB
        """

        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.tbm.get_pin_supply_values(out_name, data)
        data.open_group('pss_td')
        tvec = data['time']

        # evaluate t_start/t_stop
        if isinstance(t_start, str) or isinstance(t_stop, str):
            calc = self.get_calculator(data)
            if isinstance(t_start, str):
                t_start = calc.eval(t_start)
            if isinstance(t_stop, str):
                t_stop = calc.eval(t_stop)

        if outb_name:
            # differential
            out_vec = (data[out_name] - data[outb_name])[0]
            vth_out = 0
        else:
            # SE
            out_vec = data[out_name][0]
            vth_out = (out_1 - out_0) * thres_delay + out_0

        # We know that at most, this measurement sweep will be just for the environments
        # TODO: improve support for multiple corners
        out_c = get_all_dvdt(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                  stop=t_stop, rtol=rtol, atol=atol)
        return out_c


def get_all_dvdt(tvec: np.ndarray, yvec: np.ndarray, threshold: float,
                 start: float, stop: float, etype: EdgeType, rtol: float,
                 atol: float) -> float:
    # eliminate NaN from time vector in cases where simulation time is different between runs.
    mask = ~np.isnan(tvec)
    tvec = tvec[mask]
    yvec = yvec[mask]

    sidx = np.searchsorted(tvec, start)
    eidx = np.searchsorted(tvec, stop)
    if eidx < tvec.size and np.isclose(stop, tvec[eidx], rtol=rtol, atol=atol):
        eidx += 1

    # quantize waveform values, then detect edge.
    dvec = np.diff((yvec[sidx:eidx] >= threshold).astype(int))

    ans = np.array([])
    # interpolate to get the values
    # TODO: pretty sure this won't handle corners well
    if EdgeType.RISE in etype:
        idx_list = np.argwhere(np.maximum(dvec, 0)).flatten()
        for idx in idx_list:
            lidx = max(sidx + idx - 2, 0)
            ridx = min(sidx + idx + 3, len(yvec))
            # tzc = np.interp(0, yvec[lidx:ridx], tvec[lidx:ridx])
            dvdt = (yvec[ridx - 1] - yvec[lidx]) / (tvec[ridx - 1] - tvec[lidx])
            ans = np.append(ans, dvdt)
    if EdgeType.FALL in etype:
        idx_list = np.argwhere(np.minimum(dvec, 0)).flatten()
        for idx in idx_list:
            lidx = max(sidx + idx - 2, 0)
            ridx = min(sidx + idx + 3, len(yvec))
            # tzc = np.interp(0, yvec[lidx:ridx], tvec[lidx:ridx])
            dvdt = (yvec[ridx - 1] - yvec[lidx]) / (tvec[ridx - 1] - tvec[lidx])
            ans = np.append(ans, dvdt)
    # breakpoint()
    return ans
