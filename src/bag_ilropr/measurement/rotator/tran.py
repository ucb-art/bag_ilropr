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

from pybag.enum import DesignOutput, LogLevel

from bag.io.file import write_yaml
from bag.concurrent.util import GatherHelper
from bag.simulation.measure import MeasurementManager
from bag.simulation.data import SimData, AnalysisType
from bag.simulation.cache import SimulationDB, DesignInstance

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.base import TranTB
from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import get_first_crossings

from ..util import get_all_crossings_1d as get_all_crossings


class PhaseRotatorTranMM(MeasurementManager):
    """A central MM for setting up transient sims for phase rotator characterizations
    Measures the following characteristics:
    - amplitude
    - period, based on crossing times
    - jitter (jee, jc, and jcc) [Note, noise is not currently enabled]
    - duty cycle (dc)
    - average current
    Plotting features include summary plots and waveforms
    
    Subclasses can inherit this class and add mixins for plotting and test setup.
        Use the `plot` and `setup_code` functions, respectively.

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
    - `settle_tol`: Optional[float]
        Minimum swing allowed for check. Defaults to 5e-4

    Important values for `tbm_specs` are:
    - sim_params: Mapping[str, float]
        Simulation parameters. This should include:
            - `v_amp`: input amplitude (SE)
            - `v_dc`: input DC level (SE)
            - `t_clk_per`: input clock period
            - `t_clk_rf`: rise and fall time of input clock (`vpulse` only)
            - `t_d`: Phase offset between clk and clkb. Defaults to 0
            - `c_load`: cap load on each of `out_clk_info`

    Return:
        If one corner is provided, the return is a dictionary of the above characteristics.
        If multiple corners are provided, the return is a dictionary mapping corner to 
            subdictionaries with the above characteristics

    Note, `get_sim_info`, `initialize`, and `process_output` were deprecated in 2023.

    """
    def __init__(self, meas_specs: Mapping[str, Any], log_file: str,
                 log_level: LogLevel = LogLevel.DEBUG, precision: int = 6) -> None:
        # Hidden params
        self._sim_swp_info_struct = None
        self._save_all_dut_pins = None
        self._save_all_nets = None

        # Workaround to allow InBuffTranMM to share methods with PhaseRotatorTranMM
        self._has_code = True 

        # Put at the end to run commit() last.
        super().__init__(meas_specs, log_file, log_level, precision)

    """Harness processing functions"""
    def process_dut(self, dut: Optional[DesignInstance], sim_db: SimulationDB) -> Optional[DesignInstance]:
        return dut

    def process_harnesses(self, harnesses: Optional[Sequence[DesignInstance]] = None):
        if not harnesses:
            return
        harnesses_list = self.specs['tbm_specs'].get('harnesses_list', [])
        if not harnesses_list:
            return
        new_harnesses_list = []
        for i, har_info in enumerate(harnesses_list):
            idx = har_info.get('harness_idx', 0)
            default_conns_dict = {k: v for (k, v) in har_info.get('conns', [])}
            conns = []
            for pin in harnesses[idx].pin_names:
                if pin in default_conns_dict:
                    conns.append((pin, default_conns_dict[pin]))
                else:
                    conns.append((pin, f'nc_{i}_{pin}'))
            new_harnesses_list.append(dict(harness_idx=idx, conns=conns))
        self.specs['tbm_specs']['harnesses_list'] = new_harnesses_list

    """Measurement functions functions"""
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

        # Load specs
        if self._has_code:
            code: Mapping = self.specs['code']
            code_fmt: Mapping = self.specs['code_fmt']
        else:
            code = None
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

        # Process DUT and harnesses
        dut = self.process_dut(dut, sim_db)
        if dut is not None:
            self.specs['tbm_specs']['dut_pins'] = dut.pin_names
            tbm_specs['dut_pins'] = dut.pin_names
            if self._save_all_dut_pins:
                save_outputs = set(self.specs['tbm_specs'].get('save_outputs', []))
                self.specs['tbm_specs']['save_outputs'] = save_outputs | set(dut.pin_names)
        self.process_harnesses(harnesses)
        
        # Set up stimuli and loads
        tmp = self.get_in_outs(in_clk_info, out_clk_info, power_pins, iref_dict, input_type)
        src_list, save_outputs, load_list = tmp

        tbm_specs['save_outputs'].extend(save_outputs)        
        tbm_specs['src_list'].extend(src_list)
        tbm_specs['load_list'].extend(load_list)
        tbm_specs['sim_envs'] = [pvt]

        # Additional setup for the code
        if self._has_code:
            self.logger.info(f"Code: {code['value']}")
            tbm_specs = self.setup_code(tbm_specs)

        # Default value
        if 'td' not in tbm_specs['sim_params']:
            tbm_specs['sim_params']['td'] = 0
        
        # Run TBM
        tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
        self.tbm = tbm
        tbm_name = name
        tb_params = dict()
        nom_results = await self._run_sim(tbm_name, sim_db, sim_dir, dut, tbm, tb_params,
                                          harnesses, is_mc=False)
        ans = nom_results
        self.log(f'Measurement {name} done, recording results.')

        # Optionally run MC
        mc_params = self.specs.get('mc_params', {})
        if mc_params:
            mc_name = f'{name}_mc'
            self.log('Starting Monte Carlo simulation')
            mc_tbm_specs = tbm_specs.copy()
            mc_tbm_specs['sim_envs'] = [pvt]
            mc_tbm_specs['monte_carlo_params'] = mc_params
            mc_tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, mc_tbm_specs))
            mc_results = await self._run_sim_mc(mc_name, sim_db, sim_dir, dut, mc_tbm, tb_params, 
                                             harnesses, is_mc=True)
        else:
            mc_results={}
        ans['mc'] = mc_results
        
        if self.specs.get('plot'):
            self.plot(ans, sim_dir)
        with open(sim_dir / "ans.txt", 'w') as f:
            pprint(ans, stream=f)
        write_yaml(sim_dir / 'ans.yaml', ans)

        return ans

    async def _run_sim(self, tbm_name: str, sim_db: SimulationDB, sim_dir: Path, dut: DesignInstance,
                tbm: DigitalTranTB, tb_params: Mapping[str, Any], harnesses: List[DesignInstance],
                is_mc: bool = False):
        """Helper function to run one simulation"""
        in_clk_info: Mapping = self.specs['in_clk_info']
        out_clk_info: List[Mapping] = self.specs['out_clk_info']
        power_pins: List[str] = self.specs['power_pins']

        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                          tb_params=tb_params, harnesses=harnesses)

        # Collect data and report results
        data: SimData = sim_results.data

        # All rising and falling crossing times
        cross_dict_rise: Dict[str, np.ndarray] = {}
        cross_dict_fall: Dict[str, np.ndarray] = {}
        for cdict, edgetype in [(cross_dict_rise, EdgeType.RISE), 
                                (cross_dict_fall, EdgeType.FALL)]:
            def _process(name: str, diff: str = ""):
                tmp = self.calc_all_cross(data, name, edgetype, diff)
                
                # Settle and swing checks
                if self.specs.get('settle_tol'):
                    _settle_idx = _settle_check(np.diff(tmp), self.specs.get('settle_tol'))
                else:
                    _settle_idx = _settle_check(np.diff(tmp))
                _sig = data[name] - data[diff] if diff else data[name]
                _swing_check(_sig[0][_settle_idx:])

                # Save
                if diff:
                    cdict[name] = tmp[_settle_idx:]
                else:
                    cdict[name + "_se"] = tmp[_settle_idx:]
            
            # Process all in and out signals
            _process(in_clk_info['name'], in_clk_info.get('diff', ""))
            for info in out_clk_info:
                _process(info['name'], info.get('diff', ""))

        amp_dict: Dict[str, float] = {} # average amp during the last 10 periods
        period_dict: Dict[str, np.ndarray] = {}  # all the periods. first difference of crossings
        jee_dict: Dict[str, float] = {}  # abs jitter. std of transition
        jc_dict: Dict[str, float] = {}  # cycle jitter. std of period
        jcc_dict: Dict[str, float] = {}  # c2c jitter. std of period difference
        dc_dict: Dict[str, float] = {} # average duty cycle during the last 10 periods

        last_start_idx = -1
        # for clk_name, cross_times in cross_dict_rise.items():
        clk_list = [in_clk_info] + out_clk_info 
        for clk_info in clk_list:
            # TODO: this is ugly
            clk_name = clk_info['name']
            label_name = clk_info['name']  if clk_info.get('diff') else clk_info['name'] + '_se'

            cross_times = cross_dict_rise[label_name]
            period_list = np.diff(cross_times)
            period_dict[label_name] = period_list

            avg_period = np.mean(period_list)
            try:
                start_idx = np.where(data['time'] > data['time'][-1] - 10 * avg_period)[0][0]
                last_start_idx = start_idx
            except:
                if last_start_idx < 0:
                    raise RuntimeError("No period found")
                start_idx = last_start_idx
            wave = data[clk_name][0][start_idx:]
            wave_min, wave_max = np.min(wave), np.max(wave)
            wave_amp = wave_max - wave_min
            amp_dict[label_name] = wave_amp

            # Differential 
            # t_high = np.mean(cross_dict_fall[clk_name][-10:] - cross_times[-10:])
            # Try SE DC
            cross_hi = self.calc_all_cross(data, clk_name, EdgeType.RISE)
            cross_lo = self.calc_all_cross(data, clk_name, EdgeType.FALL)
            _idx = min(len(cross_hi), len(cross_lo))
            cross_hi, cross_lo = cross_hi[:_idx], cross_lo[:_idx]
            t_high = np.mean(cross_lo - cross_hi)
            if t_high < 0:
                t_high += avg_period
            dc_dict[label_name] = t_high / avg_period

            sample_times = np.linspace(0, avg_period * (len(cross_times) - 1), num=len(cross_times))
            samp_offset = cross_times - sample_times
            jee_dict[label_name] = np.std(samp_offset)

            jc_dict[label_name] = np.std(period_list)
            jcc_dict[label_name] = np.std(period_list[1:] - period_list[:-1])

        current_dict: Dict[str, float] = {}  # average current for each power pin
        for ppin in power_pins:
            ans = data['XDUT:' + ppin]
            ans = ans[len(ans)// 4 * 3 :]
            current_dict[ppin] = np.mean(ans)

        ans = dict(
            data=data,
            cross_dict=cross_dict_rise,
            period_dict=period_dict,
            jee=jee_dict,
            jc=jc_dict,
            jcc=jcc_dict,
            amp=amp_dict,
            current=current_dict,
            dc_dict=dc_dict,
        )
        return ans

    async def _run_sim_mc(self, tbm_name: str, sim_db: SimulationDB, sim_dir: Path, dut: DesignInstance,
                tbm: DigitalTranTB, tb_params: Mapping[str, Any], harnesses: List[DesignInstance],
                is_mc: bool = False):
        """Helper function to run MC simulations
        BZ 3/19: Much of this can be merged with _run_sim, but I can't be bothered right now
            Main issue is changing how crossings are calculated.
        """
        in_clk_info: Mapping = self.specs['in_clk_info']
        out_clk_info: List[Mapping] = self.specs['out_clk_info']
        power_pins: List[str] = self.specs['power_pins']

        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                          tb_params=tb_params, harnesses=harnesses)

        # Collect data and report results
        data: SimData = sim_results.data

        # Formatting helper function
        def _get_mc(info: Union[float, np.ndarray]):
            if is_mc:
                return dict(
                    mean=np.mean(info),
                    std=np.std(info),
                )
            else:
                if isinstance(info, np.ndarray):
                    return info[0]
            return info

        # Start time
        calc = tbm.get_calculator(data)
        t_start = self.specs.get('wait_time', 150e-12)
        if isinstance(t_start, str):
            t_start = calc.eval(t_start)[0]

        # Collect general waveform characteristics
        amp_dict: Dict[str, float] = {} # average amp during the last 10 periods
        period_dict: Dict[str, np.ndarray] = {}  # all the periods. first difference of crossings
        freq_dict: Dict[str, float] = {} # 1 / period
        dc_dict: Dict[str, float] = {} # average duty cycle during the last 10 periods

        clk_list = [in_clk_info] + out_clk_info 
        for clk_info in clk_list:
            # TODO: this is ugly
            clk_name = clk_info['name']
            clkb_name = clk_info.get('diff')
            label_name = clk_info['name']  if clk_info.get('diff') else clk_info['name'] + '_se'

            per = self.measure_period(data, tbm, clk_name, EdgeType.RISE, t_start, clkb_name)
            period_dict[label_name] = _get_mc(per)
            freq_dict[label_name] = _get_mc(1 / per)
            dc_dict[label_name] = _get_mc(self.measure_duty_cycle(data, tbm, clk_name, t_start, clkb_name))
            amp_dict[label_name] = _get_mc(measure_amplitude(data, clk_name, t_start, clkb_name))
        
        # Compute phase differences
        phase_diff_list = self.specs.get('phase_diff_list', [])
        phase_diff_ans = {}
        for item in phase_diff_list:
            _per = self.measure_period(data, tbm, item[0], EdgeType.RISE, t_start)
            td_r = self.measure_ph_diff(data, item[0], item[1], EdgeType.RISE, EdgeType.RISE, t_start)
            td_f = self.measure_ph_diff(data, item[0], item[1], EdgeType.FALL, EdgeType.FALL, t_start)
            # Correct for positive
            td_r = np.add(td_r, _per, out=td_r, where=td_r < 0)
            td_f = np.add(td_f, _per, out=td_f, where=td_f < 0)
            td_m = (td_r + td_f) / 2
            phase_diff_ans[item] = dict(rise=_get_mc(td_r), fall=_get_mc(td_f), mean=_get_mc(td_m))
            
        # average current for power pins
        current_dict: Dict[str, float] = {}
        for ppin in power_pins:
            current_dict[ppin] = _get_mc(measure_mean(data, 'XDUT:' + ppin, t_start))

        ans = dict(
            data=data,
            period=period_dict,
            freq=freq_dict,
            amp=amp_dict,
            phase_diff=phase_diff_ans,
            current=current_dict,
            dc=dc_dict,
        )
        return ans

    def setup_code(self, tbm_specs: Mapping):
        pass

    def plot(self, info: Mapping, sim_dir: Path):
        pass

    @classmethod
    def get_in_outs(cls, in_clk_info, out_clk_info, power_pins, iref_dict: Dict, input_type: str) -> Tuple[List, List, List]:
        # Define inputs
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
            raise ValueError("Core tran base: Unsupported input type: " + input_type)
        
        # Set up output definitions
        load_list = []
        load_set = []   # Avoid redundancy, for SE measurements
        def _setup_load(name):
            if name not in load_set:
                load_set.append(name)
                save_outputs.append(name)
                load_list.append(
                    dict(type='cap', value='c_load', conns={'PLUS': name, 'MINUS': 'VSS'})
                )
        for item in out_clk_info:
            _setup_load(item['name'])
            if item.get('diff'):
                _setup_load(item['diff'])

        # Save power pins to measure power
        for ppin in power_pins:
            save_outputs.append('XDUT:' + ppin)
        
        for k, v in iref_dict.items():
            src_list.append(
                # TODO: make configurable
                # dict(type='idc', value=dict(idc=v), conns={'PLUS': k, 'MINUS': 'VSS'})
                dict(type='idc', value=dict(idc=v), conns={'PLUS': 'VDD', 'MINUS': k})
            )

        return src_list, save_outputs, load_list

    def calc_all_cross(self, data: SimData, out_name: str, out_edge: EdgeType,
                       outb_name: str = '',
                       t_start: Union[np.ndarray, float, str] = 0,
                       t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        """
        calculates times of crossing, returns as an np.array. adapted from DigitalTranTB
        This function does not handle sweeps / corners. Make separate calls for each yvec
        """

        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        out_0, out_1 = self.tbm.get_pin_supply_values(out_name, data)
        data.open_analysis(AnalysisType.TRAN)
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
            # BZ 9/12, modified hack for finding the RZ point
            # TODO hack for not full swing clocks
            start_idx = len(tvec) // 5 * 4
            hi, lo = np.max(out_vec[start_idx:]), np.min(out_vec[start_idx:])
            vth_out = (hi - lo) / 2 + lo

        out_c = get_all_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                  stop=t_stop, rtol=rtol, atol=atol)
        if not out_c.size:
            self.error("No crossings found. Debug...")
        return out_c

    def calc_cross(self, data: SimData, out_name: str, out_edge: EdgeType,
                       outb_name: str = '',
                       t_start: Union[np.ndarray, float, str] = 0,
                       t_stop: Union[np.ndarray, float, str] = float('inf')) -> np.ndarray:
        """
        calculates first crossing times, returns as an np.array. adapted from DigitalTranTB
        This function does handle sweeps / corners. Make separate calls for each yvec
        """

        thres_delay = 0.5

        specs = self.specs
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        data.open_analysis(AnalysisType.TRAN)
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
            # SE - need to find the threshold
            out_vec = data[out_name][0]

            # Check each t_start
            # Logic based on get_first_crossings
            swp_shape = out_vec.shape[:-1]
            t_start_vec = np.broadcast_to(np.asarray(t_start), swp_shape)

            nlast = out_vec.shape[-1]
            _tvec = tvec.reshape(-1, nlast)
            _yvec = out_vec.reshape(-1, nlast)
            _t_start_vec = t_start_vec.flatten()
            n_swp = _t_start_vec.size
            ans = np.empty(n_swp)

            for _idx in range(n_swp):
                _tvec_idx = _tvec[_idx]
                _t_start_idx = _t_start_vec[_idx]
                start_idx = np.argwhere(_tvec_idx > _t_start_idx)[0][0]
                _yvec_idx = _yvec[_idx]
                hi, lo = np.nanmax(_yvec_idx[start_idx:]), np.nanmin(_yvec_idx[start_idx:])
                ans[_idx] = (hi - lo) / 2 + lo
            vth_out = ans.reshape(swp_shape)

            if np.isnan(vth_out).any():
                self.error("Found NAN threshold. Stopping to debug...")

        out_c = get_first_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                   stop=t_stop, rtol=rtol, atol=atol)

        return out_c
    
    def measure_period(self, data: SimData, tbm: DigitalTranTB, sig_name: str, out_edge: EdgeType,
                       t_start: float = 0, clkb_name: str = None):
        """Find the last valid crossing, and take the difference between the last cross and
        the 2nd to last crossing time. We need at least 2 good crossings to make this work."""
        last_cross = None
        curr_cross = self.calc_cross(data, sig_name, out_edge, t_start=t_start, outb_name=clkb_name)
        next_cross = self.calc_cross(data, sig_name, out_edge, t_start=curr_cross, outb_name=clkb_name)
        while np.all(np.isfinite(next_cross)):
            last_cross = curr_cross
            curr_cross = next_cross
            next_cross = self.calc_cross(data, sig_name, out_edge, t_start=curr_cross, outb_name=clkb_name)
        if not np.any(last_cross):
            raise RuntimeError("Too few valid crossings found. Extend sim time.")
        return curr_cross - last_cross


    def measure_duty_cycle(self, data: SimData, tbm: DigitalTranTB, sig_name: str, 
                           t_start: float = 0, clkb_name: str = None):
        """Find the last valid crossing, and take the difference between the last cross and
        the 2nd to last crossing time. We need at least 2 good crossings to make this work."""
        last_cross = None
        curr_cross = self.calc_cross(data, sig_name, EdgeType.RISE, t_start=t_start, outb_name=clkb_name)
        next_cross = self.calc_cross(data, sig_name, EdgeType.RISE, t_start=curr_cross, outb_name=clkb_name)
        while np.all(np.isfinite(next_cross)):
            last_cross = curr_cross
            curr_cross = next_cross
            next_cross = self.calc_cross(data, sig_name, EdgeType.RISE, t_start=curr_cross, outb_name=clkb_name)
        if not np.any(last_cross):
            raise RuntimeError("Too few valid crossings found. Extend sim time.")
        fall_cross = self.calc_cross(data, sig_name, EdgeType.FALL, t_start=last_cross, outb_name=clkb_name)
        high_time = fall_cross - last_cross
        period = curr_cross - last_cross
        return high_time / period
    
    def get_last_crossing(self, data: SimData, tbm: DigitalTranTB, sig_name: str, out_edge: EdgeType,
                       t_start: float = 0, clkb_name: str = None):
        """Find the last valid crossing"""
        last_cross = None
        curr_cross = self.calc_cross(data, sig_name, out_edge, t_start=t_start, outb_name=clkb_name)
        while np.all(np.isfinite(curr_cross)):
            last_cross = curr_cross
            curr_cross = self.calc_cross(data, sig_name, out_edge, t_start=curr_cross, outb_name=clkb_name)
        if not np.any(last_cross):
            raise RuntimeError("Too few valid crossings found. Extend sim time.")
        return last_cross
    
    def measure_ph_diff(self, data: SimData, s0: str, s1: str, in_edge: EdgeType, out_edge: EdgeType, t_start: float = 0):
        """Determine phase difference using last crossings"""
        c0 = self.get_last_crossing(data, None, s0, in_edge, t_start=t_start)
        c1 = self.get_last_crossing(data, None, s1, out_edge, t_start=t_start)
        return c1 - c0


def moving_average(a, n=5):
    """Performs a simple moving average
    Returns a vector that is `n-1` shorter than `a`
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def _settle_check(vec: np.ndarray, tol = 5e-4) -> int:
    """Returns index where the vector has 'settled'
    The assumption is the information has some transient (voltage, phase) that settle
        in an exponential fashion.
    Default tolerance is such that we have better than 10b error
        (2^-10 ~ 1e-3).
    """
    mva = moving_average(vec)
    metric = np.abs(np.diff(mva) / mva[:-1])
    idx_where = np.argwhere(metric < tol)
    if not idx_where.any():
        raise RuntimeError("Waveform not settled!")
    return idx_where[0][0]

def _swing_check(vec: np.ndarray, tol = 50e-3) -> bool:
    """Checks whether the signal acheives the minimum swing.
    Default tolerance is a (very) lower bound for V*.
    Returns True if OK. Errors if swing is not met.
    """
    _max = np.max(vec)
    _min = np.min(vec)
    if _max - _min  < tol:
        raise RuntimeError("Waveform has insufficient swing!")
    return True


def measure_amplitude(data: SimData, sig_name: str, t_start: float = 0, clk_name: str = ''):
    """Compute max - min. Handles multi-dimensional based on get_first_crossing"""
    tvec = data['time']
    yvec = data[sig_name]
    swp_shape = yvec.shape[:-1]
    shape = swp_shape
    
    t_shape = tvec.shape
    nlast = t_shape[len(t_shape) - 1]

    yvec = yvec.reshape(-1, nlast)
    tvec = tvec.reshape(-1, nlast)

    _tmp = np.zeros(swp_shape).flatten()
    n_swp = _tmp.size
    ans = np.empty(n_swp)

    idx_start = np.argwhere(tvec[0] >= t_start)[0][0]

    for idx in range(n_swp):
        wave = yvec[idx][idx_start:]
        wave_min, wave_max = np.nanmin(wave), np.nanmax(wave)
        wave_amp = wave_max - wave_min
        ans[idx] = wave_amp
    return ans.reshape(shape)

def measure_mean(data: SimData, sig_name: str, t_start: float = 0):
    """Compute mean. Handles multi-dimensional based on get_first_crossing"""
    tvec = data['time']
    yvec = data[sig_name]
    swp_shape = yvec.shape[:-1]
    shape = swp_shape
    
    t_shape = tvec.shape
    nlast = t_shape[len(t_shape) - 1]

    yvec = yvec.reshape(-1, nlast)
    tvec = tvec.reshape(-1, nlast)

    _tmp = np.zeros(swp_shape).flatten()
    n_swp = _tmp.size
    ans = np.empty(n_swp)

    idx_start = np.argwhere(tvec[0] >= t_start)[0][0]

    for idx in range(n_swp):
        wave = yvec[idx][idx_start:]
        wave_mean = np.nanmean(wave)
        ans[idx] = wave_mean
    return ans.reshape(shape)
