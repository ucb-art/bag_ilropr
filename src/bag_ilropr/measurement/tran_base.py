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

from typing import Any, Union, Sequence, Tuple, Optional, Mapping, List, Dict, cast

from pathlib import Path
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.data import SimData, AnalysisType
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.base import TranTB


class PhaseRotatorTranMM(MeasurementManager):

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo,
                     harnesses: Optional[Sequence[DesignInstance]] = None):
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
        # This is a single code measurement, so no sweeping

        code: Mapping = self.specs['code']
        code_fmt: Mapping = self.specs['code_fmt']
        in_clk_info: Mapping = self.specs['in_clk_info']
        out_clk_info: List[Mapping] = self.specs['out_clk_info']
        power_pins: List[str] = self.specs['power_pins']
        tbm_specs: Mapping = self.specs['tbm_specs']

        pin_values: Mapping[str, int] = code_to_pin_list(code, code_fmt)
        src_list, save_outputs, load_list = self.get_in_outs(in_clk_info, out_clk_info, power_pins)
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
            load_list=load_list
        )
        tbm = cast(TranTB, self.make_tbm(TranTB, tbm_specs))
        # TODO: hack
        self.tbm = tbm
        tbm_name = name
        tb_params = dict()
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir / tbm_name, dut, tbm,
                                                          tb_params=tb_params)

        # TODO: sort out corners

        # Collect data and report results
        data: SimData = sim_results.data
        cross_dict: Dict[str, np.ndarray] = {}  # all the crossing times
        if 'diff' in in_clk_info:
            tmp = self.calc_cross(data, in_clk_info['name'], EdgeType.RISE, in_clk_info['diff'])
        else:
            tmp = self.calc_cross(data, in_clk_info['name'], EdgeType.RISE)
        cross_dict[in_clk_info['name']] = tmp
        for info in out_clk_info:
            if 'diff' in info:
                tmp = self.calc_cross(data, info['name'], EdgeType.RISE, info['diff'])
            else:
                tmp = self.calc_cross(data, info['name'], EdgeType.RISE)
            cross_dict[info['name']] = tmp

        period_dict: Dict[str, np.ndarray] = {}  # all the periods. first difference of crossings
        jee_dict: Dict[str, float] = {}  # abs jitter. std of transition
        jc_dict: Dict[str, float] = {}  # cycle jitter. std of period
        jcc_dict: Dict[str, float] = {}  # c2c jitter. std of period difference
        for clk_name, cross_times in cross_dict.items():
            period_list = cross_times[1:] - cross_times[:-1]
            period_dict[clk_name] = period_list

            avg_period = np.mean(period_list)
            sample_times = np.linspace(0, avg_period * (len(cross_times) - 1), num=len(cross_times))
            samp_offset = cross_times - sample_times
            jee_dict[clk_name] = np.std(samp_offset)

            jc_dict[clk_name] = np.std(period_list)
            jcc_dict[clk_name] = np.std(period_list[1:] - period_list[:-1])

        current_dict: Dict[str, float] = {}  # average current for each power pin
        for ppin in power_pins:
            res = data['XDUT:' + ppin]
            current_dict[ppin] = np.mean(res)

        if self.specs.get('plot'):
            plt.subplot(121)
            for name in cross_dict:
                plt.plot(data['time'], data[name][0], label=name)
            plt.legend()
            plt.xlabel("time [s]")
            plt.ylabel("signal [v]")
            plt.title("Signal vs time")
            plt.grid()

            plt.subplot(122)
            for name in cross_dict:
                plt.plot(cross_dict[name][:-1], period_dict[name], 'x-', label=name)
            plt.legend()
            plt.xlabel("time [s]")
            plt.ylabel("period [s]")
            plt.title("Period vs time")
            plt.grid()
            plt.show()
            breakpoint()

        ans = dict(
            data=data,
            cross_dict=cross_dict,
            period_dict=period_dict,
            jee=jee_dict,
            jc=jc_dict,
            jcc=jcc_dict,
            current=current_dict,
        )
        return ans

    @classmethod
    def get_in_outs(cls, in_clk_info, out_clk_info, power_pins) -> Tuple[List, List, List]:
        in_name = in_clk_info['name']
        in_diff = in_clk_info.get('diff', '')
        save_outputs = [in_name]
        if in_diff:
            save_outputs.append(in_diff)

        src_list = [
            dict(type='vpulse', value=dict(v1='v_VTOP', v2='v_VBOT', tr='t_clk_rf', tf='t_clk_rf',
                 td='t_clk_per/2', per='t_clk_per', pulse='t_clk_per / 2 - t_clk_rf'),
                 conns={'PLUS': in_name, 'MINUS': 'VSS'})
        ]
        if in_diff:
            src_list.append(
                dict(type='vpulse', value=dict(v1='v_VBOT', v2='v_VTOP', tr='t_clk_rf', tf='t_clk_rf',
                     td='t_clk_per/2', per='t_clk_per', pw='t_clk_per / 2 - t_clk_rf'),
                     conns={'PLUS': in_diff, 'MINUS': 'VSS'})
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

    def calc_cross(self, data: SimData, out_name: str, out_edge: EdgeType,
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
            vth_out = (out_1 - out_0) * thres_delay + out_0

        # We know that at most, this measurement sweep will be just for the environments
        # TODO: flush this out better for corners
        out_c = get_all_crossings(tvec, out_vec, vth_out, etype=out_edge, start=t_start,
                                  stop=t_stop, rtol=rtol, atol=atol)
        return out_c


def get_all_crossings(tvec: np.ndarray, yvec: np.ndarray, threshold: float,
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
    if EdgeType.RISE in etype:
        sel_mask = np.maximum(dvec, 0)
        ans = np.append(ans, tvec[np.argwhere(sel_mask) + sidx])
    if EdgeType.FALL in etype:
        sel_mask = np.minimum(dvec, 0)
        ans = np.append(ans, tvec[np.argwhere(sel_mask) + sidx])
    return ans


class SweepSpecType(Enum):
    BIN = 0
    ONEHOT = 1
    NHOT = 2

# TODO: flush out for binary, 1-hot, and other fmts


def code_to_pin_list(code: Mapping, code_fmt: Mapping) -> Mapping[str, Union[int, str]]:
    pin_name: str = code['name']
    pin_value: int = code['value']
    pin_diff: str = code.get('diff', '')
    ans = {}

    if code_fmt['type'] == 'NHOT':
        num_hot: int = code_fmt['num']
        width: int = code_fmt['width']
        if num_hot > width:
            raise ValueError("num_hot cannot be greater than width")

        # Special case for disable
        if pin_value < 0:
            for idx in range(width):
                ans[f'{pin_name}<{idx}>'] = 0
                if pin_diff:
                    ans[f'{pin_diff}<{idx}>'] = 1
        else:
            for idx in range(width):
                hot = pin_value <= idx < pin_value + num_hot
                # Wrap around:
                if pin_value + num_hot >= width:
                    hot = hot or pin_value <= idx + width < pin_value + num_hot
                ans[f'{pin_name}<{idx}>'] = 1 if hot else 0
                if pin_diff:
                    ans[f'{pin_diff}<{idx}>'] = 0 if hot else 1
    else:
        raise RuntimeError("Code format currently not supported")

    return ans
