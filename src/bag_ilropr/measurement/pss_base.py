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


from typing import Any, Optional, Mapping, List, Union, Tuple

from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict

from bag3_testbenches.measurement.base import GenericTB

# TODO: merge with bag3_testbenches

class PSSSampledTB(GenericTB):
    """This class provide utility methods useful for all PSS/Pnoise simulations.
    This modifies the PSSTB from bag3_testbenches, to default using sampled jitter events

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    p_port/ n_port : Optional[str]/ Optional[str]
        Optional.  p_port and n_port in autonomous circuit (e.g. free-running VCO)
    fund: Union[float, str]
        Steady state analysis fundamental frequency (or its estimate for autonomous circuits).
    pss_options : Mapping[str, Any]
        Optional.  PSS simulation options dictionary. (spectre -h pss to see available parameters)
    pnoise_probe: Optional[List[Tuple]]
        Optional. A list of pnoise measurment (p_port, n_port, relative harmonic number)
    pnoise_options: Optional[Mapping[str, Any]]
        Optional.  PSS simulation options dictionary. (spectre -h pnoise to see available parameters)
    """

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        # autonomous
        p_port: Optional[str] = specs.get('p_port', None)
        n_port: Optional[str] = specs.get('n_port', None)
        # driven
        fund: Union[float, str] = specs['fund']
        harms: bool = specs['pss_options']['harms']
        pss_default = dict(
            period=1/fund,
            skipdc='yes',
            harms=10,
            saveinit='yes',
            tstab=10e-9,
            errpreset='conservative',
            maxstep=2e-12,
        )
        pnoise_default = dict(
            sweeptype='relative',
            start=1000,
            stop=1e7,
            dec=20,
            maxsideband=harms,
            noiseout='pm',
        )
        pss_options: Mapping[str, Any] = specs.get('pss_options', pss_default)
        pnoise_options: Mapping[str, Any] = specs.get('pnoise_options', pnoise_default)
        pnoise_probe: Optional[List[Tuple]] = specs.get('pnoise_probe', None)
        prio_pss = {1: pss_options, 2: pss_default}
        prio_pnoise = {1: pnoise_options, 2: pnoise_default}

        pss_dict = dict(type='PSS',
                        p_port=p_port,
                        n_port=n_port,
                        fund=fund,
                        options={**prio_pss[2], **prio_pss[1]},
                        save_outputs=self.save_outputs,
                        )
        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [pss_dict]
        sim_params = specs['sim_params']
        default_freq_sweep = dict(
            type = 'LOG',
            start = 1e3,
            stop = 1e11,
            num = 220
        )
        if pnoise_probe:
            for probe in pnoise_probe:
                p_port_noise, n_port_noise, relharmnum = probe
                vtrig = sim_params.get('vtrig', 0)  # Trigger voltage for sampling event. 0 for differential
                freq_sweep_noise: Mapping[str, Any] = specs.get('freq_sweep_noise', default_freq_sweep)
                pnoise_meas = [dict(
                    trig_p=p_port_noise,
                    trig_n=n_port_noise,
                    triggerthresh=vtrig,
                    triggernum=1,
                    triggerdir='rise',
                    targ_p=p_port_noise,
                    targ_n=n_port_noise,
                )]
                pnoise_dict = dict(type='PNOISE',
                                   p_port=p_port_noise,
                                   n_port=n_port_noise,
                                   sweep=freq_sweep_noise,
                                   measurement=pnoise_meas,
                                   options={'save': 'lvl', 'nestlvl': 1, **prio_pnoise[2], **prio_pnoise[1]},
                                   save_outputs=[],  # Remove save=selected statement
                                   )
                sim_setup['analyses'].append(pnoise_dict)
        return netlist_info_from_dict(sim_setup)



class PACTB(GenericTB):
    """This class provide utility methods useful for all PSS/Pnoise simulations.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    p_port/ n_port : Optional[str]/ Optional[str]
        Optional.  p_port and n_port in autonomous circuit (e.g. free-running VCO)
    fund: Union[float, str]
        Steady state analysis fundamental frequency (or its estimate for autonomous circuits).
    pss_options : Mapping[str, Any]
        Optional.  PSS simulation options dictionary. (spectre -h pss to see available parameters)
    pnoise_probe: Optional[List[Tuple]]
        Optional. A list of pnoise measurment (p_port, n_port, relative harmonic number)
    pnoise_options: Optional[Mapping[str, Any]]
        Optional.  PSS simulation options dictionary. (spectre -h pnoise to see available parameters)
    """

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        # autonomous
        p_port: Optional[str] = specs.get('p_port', None)
        n_port: Optional[str] = specs.get('n_port', None)
        # driven
        fund: Union[float, str] = specs['fund']
        harms: bool = specs['pss_options']['harms']
        pss_default = dict(
            period=1/fund,
            skipdc='yes',
            harms=10,
            saveinit='yes',
            tstab=10e-9,
            errpreset='conservative',
            maxstep=2e-12,
        )
        pac_default = dict(
            sweeptype='relative',
            start=1000,
            stop=1e7,
            dec=20,
            maxsideband=harms,
        )
        pss_options: Mapping[str, Any] = specs.get('pss_options', pss_default)
        pac_options: Mapping[str, Any] = specs.get('pac_options', pac_default)
        prio_pss = {1: pss_options, 2: pss_default}
        prio_ac = {1: pac_options, 2: pac_default}
        print(pac_options)

        pss_dict = dict(type='PSS',
                        p_port=p_port,
                        n_port=n_port,
                        fund=fund,
                        options={**prio_pss[2], **prio_pss[1]},
                        save_outputs=self.save_outputs,
                        )
        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [pss_dict]
        pnoise_dict = dict(type='PAC',
                            freq=fund,
                            options={**prio_ac[2], **prio_ac[1]},
                            )
        sim_setup['analyses'].append(pnoise_dict)

        return netlist_info_from_dict(sim_setup)

