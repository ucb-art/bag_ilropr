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

""" This module contains layout classes for the core + c2c + IDACs. """

from typing import Any, Dict, Type, Optional, Mapping

from pybag.enum import Orient2D, Orientation, PinMode, RoundMode, MinLenMode
from pybag.core import Transform, BBox

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing.base import TrackID, TrackManager, WDictType, SpDictType
from bag.design.module import Module

from xbase.layout.mos.top import GenericWrapper

from .core_array import CoreArray
from .c2c import CML2CMOS_Diff
from .idac_array import IDACArray
from ..schematic.core_out_dac import bag_ilropr__core_out_dac


class CoreOutDAC(TemplateBase):
    """Oscillator core, DACs, and the output C2Cs"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
    
    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__core_out_dac

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tr_widths='Track width dictionary for TrackManager',
            tr_spaces='Track spaces dictionary for TrackManager',
            core_params='Oscillator core params',
            c2c_params="C2C unit params",
            idac_params="IDAC params.",
        )
    
    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict()

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager
        num_stages = 4  # TODO: parametrize

        idac_params = self.params['idac_params']
        # Use 2 IDACs arrays to spread the area
        assert idac_params['num_stages'] == num_stages // 2, "Design intent"

        # =============== Templates and placement ===============
        core_template = self.new_template(CoreArray, params=self.params['core_params'])
        c2c_wrap_params = dict(cls_name=CML2CMOS_Diff.get_qualified_name(), 
                               params=self.params['c2c_params'], export_hidden=True)
        c2c_template = self.new_template(GenericWrapper, params=c2c_wrap_params)
        idac_wrap_params = dict(cls_name=IDACArray.get_qualified_name(), 
                                params=idac_params, export_hidden=True)
        idac_template = self.new_template(GenericWrapper, params=idac_wrap_params)

        idac_bbox = idac_template.bound_box
        core_xloc = max(0, idac_bbox.xh - core_template.bound_box.xm)

        self.conn_layer = core_template.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
        yym_layer = xxm_layer + 1
        x3m_layer = yym_layer + 1

        if core_template.bound_box.xh // 2 > idac_bbox.xh:
            # Center the IDACs with their respective half
            idac0_xloc = max(0, core_template.bound_box.xm // 2 - idac_bbox.xm)
            idac1_xloc = idac0_xloc + core_template.bound_box.xm + idac_bbox.xh
        else:
            # Center 2 IDACs and core in the x direction
            idac0_xloc = max(0, core_template.bound_box.xm - idac_bbox.xh)
            idac1_xloc = idac0_xloc +  2 * idac_bbox.xh

        # Flip MY 2nd instance to keep signals to the outside
        idac0_inst = self.add_instance(idac_template, xform=Transform(idac0_xloc, idac_bbox.yh, Orientation.MX))
        idac1_inst = self.add_instance(idac_template, xform=Transform(idac1_xloc, idac_bbox.yh, Orientation.R180))

        # Flip MX to reduce signal routing to C2C
        yloc = idac_bbox.yh + core_template.bound_box.yh
        xp, yp = self.grid.get_size_pitch(x3m_layer)
        yloc = -(-yloc // yp) * yp
        core_inst = self.add_instance(core_template, xform=Transform(core_xloc, yloc, Orientation.MX))
        
        assert c2c_template.bound_box.xh < core_template.bound_box.xh // num_stages, \
            "assumption. resize or refactor"

        c2c_list = []
        tr_pitch = self.grid.get_track_pitch(ym_layer)
        for idx in range(num_stages):
            # Align with middle of xm pin. Should be center aligned
            cntr = core_inst.get_pin(f'V<{idx}>', layer=xm_layer).middle
            xloc = cntr - c2c_template.bound_box.xm
            # quantize track position
            xloc = xloc // tr_pitch * tr_pitch
            c2c_list.append(self.add_instance(c2c_template, xform=Transform(xloc, yloc)))
        
        h_tot = c2c_list[-1].bound_box.yh
        w_tot = max(core_template.bound_box.xh, idac_bbox.xh * 2)

        self.set_size_from_bound_box(x3m_layer, BBox(0, 0, w_tot, h_tot), round_up=True)

        # ========================== Routing ==========================
        # High speed signals
        w_sig_hs_xxm = tr_manager.get_width(xxm_layer, 'sig_hs')
        w_sig_hs_yym = tr_manager.get_width(yym_layer, 'sig_hs')
        for idx, inst in enumerate(c2c_list):
            # Get target inputs and outputs
            try:
                # Mid: connect on yym
                outp_pin = core_inst.get_pin(f'V<{idx}>', layer=xxm_layer)
                outm_pin = core_inst.get_pin(f'V<{idx + num_stages}>', layer=xxm_layer)

                inp_pin = inst.get_pin('inp', layer=xxm_layer)  # on xxm
                inm_pin = inst.get_pin('inm', layer=xxm_layer)  # on xxm
                
                _, tidx_list = tr_manager.place_wires(yym_layer, ['sig_hs'] * 2, center_coord=inst.bound_box.xm)
                tid_list = [TrackID(yym_layer, tidx, w_sig_hs_yym) for tidx in tidx_list]
                self.connect_to_tracks([outp_pin, inp_pin], tid_list[0])
                self.connect_to_tracks([outm_pin, inm_pin], tid_list[1])

                # Out: ym -> yym
                outp = inst.get_pin('outp', layer=ym_layer)
                outm = inst.get_pin('outm', layer=ym_layer)
                assert outp.track_id.base_index >= outm.track_id.base_index

                ref_tidx = inst.get_all_port_pins('VSS', xxm_layer)[0][-1].track_id.base_index
                tidx = tr_manager.get_next_track(xxm_layer, ref_tidx, 'sup', 'sig_hs', up=-1)
                tidx2 = tr_manager.get_next_track(xxm_layer, tidx, 'sig_hs', 'sig_hs', up=-1)
                outp, outm = self.connect_differential_tracks(outp, outm, xxm_layer, tidx, tidx2, width=w_sig_hs_xxm)
                outp, outm = self.connect_differential_tracks(outp, outm, yym_layer, tidx_list[1], tidx_list[0],
                                                              width=w_sig_hs_yym, track_upper=self.bound_box.yh)

                self.add_pin(f'out<{idx}>', outp, mode=PinMode.UPPER)
                self.add_pin(f'out<{idx + num_stages}>', outm, mode=PinMode.UPPER)

            except:
                print("CoreOutDAC: failed to connect c2c and core on xxm_layer. Trying ym_layer")
                outp_pin = core_inst.get_pin(f'V<{idx}>', layer=ym_layer)  # on ym
                outm_pin = core_inst.get_pin(f'V<{idx + num_stages}>', layer=ym_layer)  # on ym

                inp_pin = inst.get_pin('inp')  # on xm
                inm_pin = inst.get_pin('inm')  # on xm
                
                self.connect_to_track_wires(outp_pin, inp_pin)
                self.connect_to_track_wires(outm_pin, inm_pin)

                # Export outputs
                self.add_pin(f'out<{idx}>', inst.get_pin('outp', layer=ym_layer))
                self.add_pin(f'out<{idx + num_stages}>', inst.get_pin('outm', layer=ym_layer))

            # Debug pins
            self.add_pin(f'mid<{idx}>', inp_pin)
            self.add_pin(f'mid<{idx + num_stages}>', inm_pin)

        # Injection. Signals already on xxm_layer
        self.reexport(core_inst.get_port('injp'))
        self.reexport(core_inst.get_port('injm'))

        # Tails
        w_sup_yym = tr_manager.get_width(yym_layer, 'sup')
        _order = [(idac0_inst, 1), (idac1_inst, 0), (idac1_inst, 1), (idac0_inst, 0)]
        for core_idx, (idac_inst, idac_idx) in enumerate(_order):
            ref_pin = idac_inst.get_pin(f'tail_aa<{idac_idx}>')
            _check = core_idx == 0 or core_idx == 3  # Mirror check
            if _check:
                loc = ref_pin.lower if idac_idx else ref_pin.upper
                align_idx = 2 if idac_idx else 3
                order = ['tail_inj', 'tail_aa', 'tail_injb', 'taile_inj', 'tail_aa', 'taile_injb']
            else:
                loc = ref_pin.upper if idac_idx else ref_pin.lower
                align_idx = 3 if idac_idx else 2
                order = ['taile_injb', 'tail_aa', 'taile_inj', 'tail_injb', 'tail_aa',  'tail_inj']

            tidx = self.grid.coord_to_track(yym_layer, loc, RoundMode.NEAREST)
            _, tidx_list = tr_manager.place_wires(yym_layer, ['sup'] * 6, align_track=tidx, align_idx=align_idx)
            
            for tail_name, tidx in zip(order, tidx_list):
                core_warr = core_inst.get_pin(f'{tail_name}<{core_idx}>')
                idac_warr = idac_inst.get_pin(f'{tail_name}<{idac_idx}>')
                tid = TrackID(yym_layer, tidx, width=w_sup_yym)
                self.connect_to_tracks([core_warr, idac_warr], tid)

        # Reexport enable signals
        num_en = idac_params['num_en']
        num_inj = num_en // 2
        for core_idx, (idac_inst, idac_idx) in enumerate(_order):
            for en_idx in range(num_inj):
                self.reexport(idac_inst.get_port(f'en<{en_idx + idac_idx * num_inj}>'), net_name=f'en<{num_inj * core_idx + en_idx}>')
                self.reexport(idac_inst.get_port(f'enb<{en_idx + idac_idx * num_inj}>'), net_name=f'enb<{num_inj * core_idx + en_idx}>')

                self.reexport(idac_inst.get_port(f'en<{en_idx + idac_idx * num_inj + num_inj * num_stages // 2}>'), 
                              net_name=f'en<{num_inj * core_idx + en_idx + num_inj * num_stages}>')
                self.reexport(idac_inst.get_port(f'enb<{en_idx + idac_idx * num_inj + num_inj * num_stages // 2}>'), 
                              net_name=f'enb<{num_inj * core_idx + en_idx + num_inj * num_stages}>')

        # Bias signals
        cntr_coord = (idac0_inst.bound_box.xh + idac1_inst.bound_box.xl) // 2
        tidx = self.grid.coord_to_track(yym_layer, cntr_coord, RoundMode.NEAREST)
        tid = TrackID(yym_layer, tidx, w_sup_yym)
        nbias_xxm = [idac0_inst.get_pin('NBIAS'), idac1_inst.get_pin('NBIAS')]
        nbias = self.connect_to_tracks(nbias_xxm, tid, track_lower=self.bound_box.yl)
        self.add_pin("NBIAS", nbias, mode=PinMode.LOWER, connect=True)
        self.add_pin("NBIAS", nbias_xxm, connect=True)

        # Supplies
        # Supply pins should be aligned for the C2C. Connect them
        for lay in [xm_layer, xxm_layer]:
            for sup in ['VDD', 'VSS']:
                warrs = [inst.get_all_port_pins(sup, lay) for inst in c2c_list]
                # Magic syntax for flattening a list
                warrs = [pin for pin_list in warrs for pin in pin_list]
                warrs = self.connect_wires(warrs, lower=self.bound_box.xl, upper=self.bound_box.xh)
                self.add_pin(sup, warrs, connect=True)
        
        # supply connections should all be xxm already
        vdd_xxm, vss_xxm = [], []
        for inst_grp in [[core_inst], [idac0_inst, idac1_inst], c2c_list]:
            for sup, warr_list in [('VDD', vdd_xxm), ('VSS', vss_xxm)]:
                warrs = [inst.get_all_port_pins(sup, xxm_layer) for inst in inst_grp]
                if warrs:
                    # Magic syntax for flattening a list
                    warrs = [pin for pin_list in warrs for pin in pin_list]
                    warrs = self.connect_wires(warrs, lower=self.bound_box.xl, upper=self.bound_box.xh)
                    self.add_pin(sup, warrs, connect=True)
                    warr_list.extend(warrs)

        self.c2c_bbox = BBox(self.bound_box.xl, yloc, self.bound_box.xh, self.bound_box.yh)
        self.else_bbox = BBox(self.bound_box.xl, self.bound_box.yl, self.bound_box.xh, yloc)
        # Not many yym, x3m. Just do power fill
        top_vdd, top_vss = vdd_xxm, vss_xxm            
        self.add_pin('VDD', top_vdd, connect=True)
        self.add_pin('VSS', top_vss, connect=True)

        # Sch_params 
        self.sch_params = dict(
            core_params=core_template.sch_params,
            c2c_params=c2c_template.sch_params.copy(remove=['export_mid']),
            idac_params=idac_template.sch_params,
        )
