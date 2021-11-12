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

""" This module contains top v1. Vertically compact with a horiztonal array """

from typing import Any, Dict, Type, Optional, Mapping, Union, List, Tuple

from pybag.enum import Direction, Orient2D, Orientation, PinMode, RoundMode, MinLenMode, Direction2D
from pybag.core import Transform, BBox

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing.base import TrackID, TrackManager, WDictType, SpDictType, WireArray
from bag.design.module import Module

from .util import import_params, get_closest_warr
from .in_buff import InBuff
from .core_out import CoreOutDAC
from ..schematic.top_v1 import bag_ilropr__top_v1


class TopV1(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
    
    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__top_v1

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tr_widths='Track width dictionary for TrackManager',
            tr_spaces='Track spaces dictionary for TrackManager',
            in_buff_params='Params for InBuff, either Mapping or YAML string',
            core_out_params="Params for CoreOutDAC, either Mapping or YAML string",
            inst_pullback="[Optional] Pullback (negative x-margin) between instances, to improve routing compactness",
            en_pin_ext="[Optional] Enable pin extension beyond PR boundary, for ease of routing with PNR",
            en_pin_info="[Optional] Dimensions for custom enable routing (pitch, width, via ext). If not given, uses BAG's grid",
            sup_x_margin="[Optional] X margin for supply fill",
            sup_y_margin="[Optional] Y margin for supply fill",
        )
    
    @classmethod
    def get_default_param_values(cls) ->  Dict[str, Any]:
        return dict(inst_pullback=0, en_pin_ext=0, en_pin_info=None, sup_x_margin=1000, sup_y_margin=1000)

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager
        num_stages = 4  # TODO: parametrize

        in_buff_params: Mapping = import_params(self.params['in_buff_params'])
        core_out_params: Mapping = import_params(self.params['core_out_params'])

        # =============== Templates and placement ===============
        buff_template = self.new_template(InBuff, params=in_buff_params)
        core_template = self.new_template(CoreOutDAC, params=core_out_params)

        # TODO: optimize y-placement
        buff_inst = self.add_instance(buff_template)

        # position xloc to have enough room to route en signals        
        upper = -1
        core_pin_names = core_template.port_names_iter()
        for pin in core_pin_names:
            if 'en' in pin:
                pin_tid = core_template.get_port(pin).get_pins()[0].track_id
                upper = max(upper, self.grid.track_to_coord(pin_tid.layer_id, pin_tid.base_index))
        
        en_layer_id = core_template.get_port('enb<0>').get_pins()[0].track_id.layer_id + 1
        fill_width = tr_manager.get_width(en_layer_id, 'sig')
        fill_space = tr_manager.get_sep(en_layer_id, ('sig', 'sig'))
        sep_margin = tr_manager.get_sep(en_layer_id, ('sig', ''))
        
        num_en = core_out_params['idac_params']['num_en']
        # Pull back margin to take advantage of routing availability in the edges
        pullback = self.params['inst_pullback']
        tidx_lo = self.grid.coord_to_track(en_layer_id, buff_template.bound_box.xh - pullback, RoundMode.GREATER_EQ)
        tidx_hi = tidx_lo + (fill_width + fill_space - 1) * num_en

        trs = self.get_available_tracks(en_layer_id, tidx_lo, tidx_hi, 0, upper, fill_width, 
                                        fill_space, include_last=True, sep_margin=sep_margin)
        xloc = self.grid.track_to_coord(en_layer_id, trs[-1]) - pullback
        
        core_inst = self.add_instance(core_template, xform=Transform(xloc, 0))
        track_widths = self.grid.track_to_coord(en_layer_id, trs[-1]) - self.grid.track_to_coord(en_layer_id, trs[0])
        
        top_layer = max(buff_template.top_layer, core_template.top_layer)
        h_tot = max(core_inst.bound_box.yh, buff_inst.bound_box.yh)
        w_tot = core_inst.bound_box.xh + track_widths
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot), round_up=True)

        # ========================== Routing ==========================
        # High speed signals
        # Injection. Signals are on xxm_layer. Route on x3m_layer
        buff_injp, buff_injm = buff_inst.get_pin('outp'), buff_inst.get_pin('outm')
        xxm_layer = buff_injp.layer_id
        yym_layer = xxm_layer + 1
        x3m_layer = yym_layer + 1
        w_sig_hs_yym = tr_manager.get_width(yym_layer, 'sig_hs')
        ref_tidx = self.grid.coord_to_track(yym_layer, buff_inst.bound_box.xh, RoundMode.LESS)
        _, tidx_list = tr_manager.place_wires(yym_layer, ['sig_hs', 'sig_hs'], align_track=ref_tidx, align_idx=-1)
        buff_injp, buff_injm = self.connect_differential_tracks(buff_injp, buff_injm, yym_layer, tidx_list[0],
                                                                tidx_list[1], width=w_sig_hs_yym)

        core_injp, core_injm = core_inst.get_pin('injp'), core_inst.get_pin('injm')
        _, tidx_list = tr_manager.place_wires(yym_layer, ['sig_hs', 'sig_hs'], center_coord=core_injp.middle)
        core_injp, core_injm = self.connect_differential_tracks(core_injp, core_injm, yym_layer, tidx_list[0],
                                                                tidx_list[1], width=w_sig_hs_yym)

        ref_tidx = core_inst.get_pin('injm').track_id.base_index
        ptidx = tr_manager.get_next_track(x3m_layer, ref_tidx, 'sig_hs', 'sig_hs', up=False)
        mtidx = tr_manager.get_next_track(x3m_layer, ptidx, 'sig_hs', 'sig_hs', up=False)
        w_sig_hs_x3m = tr_manager.get_width(x3m_layer, 'sig_hs')
        injp, injm = self.connect_differential_tracks(buff_injp, buff_injm, x3m_layer,
                                                      ptidx, mtidx, width=w_sig_hs_x3m)
        self.connect_differential_wires(core_injp, core_injm, injp, injm)
        
        # Other buffer signals
        self.reexport(buff_inst.get_port('bias_out'), net_name='VCM')  # Name based on schematics
        self.add_pin('injpb', buff_injp)
        self.add_pin('injmb', buff_injm)
        
        self.reexport(buff_inst.get_port('inp'), net_name='injp')
        self.reexport(buff_inst.get_port('inm'), net_name='injm')

        self.reexport(buff_inst.get_port('inbp'))
        self.reexport(buff_inst.get_port('inbm'))

        # Core high speed signals
        num_out = 2 * num_stages
        for sig in ['mid']:
            for idx in range(num_out):
                self.reexport(core_inst.get_port(f'{sig}<{idx}>'))
        # bring outputs up to yym and to the top edge
        for sig in ['out']:
            for idx in range(num_out):
                pin_name = f'{sig}<{idx}>'
                out_pin = core_inst.get_pin(pin_name)
                if out_pin.layer_id < yym_layer:
                    out_pin = self.connect_via_stack(tr_manager, out_pin, yym_layer, 'sig_hs')
                out_pin = self.extend_wires(out_pin, upper=self.bound_box.yh)
                self.add_pin(pin_name, out_pin, mode=PinMode.UPPER)

        # Core enable signals
        vm_layer = en_layer_id
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        # Compute each group's track locations for enable
        vm_ym_trs = []
        # Group 0
        xloc = (core_inst.bound_box.xm - core_inst.bound_box.xl) // 2  + core_inst.bound_box.xl
        _, vm_tidx_list = tr_manager.place_wires(vm_layer, ['sig'] * num_en, center_coord=xloc)
        _, ym_tidx_list = tr_manager.place_wires(ym_layer, ['sig'] * num_en, center_coord=xloc)
        vm_ym_trs.append((vm_tidx_list, ym_tidx_list))

        # Group 1
        xloc = core_inst.bound_box.xh
        align_track = self.grid.coord_to_track(vm_layer, xloc, RoundMode.NEAREST)
        _, vm_tidx_list = tr_manager.place_wires(vm_layer, ['sig'] * num_en, align_idx=0, align_track=align_track)
        align_track = self.grid.coord_to_track(vm_layer, xloc, RoundMode.NEAREST)
        _, ym_tidx_list = tr_manager.place_wires(ym_layer, ['sig'] * num_en, align_idx=0, align_track=align_track)
        vm_ym_trs.append((vm_tidx_list, ym_tidx_list))

        # Group 2
        xloc = (core_inst.bound_box.xm - core_inst.bound_box.xl) // 2  + core_inst.bound_box.xm
        _, vm_tidx_list = tr_manager.place_wires(vm_layer, ['sig'] * num_en, center_coord=xloc)
        _, ym_tidx_list = tr_manager.place_wires(ym_layer, ['sig'] * num_en, center_coord=xloc)
        vm_ym_trs.append((vm_tidx_list, ym_tidx_list))
        
        # Group 3
        xloc = (core_inst.bound_box.xl - buff_inst.bound_box.xh) // 2  + buff_inst.bound_box.xh
        _, vm_tidx_list = tr_manager.place_wires(vm_layer, ['sig'] * num_en, center_coord=xloc)
        _, ym_tidx_list = tr_manager.place_wires(ym_layer, ['sig'] * num_en, center_coord=xloc)
        vm_ym_trs.append((vm_tidx_list, ym_tidx_list))
        
        # Connect enable pins
        w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        w_sig_ym = tr_manager.get_width(ym_layer, 'sig')
        num_en_2 = num_en // 2
        
        en_pin_ext: int = self.params['en_pin_ext']
        en_pin_info: Optional[Tuple] = self.params['en_pin_info']
        if not en_pin_info:
            # Default method: connect to placed wires and extend
            for dac_idx, (vm_tidx_list, ym_tidx_list) in enumerate(vm_ym_trs):
                en_idx_list = list(range(dac_idx * num_en_2, (dac_idx + 1) * num_en_2)) + \
                    list(range((num_stages + dac_idx) * num_en_2, (num_stages + dac_idx + 1) * num_en_2))
                lower = self.bound_box.xl - en_pin_ext
                for idx in range(num_en):
                    en_idx = en_idx_list[idx]
                    vm_tid = TrackID(vm_layer, vm_tidx_list[idx], w_sig_vm)
                    ym_tid = TrackID(ym_layer, ym_tidx_list[idx], w_sig_ym)
                    enb = self.connect_to_tracks(core_inst.get_pin(f'enb<{en_idx}>'), vm_tid, track_lower=lower)
                    en = self.connect_to_tracks(core_inst.get_pin(f'en<{en_idx}>'), ym_tid, track_lower=lower)
                    self.add_pin(f'enb<{en_idx}>', enb, mode=PinMode.LOWER)
                    self.add_pin(f'en<{en_idx}>', en, mode=PinMode.LOWER)
        else:
            # Alt method: use BBoxs to align to PNR grid, for easier abutting
            _vm_pitch, _vm_width, _via_ext = en_pin_info
            if not en_pin_ext: # Add a default to avoid no pin getting drawn.
                en_pin_ext = _vm_pitch

            vm_lp = self.grid.tech_info.get_lay_purp_list(vm_layer)[0]
            ym_lp = self.grid.tech_info.get_lay_purp_list(ym_layer)[0]
            
            for dac_idx in range(num_stages):
                en_idx_list = list(range(dac_idx * num_en_2, (dac_idx + 1) * num_en_2)) + \
                    list(range((num_stages + dac_idx) * num_en_2, (num_stages + dac_idx + 1) * num_en_2))
                lower = self.bound_box.xl - en_pin_ext
                ref_tidx = vm_ym_trs[dac_idx][1][0]
                xloc_init = -(-self.grid.track_to_coord(ym_layer, ref_tidx) // _vm_pitch) * _vm_pitch
                xloc = xloc_init
                for idx in range(num_en):
                    en_idx = en_idx_list[idx]
                    enb = core_inst.get_pin(f'enb<{en_idx}>')
                    en = core_inst.get_pin(f'en<{en_idx}>')
                    upper = max(self.grid.track_to_coord(en.layer_id, en.track_id.base_index),
                                self.grid.track_to_coord(enb.layer_id, enb.track_id.base_index))
                    upper += _via_ext
                    bbox = BBox(xloc - _vm_width // 2,  lower, xloc + _vm_width // 2, upper)
                    enb = self.connect_bbox_to_track_wires(Direction.UPPER, vm_lp, bbox, enb)
                    en = self.connect_bbox_to_track_wires(Direction.UPPER, ym_lp, bbox, en)
                    pin_bbox = BBox(xloc - _vm_width // 2,  lower, xloc + _vm_width // 2, lower + en_pin_ext)
                    self.add_pin_primitive(f'enb<{en_idx}>', vm_lp[0], pin_bbox)
                    self.add_pin_primitive(f'en<{en_idx}>', ym_lp[0], pin_bbox)
                    xloc += _vm_pitch

        # Current DAC bias
        buff_bias = buff_inst.get_pin('NBIAS')
        core_bias = core_inst.get_pin('NBIAS', layer=yym_layer)
        yym_layer = core_bias.layer_id
        x3m_layer = yym_layer + 1
        loc = 0
        tidx = self.grid.coord_to_track(x3m_layer, loc, RoundMode.GREATER_EQ)
        tid = TrackID(x3m_layer, tidx, tr_manager.get_width(x3m_layer, 'sup'))
        nbias = self.connect_to_tracks([buff_bias, core_bias], tid)
        self.add_pin("NBIAS", nbias, mode=PinMode.MIDDLE)

        # Supplies
        vdd_xxm, vss_xxm = [], []
        for inst in [core_inst, buff_inst]:
            for sup, warr_list in [('VDD', vdd_xxm), ('VSS', vss_xxm)]:
                if inst.has_port(sup):
                    warr_list.extend(inst.get_all_port_pins(sup, xxm_layer))

        # Not many yym, x3m. Just do power fill
        top_vdd, top_vss = vdd_xxm, vss_xxm
        vdd_yym, vss_yym = [], []
        x_margin = self.params['sup_x_margin']
        y_margin = self.params['sup_y_margin']
        for _idx, _bbox in enumerate([buff_inst.bound_box, \
                     core_template.c2c_bbox.get_move_by(dx=xloc), \
                     core_template.else_bbox.get_move_by(dx=xloc)]):
            _vdd, _vss = self.do_power_fill(yym_layer, tr_manager, top_vdd, top_vss, bound_box=_bbox, 
                                            x_margin=x_margin, y_margin=y_margin)
            # Extra help to connect to the buffer instance
            if _idx == 0:
                # For each xm warr, find the closest warr in the list, and connect
                for _warr in buff_inst.get_all_port_pins('VDD', xxm_layer):
                    closest_warr = get_closest_warr(self.grid, _vdd, _warr)
                    self.connect_to_track_wires(closest_warr, _warr)
                for _warr in buff_inst.get_all_port_pins('VSS', xxm_layer):
                    closest_warr = get_closest_warr(self.grid, _vss, _warr)
                    self.connect_to_track_wires(closest_warr, _warr)
            vdd_yym.extend(_vdd)
            vss_yym.extend(_vss)
        
        top_vdd, top_vss = self.do_power_fill(x3m_layer, tr_manager, vdd_yym, vss_yym)
        self.add_pin('VDD', top_vdd, connect=True)
        self.add_pin('VSS', top_vss, connect=True)

        # Sch_params 
        self.sch_params = dict(
            in_buff_params=buff_template.sch_params,
            core_out_params=core_template.sch_params,
        )
