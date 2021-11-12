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

# Core array, routed as a row

from typing import Any, Dict, Type, Optional, Mapping

from pybag.enum import PinMode, RoundMode, MinLenMode
from pybag.core import Transform, BBox

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.core import PyLayInstance
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing.base import TrackID, TrackManager, WDictType, SpDictType
from bag.design.module import Module

from .core_unit import CoreUnit
from ..schematic.core import bag_ilropr__core
from .util import track_to_track


class CoreArray(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        self.conn_layer = -1
        self.unit_cell = CoreUnit
        self.inst_dict = None

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tr_widths='Track width dictionary for TrackManager',
            tr_spaces='Track spaces dictionary for TrackManager',
            core_unit_params='Core unit params',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict()

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager
        num_stages = 4  # TODO: parametrize

        # Create and place templates
        core_unit_params: Mapping[str, Any] = self.params['core_unit_params']
        
        cur_xloc=0
        inst_dict: Mapping[int, PyLayInstance] = {}
        self.inst_dict = inst_dict
        # placement order: 0 3 1 2
        for idx in range(num_stages):
            sig_locs = {'outp': 2*(idx % 2), 'outm': 2*(idx % 2) + 1}
            core_params_inst = core_unit_params.copy(append=dict(sig_locs=sig_locs))

            core_template = self.new_template(self.unit_cell, params=core_params_inst)
            core_bbox = core_template.bound_box

            _idx = (num_stages - idx // 2 - 1) if idx % 2 else idx // 2
            inst = self.add_instance(core_template, inst_name=f'XSTG{_idx}', 
                                     xform=Transform(dx=cur_xloc))
            inst_dict[_idx] = inst
            cur_xloc += core_bbox.xh
        
        # Set size
        self.conn_layer = core_template.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        h_tot = core_bbox.yh
        w_tot = cur_xloc
        self.set_size_from_bound_box(xxm_layer, BBox(0, 0, w_tot, h_tot), round_up=True)

        # Routing ====================================================

        # High speed ring signals
        for idx, inst in inst_dict.items():
            # Get target inputs and outputs
            outp_pin = inst.get_pin('outp', layer=xm_layer)  # on xm
            outm_pin = inst.get_pin('outm', layer=xm_layer)  # on xm
            in_inst = inst_dict[idx + 1]  if idx < num_stages - 1 else inst_dict[0]
            inp_pin = in_inst.get_pin('inp')  # on hm
            inm_pin = in_inst.get_pin('inm')  # on hm

            # Get vm locations
            # Strategy: route out on xm up to the edge of the input inst
            # Then route on vm tracks on the inner side area of the input inst
            # Assume we have the routing area
            # TODO: add check to verify tracks are available. Or modify the strategy

            if idx < num_stages // 2:
                coord = in_inst.bound_box.xl
                vm_tidx_ref = self.grid.coord_to_track(vm_layer, coord, mode=RoundMode.GREATER)
                vm_tidx_1 = tr_manager.get_next_track(vm_layer, vm_tidx_ref, 'sig_hs', 'sig_hs')
                vm_tidx_2 = tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig_hs', 'sig_hs')
            else:
                coord = in_inst.bound_box.xh
                vm_tidx_ref = self.grid.coord_to_track(vm_layer, coord, mode=RoundMode.LESS)
                vm_tidx_1 = tr_manager.get_next_track(vm_layer, vm_tidx_ref, 'sig_hs', 'sig_hs', up=-1)
                vm_tidx_2 = tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig_hs', 'sig_hs', up=-1)
            
            # Account for last stage swap
            _inp, _inm = (inm_pin, inp_pin) if idx == num_stages - 1 else (inp_pin, inm_pin)
            self.connect_differential_tracks([outp_pin, _inp], [outm_pin, _inm], vm_layer, vm_tidx_1, 
                                             vm_tidx_2, width=tr_manager.get_width(vm_layer, 'sig_hs'))

            # HACK: Extend routes for cap symmetriy
            if idx == num_stages // 2 - 1:
                point = in_inst.bound_box.xm
                self.extend_wires(outp_pin, upper=point)
                self.extend_wires(outm_pin, upper=point)
            if idx == num_stages - 1:
                point = in_inst.bound_box.xm
                self.extend_wires(outp_pin, lower=point)
                self.extend_wires(outm_pin, lower=point)

            # Export upper layer pins
            self.reexport(inst.get_port('outp'), net_name=f'V<{idx}>', connect=True)
            self.reexport(inst.get_port('outm'), net_name=f'V<{num_stages + idx}>', connect=True)

        # Injection
        # placement order: 0 3 1 2
        # Pins are on xm. Use H route strategy 
        # Connect xm pins to x3m
        injp_xm = [[inst_dict[0].get_pin('injp'), inst_dict[3].get_pin('injp')], 
                   [inst_dict[1].get_pin('injp'), inst_dict[2].get_pin('injp')]]
        injm_xm = [[inst_dict[0].get_pin('injm'), inst_dict[3].get_pin('injm')], 
                   [inst_dict[1].get_pin('injm'), inst_dict[2].get_pin('injm')]]
        xlocs = inst_dict[0].bound_box.xh, inst_dict[1].bound_box.xh
        ref_warr = inst_dict[0].get_pin('VSS', layer=xm_layer)[-1]
        yloc = self.grid.track_to_coord(xm_layer, ref_warr.track_id.base_index)
        x3m_layer = xm_layer + 4
        injp, injm = injp_xm, injm_xm
        for layer in range(ym_layer, x3m_layer + 1):
            w_sig_hs_mm = tr_manager.get_width(layer, 'sig_hs')
            if layer % 2:
                _, tidx_list0 = tr_manager.place_wires(layer, ['sig_hs', 'sup', 'sig_hs'], center_coord=xlocs[0])
                _, tidx_list1 = tr_manager.place_wires(layer, ['sig_hs', 'sup', 'sig_hs'], center_coord=xlocs[1])
            else:
                _, tidx_list0 = tr_manager.place_wires(layer, ['sig_hs', 'sup', 'sig_hs'], center_coord=yloc)
                tidx_list1 = tidx_list0
            injp[0], injm[0] = self.connect_differential_tracks(injp[0], injm[0], layer, tidx_list0[0],
                                                                tidx_list0[-1], width=w_sig_hs_mm)
            injp[1], injm[1] = self.connect_differential_tracks(injp[1], injm[1], layer, tidx_list1[0],
                                                                tidx_list1[-1], width=w_sig_hs_mm)
        injp = self.connect_wires(injp)[0]
        injm = self.connect_wires(injm)[0]
        self.add_pin('injp', injp, mode=PinMode.MIDDLE)
        self.add_pin('injm', injm, mode=PinMode.MIDDLE)

        # Supplies
        # Supply pins should be aligned on hm, xm, and xxm
        for sup_name in ['VDD', 'VSS']:
            for layer in [hm_layer, xm_layer, xxm_layer]:
                sup_warrs = [inst.get_all_port_pins(sup_name, layer) for inst in inst_dict.values()]
                # Magic syntax for flattening a list
                sup_warrs = [pin for pin_list in sup_warrs for pin in pin_list]
                sup_warrs = self.connect_wires(sup_warrs)
                self.add_pin(sup_name, sup_warrs, connect=True)
                
        # Expose Tails for higher level routing
        tail_names = ['tail_aa', 'tail_injb', 'tail_inj', 'taile_injb', 'taile_inj']
        for idx, inst in inst_dict.items():
            for tail_name in tail_names:
                self.reexport(inst.get_port(tail_name), net_name=f'{tail_name}<{idx}>')

        # Sch_params 
        self.sch_params = dict(
            num_stage=num_stages,
            stg_params=core_template.sch_params,
        )
