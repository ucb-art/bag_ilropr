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


from typing import Any, Dict, Type, Optional, Mapping
from bag.layout.core import PyLayInstance

from pybag.enum import RoundMode, MinLenMode, Orientation
from pybag.core import Transform, BBox

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing.base import TrackID, TrackManager, WDictType, SpDictType
from bag.design.module import Module

from xbase.layout.mos.top import GenericWrapper
from xbase.layout.array.top import ArrayBaseWrapper

from bag3_analog.layout.res.diff_res import DiffRes

from .util import track_to_track
from .diffpair import DiffPair, DiffPairBias
from ..schematic.buf_unit import bag_ilropr__buf_unit
from ..schematic.buf_core import bag_ilropr__buf_core


class ReplicaBias(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        self.conn_layer = -1
        

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__buf_unit

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tr_widths='Track width dictionary for TrackManager',
            tr_spaces='Track spaces dictionary for TrackManager',
            core_params='Core diff pair params',
            res_params='Differential resistor params',
            sig_locs='Signal locations for top horizontal metal layer pins'\
                     'This should contain str->int for the output pins, e.g {"outp": 0}',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(sig_locs={})

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager

        core_params: Mapping[str, Any] = self.params['core_params']
        res_params: Mapping[str, Any] = self.params['res_params']

        core_params = core_params.copy(append=dict(sig_locs=self.params['sig_locs']))

        # Make templates
        core_wrap_params = dict(cls_name=DiffPairBias.get_qualified_name(), 
                                params=core_params, export_hidden=True)
        core_template = self.new_template(GenericWrapper, params=core_wrap_params)
        res_wrap_params = dict(cls_name=DiffRes.get_qualified_name(), 
                               params=res_params, export_hidden=True)
        res_template = self.new_template(ArrayBaseWrapper, params=res_wrap_params)
        core_bbox = core_template.bound_box
        res_bbox = res_template.bound_box

        core_pinfo = core_template.core.place_info
        top_layer = core_pinfo.top_layer

        # Determine placement by aligning centers
        core_xloc = max(0, res_bbox.xm - core_bbox.xm)
        res_xloc = max(0, core_bbox.xm - res_bbox.xm)
        core_yloc = 0
        res_yloc = core_bbox.yh
        
        # Placement
        core_inst = self.add_instance(core_template, xform=Transform(core_xloc, core_yloc))
        res_inst = self.add_instance(res_template, xform=Transform(res_xloc, res_yloc))

        h_tot = res_inst.bound_box.yh
        w_tot = max(res_inst.bound_box.xh, core_inst.bound_box.xh)
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot), round_up=True)

        # ======== Routing ========
        self.conn_layer = core_template.core.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
        
        # Connect res to diff pair
        for res_out, core_out in [('rout', 'out'), ('rout_b', 'out')]:
            res_warr = res_inst.get_pin(res_out)
            if core_inst.get_all_port_pins(core_out, layer=ym_layer):
                core_warr = core_inst.get_all_port_pins(core_out, ym_layer)
                self.connect_to_track_wires(core_warr, res_warr)
            else:
                # Place as many wires as possible on ym
                core_warr = core_inst.get_pin(core_out, layer=xm_layer)
                lower_tidx = self.grid.coord_to_track(ym_layer, res_warr.lower, mode=RoundMode.GREATER_EQ)
                upper_tidx = self.grid.coord_to_track(ym_layer, res_warr.upper, mode=RoundMode.LESS_EQ)
                num_wires_btw = tr_manager.get_num_wires_between(
                    ym_layer, 'sup', lower_tidx, 'sup', upper_tidx, 'sup')

                num_wires = max(2, (2 + num_wires_btw) // 2)
                w_sup_ym = tr_manager.get_width(ym_layer, 'sup')
                tidx_list = tr_manager.spread_wires(ym_layer, ['sup'] * num_wires, 
                                                    lower_tidx, upper_tidx, ('sup', 'sup'))
                for tidx in tidx_list:
                    tid = TrackID(ym_layer, tidx, w_sup_ym)
                    self.connect_to_tracks([res_warr, core_warr], tid)     

        for pin_name in ['tail']:
            self.reexport(core_inst.get_port(pin_name))
        
        for pin_name in ['out']:
            self.add_pin(pin_name, core_inst.get_pin(pin_name, layer=xm_layer))

        # Diff pair should already have supply on xxm
        if core_inst.has_port('VSS'):
            self.reexport(core_inst.get_port('VSS'), connect=True)
        if core_inst.has_port('VDD'):
            self.reexport(core_inst.get_port('VDD'), connect=True)

        # Bring res pair supplies from xm to xxm
        # We expect only one of these, which provides both supply and well bias
        res_sup = res_template.sch_params['bias_node']
        warr = res_inst.get_all_port_pins(res_sup, xm_layer)
        bot_warr, top_warr = warr[0], warr[1]
        # Fully connect top rail
        ret_dict = {}
        top_xxm = self.connect_via_stack(tr_manager, top_warr, xxm_layer, 'sup', ret_warr_dict=ret_dict)
        # Only connect bottom ym on the ends to make room for routing
        warr = self.connect_wires([ret_dict[ym_layer][0], ret_dict[ym_layer][-1]])[0]
        bot_warr_ym = self.connect_to_track_wires(bot_warr, warr)
        tidx = track_to_track(self.grid, bot_warr, xm_layer, xxm_layer)
        w_sup_xxm = tr_manager.get_width(xxm_layer, 'sup')
        bot_xxm = self.connect_to_tracks(bot_warr_ym, TrackID(xxm_layer, tidx, w_sup_xxm))
        self.add_pin(res_sup, [bot_xxm, top_xxm], connect=True)

        # Sch_params 
        self.sch_params = dict(
            core_params=core_template.sch_params,
            res_params=res_template.sch_params,
            is_bias=True,
        )


class ReplicaUnit(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        self.conn_layer = -1        

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__buf_unit

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tr_widths='Track width dictionary for TrackManager',
            tr_spaces='Track spaces dictionary for TrackManager',
            core_params='Core diff pair params',
            res_params='Differential resistor params',
            sig_locs='Signal locations for top horizontal metal layer pins'\
                     'This should contain str->int for the output pins, e.g {"outp": 0}',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(sig_locs={})

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager

        core_params: Mapping[str, Any] = self.params['core_params']
        res_params: Mapping[str, Any] = self.params['res_params']

        core_params = core_params.copy(append=dict(sig_locs=self.params['sig_locs']))

        # Make templates
        core_wrap_params = dict(cls_name=DiffPair.get_qualified_name(), 
                                params=core_params, export_hidden=True)
        core_template = self.new_template(GenericWrapper, params=core_wrap_params)
        res_wrap_params = dict(cls_name=DiffRes.get_qualified_name(), 
                               params=res_params, export_hidden=True)
        res_template = self.new_template(ArrayBaseWrapper, params=res_wrap_params)
        core_bbox = core_template.bound_box
        res_bbox = res_template.bound_box

        core_pinfo = core_template.core.place_info
        top_layer = core_pinfo.top_layer

        # Determine placement by aligning centers
        core_xloc = max(0, res_bbox.xm - core_bbox.xm)
        res_xloc = max(0, core_bbox.xm - res_bbox.xm)
        core_yloc = 0
        res_yloc = core_bbox.yh
        
        # Placement
        core_inst = self.add_instance(core_template, inst_name='XDIFFPAIR',
                                      xform=Transform(core_xloc, core_yloc))
        res_inst = self.add_instance(res_template, inst_name='XDIFFRES',
                                     xform=Transform(res_xloc, res_yloc))

        h_tot = res_inst.bound_box.yh
        w_tot = max(res_inst.bound_box.xh, core_inst.bound_box.xh)
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot), round_up=True)

        # ======== Routing ========
        self.conn_layer = core_template.core.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        tail_xxm = self.connect_via_stack(tr_manager, core_inst.get_pin('tail', layer=xm_layer), xxm_layer, 'sup')
        self.add_pin('tail', tail_xxm)
        tidx = tail_xxm.track_id.base_index
        out_tidx0 = tr_manager.get_next_track(xxm_layer, tidx, 'sup', 'sig_hs')
        out_tidx1 = tr_manager.get_next_track(xxm_layer, out_tidx0, 'sup', 'sig_hs')

        # Connect res to diff pair
        w_sup_ym = tr_manager.get_width(ym_layer, 'sup')
        w_sig_hs_xxm = tr_manager.get_width(xxm_layer, 'sig_hs')
        out_ym = []
        for res_out, core_out in [('rout', 'outp'), ('rout_b', 'outm')]:
            res_warr = res_inst.get_pin(res_out)
            if core_inst.get_all_port_pins(core_out, layer=ym_layer):
                core_warr = core_inst.get_all_port_pins(core_out, ym_layer)
                self.connect_to_track_wires(core_warr, res_warr)
            else:
                # Place as many wires as possible on ym
                core_warr = core_inst.get_pin(core_out, layer=xm_layer)
                lower_tidx = self.grid.coord_to_track(ym_layer, res_warr.lower, mode=RoundMode.GREATER_EQ)
                upper_tidx = self.grid.coord_to_track(ym_layer, res_warr.upper, mode=RoundMode.LESS_EQ)
                num_wires_btw = tr_manager.get_num_wires_between(
                    ym_layer, 'sup', lower_tidx, 'sup', upper_tidx, 'sup')

                num_wires = max(2, (2 + num_wires_btw) // 2)
                tidx_list = tr_manager.spread_wires(ym_layer, ['sup'] * num_wires, 
                                                    lower_tidx, upper_tidx, ('sup', 'sup'))
                warr_list = []
                for tidx in tidx_list:
                    tid = TrackID(ym_layer, tidx, w_sup_ym)
                    warr_list.append(self.connect_to_tracks([res_warr, core_warr], tid))
                out_ym.append(self.connect_wires(warr_list))
                self.add_pin(core_out, warr_list, connect=True)

        # Add xxm_layer
        outp, outm = self.connect_differential_tracks(out_ym[0], out_ym[1], xxm_layer, out_tidx0,
                                                      out_tidx1, width=w_sig_hs_xxm)
        self.add_pin('outp', outp, connect=True)
        self.add_pin('outm', outm, connect=True)

        for pin_name in ['inp', 'inm']:
            self.reexport(core_inst.get_port(pin_name))
        
        for pin_name in ['outp', 'outm']:
            self.add_pin(pin_name, core_inst.get_pin(pin_name, layer=xm_layer), connect=True)

        # Diff pair should already have supply on xxm
        if core_inst.has_port('VSS'):
            self.reexport(core_inst.get_port('VSS'), connect=True)
        if core_inst.has_port('VDD'):
            self.reexport(core_inst.get_port('VDD'), connect=True)

        # Bring res pair supplies from xm to xxm
        # We expect only one of these, which provides both supply and well bias
        res_sup = res_template.sch_params['bias_node']
        warr = res_inst.get_all_port_pins(res_sup, xm_layer)
        bot_warr, top_warr = warr[0], warr[1]
        # Fully connect top rail
        ret_dict = {}
        top_xxm = self.connect_via_stack(tr_manager, top_warr, xxm_layer, 'sup', ret_warr_dict=ret_dict)
        # Only connect bottom ym on the ends to make room for routing
        warr = self.connect_wires([ret_dict[ym_layer][0], ret_dict[ym_layer][-1]])[0]
        bot_warr_ym = self.connect_to_track_wires(bot_warr, warr)
        tidx = track_to_track(self.grid, bot_warr, xm_layer, xxm_layer)
        w_sup_xxm = tr_manager.get_width(xxm_layer, 'sup')
        bot_xxm = self.connect_to_tracks(bot_warr_ym, TrackID(xxm_layer, tidx, w_sup_xxm))
        self.add_pin(res_sup, [bot_xxm, top_xxm], connect=True)

        # Sch_params 
        self.sch_params = dict(
            core_params=core_template.sch_params,
            res_params=res_template.sch_params,
        )


class ReplicaBuffer(TemplateBase):
    """Replicas of the core, used for input buffering"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        self.conn_layer = -1
        

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__buf_core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            num_stages="Number of input buffer stages",
            tr_widths='Track width dictionary for TrackManager',
            tr_spaces='Track spaces dictionary for TrackManager',
            core_unit_params='Core unit params',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(num_stages=2)

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager
        num_stages = self.params['num_stages']

        # Create and place templates
        core_unit_params: Mapping[str, Any] = self.params['core_unit_params']
        core_template = self.new_template(ReplicaUnit, params=core_unit_params)
        core_bbox = core_template.bound_box

        inst_dict: Mapping[int, PyLayInstance] = {}
        for idx in range(num_stages):
            dx =  idx // 2 * core_bbox.xh
            orient = Orientation.MX if idx % 2 else Orientation.R0
            inst_dict[idx] = self.add_instance(
                core_template, xform=Transform(dx=dx, dy=core_bbox.yh, mode=orient))
        
        top_layer = core_template.top_layer

        h_tot = core_bbox.yh * 2
        w_tot = core_bbox.xh * num_stages // 2
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot), round_up=True)
        
        # ======== Routing ========
        self.conn_layer = core_template.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        # High speed ring signals
        for idx, inst in inst_dict.items():
            if idx == num_stages - 1:
                continue
            
            # Get target inputs and outputs
            in_inst = inst_dict[idx + 1]
            outp_pin = inst.get_pin('outp', layer=xm_layer)  # on xm
            outm_pin = inst.get_pin('outm', layer=xm_layer)  # on xm
            inp_pin = in_inst.get_pin('inp')  # on hm
            inm_pin = in_inst.get_pin('inm')  # on hm

            # Get vm locations            
            # Place close to but outside the diff pair
            if idx % 2:
                coord = inst.get_pin('VSS', layer=xm_layer).lower
                vm_tidx_ref = self.grid.coord_to_track(vm_layer, coord, mode=RoundMode.NEAREST)
                vm_tidx_1 = tr_manager.get_next_track(vm_layer, vm_tidx_ref, 'sup', 'sig_hs', up=-2)
                vm_tidx_2 = tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig_hs', 'sig_hs', up=-1)
            else:
                coord = inst.get_pin('VSS', layer=xm_layer).upper
                vm_tidx_ref = self.grid.coord_to_track(vm_layer, coord, mode=RoundMode.NEAREST)
                vm_tidx_1 = tr_manager.get_next_track(vm_layer, vm_tidx_ref, 'sup', 'sig_hs', up=2)
                vm_tidx_2 = tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig_hs', 'sig_hs')
            
            self.connect_differential_tracks([outp_pin, inp_pin], [outm_pin, inm_pin], vm_layer, vm_tidx_1, 
                                             vm_tidx_2, width=tr_manager.get_width(vm_layer, 'sig_hs'))

        # Connect input to xm on left edge
        w_sig_hs_vm = tr_manager.get_width(vm_layer, 'sig_hs')
        w_sig_hs_xm = tr_manager.get_width(xm_layer, 'sig_hs')
        inp_hm, inm_hm = inst_dict[0].get_pin('inp'), inst_dict[0].get_pin('inm')
        cntr_coord = self.bound_box.xl
        _, tidx_list = tr_manager.place_wires(vm_layer, ['sig_hs', 'sup', 'sig_hs'], center_coord=cntr_coord)
        inp_vm, inm_vm = self.connect_differential_tracks(inp_hm, inm_hm, vm_layer, tidx_list[0],
                                                          tidx_list[-1], width=w_sig_hs_vm)
        cntr_coord = (inp_hm.bound_box.ym + inm_hm.bound_box.ym) // 2
        _, tidx_list = tr_manager.place_wires(xm_layer, ['sig_hs', 'sig_hs'], center_coord=cntr_coord)
        inp_xm, inm_xm = self.connect_differential_tracks(inp_vm, inm_vm, xm_layer, tidx_list[0],
                                                          tidx_list[-1], width=w_sig_hs_xm)
        self.add_pin('inp', inp_xm)
        self.add_pin('inm', inm_xm)

        # Output is already on xxm
        self.add_pin('outp', inst_dict[num_stages-1].get_pin('outp', layer=xxm_layer))
        self.add_pin('outm', inst_dict[num_stages-1].get_pin('outm', layer=xxm_layer))

        # Supplies
        # Supply pins should be aligned (separately at the top and bottom)
        vdd_warrs = [inst.get_all_port_pins('VDD', xm_layer) for inst in inst_dict.values()]
        # Magic syntax for flattening a list
        vdd_warrs = [pin for pin_list in vdd_warrs for pin in pin_list]
        self.connect_wires(vdd_warrs)
        vss_warrs = [inst.get_all_port_pins('VSS', xm_layer) for inst in inst_dict.values()]
        vss_warrs = [pin for pin_list in vss_warrs for pin in pin_list]
        self.connect_wires(vss_warrs)

        # Reexport pins
        for idx, inst in inst_dict.items():
            self.reexport(inst.get_port('VDD'), connect=True)
            self.reexport(inst.get_port('VSS'), connect=True)

        # Tails
        tail_name = 'tail'
        for idx, inst in inst_dict.items():
            self.reexport(inst.get_port(tail_name), net_name=f'{tail_name}<{idx}>')

        # Sch_params 
        self.sch_params = dict(
            num_stage=num_stages,
            stg_params=core_template.sch_params,
        )
