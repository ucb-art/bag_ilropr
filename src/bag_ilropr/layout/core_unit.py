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
# A diff pair unit and diff pair resistor

from typing import Any, Dict, Type, Optional, Mapping, cast

from pybag.enum import RoundMode, MinLenMode
from pybag.core import Transform, BBox

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing.base import TrackID, TrackManager, WDictType, SpDictType
from bag.design.module import Module

from xbase.layout.mos.top import GenericWrapper
from xbase.layout.array.top import ArrayBaseWrapper
from xbase.layout.res.base import ResBasePlaceInfo, ResArrayBase, ResTermType

from bag3_analog.layout.res.diff_res import DiffRes

from .diffpair import CoreDiffPairInj, CoreDiffPairInjNoInt
from ..schematic.core_unit import bag_ilropr__core_unit
from .util import track_to_track

class CoreUnit(TemplateBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        self.conn_layer = -1
        self.res_class = DiffRes
        self.res_inst = None
        self.core_inst = None

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__core_unit

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
        core_wrap_params = dict(cls_name=CoreDiffPairInjNoInt.get_qualified_name(), 
                                params=core_params, export_hidden=True)
        core_template = self.new_template(GenericWrapper, params=core_wrap_params)
        res_wrap_params = dict(cls_name=self.res_class.get_qualified_name(), 
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
        self.core_inst = core_inst
        self.res_inst = res_inst

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
        for res_out, core_out in [('rout', 'outp'), ('rout_b', 'outm')]:
            res_warr = res_inst.get_pin(res_out)
            if core_inst.get_all_port_pins(core_out, layer=ym_layer):
                core_warr = core_inst.get_all_port_pins(core_out, ym_layer)
                self.connect_to_track_wires(core_warr, res_warr)
                self.add_pin(core_out, core_warr, connect=True)
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
                w_sup_ym = tr_manager.get_width(ym_layer, 'sup')
                warr_list = []
                for tidx in tidx_list:
                    tid = TrackID(ym_layer, tidx, w_sup_ym)
                    warr_list.append(self.connect_to_tracks([res_warr, core_warr], tid))
                self.add_pin(core_out, warr_list, connect=True)


        for pin_name in ['tail_aa', 'tail_injb', 'tail_inj', 'taile_injb', 'taile_inj']:
            self.reexport(core_inst.get_port(pin_name))

        for pin_name in ['inp', 'inm', 'injp', 'injm']:
            self.reexport(core_inst.get_port(pin_name))
        
        for pin_name in ['outp', 'outm']:
            self.add_pin(pin_name, core_inst.get_pin(pin_name, layer=xm_layer), connect=True)
            self.add_pin(pin_name, core_inst.get_pin(pin_name, layer=xxm_layer), connect=True)

        # Diff pair should already have supply on xxm
        if core_inst.has_port('VSS'):
            self.reexport(core_inst.get_port('VSS'), connect=True)
        if core_inst.has_port('VDD'):
            self.reexport(core_inst.get_port('VDD'), connect=True)

        # Bring res pair supplies from xm to xxm
        # We expect only one of these, which provides both supply and well bias
        sub_type = cast(ResBasePlaceInfo, res_template.core.place_info).res_config['sub_type_default']
        sup_name = 'VDD' if sub_type == 'ntap' else 'VSS'

        sup = res_inst.get_all_port_pins(sup_name)
        bot_warr, top_warr = sup[0], sup[1]
        top_xxm = self.connect_via_stack(tr_manager, top_warr, xxm_layer, 'sup')
        self.add_pin(sup_name, top_xxm, connect=True)       
        
        # Skip bottom ym. It would only be used for biasing the dummies,
        # which is not critical and only adds congestion.

        # Sch_params 
        self.sch_params = dict(
            core_params=core_template.sch_params,
            res_params=res_template.sch_params,
        )
