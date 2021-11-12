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

"""This module contains layout classes for the full replica input buffer structure"""

from modulefinder import LOAD_CONST
from typing import Any, Dict, Type, Optional, Mapping

from pybag.enum import Orientation, PinMode, RoundMode
from pybag.core import Transform, BBox

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.template import TemplateBase, TemplateDB
from bag.layout.routing.base import TrackID, TrackManager, WDictType, SpDictType
from bag.design.module import Module

from xbase.layout.mos.top import GenericWrapper
from xbase.layout.array.top import ArrayBaseWrapper

from bag3_analog.layout.highpass import HighPassDiffCore

from .util import track_to_track
from .replica_buffer import ReplicaBias, ReplicaBuffer
from .idac_array import IMIRRArray
from ..schematic.in_buff_core import bag_ilropr__in_buff_core
from ..schematic.in_buff import bag_ilropr__in_buff


class InBuffCore(TemplateBase):
    """High pass filter, replica bias gen, and replica input buffer"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        self.num_stages = -1
        self.conn_layer = -1

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__in_buff_core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tr_widths='Track width dictionary for TrackManager',
            tr_spaces='Track spaces dictionary for TrackManager',
            hpf_params='High pass filter params',
            bias_params="Replica bias gen params",
            buff_params="Replica buffer params",
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict()

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager

        self.num_stages = self.params['buff_params']['num_stages']

        # Make templates
        hpf_wrap_params = dict(cls_name=HighPassDiffCore.get_qualified_name(), 
                              params=self.params['hpf_params'], export_hidden=True)
        hpf_template = self.new_template(ArrayBaseWrapper, params=hpf_wrap_params)
        bias_template = self.new_template(ReplicaBias, params=self.params['bias_params'])
        buff_template = self.new_template(ReplicaBuffer, params=self.params['buff_params'])
        
        # Get properties        
        hpf_bbox = hpf_template.bound_box
        bias_bbox = bias_template.bound_box
        buff_bbox = buff_template.bound_box

        # Align bias and HPF centers
        bias_xloc = max(0, hpf_bbox.xm - bias_bbox.xm)
        hpf_xloc = max(0, bias_bbox.xm - hpf_bbox.xm)
        # Align buffer to RFS
        buff_xloc = max(bias_bbox.xh, hpf_bbox.xh)

        # bias need MX orietation
        bias_yloc = bias_bbox.yh
        hpf_yloc = bias_yloc
        
        # ======== Placement ======== #
        bias_inst = self.add_instance(bias_template, xform=Transform(bias_xloc, bias_yloc, Orientation.MX))
        hpf_inst = self.add_instance(hpf_template, xform=Transform(hpf_xloc, hpf_yloc))
        buff_inst = self.add_instance(buff_template, xform=Transform(buff_xloc, 0))

        h_tot = max(hpf_inst.bound_box.yh, buff_inst.bound_box.yh)
        w_tot = buff_inst.bound_box.xh
        
        self.conn_layer = bias_template.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
        yym_layer = xxm_layer + 1
        x3m_layer = yym_layer + 1

        self.set_size_from_bound_box(x3m_layer, BBox(0, 0, w_tot, h_tot), round_up=True)

        # ======== Routing ======== #
        
        # Bring HPF input from xxm to x3m
        # Rename inn to inm to match preferred naming scheme
        for pin_name, in_name in [('inp', 'inp'), ('inn', 'inm')]:
            in_warr = hpf_inst.get_pin(pin_name, layer=xxm_layer)
            in_warr = self.connect_via_stack(tr_manager, in_warr, x3m_layer, 'sig_hs')
            # Extend to leftside
            in_warr = self.extend_wires(in_warr, lower=self.bound_box.xl)
            self.add_pin(in_name, in_warr, mode=PinMode.LOWER)

        # bias out xm -> HPF bias
        self.connect_to_track_wires([hpf_inst.get_pin('biasp'), hpf_inst.get_pin('biasn')], bias_inst.get_pin('out'))
        self.reexport(bias_inst.get_port('out'), net_name='bias_out')

        # HPF out -> buff in
        cntr_coord = buff_xloc
        _, tidx_list = tr_manager.place_wires(ym_layer, ['sig_hs', 'sup', 'sig_hs'], center_coord=cntr_coord)
        pwarrs = [hpf_inst.get_pin('outp'), buff_inst.get_pin('inp')]
        nwarrs = [hpf_inst.get_pin('outn'), buff_inst.get_pin('inm')]
        w_sig_hs_ym = tr_manager.get_width(ym_layer, 'sig_hs')
        self.connect_differential_tracks(pwarrs, nwarrs, ym_layer, tidx_list[0], 
                                         tidx_list[-1], width=w_sig_hs_ym)
        # debug
        self.reexport(buff_inst.get_port('inp'), net_name='inbp')
        self.reexport(buff_inst.get_port('inm'), net_name='inbm')

        # buff out xm -> xxm
        outp = buff_inst.get_pin('outp')
        outm = buff_inst.get_pin('outm')
        outp = self.connect_via_stack(tr_manager, outp, xxm_layer, 'sig_hs')
        outm = self.connect_via_stack(tr_manager, outm, xxm_layer, 'sig_hs')
        self.add_pin('outp', outp)
        self.add_pin('outm', outm)

        # Export tail nodes
        self.reexport(bias_inst.get_port('tail'), net_name='bias_tail')
        for idx in range(self.num_stages):
            self.reexport(buff_inst.get_port(f'tail<{idx}>'))
    
        # supply strategy
        # Bring everything up to xxm_layer
        for sup in ['VDD', 'VSS']:
            warr_list = hpf_inst.get_all_port_pins(sup, xm_layer)
            for warr in warr_list:
                warr = self.connect_via_stack(tr_manager, warr, xxm_layer, 'sup')
                self.add_pin(sup, warr, connect=True)

        for inst in [buff_inst, bias_inst]:
            for sup in ['VDD', 'VSS']:
                if inst.has_port(sup):
                    self.reexport(inst.get_port(sup), connect=True)

        # sch_params
        self.sch_params=dict(
            hpf_params=hpf_template.sch_params,
            bias_params=bias_template.sch_params,
            buff_params=buff_template.sch_params,
        )


class InBuff(TemplateBase):
    """InBuffCore + IMirr"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        self._tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        self.core_class = InBuffCore
        self.core_inst = None

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__in_buff

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            imirr_params='Current mirror params',
            **InBuffCore.get_params_info()
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return InBuffCore.get_default_param_values()

    def draw_layout(self) -> None:
        tr_manager = self._tr_manager

        # Make templates
        imirr_wrap_params = dict(cls_name=IMIRRArray.get_qualified_name(), 
                                 params=self.params['imirr_params'], export_hidden=True)
        imirr_template = self.new_template(GenericWrapper, params=imirr_wrap_params)
        core_params = self.params.copy(remove=['imirr_params'])
        core_template = self.new_template(self.core_class, params=core_params)
                
        # Get properties        
        imirr_bbox = imirr_template.bound_box
        core_bbox = core_template.bound_box

        # Center align
        imirr_xloc = max(0, core_bbox.xm - imirr_bbox.xm)
        core_xloc = max(0, imirr_bbox.xm - core_bbox.xm)
        
        # ======== Placement ======== #
        imirr_inst = self.add_instance(imirr_template, xform=Transform(imirr_xloc, 0))
        core_inst = self.add_instance(core_template, xform=Transform(core_xloc, imirr_inst.bound_box.yh))
        self.core_inst = core_inst

        w_tot = max(core_inst.bound_box.xh, imirr_inst.bound_box.xh)
        h_tot = core_inst.bound_box.yh

        self.conn_layer = core_template.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
        yym_layer = xxm_layer + 1
        x3m_layer = yym_layer + 1

        self.set_size_from_bound_box(x3m_layer, BBox(0, 0, w_tot, h_tot), 
                                     half_blk_x=False, half_blk_y=False, round_up=True)

        # ======== Routing ======== #
        
        # Export core signals
        for sig in ['inp', 'inm', 'bias_out', 'outp', 'outm', 'inbp', 'inbm']:
            self.reexport(core_inst.get_port(sig))

        # Connect bias tail
        w_sup_yym = tr_manager.get_width(yym_layer, 'sup')
        bias_tail = core_inst.get_pin('bias_tail')
        tidx = self.grid.coord_to_track(yym_layer, bias_tail.middle, mode=RoundMode.NEAREST)
        tid = TrackID(yym_layer, tidx, w_sup_yym)
        self.connect_to_tracks([bias_tail, imirr_inst.get_pin('tail_aa<0>')], tid)

        # Connect stage tails
        for idx in range(core_template.num_stages):
            # select tail for improving length match
            _idx = idx // 2 * 2 + (1 - idx % 2)
            tail_warr = core_inst.get_pin(f'tail<{_idx}>')
            _, tidx_list = tr_manager.place_wires(yym_layer, ['sup', 'sup'], center_coord=tail_warr.middle)
            tid = TrackID(yym_layer, tidx_list[idx % 2], w_sup_yym)
            self.connect_to_tracks([tail_warr, imirr_inst.get_pin(f'tail_aa<{idx + 1}>')], tid)

        # Connect NBIAS
        nbias_list = []
        tidx1 = self.grid.coord_to_track(yym_layer, self.bound_box.xl, mode=RoundMode.GREATER)
        tidx2 = self.grid.coord_to_track(yym_layer, self.bound_box.xh, mode=RoundMode.LESS)
        for tidx in [tidx1, tidx2]:
            tid = TrackID(yym_layer, tidx, w_sup_yym)
            nbias_list.append(self.connect_to_tracks(imirr_inst.get_all_port_pins('NBIAS'), tid))
        self.add_pin('NBIAS', self.connect_wires(nbias_list))
    
        # supply strategy
        vdd_xxm, vss_xxm = [], []
        for inst in [imirr_inst, core_inst]:
            for sup, warr_list in [('VDD', vdd_xxm), ('VSS', vss_xxm)]:
                if inst.has_port(sup):
                    self.reexport(inst.get_port(sup), connect=True)
                    warr_list.extend(inst.get_all_port_pins(sup, xxm_layer))

        # Not many yym, x3m. Just do power fill
        top_vdd, top_vss = vdd_xxm, vss_xxm
        self.add_pin('VDD', top_vdd, connect=True)
        self.add_pin('VSS', top_vss, connect=True)

        # sch_params
        self.sch_params=dict(
            core_params=core_template.sch_params,
            imirr_params=imirr_template.sch_params,
        )
