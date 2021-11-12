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

from typing import Any, Dict, Type, Optional, List, Sequence

from pybag.enum import RoundMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.module import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ..schematic.core_diff_pair import bag_ilropr__core_diff_pair
from ..schematic.buf_diff_pair import bag_ilropr__buf_diff_pair
from .util import track_to_track

class CoreDiffPairInj(MOSBase):
    """Core diff pairs for ILROPR core.
    - 2 transistor rows with substrate rows on the top + bottom
    - Draws up to xm_layer
    - Inputs on hm_layer in the center
    - Outputs on xm_layer in the center
    - Tails on xm_layer on the outsides
    - Transistors are unit-cell based and interdigitated for matching
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__core_diff_pair

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            unit_seg="Unit transistor segments",
            sig_locs='Signal locations for top horizontal metal layer pins'\
                     'This should contain str->int for the output pins, e.g {"outp": 0}',
            add_od_dummy="True to add dummy devices between core devices for continuous OD."\
                         "Otherwise, draws MOSSpace",
            inj_xm="True to connect injp/m to xm, else on hm"
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            unit_seg=2,
            sig_locs={},
            add_od_dummy=False,
            inj_xm=False
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        add_od_dummy: bool = self.params['add_od_dummy']
        sig_locs: Dict[str, int] = self.params['sig_locs']
        unit_seg: int = self.params['unit_seg']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
    
        seg_dp = seg_dict['seg_dp']
        seg_inj = seg_dict['seg_inj']
        seg_dum = seg_dict.get('seg_dum', 0)
        if not seg_dum:
            seg_dum = 2 * unit_seg
        assert seg_dum > 0

        # Check that everything is a multiple of unit_seg
        assert not seg_dp % unit_seg, "Must be a multiple of unit_seg"
        assert not seg_inj % unit_seg, "Must be a multiple of unit_seg"
        assert not seg_dum % unit_seg, "Must be a multiple of unit_seg"

        # Compute seg info
        num_aa = seg_dp // unit_seg
        num_inj = seg_inj // unit_seg
        num_dev = num_aa + 4 * num_inj
        seg_dev = seg_dp + 4 * seg_inj
        
        num_dum = seg_dum // unit_seg
        num_tot = num_dev + num_dum

        unit_seg_od = self.min_sep_col
        seg_od_dum = unit_seg_od * (num_tot - 1)
        
        seg_tot = seg_dev + seg_dum + seg_od_dum
        
        # set total number of columns
        # TODO Total width can be limited by either transistor size or by vertical metal size
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        grid = self.grid
        tr_manager = self.tr_manager

        w_sig_hm = tr_manager.get_width(hm_layer, 'sig')
        w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        w_sig_xm = tr_manager.get_width(xm_layer, 'sig')

        w_sup_hm = tr_manager.get_width(hm_layer, 'sup')
        w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
        w_sup_xm = tr_manager.get_width(xm_layer, 'sup')
        w_sup_ym = tr_manager.get_width(ym_layer, 'sup')
        w_sup_xxm = tr_manager.get_width(xxm_layer, 'sup')

        # Place and connect taps up to xm_layer
        vss_tap_bot = self.add_substrate_contact(row_idx=0, col_idx=0, seg=seg_tot)
        bot_tidx = grid.coord_to_track(hm_layer, vss_tap_bot.middle, mode=RoundMode.NEAREST)
        vss_bot = self.connect_to_tracks(vss_tap_bot, TrackID(hm_layer, bot_tidx, w_sup_hm))
        ret_warr_dict_bot, ret_warr_dict_top = {}, {}
        vss_bot_xm = self.connect_via_stack(tr_manager, vss_bot, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_bot)

        vss_tap_top = self.add_substrate_contact(row_idx=-1, col_idx=0, seg=seg_tot)
        top_tidx = grid.coord_to_track(hm_layer, vss_tap_top.middle, mode=RoundMode.NEAREST)
        vss_top = self.connect_to_tracks(vss_tap_top, TrackID(hm_layer, top_tidx, w_sup_hm))
        vss_top_xm = self.connect_via_stack(tr_manager, vss_top, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_top)
        
        # Connect up to xxm_layer
        # Fully connect the bottom 
        vss_bot_xxm = self.connect_via_stack(tr_manager, vss_bot_xm, xxm_layer, 
                                             w_type='sup', ret_warr_dict=ret_warr_dict_bot)
        # Only connect top ym on the ends to make room for routing
        # Since this is a tap bias, we don't need very well connected
        # Skip vm to leave room for inter-stage routing
        warr = self.connect_wires([ret_warr_dict_bot[ym_layer][0], ret_warr_dict_bot[ym_layer][-1]])[0]
        vss_top_ym = self.connect_to_track_wires(vss_top_xm, warr)
        tidx = track_to_track(self.grid, vss_top_xm, xm_layer, xxm_layer)
        vss_top_xxm = self.connect_to_tracks(vss_top_ym, TrackID(xxm_layer, tidx, w_sup_xxm))

        # Add_pins on hm and xm
        vss_hm = self.connect_wires([vss_bot, vss_top])
        self.add_pin("VSS", vss_hm, connect=True)
        vss_xm = self.connect_wires([vss_bot_xm, vss_top_xm])
        self.add_pin("VSS", vss_xm, connect=True)
        vss_xxm = self.connect_wires([vss_bot_xxm, vss_top_xxm])
        self.add_pin("VSS", vss_xxm, connect=True)
        
        # Place devices in the main rows
        g_on_s = True
        bot_row, top_row, bot_dummy, top_dummy = [], [], [], []
        bot_od_dummy, top_od_dummy = [], []
        ridx_bot, ridx_top = 1, 2
        col_idx = 0
        for idx in range(num_tot):
            if idx < num_dum // 2 or idx >= num_dum // 2 + num_dev:
                _bot, _top = bot_dummy, top_dummy
            else:
                _bot, _top = bot_row, top_row
            _bot.append(self.add_mos(row_idx=ridx_bot, col_idx=col_idx, seg=unit_seg, 
                                        g_on_s=g_on_s, sep_g=False, draw_g2=False))
            _top.append(self.add_mos(row_idx=ridx_top, col_idx=col_idx, seg=unit_seg, 
                                        g_on_s=g_on_s, sep_g=False, draw_g2=False))
            col_idx += unit_seg
            if add_od_dummy and idx < num_dev - 1:
                bot_od_dummy.append(self.add_mos(row_idx=ridx_bot, col_idx=col_idx, 
                                              seg=unit_seg_od, draw_g=False))
                top_od_dummy.append(self.add_mos(row_idx=ridx_top, col_idx=col_idx, 
                                              seg=unit_seg_od, draw_g=False))
            col_idx += unit_seg_od

        # OD dummies: Connect gates to tap vss
        # should short drain along the way
        # do connect_wires instead of connect_to_track_wires to avoid adding additional vias
        if add_od_dummy:
            self.connect_wires([dev.g2 for dev in bot_od_dummy] + [vss_tap_bot], lower=vss_tap_bot.lower)
            self.connect_wires([dev.g2 for dev in top_od_dummy] + [vss_tap_top], upper=vss_tap_top.upper)

        # Dummy units: Connect devices gates to tap vss
        # should short source along the way
        self.connect_wires([dev.g for dev in bot_dummy] + [vss_tap_bot], lower=vss_tap_bot.lower)
        self.connect_wires([dev.g for dev in top_dummy] + [vss_tap_top], upper=vss_tap_top.upper)
        self.connect_wires([dev.d for dev in bot_dummy] + [vss_tap_bot], lower=vss_tap_bot.lower)
        self.connect_wires([dev.d for dev in top_dummy] + [vss_tap_top], upper=vss_tap_top.upper)

        # Main and dummy units: Connect separated gates on HM
        wire_type = MOSWireType.G
        wire_name = 'sig'
        g_bot = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        g_top = self.get_track_id(ridx_top, wire_type, wire_name, -1)
        for ports in bot_row + bot_dummy:
            self.connect_to_tracks(ports.g, g_bot)
        for ports in top_row + top_dummy:
            self.connect_to_tracks(ports.g, g_top)
        
        # Main devices: Connect inputs to gates (in and inj)
        # 4 tracks need to be available in hm_layer
        wire_type = MOSWireType.G_MATCH
        wire_name = 'sig_hs'
        inp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        inm_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 1)
        injp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 2)
        injm_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 3)

        tid_list = [inp_tid, inm_tid, injp_tid, injm_tid]
        warr_list = [[], [], [], []]
        name_list = ['inp', 'inm', 'injp', 'injm']

        # Order: aa, injb, einjb, inj, einj, aa
        # For simplicity, keep top and bottom as pairs
        # TODO: further enhance matching strategies
        top_row_order = [0, 1, 0, 3, 2, 1]
        bot_row_order = [1, 0, 1, 2, 3, 0]
        
        for idx, dev in enumerate(top_row):
            idx_6 = idx % 6
            warr_list[top_row_order[idx_6]].append(
                self.connect_to_tracks(dev.g0, tid_list[top_row_order[idx_6]]))
            warr_list[bot_row_order[idx_6]].append(
                self.connect_to_tracks(bot_row[idx].g1, tid_list[bot_row_order[idx_6]]))

        warr_list2 = []
        for warr_l in warr_list:
            warr_list2.append(self.connect_wires(warr_l)[0])
        
        min_edge = min([warr.lower for warr in warr_list2])
        max_edge = max([warr.upper for warr in warr_list2])
        warr_list2 = self.extend_wires(warr_list2, lower=min_edge, upper=max_edge)
        for idx, warr in enumerate(warr_list2[:2]):
            # Don't add pins to inj. Deal with it later
            self.add_pin(name_list[idx], warr)

        # Place xm wires for tails and outputs        
        w_sig_hs_vm = tr_manager.get_width(vm_layer, 'sig_hs')
        w_sig_hs_xm = tr_manager.get_width(xm_layer, 'sig_hs')

        dl_tid = self.get_track_id(ridx_bot, MOSWireType.DS, 'sig', -1)
        sl_tid = self.get_track_id(ridx_bot, MOSWireType.DS, 'sig', 0)

        du_tid = self.get_track_id(ridx_top, MOSWireType.DS, 'sig', 0)
        su_tid = self.get_track_id(ridx_top, MOSWireType.DS, 'sig', -1)

        # wire_list = ['sup'] * 3 + ['sig_hs'] * 4 + ['sup'] * 2
        wire_list = ['sig_hs'] * 6
        _, xm_tidx_list = tr_manager.place_wires(xm_layer, wire_list, center_coord=self.bound_box.ym)

        xm_out_tidx_list = xm_tidx_list[:4]
        xm_inj_tidx_list = xm_tidx_list[4:6]
        
        # xm inj
        injp_hm = self.connect_wires(warr_list[2])
        injm_hm = self.connect_wires(warr_list[3])
        vm_tidx_list = []
        for idx, _ in enumerate(bot_row):
            col_mid = (num_dum // 2 + idx) * (unit_seg + unit_seg_od) + unit_seg // 2
            vm_tidx_list.append(self.arr_info.col_to_track(vm_layer, col_mid, RoundMode.NEAREST))
        jump = 2  # Hueristic
        assert jump >= 2
        injp_tidx_vm = vm_tidx_list[::jump]
        injm_tidx_vm = vm_tidx_list[1::jump]
        assert len(injp_tidx_vm) == len(injm_tidx_vm)
        injp_vm = [self.connect_to_tracks(injp_hm, TrackID(vm_layer, tidx)) for tidx in injp_tidx_vm]
        injm_vm = [self.connect_to_tracks(injm_hm, TrackID(vm_layer, tidx)) for tidx in injm_tidx_vm]
        injp_vm = self.connect_wires(injp_vm)
        injm_vm = self.connect_wires(injm_vm)
        injp_xm, injm_xm = self.connect_differential_tracks(injp_vm, injm_vm, xm_layer, xm_inj_tidx_list[0],
                                                            xm_inj_tidx_list[1], width=w_sig_hs_xm)
        self.add_pin('injp', injp_xm)
        self.add_pin('injm', injm_xm)

        # xm outputs 
        p_tidx = xm_out_tidx_list[sig_locs.get('outp', 0)]
        m_tidx = xm_out_tidx_list[sig_locs.get('outm', 1)]

        outp_xm_list = []
        outm_xm_list = []

        # Order: aa, injb, einjb, inj, einj, aa
        sup_xm_list = [[], [], [], [], []]
        s_order = [0, 1, 2, 3, 4, 0]
        d_order = [0, 1, 0, 1, 1, 1] 

        wire_list = ['sup'] * 5
        _, xxm_tidx_list = tr_manager.place_wires(xxm_layer, wire_list, center_coord=self.bound_box.ym)

        last_vm_tidx = -10
        for idx, dev in enumerate(bot_row):
            # for each device, add 2 M2: one for drain, one for source
            dl_warr = self.connect_to_tracks(dev.d, dl_tid)
            sl_warr = self.connect_to_tracks(dev.s, sl_tid)

            du_warr = self.connect_to_tracks(top_row[idx].d, du_tid)
            su_warr = self.connect_to_tracks(top_row[idx].s, su_tid)

            # for each device column, add 3 M3: drain, source, drain
            col_mid = (num_dum // 2 + idx) * (unit_seg + unit_seg_od) + unit_seg // 2
            cntr_coord = self.place_info.col_to_coord(col_mid)
            _, tidx_list = tr_manager.place_wires(vm_layer, ['sig_hs', 'sig_hs'], center_coord=cntr_coord)
            dl_vm_tidx, du_vm_tidx = tidx_list
            if dl_vm_tidx < last_vm_tidx:
                raise ValueError("VM collision. Either shrink vm width, or increase unit seg")
            last_vm_tidx = du_vm_tidx

            # connect to vm drain (output) and sources (tails)
            du_vm_tid = TrackID(vm_layer, du_vm_tidx, w_sig_vm)
            dl_vm_tid = TrackID(vm_layer, dl_vm_tidx, w_sig_vm)
            dl_vm_warr = self.connect_to_tracks(dl_warr, dl_vm_tid)
            du_vm_warr = self.connect_to_tracks(du_warr, du_vm_tid)            
            
            # Connect to xm outp / outm
            idx_6 = idx % 6
            du_xm_tidx, dl_xm_tidx = (p_tidx, m_tidx) if d_order[idx_6] else (m_tidx, p_tidx)
            du_xm_list, dl_xm_list = (outp_xm_list, outm_xm_list) if d_order[idx_6] \
                else (outm_xm_list, outp_xm_list)
            du_xm_tid = TrackID(xm_layer, du_xm_tidx, w_sig_hs_xm)
            dl_xm_tid = TrackID(xm_layer, dl_xm_tidx, w_sig_hs_xm)
            du_xm_list.append(self.connect_to_tracks(du_vm_warr, du_xm_tid))
            dl_xm_list.append(self.connect_to_tracks(dl_vm_warr, dl_xm_tid))

            # Connect to xm tails
            # Order: aa, injb, einjb, inj, einj, aa
            ym_layer = xm_layer + 1
            sl_ym_warr = self.connect_via_stack(tr_manager, sl_warr, ym_layer, 'sup')
            su_ym_warr = self.connect_via_stack(tr_manager, su_warr, ym_layer, 'sup')
            s_xxm_tid = TrackID(xxm_layer, xxm_tidx_list[s_order[idx_6]], w_sup_xxm)
            sup_xm_list[s_order[idx_6]].append(self.connect_to_tracks([sl_ym_warr, su_ym_warr], s_xxm_tid))
            
        # add pins for xm tails
        names = ['tail_aa', 'tail_injb', 'taile_injb', 'tail_inj', 'taile_inj']
        for idx in range(len(names)):
            xm_warr = self.connect_wires(sup_xm_list[idx])[0]
            self.add_pin(names[idx], xm_warr)

        lower = min([warr.lower for warr in outp_xm_list] + [warr.lower for warr in outm_xm_list])
        upper = max([warr.upper for warr in outp_xm_list] + [warr.upper for warr in outm_xm_list])
        outp = self.connect_wires(outp_xm_list, lower=lower, upper=upper)
        outm = self.connect_wires(outm_xm_list, lower=lower, upper=upper)
        self.add_pin('outp', outp)
        self.add_pin('outm', outm)
        # Add wires xm wires for matching
        # Note: for better matching, need to add dummy wires outside of these 4
        for tidx in xm_out_tidx_list:
            self.add_wires(xm_layer, tidx, lower, upper, width=w_sig_hs_xm)

        # Create ym pins that don't conflict
        out_ym_tidx = []
        for idx, _ in enumerate(bot_row[:-1]):
            col_mid = (num_dum // 2 + idx) * (unit_seg + unit_seg_od) + unit_seg // 2 + unit_seg
            ym_tidx = self.arr_info.col_to_track(ym_layer, col_mid, mode=RoundMode.NEAREST)
            out_ym_tidx.append(ym_tidx)
        
        _len_2 = len(out_ym_tidx) // 2
        if len(out_ym_tidx) % 2:
            outp_ym_tidx = out_ym_tidx[:_len_2]
            outm_ym_tidx = out_ym_tidx[_len_2+1:]
        else:
            outp_ym_tidx = out_ym_tidx[:_len_2]
            outm_ym_tidx = out_ym_tidx[_len_2:]
        for warr, tidx_list, name in [(outp, outp_ym_tidx, 'outp'), (outm, outm_ym_tidx, 'outm')]:
            _warr_list = []
            for tidx in tidx_list:
                tid = TrackID(ym_layer, tidx, w_sup_ym)
                _warr_list.append(self.connect_to_tracks(warr, tid))
            _warr_list = self.connect_wires(_warr_list)
            self.add_pin(name, _warr_list)

        # TODO: od dummy params
        rpinfo = self.get_row_info(1)
        self.sch_params = dict(
            lch=rpinfo.lch,
            w=rpinfo.width,
            th=rpinfo.threshold,
            unit_seg=unit_seg,
            num_aa=num_aa,
            num_inj=num_inj,
            num_dum=num_dum,
        )
        

class CoreDiffPairInjNoInt(MOSBase):
    """Core diff pairs for ILROPR core. No interdigitating
    - 2 transistor rows with substrate rows on the top + bottom
    - Draws up to xm_layer
    - Inputs on hm_layer in the center
    - Outputs on xm_layer in the center
    - Tails on xm_layer on the outsides
    - Transistors are unit-cell based and interdigitated for matching
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__core_diff_pair

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            sig_locs='Signal locations for top horizontal metal layer pins'\
                     'This should contain str->int for the output pins, e.g {"outp": 0}',
            inj_xm="True to connect injp/m to xm, else on hm"
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            sig_locs={},
            inj_xm=False
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        sig_locs: Dict[str, int] = self.params['sig_locs']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
    
        seg_dp = seg_dict['seg_dp']
        seg_inj = seg_dict['seg_inj']
        seg_dum = seg_dict.get('seg_dum', 4)
        assert seg_dum > 0

        # Compute seg info
        seg_tot = seg_dum + seg_inj * 4 + seg_dp + 7 * self.min_sep_col
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        grid = self.grid
        tr_manager = self.tr_manager

        w_sig_hm = tr_manager.get_width(hm_layer, 'sig')
        w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        w_sig_xm = tr_manager.get_width(xm_layer, 'sig')

        w_sup_hm = tr_manager.get_width(hm_layer, 'sup')
        w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
        w_sup_xm = tr_manager.get_width(xm_layer, 'sup')
        w_sup_ym = tr_manager.get_width(ym_layer, 'sup')
        w_sup_xxm = tr_manager.get_width(xxm_layer, 'sup')

        # Place and connect taps up to xm_layer
        vss_tap_bot = self.add_substrate_contact(row_idx=0, col_idx=0, seg=seg_tot)
        bot_tidx = grid.coord_to_track(hm_layer, vss_tap_bot.middle, mode=RoundMode.NEAREST)
        vss_bot = self.connect_to_tracks(vss_tap_bot, TrackID(hm_layer, bot_tidx, w_sup_hm))
        ret_warr_dict_bot, ret_warr_dict_top = {}, {}
        vss_bot_xm = self.connect_via_stack(tr_manager, vss_bot, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_bot)

        vss_tap_top = self.add_substrate_contact(row_idx=-1, col_idx=0, seg=seg_tot)
        top_tidx = grid.coord_to_track(hm_layer, vss_tap_top.middle, mode=RoundMode.NEAREST)
        vss_top = self.connect_to_tracks(vss_tap_top, TrackID(hm_layer, top_tidx, w_sup_hm))
        vss_top_xm = self.connect_via_stack(tr_manager, vss_top, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_top)
        
        # Connect up to xxm_layer
        # Fully connect the bottom 
        vss_bot_xxm = self.connect_via_stack(tr_manager, vss_bot_xm, xxm_layer, 
                                             w_type='sup', ret_warr_dict=ret_warr_dict_bot)
        # Only connect top ym on the ends to make room for routing
        # Since this is a tap bias, we don't need very well connected
        # Skip vm to leave room for inter-stage routing
        warr = self.connect_wires([ret_warr_dict_bot[ym_layer][0], ret_warr_dict_bot[ym_layer][-1]])[0]
        vss_top_ym = self.connect_to_track_wires(vss_top_xm, warr)
        tidx = track_to_track(self.grid, vss_top_xm, xm_layer, xxm_layer)
        vss_top_xxm = self.connect_to_tracks(vss_top_ym, TrackID(xxm_layer, tidx, w_sup_xxm))

        # Add_pins on hm and xm
        vss_hm = self.connect_wires([vss_bot, vss_top])
        self.add_pin("VSS", vss_hm, connect=True)
        vss_xm = self.connect_wires([vss_bot_xm, vss_top_xm])
        self.add_pin("VSS", vss_xm, connect=True)
        vss_xxm = self.connect_wires([vss_bot_xxm, vss_top_xxm])
        self.add_pin("VSS", vss_xxm, connect=True)

        # Place devices in the main rows
        bot_row, top_row, bot_dummy, top_dummy = [], [], [], []
        ridx_bot, ridx_top = 1, 2
        col_idx = 0
        seg_list = [seg_dum // 2, seg_dp // 2, seg_inj, seg_inj, seg_inj, seg_inj, seg_dp // 2, seg_dum // 2]
        for idx, seg in enumerate(seg_list):
            is_dummy = idx == 0 or idx == len(seg_list) - 1
            _bot, _top = (bot_dummy, top_dummy) if is_dummy else (bot_row, top_row)
            g_on_s = is_dummy or (seg_inj // 4) % 2
            _bot.append(self.add_mos(row_idx=ridx_bot, col_idx=col_idx, seg=seg, 
                                     g_on_s=g_on_s, sep_g=False, draw_g2=False))
            _top.append(self.add_mos(row_idx=ridx_top, col_idx=col_idx, seg=seg, 
                                     g_on_s=g_on_s, sep_g=False, draw_g2=False))
            col_idx += seg + self.min_sep_col
        
        # Dummy units: Connect devices gates to tap vss
        # should short source along the way
        self.connect_wires([dev.g for dev in bot_dummy] + [vss_tap_bot], lower=vss_tap_bot.lower)
        self.connect_wires([dev.g for dev in top_dummy] + [vss_tap_top], upper=vss_tap_top.upper)
        self.connect_wires([dev.d for dev in bot_dummy] + [vss_tap_bot], lower=vss_tap_bot.lower)
        self.connect_wires([dev.d for dev in top_dummy] + [vss_tap_top], upper=vss_tap_top.upper)

        # Main and dummy units: Connect separated gates on HM
        wire_type = MOSWireType.G
        wire_name = 'sig'
        g_bot = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        g_top = self.get_track_id(ridx_top, wire_type, wire_name, -1)
        for ports in bot_row + bot_dummy:
            self.connect_to_tracks(ports.g, g_bot)
        for ports in top_row + top_dummy:
            self.connect_to_tracks(ports.g, g_top)
        
        # Main devices: Connect inputs to gates (in and inj)
        # 4 tracks need to be available in hm_layer
        wire_type = MOSWireType.G_MATCH
        wire_name = 'sig_hs'
        inp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 3)
        inm_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        injp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 2)
        injm_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 1)

        tid_list = [inp_tid, inm_tid, injp_tid, injm_tid]
        warr_list = [[], [], [], []]
        name_list = ['inp', 'inm', 'injp', 'injm']

        # For simplicity, keep top and bottom as pairs
        gate_order = [0, 0, 1, 1, 0, 0]  # 8/31

        for idx, dev in enumerate(top_row):
            warr_list[gate_order[idx] * 2].append(
                self.connect_to_tracks(dev.g0, tid_list[gate_order[idx] * 2]))
            warr_list[gate_order[idx] * 2 + 1].append(
                self.connect_to_tracks(bot_row[idx].g1, tid_list[gate_order[idx] * 2 + 1]))

        warr_list2 = []
        for warr_l in warr_list:
            warr_list2.append(self.connect_wires(warr_l)[0])
        
        min_edge = min([warr.lower for warr in warr_list2])
        max_edge = max([warr.upper for warr in warr_list2])
        warr_list2 = self.extend_wires(warr_list2, lower=min_edge, upper=max_edge)
        for idx, warr in enumerate(warr_list2[:2]):
            # Don't add pins to inj. Handled below
            self.add_pin(name_list[idx], warr)
        
        # Place xm wires for tails and outputs        
        w_sig_hs_vm = tr_manager.get_width(vm_layer, 'sig_hs')
        w_sig_hs_xm = tr_manager.get_width(xm_layer, 'sig_hs')

        dl_tid = self.get_track_id(ridx_bot, MOSWireType.DS, 'sig_hs', -1)
        sl_tid = self.get_track_id(ridx_bot, MOSWireType.DS, 'sup', 0)

        du_tid = self.get_track_id(ridx_top, MOSWireType.DS, 'sig_hs', 0)
        su_tid = self.get_track_id(ridx_top, MOSWireType.DS, 'sup', -1)

        # wire_list = ['sup'] * 3 + ['sig_hs'] * 4 + ['sup'] * 2
        wire_list = ['sig_hs'] * 6
        _, xm_tidx_list = tr_manager.place_wires(xm_layer, wire_list, center_coord=self.bound_box.ym)

        xm_out_tidx_list = xm_tidx_list[:2] + xm_tidx_list[4:6]
        xm_inj_tidx_list = xm_tidx_list[2:4]
        
        # xm inj
        injp_hm = self.connect_wires(warr_list[2])
        injm_hm = self.connect_wires(warr_list[3])
        vm_tidx_list = []
        
        def _compute_col_mid(idx):
            _idx = idx + 1
            return sum(seg_list[:_idx]) + self.min_sep_col * (_idx) + seg_list[_idx] // 2 

        for idx, _ in enumerate(bot_row):
            col_mid = _compute_col_mid(idx)
            vm_tidx_list.append(self.arr_info.col_to_track(vm_layer, col_mid, RoundMode.NEAREST))
        jump = 2  # Hueristic
        assert jump >= 2
        injp_tidx_vm = vm_tidx_list[::jump]
        injm_tidx_vm = vm_tidx_list[1::jump]
        assert len(injp_tidx_vm) == len(injm_tidx_vm)
        injp_vm = [self.connect_to_tracks(injp_hm, TrackID(vm_layer, tidx)) for tidx in injp_tidx_vm]
        injm_vm = [self.connect_to_tracks(injm_hm, TrackID(vm_layer, tidx)) for tidx in injm_tidx_vm]
        injp_vm = self.connect_wires(injp_vm)
        injm_vm = self.connect_wires(injm_vm)
        injp_xm, injm_xm = self.connect_differential_tracks(injp_vm, injm_vm, xm_layer, xm_inj_tidx_list[0],
                                                            xm_inj_tidx_list[1], width=w_sig_hs_xm)
        self.add_pin('injp', injp_xm)
        self.add_pin('injm', injm_xm)

        # xm outputs 
        p_tidx = xm_out_tidx_list[sig_locs.get('outp', 0)]
        m_tidx = xm_out_tidx_list[sig_locs.get('outm', 1)]

        outp_xm_list = []
        outm_xm_list = []
        # Order: aa, injb, einjb, inj, einj, aa
        tail_order = ['aa', 'injb', 'inj', 'e_inj', 'e_injb', 'aa']
        sup_xm_list = [[], [], [], [], []]
        s_order = [0, 1, 3, 4, 2, 0] # 8/31
        d_order = [0, 0, 0, 1, 0, 0] # 8/31

        wire_list = ['sig_hs'] * 2 + ['sup'] * 5 
        _, xxm_tidx_list = tr_manager.place_wires(xxm_layer, wire_list, center_coord=self.bound_box.ym)
        sup_xxm_tidx = xxm_tidx_list[2:]

        last_vm_tidx = -10
        for idx, dev in enumerate(bot_row):
            # for each device, add 2 M2: one for drain, one for source
            dl_warr = self.connect_to_tracks(dev.d, dl_tid)
            sl_warr = self.connect_to_tracks(dev.s, sl_tid)

            du_warr = self.connect_to_tracks(top_row[idx].d, du_tid)
            su_warr = self.connect_to_tracks(top_row[idx].s, su_tid)

            # for each device column, add 3 M3: drain, source, drain
            col_mid = _compute_col_mid(idx) 
            cntr_coord = self.place_info.col_to_coord(col_mid)
            _, tidx_list = tr_manager.place_wires(vm_layer, ['sig_hs', 'sig_hs', 'sig_hs'], center_coord=cntr_coord)
            dl_vm_tidx, du_vm_tidx = tidx_list[0], tidx_list[-1]
            if dl_vm_tidx < last_vm_tidx:
                raise ValueError("VM collision. Either shrink vm width, or increase unit seg")
            last_vm_tidx = du_vm_tidx

            # connect to vm drain (output) and sources (tails)
            du_vm_tid = TrackID(vm_layer, du_vm_tidx, w_sig_hs_vm)
            dl_vm_tid = TrackID(vm_layer, dl_vm_tidx, w_sig_hs_vm)
            dl_vm_warr = self.connect_to_tracks(dl_warr, dl_vm_tid)
            du_vm_warr = self.connect_to_tracks(du_warr, du_vm_tid)
            
            # vm Cap match 
            lower = min(dl_vm_warr.lower, du_vm_warr.lower)
            upper = max(dl_vm_warr.upper, du_vm_warr.upper)
            self.extend_wires([dl_vm_warr, du_vm_warr], lower=lower, upper=upper)
            
            # Connect to xm outp / outm
            du_xm_tidx, dl_xm_tidx = (p_tidx, m_tidx) if d_order[idx] else (m_tidx, p_tidx)
            du_xm_list, dl_xm_list = (outp_xm_list, outm_xm_list) if d_order[idx] \
                else (outm_xm_list, outp_xm_list)
            du_xm_tid = TrackID(xm_layer, du_xm_tidx, w_sig_hs_xm)
            dl_xm_tid = TrackID(xm_layer, dl_xm_tidx, w_sig_hs_xm)
            du_xm_list.append(self.connect_to_tracks(du_vm_warr, du_xm_tid))
            dl_xm_list.append(self.connect_to_tracks(dl_vm_warr, dl_xm_tid))

            # Connect to xm tails            
            # TODO: resolve this with the ym pins to not conflict
            sl_xm_warr = self.connect_via_stack(tr_manager, sl_warr, xm_layer, 'sup')
            su_xm_warr = self.connect_via_stack(tr_manager, su_warr, xm_layer, 'sup')
            tidx_l = self.grid.coord_to_track(ym_layer, sl_xm_warr.lower, RoundMode.GREATER_EQ)
            tidx_h = self.grid.coord_to_track(ym_layer, sl_xm_warr.upper, RoundMode.LESS_EQ)
            tid_l = TrackID(ym_layer, tidx_l, w_sup_ym)
            tid_h = TrackID(ym_layer, tidx_h, w_sup_ym)
            sl_ym_warr = [self.connect_to_tracks(sl_xm_warr, tid) for tid in [tid_l, tid_h]]
            su_ym_warr = [self.connect_to_tracks(su_xm_warr, tid) for tid in [tid_l, tid_h]]
            sl_ym_warr = self.connect_wires(sl_ym_warr)[0]
            su_ym_warr = self.connect_wires(su_ym_warr)[0]
            # Connect to xxm
            s_xxm_tid = TrackID(xxm_layer, sup_xxm_tidx[s_order[idx]], w_sup_xxm)
            sup_xm_list[s_order[idx]].append(self.connect_to_tracks([sl_ym_warr, su_ym_warr], s_xxm_tid))
            
        # add pins for xm tails
        names = ['tail_aa', 'tail_injb', 'taile_injb', 'tail_inj', 'taile_inj']
        for idx in range(len(names)):
            xm_warr = self.connect_wires(sup_xm_list[idx])[0]
            self.add_pin(names[idx], xm_warr)

        # Connect outputs
        lower = min([warr.lower for warr in outp_xm_list] + [warr.lower for warr in outm_xm_list])
        upper = max([warr.upper for warr in outp_xm_list] + [warr.upper for warr in outm_xm_list])
        outp = self.connect_wires(outp_xm_list, lower=lower, upper=upper)
        outm = self.connect_wires(outm_xm_list, lower=lower, upper=upper)
        self.add_pin('outp', outp)
        self.add_pin('outm', outm)
        # Add wires xm wires for matching
        # Note: for better matching, need to add dummy wires outside of these 4
        for tidx in xm_out_tidx_list:
            self.add_wires(xm_layer, tidx, lower, upper, width=w_sig_hs_xm)
       
        # Create ym pins that don't conflict
        out_ym_tidx = []
        for idx, _ in enumerate(bot_row):
            col_mid = _compute_col_mid(idx)
            ym_tidx = self.arr_info.col_to_track(ym_layer, col_mid, mode=RoundMode.NEAREST)
            out_ym_tidx.append(ym_tidx)
        
        _len_2 = len(out_ym_tidx) // 2
        if len(out_ym_tidx) % 2:
            outp_ym_tidx = out_ym_tidx[:_len_2]
            outm_ym_tidx = out_ym_tidx[_len_2+1:]
        else:
            outp_ym_tidx = out_ym_tidx[:_len_2]
            outm_ym_tidx = out_ym_tidx[_len_2:]
        
        # Connect outputs up to xxm_layer
        ptidx, mtidx = xxm_tidx_list[0], xxm_tidx_list[0]
        w_sig_hs_xxm = tr_manager.get_width(xxm_layer, 'sig_hs')
        for warr, tidx_list, name, xxm_tidx in \
            [(outp, outp_ym_tidx, 'outp', ptidx), (outm, outm_ym_tidx, 'outm', mtidx)]:
            _warr_list = []
            for tidx in tidx_list:
                tid = TrackID(ym_layer, tidx, w_sup_ym)
                _warr_list.append(self.connect_to_tracks(warr, tid))
            _warr_list = self.connect_wires(_warr_list)
            self.add_pin(name, _warr_list)

            xxm_tid = TrackID(xxm_layer, xxm_tidx, w_sig_hs_xxm)
            warr = self.connect_to_tracks(_warr_list, xxm_tid)
            self.add_pin(name, warr)

        # Sch Params
        rpinfo = self.get_row_info(1)
        self.sch_params = dict(
            lch=rpinfo.lch,
            w=rpinfo.width,
            th=rpinfo.threshold,
            seg_dict=dict(
                seg_dp=seg_dp,
                seg_inj=seg_inj,
                seg_dum=seg_dum,
            )
        )


class DiffPair(MOSBase):
    """Simple diff pair, used for input buffer
    - 2 transistor rows with substrate rows on the top + bottom
    - Draws up to xm_layer
    - Inputs on hm_layer in the center
    - Outputs on xm_layer in the center
    - Tails on xm_layer on the outsides
    - Transistors are placed in 4 sections:
        XP - XM
        XM - XP
    - NMOS is assumed
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__buf_diff_pair

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            sig_locs='Signal locations for top horizontal metal layer pins'\
                     'This should contain str->int for the output pins, e.g {"outp": 0}',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            sig_locs={},
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        sig_locs: Dict[str, int] = self.params['sig_locs']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
    
        seg_dp = seg_dict['seg_dp']
        seg_dum = seg_dict.get('seg_dum', 8)

        assert seg_dum > 0

        seg_tot = seg_dp + seg_dum
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        grid = self.grid
        tr_manager = self.tr_manager

        w_sig_hm = tr_manager.get_width(hm_layer, 'sig')
        w_sup_hm = tr_manager.get_width(hm_layer, 'sup')
        w_sup_ym = tr_manager.get_width(ym_layer, 'sup')
        w_sup_xxm = tr_manager.get_width(xxm_layer, 'sup')
        
        # Place and connect taps up to xm_layer
        vss_tap_bot = self.add_substrate_contact(row_idx=0, col_idx=0, seg=seg_tot)
        bot_tidx = grid.coord_to_track(hm_layer, vss_tap_bot.middle, mode=RoundMode.NEAREST)
        vss_bot = self.connect_to_tracks(vss_tap_bot, TrackID(hm_layer, bot_tidx, w_sup_hm))
        ret_warr_dict_bot, ret_warr_dict_top = {}, {}
        vss_bot_xm = self.connect_via_stack(tr_manager, vss_bot, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_bot)

        vss_tap_top = self.add_substrate_contact(row_idx=-1, col_idx=0, seg=seg_tot)
        top_tidx = grid.coord_to_track(hm_layer, vss_tap_top.middle, mode=RoundMode.NEAREST)
        vss_top = self.connect_to_tracks(vss_tap_top, TrackID(hm_layer, top_tidx, w_sup_hm))
        vss_top_xm = self.connect_via_stack(tr_manager, vss_top, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_top)
        
        # Connect up to xxm_layer
        # Fully connect the bottom 
        vss_bot_xxm = self.connect_via_stack(tr_manager, vss_bot_xm, xxm_layer, 
                                             w_type='sup', ret_warr_dict=ret_warr_dict_bot)
        
        # Add_pins on hm and xm
        vss_hm = self.connect_wires([vss_bot, vss_top])
        self.add_pin("VSS", vss_hm, connect=True)
        vss_xm = self.connect_wires([vss_bot_xm, vss_top_xm])
        self.add_pin("VSS", vss_xm, connect=True)
        # vss_xxm = self.connect_wires([vss_bot_xxm, vss_top_xxm])
        vss_xxm = vss_bot_xxm
        self.add_pin("VSS", vss_xxm, connect=True)
        
        # Connect tap vm_layer on the ends
        for _idx in [0, -1]:
            self.connect_wires([ret_warr_dict_bot[vm_layer][_idx], ret_warr_dict_top[vm_layer][_idx]])

        # Place devices in the main rows
        seg_dp_2 = seg_dp // 2
        seg_dum_2 = seg_dum // 2

        ridx_bot = 1
        dum1_l = self.add_mos(ridx_bot, 0, seg_dum_2)
        xp1 = self.add_mos(ridx_bot, seg_dum_2, seg_dp_2)
        xn1 = self.add_mos(ridx_bot, seg_dum_2 + seg_dp_2, seg_dp_2)
        dum1_r = self.add_mos(ridx_bot, seg_dum_2 + seg_dp_2 * 2, seg_dum_2)

        ridx_top = 2
        dum2_l = self.add_mos(ridx_top, 0, seg_dum_2)
        xn2 = self.add_mos(ridx_top, seg_dum_2, seg_dp_2)
        xp2 = self.add_mos(ridx_top, seg_dum_2 + seg_dp_2, seg_dp_2)
        dum2_r = self.add_mos(ridx_top, seg_dum_2 + seg_dp_2 * 2, seg_dum_2)

        # --- Routing --- #
        w_sig_hs_hm = tr_manager.get_width(hm_layer, 'sig_hs')
        w_sig_hs_vm = tr_manager.get_width(vm_layer, 'sig_hs')
        w_sig_hs_xm = tr_manager.get_width(xm_layer, 'sig_hs')

        # dummies: Connect gates
        wire_type = MOSWireType.G
        wire_name = 'sig_hs'
        g_bot = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        g_top = self.get_track_id(ridx_top, wire_type, wire_name, -1)
        vm_warr_l = ret_warr_dict_bot[vm_layer][0]
        vm_warr_h = ret_warr_dict_bot[vm_layer][-1]

        for ports, vm_warr in [(dum1_l, vm_warr_l), (dum1_r, vm_warr_h)]:
            warr = self.connect_to_tracks(ports.g, g_bot)
            self.connect_to_track_wires(vm_warr, warr)
        for ports, vm_warr in [(dum2_l, vm_warr_l), (dum2_r, vm_warr_h)]:
            warr = self.connect_to_tracks(ports.g, g_top)
            self.connect_to_track_wires(vm_warr, warr)
        
        # Dummies: connect drains to gates
        for dum in [dum1_l, dum1_r, dum2_l, dum2_r]:
            self.connect_wires([dum.d, dum.g])
        
        # Main: Connect separate gates
        wire_type = MOSWireType.G_MATCH
        wire_name = 'sig_hs'
        inp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        inm_tid = self.get_track_id(ridx_top, wire_type, wire_name, -1)
        self.connect_to_tracks(xp1.g, inp_tid)
        self.connect_to_tracks(xn1.g, inp_tid)
        self.connect_to_tracks(xp2.g, inm_tid)
        self.connect_to_tracks(xn2.g, inm_tid)

        # Main: Connect gates between rows
        inp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, -1)
        inm_tid = self.get_track_id(ridx_top, wire_type, wire_name, 0)
        pwarr = [xp1.g0, xp2.g1] if (seg_dp // 4) % 2 else [xp1.g0, xp2.g0]
        nwarr = [xn2.g1, xn1.g0] if (seg_dp // 4) % 2 else [xn2.g1, xn1.g1]
        inp = self.connect_to_tracks(pwarr, inp_tid)
        inm = self.connect_to_tracks(nwarr, inm_tid)

        self.add_pin('inp', inp)
        self.add_pin('inm', inm)

        # Main: Connect drains
        wire_type = MOSWireType.DS
        wire_name = 'sig_hs'
        out1_tid = self.get_track_id(ridx_bot, wire_type, wire_name, -1)
        out2_tid = self.get_track_id(ridx_top, wire_type, wire_name, 0)

        outp1 = self.connect_to_tracks(xp1.d, out1_tid)
        outp2 = self.connect_to_tracks(xp2.d, out2_tid)
        outm1 = self.connect_to_tracks(xn1.d, out1_tid)
        outm2 = self.connect_to_tracks(xn2.d, out2_tid)

        # Main: Connect sources to tail. include dummies
        wire_type = MOSWireType.DS
        wire_name = 'sup'
        tailp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        tailm_tid = self.get_track_id(ridx_top, wire_type, wire_name, -1)
        
        dev1_list = [xp1, xn1, dum1_l, dum1_r]
        dev2_list = [xp2, xn2, dum2_l, dum2_r]
        tailp_hm = self.connect_to_tracks([dev.s for dev in dev1_list], tailp_tid)
        tailm_hm = self.connect_to_tracks([dev.s for dev in dev2_list], tailm_tid)

        # Determine vm_tracks
        vm_l, vm_m, vm_h = self._place_vm_tracks(vm_layer, seg_dum_2)
        num_tracks = len(vm_l)

        # Connect tails to vm_m
        tail_vm = [self.connect_to_tracks([tailp_hm, tailm_hm], tid) for tid in vm_m]
        tail_vm = self.connect_wires(tail_vm)

        # Connect outp and outm
        outp1_vm = self.connect_wires([self.connect_to_tracks(outp1, tid) for tid in vm_l[:num_tracks // 2]])
        outp2_vm = self.connect_wires([self.connect_to_tracks(outp2, tid) for tid in vm_l[num_tracks // 2:]])
        outm2_vm = self.connect_wires([self.connect_to_tracks(outm2, tid) for tid in vm_h[:num_tracks // 2]])
        outm1_vm = self.connect_wires([self.connect_to_tracks(outm1, tid) for tid in vm_h[num_tracks // 2:]])
        
        # Connect to xm
        outp_xm_tidx = self.get_hm_track_index(xm_layer, 'sig_hs', 0)
        tail_xm_tid = self.get_hm_track_id(xm_layer, 'sup', 0)
        outm_xm_tidx = self.get_hm_track_index(xm_layer, 'sig_hs', -1)

        outp, outm = self.connect_differential_tracks(outp1_vm + outp2_vm, outm1_vm + outm2_vm, xm_layer,
                                                      outp_xm_tidx, outm_xm_tidx, width=w_sig_hs_xm)
        tail_xm = self.connect_to_tracks(tail_vm, tail_xm_tid)

        self.add_pin('outp', outp)
        self.add_pin('outm', outm)
        self.add_pin('tail', tail_xm)

        rpinfo = self.get_row_info(1)
        self.sch_params = dict(
            lch=rpinfo.lch,
            w=rpinfo.width,
            th=rpinfo.threshold,
            seg_dp=seg_dp,
            seg_dum=seg_dum,
        )

    def _place_vm_tracks(self, vm_layer: int, seg_dum_2: int):
        tr_manager = self.tr_manager

        w_sig_hs_vm = tr_manager.get_width(vm_layer, 'sig_hs')
        w_sup_vm = tr_manager.get_width(vm_layer, 'sup')

        wire_base = ['sig_hs', 'sup', 'sig_hs'] * 2
        cntr_coord = self.bound_box.xm
        lower_coord = self.place_info.col_to_coord(seg_dum_2)
        
        num = 1
        last_tidx_list = []
        while True:
            _, tidx_list = tr_manager.place_wires(vm_layer, wire_base * num, center_coord=cntr_coord)
            if self.grid.track_to_coord(vm_layer, tidx_list[0]) < lower_coord:
                break
            num += 1
            last_tidx_list = tidx_list
        if not last_tidx_list:
            raise RuntimeError("diff pair : Could not place vm_tracks. Increase seg_dp")
        
        vm_l = last_tidx_list[::3]
        vm_m = last_tidx_list[1::3]
        vm_h = last_tidx_list[2::3]
        
        vm_l = [TrackID(vm_layer, tidx, w_sig_hs_vm) for tidx in vm_l]
        vm_m = [TrackID(vm_layer, tidx, w_sup_vm) for tidx in vm_m]
        vm_h = [TrackID(vm_layer, tidx, w_sig_hs_vm) for tidx in vm_h]

        return vm_l, vm_m, vm_h


class DiffPairBias(MOSBase):
    """Replica bias diff pair
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            sig_locs='Signal locations for top horizontal metal layer pins'\
                     'This should contain str->int for the output pins, e.g {"outp": 0}',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            sig_locs={},
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        sig_locs: Dict[str, int] = self.params['sig_locs']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1
    
        seg_dp = seg_dict['seg_dp']
        seg_dum = seg_dict.get('seg_dum', 8)

        assert seg_dum > 0

        seg_tot = seg_dp + seg_dum
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        grid = self.grid
        tr_manager = self.tr_manager

        w_sig_hm = tr_manager.get_width(hm_layer, 'sig')
        w_sup_hm = tr_manager.get_width(hm_layer, 'sup')
        w_sup_xxm = tr_manager.get_width(xxm_layer, 'sup')
        
        # Place and connect taps up to xm_layer
        vss_tap_bot = self.add_substrate_contact(row_idx=0, col_idx=0, seg=seg_tot)
        bot_tidx = grid.coord_to_track(hm_layer, vss_tap_bot.middle, mode=RoundMode.NEAREST)
        vss_bot = self.connect_to_tracks(vss_tap_bot, TrackID(hm_layer, bot_tidx, w_sup_hm))
        ret_warr_dict_bot, ret_warr_dict_top = {}, {}
        vss_bot_xm = self.connect_via_stack(tr_manager, vss_bot, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_bot)

        vss_tap_top = self.add_substrate_contact(row_idx=-1, col_idx=0, seg=seg_tot)
        top_tidx = grid.coord_to_track(hm_layer, vss_tap_top.middle, mode=RoundMode.NEAREST)
        vss_top = self.connect_to_tracks(vss_tap_top, TrackID(hm_layer, top_tidx, w_sup_hm))
        vss_top_xm = self.connect_via_stack(tr_manager, vss_top, xm_layer, 
                                         w_type='sup', ret_warr_dict=ret_warr_dict_top)
        
        # Connect up to xxm_layer
        # Fully connect the bottom 
        vss_bot_xxm = self.connect_via_stack(tr_manager, vss_bot_xm, xxm_layer, 
                                             w_type='sup', ret_warr_dict=ret_warr_dict_bot)
        
        # Add_pins on hm and xm
        vss_hm = self.connect_wires([vss_bot, vss_top])
        self.add_pin("VSS", vss_hm, connect=True)
        vss_xm = self.connect_wires([vss_bot_xm, vss_top_xm])
        self.add_pin("VSS", vss_xm, connect=True)
        # vss_xxm = self.connect_wires([vss_bot_xxm, vss_top_xxm])
        vss_xxm = vss_bot_xxm
        self.add_pin("VSS", vss_xxm, connect=True)
        
        # Connect tap vm_layer on the ends
        for _idx in [0, -1]:
            self.connect_wires([ret_warr_dict_bot[vm_layer][_idx], ret_warr_dict_top[vm_layer][_idx]])

        # Place devices in the main rows
        seg_dp_2 = seg_dp // 2
        seg_dum_2 = seg_dum // 2

        ridx_bot = 1
        dum1_l = self.add_mos(ridx_bot, 0, seg_dum_2)
        xp1 = self.add_mos(ridx_bot, seg_dum_2, seg_dp_2)
        xn1 = self.add_mos(ridx_bot, seg_dum_2 + seg_dp_2, seg_dp_2)
        dum1_r = self.add_mos(ridx_bot, seg_dum_2 + seg_dp_2 * 2, seg_dum_2)

        ridx_top = 2
        dum2_l = self.add_mos(ridx_top, 0, seg_dum_2)
        xn2 = self.add_mos(ridx_top, seg_dum_2, seg_dp_2)
        xp2 = self.add_mos(ridx_top, seg_dum_2 + seg_dp_2, seg_dp_2)
        dum2_r = self.add_mos(ridx_top, seg_dum_2 + seg_dp_2 * 2, seg_dum_2)

        # --- Routing --- #

        # dummies: Connect gates
        wire_type = MOSWireType.G
        wire_name = 'sig_hs'
        g_bot = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        g_top = self.get_track_id(ridx_top, wire_type, wire_name, -1)
        vm_warr_l = ret_warr_dict_bot[vm_layer][0]
        vm_warr_h = ret_warr_dict_bot[vm_layer][-1]

        for ports, vm_warr in [(dum1_l, vm_warr_l), (dum1_r, vm_warr_h)]:
            warr = self.connect_to_tracks(ports.g, g_bot)
            self.connect_to_track_wires(vm_warr, warr)
        for ports, vm_warr in [(dum2_l, vm_warr_l), (dum2_r, vm_warr_h)]:
            warr = self.connect_to_tracks(ports.g, g_top)
            self.connect_to_track_wires(vm_warr, warr)
        
        # Dummies: connect drains to gates
        for dum in [dum1_l, dum1_r, dum2_l, dum2_r]:
            self.connect_wires([dum.d, dum.g])
        
        # Main: Connect separate gates
        wire_type = MOSWireType.G_MATCH
        wire_name = 'sig_hs'
        in_tid = self.get_track_id(ridx_bot, wire_type, wire_name, -1)
        in_warr = self.connect_to_tracks([xp1.g, xp2.g, xn1.g, xn2.g], in_tid)

        # Main: Connect drains
        wire_type = MOSWireType.DS
        wire_name = 'sig_hs'
        out1_tid = self.get_track_id(ridx_bot, wire_type, wire_name, -1)
        out2_tid = self.get_track_id(ridx_top, wire_type, wire_name, 0)

        outp1 = self.connect_to_tracks(xp1.d, out1_tid)
        outp2 = self.connect_to_tracks(xp2.d, out2_tid)
        outm1 = self.connect_to_tracks(xn1.d, out1_tid)
        outm2 = self.connect_to_tracks(xn2.d, out2_tid)

        # Main: Connect sources to tail. include dummies
        wire_type = MOSWireType.DS
        wire_name = 'sup'
        tailp_tid = self.get_track_id(ridx_bot, wire_type, wire_name, 0)
        tailm_tid = self.get_track_id(ridx_top, wire_type, wire_name, -1)
        
        dev1_list = [xp1, xn1, dum1_l, dum1_r]
        dev2_list = [xp2, xn2, dum2_l, dum2_r]
        tailp_hm = self.connect_to_tracks([dev.s for dev in dev1_list], tailp_tid)
        tailm_hm = self.connect_to_tracks([dev.s for dev in dev2_list], tailm_tid)

        # Determine vm_tracks
        vm_sig, vm_sup = self._place_vm_tracks(vm_layer, seg_dum_2)

        # Connect tails to vm_m
        tail_vm = [self.connect_to_tracks([tailp_hm, tailm_hm], tid) for tid in vm_sup]
        tail_vm = self.connect_wires(tail_vm)

        # Connect out
        hm_warrs = [in_warr, outp1, outp2, outm1, outm2]
        vm_warrs = [self.connect_to_tracks(hm_warrs, tid) for tid in vm_sig]
        
        # Connect to xm
        out_xm_tid = self.get_hm_track_id(xm_layer, 'sig_hs', 0)
        tail_xm_tid = self.get_hm_track_id(xm_layer, 'sup', 0)
        outm_xm_tidx = self.get_hm_track_index(xm_layer, 'sig_hs', -1)

        out = self.connect_to_tracks(vm_warrs, out_xm_tid)
        tail_xm = self.connect_to_tracks(tail_vm, tail_xm_tid)
        tail_xxm = self.connect_via_stack(tr_manager, tail_xm, xxm_layer, 'sup')

        self.add_pin('out', out)
        self.add_pin('tail', tail_xxm)

        rpinfo = self.get_row_info(1)
        self.sch_params = dict(
            lch=rpinfo.lch,
            w=rpinfo.width,
            th=rpinfo.threshold,
            seg_dp=seg_dp,
            seg_dum=seg_dum,
        )

    def _place_vm_tracks(self, vm_layer: int, seg_dum_2: int):
        tr_manager = self.tr_manager

        w_sig_hs_vm = tr_manager.get_width(vm_layer, 'sig_hs')
        w_sup_vm = tr_manager.get_width(vm_layer, 'sup')

        wire_base = ['sig_hs', 'sup', 'sig_hs']
        cntr_coord = self.bound_box.xm
        lower_coord = self.place_info.col_to_coord(seg_dum_2)
        
        num = 1
        last_tidx_list = []
        while True:
            _, tidx_list = tr_manager.place_wires(vm_layer, wire_base * num, center_coord=cntr_coord)
            if self.grid.track_to_coord(vm_layer, tidx_list[0]) < lower_coord:
                break
            wire_base += ['sup', 'sig_hs']
            last_tidx_list = tidx_list
        if not last_tidx_list:
            raise RuntimeError("diff pair : Could not place vm_tracks. Increase seg_dp")
        
        vm_sig = [TrackID(vm_layer, tidx, w_sup_vm) for tidx in last_tidx_list[::2]]
        vm_sup = [TrackID(vm_layer, tidx, w_sig_hs_vm) for tidx in last_tidx_list[1::2]]

        return vm_sig, vm_sup
