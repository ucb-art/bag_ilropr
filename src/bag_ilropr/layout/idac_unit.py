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

from math import log2
from typing import Any, Dict, Type, Optional, List, Sequence, Tuple, Union

from pybag.enum import MinLenMode, RoundMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID, WireArray
from bag.design.module import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from .util import track_to_track
from ..schematic.idac_unit import bag_ilropr__idac_unit


class IDACUnit(MOSBase):
    """Unit IDAC system, for one stage"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self.num_tiles = -1
    
    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__idac_unit

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_tail='Segments for the unit device. Value shared by the CS and switch',
            cs_stack='Current source stack height',
            num_diode='Number of units in the diode connected device',
            num_en='Number of en / enb bits.',
            num_aa='Number of always on units.',
            num_dum='Number of dummy units.',
            connect_out_lr="True to connect the left and right outs",
            has_top_sub="True if has top substrate"
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            cs_stack=1,
            num_diode=0,
            num_aa=0,
            num_dum=2,
            connect_out_lr=False,
            has_top_sub=True,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_tail: int = self.params['seg_tail']
        cs_stack: int = self.params['cs_stack']
        num_diode: int = self.params['num_diode']
        num_en: int = self.params['num_en']
        num_aa: int = self.params['num_aa']
        num_dummy: int = self.params['num_dum']
        connect_out_lr: bool = self.params['connect_out_lr']
        has_top_sub: bool = self.params['has_top_sub']

        """
        Pinfo is expected to be in tile format, and include:
        - ptap bottom row
        - 2 rows of nch switches, followed by...
        - ...`cs_stack` rows of nch current source
        - ptap top row, if has_top_sub
        """
        num_tiles = 4 + cs_stack
        self.num_tiles = num_tiles

        assert seg_tail % 2 == 0, "Odd tail segments currently not supported"

        if not num_diode:
            num_diode = max(2, (num_en + num_aa) // 16)
        assert num_en / num_diode <= 20, "Design check for mirror ratio"

        assert num_aa % 2 == 0, "Design check for num_aa"
        assert num_en % 2 == 0, "Design check for num_en"

        # This is not a generator requirement, but a design expectation
        _num = 2 ** int(log2(num_en + num_aa))
        assert (num_en + num_aa) - _num == 0, "Design check, num_en should be a power of 2"

        assert cs_stack >= 1

        # ========= Placement =========
        num_units = num_en + num_diode + num_aa + num_dummy
        seg_space = self.min_sep_col
        total_segments = num_units * seg_tail + (num_units - 1) * seg_space
        self.set_mos_size(total_segments, num_tiles - 1)

        # Draw taps
        bot_tap_tile = 0
        top_tap_tile = num_tiles - 1 
        bot_vss = self.add_substrate_contact(row_idx=0, col_idx=0, tile_idx=bot_tap_tile, seg=total_segments)
        if has_top_sub:
            top_vss = self.add_substrate_contact(row_idx=0, col_idx=0, tile_idx=top_tap_tile, seg=total_segments)

        hm_layer = bot_vss.layer_id + 1
        bot_vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=bot_tap_tile)
        top_vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=top_tap_tile)
        bot_vss_hm = self.connect_to_tracks(bot_vss, bot_vss_hm_tid)
        if has_top_sub:
            top_vss_hm = self.connect_to_tracks(top_vss, top_vss_hm_tid)
        else:
            top_vss_hm = self.add_wires(hm_layer, top_vss_hm_tid.base_index, bot_vss_hm.lower, bot_vss_hm.upper, 
                                        width=self.tr_manager.get_width(hm_layer, 'sup'))
        
        # Draw columns
        en_idx = 0
        col_idx = 0
        l_out, l_outb, r_out, r_outb = [], [], [], []
        l_out_vm, l_outb_vm, r_out_vm, r_outb_vm = [], [], [], []
        bot_hm, mirr_hm, mirr_vm_list = [], [], []
        aa_out_hm, aa_out_vm, aa_g_vm = [], [], []

        col_mirr_start = (seg_tail + seg_space) * (num_dummy + num_en + num_aa) // 2 
        col_mirr_stop = col_mirr_start + (seg_tail + seg_space) * num_diode

        while col_idx < total_segments:
            is_dummy = False
            if col_idx < num_dummy // 2 * (seg_tail + seg_space) or \
                col_idx >= total_segments - num_dummy // 2 * (seg_tail + seg_space):
                # Dummy columns on the outside
                self._draw_dummy_column(col_idx)
                is_dummy = True
            elif col_mirr_start <= col_idx < col_mirr_stop:
                # Diode mirror columns in the center
                bot, mirr, mirr_vm, last_drain = self._draw_mirr_column(col_idx)
            elif col_idx >= (seg_tail + seg_space) * (num_dummy + num_en) // 2 and \
                col_idx < col_mirr_stop + (seg_tail + seg_space) * (num_aa) // 2:
                # Always on columns
                # TODO: add better interleaving
                tmp = self._draw_aa_column(col_idx)
                en, out, out_vm, bot, mirr, mirr_vm, last_drain = tmp
                aa_g_vm.append(en)
                aa_out_hm.append(out)
                aa_out_vm.append(out_vm)
            else:
                # Steered columns on the outside
                tmp = self._draw_column(col_idx, en_idx % 2)
                en, enb, out, outb, out_vm, outb_vm, bot, mirr, mirr_vm, last_drain = tmp
                
                # TODO: add better interleaving
                self.add_pin(f'en<{en_idx}>', en)
                self.add_pin(f'enb<{en_idx}>', enb)
                en_idx += 1
                
                if col_idx < col_mirr_stop:
                    l_out.append(out)
                    l_outb.append(outb)
                    l_out_vm.append(out_vm)
                    l_outb_vm.append(outb_vm)
                else:
                    r_out.append(out)
                    r_outb.append(outb)
                    r_out_vm.append(out_vm)
                    r_outb_vm.append(outb_vm)
            
            if not is_dummy:
                mirr_vm_list += mirr_vm
                mirr_hm += mirr
                bot_hm.append(bot)

            col_idx += seg_tail + seg_space

        # ========= Routing =========
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        w_sup_xm = self.tr_manager.get_width(xm_layer, 'sup')

        # connect wires on hm
        self.connect_wires(l_out)
        self.connect_wires(l_outb)
        self.connect_wires(r_out)
        self.connect_wires(r_outb)
        self.connect_wires(mirr_hm)
        self.connect_wires(bot_hm)
        self.connect_wires(aa_out_hm)

        # Connect taps up to vm...
        col_space_list = [(seg_tail + seg_space) * (idx) - seg_space // 2 \
                            for idx in range(num_dummy // 2, num_units - num_dummy // 2 + 1)]
        vm_tidx_list = [self.arr_info.col_to_track(vm_layer, col_idx, mode=RoundMode.NEAREST) \
                            for col_idx in col_space_list]
        vm_tid_list = [TrackID(vm_layer, tidx) for tidx in vm_tidx_list]
        bot_vss_vm = [self.connect_to_tracks(bot_vss_hm, tid) for tid in vm_tid_list]
        top_vss_vm = [self.connect_to_tracks(top_vss_hm, tid) for tid in vm_tid_list]
        # ... and then to xm
        bot_xm_tidx = track_to_track(self.grid, bot_vss_hm, hm_layer, xm_layer)
        top_xm_tidx = track_to_track(self.grid, top_vss_hm, hm_layer, xm_layer)
        bot_xm_tid = TrackID(xm_layer, bot_xm_tidx, w_sup_xm)
        top_xm_tid = TrackID(xm_layer, top_xm_tidx, w_sup_xm)
        bot_xm = self.connect_to_tracks(bot_vss_vm, bot_xm_tid)
        top_xm = self.connect_to_tracks(top_vss_vm, top_xm_tid)
        vss = self.connect_wires([top_xm, bot_xm], lower=bot_vss_hm.lower, upper=bot_vss_hm.upper)
        self.add_pin('VSS', vss, connect=True)      

        # connect wires up to xm
        if num_en > 0:
            out_xm_tidx = track_to_track(self.grid, l_out[0], hm_layer, xm_layer)
            outb_xm_tidx = track_to_track(self.grid, l_outb[0], hm_layer, xm_layer)
            out_xm_tid = TrackID(xm_layer, out_xm_tidx, w_sup_xm)
            outb_xm_tid = TrackID(xm_layer, outb_xm_tidx, w_sup_xm)

            outl_xm = self.connect_to_tracks(l_out_vm, out_xm_tid)
            outbl_xm = self.connect_to_tracks(l_outb_vm, outb_xm_tid)
            outr_xm = self.connect_to_tracks(r_out_vm, out_xm_tid)
            outbr_xm = self.connect_to_tracks(r_outb_vm, outb_xm_tid)
        
            if connect_out_lr:
                out = self.connect_wires([outl_xm, outr_xm])[0]
                outb = self.connect_wires([outbl_xm, outbr_xm])[0]
                self.add_pin('out', out)
                self.add_pin('outb', outb)
            else:
                self.add_pin('tail_inj', outl_xm)
                self.add_pin('tail_injb', outbl_xm)
                self.add_pin('taile_inj', outr_xm)
                self.add_pin('taile_injb', outbr_xm)
        else:
            out_xm_tidx = track_to_track(self.grid, out, hm_layer, xm_layer)
            out_xm_tid = TrackID(xm_layer, out_xm_tidx, w_sup_xm)
            outb_xm_tid = out_xm_tid

        aa_out_xm_tid = self.tr_manager.get_next_track_obj(
            out_xm_tid, 'sup', 'sup', count_rel_tracks=-1)
        if num_aa > 0:
            # Connect aa_wires on xm_layer
            # gate is a signal here, not a supply
            aa_g_xm_tid = self.tr_manager.get_next_track_obj(out_xm_tid, 'sup', 'sig')
            aa_g_xm = self.connect_to_tracks(aa_g_vm, aa_g_xm_tid)
            self.add_pin('VDD', aa_g_xm)
        
            aa_out_xm = self.connect_to_tracks(aa_out_vm, aa_out_xm_tid)
            self.add_pin('tail_aa', aa_out_xm)
        
        mirr_xm_tid = self.tr_manager.get_next_track_obj(
                aa_out_xm_tid, 'sup', 'sup', count_rel_tracks=-2)
        mirr_xm = self.connect_to_tracks(mirr_vm_list, mirr_xm_tid)
        self.add_pin('NBIAS', mirr_xm)

        # sch params
        col_dummies = num_dummy * (2 + cs_stack)
        inner_dummies = num_aa + num_diode * 2
        rpinfo = self.get_tile_pinfo(1).get_row_place_info(0).row_info
        self._sch_params = dict(
            num_aa=num_aa,
            num_en=num_en,
            num_diode=num_diode,
            num_dum = col_dummies + inner_dummies,
            seg_tail = seg_tail,
            cs_stack = cs_stack,
            w_tail=rpinfo.width,
            lch=rpinfo.lch,
            th_tail=rpinfo.threshold
        )

    def _draw_cs(self, col_idx, diode_connect=False) -> Tuple[WireArray, List[WireArray], List[WireArray]]:
        """Draw the current source with optional stacking and diode connect"""
        seg_tail = self.params['seg_tail']
        cs_stack = self.params['cs_stack']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        
        # Get VM tracks
        vm_tidx_1 = self.arr_info.col_to_track(vm_layer, col_idx + seg_tail // 2, mode=RoundMode.GREATER_EQ)
        vm_tidx_2 = self.tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig', 'sig', up=False)
        vm_tidx_3 = self.tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig', 'sig', up=True)
        vm_tid_1 = TrackID(vm_layer, vm_tidx_1)
        vm_tid_2 = TrackID(vm_layer, vm_tidx_2)
        vm_tid_3 = TrackID(vm_layer, vm_tidx_3)

        bot = None
        mirr = []
        last_drain = None
        for tile_idx in range(1, 1 + cs_stack):
            ports = self.add_mos(row_idx=0, col_idx=col_idx, seg=seg_tail, tile_idx=tile_idx)
            if tile_idx == 1:
                s_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=0)
            else:
                s_tid = self.get_track_id(0, MOSWireType.DS, 'sig', 1, tile_idx=tile_idx)
            d_tid = self.get_track_id(0, MOSWireType.DS, 'sig', 0, tile_idx=tile_idx)
            g_tid = self.get_track_id(0, MOSWireType.G, 'sig', 0, tile_idx=tile_idx)
            cs_s_hm = self.connect_to_tracks(ports.s, s_tid)
            cs_d_hm = self.connect_to_tracks(ports.d, d_tid)
            cs_g_hm = self.connect_to_tracks(ports.g, g_tid)
            
            if not bot:
                bot = cs_s_hm
            
            if last_drain:
                tid = vm_tid_2 if (tile_idx % 2 ^ cs_stack % 2) else vm_tid_1
                self.connect_to_tracks([cs_s_hm, last_drain], tid)
            last_drain = cs_d_hm

            mirr.append(cs_g_hm)

        mirr_vm = []

        if cs_stack > 1:
            mirr_vm.append(self.connect_to_tracks(mirr, vm_tid_3))
        
        if diode_connect:
            # Connect the last drain and gate
            # use the same track as the stack, so that it intersects
            mirr_vm.append(self.connect_to_tracks([cs_d_hm, cs_g_hm], vm_tid_3))

        return bot, mirr, mirr_vm, last_drain

    def _draw_column(self, col_idx, even: bool, is_ao: bool = False, 
                     is_mirr: bool = False) -> Tuple[WireArray, WireArray, \
        WireArray, WireArray, WireArray, WireArray, WireArray, List[WireArray], List[WireArray]]:
        """Draw full column and add connection, based on options
        ------------------------------------
        Parameters:
        ------------------------------------
        col_idx: int
            Column index to draw the transistors
        even: bool
            True if the unit idx is even. Used to alternate gate tracks
        is_ao: bool
            True if the column is always on. Changes SW2 to a dummy
        is_mirr: bool
            True if the column is a diode-connected mirror. 
            Changes both SW1 and SW2 to dummies.
        """
        seg_tail = self.params['seg_tail']
        cs_stack = self.params['cs_stack']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        
        bot, mirr, mirr_vm, last_drain = self._draw_cs(col_idx, diode_connect=is_mirr)

        # SW1
        d_idx = -1 if is_ao else 1  # 1 for mirror and for full col. only -1 for AO
        tile_idx = 1 + cs_stack
        ports = self.add_mos(row_idx=0, col_idx=col_idx, seg=seg_tail, tile_idx=tile_idx)
        s_tid = self.get_track_id(0, MOSWireType.DS, 'sig', 0, tile_idx=tile_idx)
        d_tid = self.get_track_id(0, MOSWireType.DS, 'sig', d_idx, tile_idx=tile_idx)
        g_tid = self.get_track_id(0, MOSWireType.G, 'sig', even, tile_idx=tile_idx)
        sw1_s_hm = self.connect_to_tracks(ports.d, s_tid)
        sw1_d_hm = self.connect_to_tracks(ports.s, d_tid)
        sw1_g_hm = self.connect_to_tracks(ports.g, g_tid)

        # SW2
        tile_idx += 1
        ports = self.add_mos(row_idx=0, col_idx=col_idx, seg=seg_tail, tile_idx=tile_idx)
        s_tid = self.get_track_id(0, MOSWireType.DS, 'sig', 0, tile_idx=tile_idx)
        d_tid = self.get_track_id(0, MOSWireType.DS, 'sig', 1, tile_idx=tile_idx)
        g_tid = self.get_track_id(0, MOSWireType.G, 'sig', even, tile_idx=tile_idx)
        sw2_s_hm = self.connect_to_tracks(ports.d, s_tid)
        sw2_d_hm = self.connect_to_tracks(ports.s, d_tid)
        sw2_g_hm = self.connect_to_tracks(ports.g, g_tid)

        # Get VM tracks
        vm_tidx_1 = self.arr_info.col_to_track(vm_layer, col_idx + seg_tail // 2, mode=RoundMode.GREATER_EQ)
        vm_tidx_2 = self.tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig', 'sig', up=False)
        vm_tidx_3 = self.tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig', 'sig', up=True)
        vm_tid_1 = TrackID(vm_layer, vm_tidx_1)
        vm_tid_2 = TrackID(vm_layer, vm_tidx_2)
        vm_tid_3 = TrackID(vm_layer, vm_tidx_3)

        # Process SW1
        if is_mirr:
            # Treat as dummy
            top_vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=self.num_tiles-1)
            vss_gs = self.connect_to_tracks([sw1_s_hm, sw1_g_hm], vm_tid_2)
            self.connect_to_tracks(vss_gs, top_vss_hm_tid)
            vss_d = self.connect_to_tracks([sw1_d_hm], vm_tid_3)
            self.connect_to_tracks(vss_d, top_vss_hm_tid)
        else:
            # Connect mid, en, out
            self.connect_to_tracks([last_drain, sw1_s_hm], vm_tid_2)
            if is_ao:
                # Don't extend
                en = self.connect_to_tracks(sw1_g_hm, vm_tid_1)
            else:
                en = self.connect_to_tracks(sw1_g_hm, vm_tid_1, track_upper=self.bound_box.yh)
            out = sw1_d_hm
            out_vm = self.connect_to_tracks(out, vm_tid_3, min_len_mode=MinLenMode.MIDDLE)

        # Process SW2
        if is_ao or is_mirr:
            # Treat as dummy
            top_vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=self.num_tiles-1)
            vss_gs = self.connect_to_tracks([sw2_s_hm, sw2_g_hm], vm_tid_2)
            self.connect_to_tracks(vss_gs, top_vss_hm_tid)
            vss_d = self.connect_to_tracks([sw2_d_hm], vm_tid_3)
            self.connect_to_tracks(vss_d, top_vss_hm_tid)
        else:
            # Connect mid, en, out
            self.connect_to_tracks([last_drain, sw2_s_hm], vm_tid_2)
            enb = self.connect_to_tracks(sw2_g_hm, vm_tid_2, track_upper=self.bound_box.yh)
            outb = sw2_d_hm
            outb_vm = self.connect_to_tracks(outb, vm_tid_3, min_len_mode=MinLenMode.MIDDLE)

        # Return statements, for legacy compatibility
        if is_mirr:
            return bot, mirr, mirr_vm, last_drain
        elif is_ao:
            return en, out, out_vm, bot, mirr, mirr_vm, last_drain
        else:
            return en, enb, out, outb, out_vm, outb_vm, bot, mirr, mirr_vm, last_drain

    def _draw_mirr_column(self, col_idx):
        # Utility function
        return self._draw_column(col_idx, even=False, is_ao=False, is_mirr=True)

    def _draw_aa_column(self, col_idx):
        # Utility function
        return self._draw_column(col_idx, even=False, is_ao=True, is_mirr=False)

    def _draw_dummy_column(self, col_idx) -> None:
        """Draw a column of all dummies"""
        seg_tail = self.params['seg_tail']
        cs_stack = self.params['cs_stack']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        
        g_hm, s_hm, d_hm = [], [], []

        for tile_idx in range(1, 1 + cs_stack + 2):    
            ports = self.add_mos(row_idx=0, col_idx=col_idx, seg=seg_tail, tile_idx=tile_idx)
            s_tid = self.get_track_id(0, MOSWireType.DS, 'sig', 0, tile_idx=tile_idx)
            d_tid = self.get_track_id(0, MOSWireType.DS, 'sig', 1, tile_idx=tile_idx)
            g_tid = self.get_track_id(0, MOSWireType.G, 'sig', 0, tile_idx=tile_idx)
            s_hm.append(self.connect_to_tracks(ports.d, s_tid))
            d_hm.append(self.connect_to_tracks(ports.s, d_tid))
            g_hm.append(self.connect_to_tracks(ports.g, g_tid))

        # Get VM tracks
        vm_tidx_1 = self.arr_info.col_to_track(vm_layer, col_idx + seg_tail // 2, mode=RoundMode.GREATER_EQ)
        vm_tidx_2 = self.tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig', 'sig', up=False)
        vm_tidx_3 = self.tr_manager.get_next_track(vm_layer, vm_tidx_1, 'sig', 'sig', up=True)
        vm_tid_1 = TrackID(vm_layer, vm_tidx_1, width=self.tr_manager.get_width(vm_layer, 'sup'))
        vm_tid_2 = TrackID(vm_layer, vm_tidx_2)
        vm_tid_3 = TrackID(vm_layer, vm_tidx_3)

        # Connect SW1 + SW2 dummies
        bot_vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=0)
        top_vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=self.num_tiles-1)
        
        vss_gs = self.connect_to_tracks(g_hm + s_hm, vm_tid_2)
        self.connect_to_tracks(vss_gs, top_vss_hm_tid)
        self.connect_to_tracks(vss_gs, bot_vss_hm_tid)
        vss_d = self.connect_to_tracks(d_hm, vm_tid_3)
        self.connect_to_tracks(vss_d, top_vss_hm_tid)
        self.connect_to_tracks(vss_d, bot_vss_hm_tid)

        self.add_pin('VSS', [vss_gs, vss_d], connect=True)
