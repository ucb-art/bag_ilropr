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

"""This library contains classes for the CML2CMOS - AC coupled TIA"""

from typing import Any, Dict, Type, Optional, List, Sequence, Mapping

from pyparsing import col

from pybag.enum import MinLenMode, PinMode, RoundMode

from bag.util.immutable import Param
from bag.layout.routing.base import TrackID
from bag.design.module import Module
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.module import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.cap.mos import MOMCapOnMOS

from bag3_digital.layout.stdcells.gates import InvCore

from ..schematic.cml2cmos_diff_ac import bag_ilropr__cml2cmos_diff_ac
from ..schematic.c2c_array import bag_ilropr__c2c_array
from .util import track_to_track


class CML2CMOS_MOS(MOSBase):
    """Single Inverter pair and stacked CMOS resistors
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        
    @classmethod
    def tiles(cls) -> List[Dict[str, str]]:
        return [dict(name='logic_tile'), dict(name='logic_tile', flip=True)]

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            inv_seg_list='Segments for the inverter pair',
            res_params='Parameters for CMOS resistors',            
            res_nser='Number of resistors in series',
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        inv_seg_list: List[int] = self.params['inv_seg_list']
        res_params: Dict[str, Any] = self.params['res_params']
        res_nser: int = self.params['res_nser']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        assert res_nser % 2 == 0  # Even number of resistors, for even rows

        # set total number of columns
        res_stack = res_params['stack']

        col_res = res_nser // 2 * res_stack
        col_inv = max(inv_seg_list)
        seg_tot = col_inv + self.min_sep_col + col_res
        seg_tot += self.sub_sep_col + self.min_sub_col
        self.set_mos_size(seg_tot, num_tiles=2)

        # --- Placement --- #
        grid = self.grid
        tr_manager = self.tr_manager

        w_sig_hm = tr_manager.get_width(hm_layer, 'sig')
        w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        w_sig_xm = tr_manager.get_width(xm_layer, 'sig')

        w_sig_hs_hm = tr_manager.get_width(hm_layer, 'sig_hs')
        w_sig_hs_vm = tr_manager.get_width(vm_layer, 'sig_hs')
        w_sig_hs_xm = tr_manager.get_width(xm_layer, 'sig_hs')

        w_sup_hm = tr_manager.get_width(hm_layer, 'sup')
        w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
        w_sup_xm = tr_manager.get_width(xm_layer, 'sup')

        # Create and place inv templates
        inv_tiles = ['logic_tile']
        inv_pinfo = self.params['pinfo'].copy(append=dict(tiles=inv_tiles))

        inv0_params = dict(pinfo=inv_pinfo, seg=inv_seg_list[0], vertical_out=False)
        inv0_template = self.new_template(InvCore, params=inv0_params)

        in_tidx = self.get_track_index(1, MOSWireType.G, 'sig', 0)
        inv1_params = dict(pinfo=inv_pinfo, seg=inv_seg_list[1], vertical_out=False, sig_locs={'in': in_tidx})
        inv1_template = self.new_template(InvCore, params=inv1_params)

        col_idx = 0
        tile0, tile1 = 0, 1
        inv0_inst = self.add_tile(inv0_template, tile_idx=tile0, col_idx=col_idx)
        inv1_inst = self.add_tile(inv1_template, tile_idx=tile1, col_idx=col_idx)
        col_idx += col_inv + self.min_sep_col

        # Create and place CMOS res stacks
        seg_res = 1
        res_ports = []
        for tile_idx in [tile0, tile1]:
            nport = self.add_mos(0, col_idx=col_idx, tile_idx=tile_idx, seg=seg_res, stack=res_stack)
            pport = self.add_mos(1, col_idx=col_idx, tile_idx=tile_idx, seg=seg_res, stack=res_stack)
            res_ports.append((nport, pport))

        col_idx += res_stack * seg_res + self.sub_sep_col + self.min_sub_col

        # Create and place substrate contacts
        sub0_vss = self.add_substrate_contact(0, col_idx, tile_idx=tile0, seg=self.min_sub_col, flip_lr=True)
        sub0_vdd = self.add_substrate_contact(1, col_idx, tile_idx=tile0, seg=self.min_sub_col, flip_lr=True)

        sub1_vss = self.add_substrate_contact(0, col_idx, tile_idx=tile1, seg=self.min_sub_col, flip_lr=True)
        sub1_vdd = self.add_substrate_contact(1, col_idx, tile_idx=tile1, seg=self.min_sub_col, flip_lr=True)
        
        # connect contacts
        inv_vdd = self.connect_wires([inv0_inst.get_pin('VDD'), inv1_inst.get_pin('VDD')])[0]
        inv_vss_bot = inv0_inst.get_pin('VSS')
        inv_vss_top = inv1_inst.get_pin('VSS')

        inv_vdd = self.connect_to_track_wires([sub0_vdd, sub1_vdd], inv_vdd)
        inv_vss_bot = self.connect_to_track_wires(sub0_vss, inv_vss_bot)
        inv_vss_top = self.connect_to_track_wires(sub1_vss, inv_vss_top)

        # connect inv and supply vm
        lower_tidx = self.grid.coord_to_track(vm_layer, self.place_info.col_to_coord(0))
        _, tidx_list = self.tr_manager.place_wires(vm_layer, ['sup', 'sig_hs', 'sig_hs'], 
                                                   align_track=lower_tidx, align_idx=0)
        in_tid = TrackID(vm_layer, tidx_list[1], w_sig_hs_vm)
        mid_tid = TrackID(vm_layer, tidx_list[2], w_sig_hs_vm)
        in_warr = self.connect_to_tracks(inv0_inst.get_pin('in'), in_tid, track_lower=self.bound_box.yl)
        mid_warr = self.connect_to_tracks(
            [inv1_inst.get_pin('in'), inv0_inst.get_pin('nout'), inv0_inst.get_pin('pout')], mid_tid)
        out_warr = self.connect_to_tracks([inv1_inst.get_pin('nout'), inv1_inst.get_pin('pout')], 
                                          in_tid, track_upper=self.bound_box.yh)
        self.add_pin('in', in_warr)
        self.add_pin('out', out_warr)

        vss_vm, vdd_vm = [], []
        # free vm tracks are sparse. Use what is available without conflicting.
        tidx_h = self.grid.coord_to_track(vm_layer, self.place_info.col_to_coord(seg_tot))
        for _idx, tidx in enumerate([tidx_list[0], tidx_h]):
            tid = TrackID(vm_layer, tidx, w_sup_vm)
            vss_vm.append(self.connect_to_tracks([inv_vss_bot, inv_vss_top], tid))
        
        tid = TrackID(vm_layer, tidx_list[1], w_sig_hs_vm)
        vdd_vm.append(self.connect_to_tracks(inv_vdd, tid))

        # Connect res gates
        cntr_coord = self.place_info.col_to_coord(col_inv + self.min_sep_col + (res_stack * seg_res) // 2)
        _, res_tidx_list = self.tr_manager.place_wires(vm_layer, ['sig', 'sig'], center_coord=cntr_coord)
        for tile_idx, vss_hm, vdd_hm in [(tile0, inv_vss_bot, inv_vdd), (tile1, inv_vss_top, inv_vdd)]:
            g_hm0 = self.get_track_id(0, MOSWireType.G, 'sig', 1, tile_idx=tile_idx)
            g_hm1 = self.get_track_id(1, MOSWireType.G, 'sig', -1, tile_idx=tile_idx)
            g_hm0 = self.connect_to_tracks(res_ports[tile_idx][0].g, g_hm0)
            g_hm1 = self.connect_to_tracks(res_ports[tile_idx][1].g, g_hm1)
            g_vm0 = TrackID(vm_layer, res_tidx_list[0], w_sig_vm)
            g_vm1 = TrackID(vm_layer, res_tidx_list[1], w_sig_vm)
            vdd_vm.append(self.connect_to_tracks([g_hm0, vdd_hm], g_vm0))
            vss_vm.append(self.connect_to_tracks([g_hm1, vss_hm], g_vm1))

        # Connect res mid
        coord = self.place_info.col_to_coord(col_inv + self.min_sep_col + (res_stack * seg_res))
        tidx = self.grid.coord_to_track(vm_layer, coord, RoundMode.GREATER_EQ)
        tid = TrackID(vm_layer, tidx, w_sig_hs_vm)
        hm_list = []
        for tile_idx in [tile0, tile1]:
            d_hm0 = self.get_track_id(0, MOSWireType.DS, 'sig', 1, tile_idx=tile_idx)
            d_hm1 = self.get_track_id(1, MOSWireType.DS, 'sig', 0, tile_idx=tile_idx)
            # Drain is on the right
            d_hm0 = self.connect_to_tracks(res_ports[tile_idx][0].d, d_hm0)
            d_hm1 = self.connect_to_tracks(res_ports[tile_idx][1].d, d_hm1)
            hm_list.extend([d_hm0, d_hm1])
        self.connect_to_tracks(hm_list, tid)

        # Connect res in and out
        # Place to the left of source to satisfy DRC in some techs
        coord = self.place_info.col_to_coord(col_inv + self.min_sep_col)
        _tidx = self.grid.coord_to_track(vm_layer, coord, mode=RoundMode.LESS_EQ)
        _tidx = tr_manager.get_next_track(vm_layer, _tidx, 'sig', 'sig', up=-1)
        tid = TrackID(vm_layer, _tidx, w_sig_vm)
        for tile_idx, in_warr in [(tile0, inv0_inst.get_pin('in')), 
                                  (tile1, inv1_inst.get_pin('in'))]:
            s_hm0 = self.get_track_id(0, MOSWireType.DS, 'sig', 0, tile_idx=tile_idx)
            s_hm1 = self.get_track_id(1, MOSWireType.DS, 'sig', 1, tile_idx=tile_idx)
            # Source is on the left
            s_hm0 = self.connect_to_tracks(res_ports[tile_idx][0].s, s_hm0)
            s_hm1 = self.connect_to_tracks(res_ports[tile_idx][1].s, s_hm1)
            self.connect_to_tracks([s_hm0, s_hm1, in_warr], tid)            

        # free vm tracks are sparse. Use what is available without conflicting.
        vdd_vm.append(self.connect_to_tracks(inv_vdd, tid))

        # Add supply xm tracks
        vdd_xm_tidx = track_to_track(self.grid, inv_vdd, hm_layer, xm_layer)
        vss0_xm_tidx = track_to_track(self.grid, inv_vss_bot, hm_layer, xm_layer)
        vss1_xm_tidx = track_to_track(self.grid, inv_vss_top, hm_layer, xm_layer)

        vdd_xm = self.connect_to_tracks(vdd_vm, TrackID(xm_layer, vdd_xm_tidx, w_sup_xm))
        self.add_pin('VDD', vdd_xm, connect=True)
        self.add_pin('VDD', [inv_vdd], connect=True)
        vss0_xm = self.connect_to_tracks(vss_vm, TrackID(xm_layer, vss0_xm_tidx, w_sup_xm))
        vss1_xm = self.connect_to_tracks(vss_vm, TrackID(xm_layer, vss1_xm_tidx, w_sup_xm))
        vss_xm = self.connect_wires([vss0_xm, vss1_xm])
        self.add_pin('VSS', vss_xm, connect=True)
        self.add_pin('VSS', [inv_vss_top, inv_vss_bot], connect=True)

        # free vm tracks are sparse. Use what is available without conflicting.
        self.connect_to_tracks([vss0_xm, inv_vss_bot], tid, min_len_mode=MinLenMode.MIDDLE)
        self.connect_to_tracks([vss1_xm, inv_vss_top], tid, min_len_mode=MinLenMode.MIDDLE)
        tid = TrackID(vm_layer, tidx_list[-1], w_sig_hs_vm)
        self.connect_to_tracks([vss0_xm, inv_vss_bot], tid, min_len_mode=MinLenMode.MIDDLE)
        self.connect_to_tracks([vss1_xm, inv_vss_top], tid, min_len_mode=MinLenMode.MIDDLE)
        
        # Sch params
        rninfo = self.get_row_info(0, tile_idx=0)
        rpinfo = self.get_row_info(1, tile_idx=0)
        inv_params = dict(
            seg_n=inv_seg_list[0],
            seg_p=inv_seg_list[0],
            w_n=rninfo.width,
            w_p=rpinfo.width,
            lch=rninfo.lch,
            th_n=rninfo.threshold,
            th_p=rpinfo.threshold
        )
        invout_params = inv_params.copy()
        invout_params.update(dict(seg_n=inv_seg_list[1], seg_p=inv_seg_list[1]))
        res_params = dict(
            res_type='mos',
            nser=res_nser,
            unit_params=dict(
                lch=rninfo.lch,
                w_dict=dict(n=rninfo.width, p=rpinfo.width),
                th_dict=dict(n=rninfo.threshold, p=rpinfo.threshold),
                seg=1,
                stack=res_stack
            )
        )
        self.sch_params = dict(
            inv_params=inv_params,
            invout_params=invout_params,
            has_invout=True,
            res_params=res_params
        )


class CML2CMOS_MOSDiff(MOSBase):
    """Differential pair of CML2CMOS_MOS"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
    
    @classmethod
    def tiles(cls):
        return CML2CMOS_MOS.tiles()

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return CML2CMOS_MOS.get_params_info()

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        tr_manager = self.tr_manager

        half_template: CML2CMOS_MOS = self.new_template(CML2CMOS_MOS, params=self.params)
        half_col = half_template.num_cols
        self.set_mos_size(half_col * 2, num_tiles=2)

        # --- Placement --- #
        # Create and place inv templates
        inv0_inst = self.add_tile(half_template, tile_idx=0, col_idx=half_col, flip_lr=True)
        inv1_inst = self.add_tile(half_template, tile_idx=0, col_idx=half_col)

        # --- Routing  --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        # Bring in from vm up to xm... 
        inp, inm = inv0_inst.get_pin('in'), inv1_inst.get_pin('in')
        vss_warr = inv0_inst.get_pin('VSS', layer=xm_layer)[0]
        tidx0 = tr_manager.get_next_track(xm_layer, vss_warr.track_id.base_index, 'sup', 'sig_hs')
        tidx1 = tr_manager.get_next_track(xm_layer, tidx0, 'sig_hs', 'sig_hs')
        inp, inm = self.connect_differential_tracks(inp, inm, xm_layer, tidx0, tidx1,
                                                    width=tr_manager.get_width(xm_layer, 'sig_hs'))
        
        # ... and up to ym
        cntr_coord = self.place_info.col_to_coord(half_col)
        _, tidx_list = tr_manager.place_wires(ym_layer, ['sig_hs', 'sup', 'sig_hs'], center_coord=cntr_coord)
        w_sig_hs_ym = tr_manager.get_width(ym_layer, 'sig_hs')
        tid0 = TrackID(ym_layer, tidx_list[0], w_sig_hs_ym)
        tid1 = TrackID(ym_layer, tidx_list[-1], w_sig_hs_ym)

        inp = self.connect_to_tracks(inp, tid0)
        inm = self.connect_to_tracks(inm, tid1)
        
        self.add_pin('inp', inp)
        self.add_pin('inm', inm)

        # Bring out from vm up to xm...
        outm, outp  = inv0_inst.get_pin('out'), inv1_inst.get_pin('out')
        vss_warr = inv0_inst.get_pin('VSS', layer=xm_layer)[-1]
        tidx0 = tr_manager.get_next_track(xm_layer, vss_warr.track_id.base_index, 'sup', 'sig_hs', up=-1)
        tidx1 = tr_manager.get_next_track(xm_layer, tidx0, 'sig_hs', 'sig_hs', up=-1)
        outm, outp = self.connect_differential_tracks(outm, outp, xm_layer, tidx0, tidx1, 
                                                      width=tr_manager.get_width(xm_layer, 'sig_hs'))
        
        # ...and up to ym
        outm, outp = self.connect_differential_tracks(outm, outp, ym_layer, tidx_list[0], tidx_list[-1], 
                                                      width=tr_manager.get_width(ym_layer, 'sig_hs'),
                                                      track_upper=self.bound_box.yh)
        self.add_pin('outm', outm, mode=PinMode.UPPER)
        self.add_pin('outp', outp, mode=PinMode.UPPER)

        # Connect supplies
        sup_warr = []
        for lay in [hm_layer, xm_layer]:
            for sup in ['VDD', 'VSS']:
                warr = self.connect_wires([inv0_inst.get_all_port_pins(sup, layer=lay)[0], 
                                           inv1_inst.get_all_port_pins(sup, layer=lay)[0]])[0]
                sup_warr.append(warr)
                self.add_pin(sup, warr, connect=True)
        
        # Connect VSS up to xxm
        vss_xm = warr
        vss_xxm = []
        w_sup_xm = tr_manager.get_width(xm_layer, 'sup')
        for _warr in [vss_xm[0], vss_xm[1]]:
            for ref_tidx, up in [(tidx_list[0], -1), (tidx_list[-1], +1)]:
                nxt_tidx = tr_manager.get_next_track(ym_layer, ref_tidx, 'sig_hs', 'sup', up)
                coord = self.grid.track_to_coord(ym_layer, nxt_tidx)
                if up > 0:
                    lower, upper = coord, self.bound_box.xh
                else:
                    lower, upper = self.bound_box.xl, coord
                xm_warr = self.add_wires(xm_layer, _warr.track_id.base_index, lower, upper, width=w_sup_xm)
                vss_xxm.append(self.connect_via_stack(tr_manager, xm_warr, xxm_layer, 'sup'))
        vss_xxm = self.connect_wires(vss_xxm)[0]
        # also include the middle tidx
        tid = TrackID(ym_layer, tidx_list[1], tr_manager.get_width(ym_layer, 'sup'))
        self.connect_to_tracks([vss_xm[0], vss_xxm[0]], tid)
        self.connect_to_tracks([vss_xm[1], vss_xxm[1]], tid)
        self.add_pin('VSS', vss_xxm, connect=True)

        # Connect VDD up to xxm
        vdd_xm = sup_warr[-2]
        vdd_xxm = self.connect_via_stack(tr_manager, vdd_xm, xxm_layer, 'sup')
        self.add_pin('VDD', vdd_xxm, connect=True)

        self.sch_params = half_template.sch_params


class CML2CMOS_Diff(MOSBase):
    """CML2CMOS MOS with Cap inputs"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__cml2cmos_diff_ac

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cap_params="Parameters for MOMCapOverMOS",
            sch_cap_value="Cap value for schematic simulations",
            export_mid="True to export midp/m",
            **CML2CMOS_MOS.get_params_info()
        )

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(export_mid=False, sch_cap_value=0)

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        grid = self.grid
        tr_manager = self.tr_manager

        core_params = self.params.copy(remove=['cap_params'])
        core_pinfo = self.params['pinfo'].copy(append=dict(tiles=CML2CMOS_MOSDiff.tiles()))
        core_params = core_params.copy(append=dict(pinfo=core_pinfo))
        core_template: CML2CMOS_MOSDiff = self.new_template(CML2CMOS_MOSDiff, params=core_params)

        cap_params: Mapping[str, Any] = self.params['cap_params']
        cap_pinfo = self.get_tile_info(1)
        cap_params = cap_params.copy(append=dict(pinfo=cap_pinfo))
        cap_template: MOMCapOnMOS = self.new_template(MOMCapOnMOS, params=cap_params)
        
        num_cols = max(core_template.num_cols, cap_template.num_cols * 2)

        # --- Placement --- #

        # Create and place templates
        core_inst = self.add_tile(core_template, tile_idx=3, col_idx=num_cols // 2 - core_template.num_cols // 2)
        cap0_inst = self.add_tile(cap_template, tile_idx=1, col_idx=num_cols // 2, flip_lr=True)
        cap1_inst = self.add_tile(cap_template, tile_idx=1, col_idx=num_cols // 2)

        # --- Routing  --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        # Add substrate taps
        vss_hm = []
        for sub_tile_idx in [0, 2]:
            vss_conn = self.add_substrate_contact(0, 0, tile_idx=sub_tile_idx, seg=num_cols)
            vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=sub_tile_idx)
            vss_hm.append(self.connect_to_tracks(vss_conn, vss_hm_tid))
        
        vss_vm_col = [0, num_cols // 2, num_cols]
        vss_vm_tidx = [grid.coord_to_track(vm_layer, self.place_info.col_to_coord(col)) for col in vss_vm_col]
        w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
        vss_vm_tid = [TrackID(vm_layer, tidx, w_sup_vm) for tidx in vss_vm_tidx]
        vss_vm = [self.connect_to_tracks(vss_hm, tid) for tid in vss_vm_tid]
        self.connect_to_track_wires(vss_vm, core_inst.get_all_port_pins('VSS', xm_layer))

        # self.add_pin('VSS', [vss0, vss1], connect=True)

        self.set_mos_size(num_cols=num_cols)

        # Export output
        self.reexport(core_inst.get_port('outp'))
        self.reexport(core_inst.get_port('outm'))

        # Connect intermediate node        
        midp = self.connect_to_track_wires(cap0_inst.get_pin('minus', layer=xm_layer), core_inst.get_pin('inp'))
        midm = self.connect_to_track_wires(cap1_inst.get_pin('minus', layer=xm_layer), core_inst.get_pin('inm'))
        if self.params['export_mid']:   
            self.add_pin('midp', midp)
            self.add_pin('midm', midm)

        # Export input
        # inp = self.connect_to_tracks(cap0_inst.get_pin('plus', layer=xm_layer), tid0, track_lower=self.bound_box.yl)
        # inm = self.connect_to_tracks(cap1_inst.get_pin('plus', layer=xm_layer), tid1, track_lower=self.bound_box.yl)
        inp = cap0_inst.get_pin('plus', layer=xm_layer)
        inm = cap1_inst.get_pin('plus', layer=xm_layer)
        inp = self.connect_via_stack(tr_manager, inp, xxm_layer, 'sig_hs')
        inm = self.connect_via_stack(tr_manager, inm, xxm_layer, 'sig_hs')
        
        self.add_pin('inp', inp)
        self.add_pin('inm', inm)

        self.reexport(core_inst.get_port('VDD'))
        self.reexport(core_inst.get_port('VSS'), connect=True)

        # Sch_params 
        cap_params = cap_template.sch_params
        if self.params['sch_cap_value']:
            cap_params = cap_params.copy(append=dict(value=self.params['sch_cap_value']))
        self.sch_params = dict(
            cap_params=cap_params,
            export_mid=self.params['export_mid'],
            **core_template.sch_params
        )


class CML2CMOS_Array(MOSBase):
    """Array of C2C Diff, for compact layout"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__c2c_array

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            num_units="Number of C2C Diff units",
            **CML2CMOS_Diff.get_params_info()
        )

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return CML2CMOS_Diff.get_default_param_values()

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        num_units = self.params['num_units']
        assert num_units > 0

        unit_params = self.params.copy(remove=['num_units'])
        unit_template: CML2CMOS_Diff = self.new_template(CML2CMOS_Diff, params=unit_params)
        
        num_cols = unit_template.num_cols * num_units
        self.set_mos_size(num_cols=num_cols)

        # --- Placement --- #

        # Create and place unit_templates
        inst_list = []
        for idx in range(num_units):
            col_idx = unit_template.num_cols * idx
            inst_list.append(self.add_tile(unit_template, 0, col_idx))
        
        # --- Routing --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        # Connect supplies
        for sup in ['VDD', 'VSS']:
            for layer in [xm_layer, xxm_layer]:
                sup_list = []
                for inst in inst_list:
                    sup_list.extend(inst.get_all_port_pins(sup, layer))
                sup_warr = self.connect_wires(sup_list)
                self.add_pin(sup, sup_warr, connect=True)

        # Export pins
        terms = ['inp', 'inm', 'outp', 'outm']
        if self.params['export_mid']:
            terms += ['midp', 'midm']
        for idx, inst in enumerate(inst_list):
            for term in terms:
                pin_name = f'{term}<{idx}>' if num_units > 1 else term
                self.add_pin(pin_name, inst.get_pin(term))

        self.sch_params = dict(
            num_units=num_units,
            **unit_template.sch_params
        )
