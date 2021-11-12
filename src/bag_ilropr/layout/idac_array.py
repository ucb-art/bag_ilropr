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

from pybag.enum import MinLenMode, PinMode, RoundMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID, WireArray
from bag.design.module import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from .util import track_to_track
from .idac_unit import IDACUnit
from ..schematic.idac_array import bag_ilropr__idac_array


class IDACArray(MOSBase):
    """Array of IDACUnits"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
    
    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__idac_array

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            num_stages='',
            pinfo='The MOSBasePlaceInfo object.',
            seg_tail='Segments for the unit device. Value shared by the CS and switch',
            cs_stack='Current source stack height',
            num_diode='Number of units in the diode connected device',
            num_en='Number of en / enb bits.',
            num_aa='Number of always on units.',
            num_dum='Number of dummy units.',
            connect_out_lr="True to connect the left and right outs",
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            num_stages=4,
            cs_stack=1,
            num_diode=0,
            num_aa=0,
            num_dum=2,
            connect_out_lr=False
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        num_stages: int = self.params['num_stages']
        seg_tail: int = self.params['seg_tail']
        cs_stack: int = self.params['cs_stack']
        num_diode: int = self.params['num_diode']
        num_en: int = self.params['num_en']
        num_aa: int = self.params['num_aa']
        num_dummy: int = self.params['num_dum']
        connect_out_lr: bool = self.params['connect_out_lr']

        """
        Pinfo is expected to be in tile format, and include:
        - ptap bottom row
        - 2 rows of nch switches, followed by...
        - ...`cs_stack` rows of nch current source
        - ptap top row
        """
        num_tiles_unit = 4 + cs_stack + 1
        num_tiles = num_tiles_unit * num_stages + 1

        # ========= Placement =========

        # Create IDAC unit templates
        unit_params = self.params.copy(remove=['num_stages'])
        unit_template = self.new_template(IDACUnit, params=unit_params)
        inst_list = []
        for idx in range(num_stages):
            tile_idx = idx * num_tiles_unit
            inst_list.append(self.add_tile(unit_template, tile_idx, 0))

        self.set_mos_size(unit_template.num_cols, num_tiles)

        # ========= Routing =========
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        w_sup_xm = self.tr_manager.get_width(xm_layer, 'sup')

        # route enables
        for idx, inst in enumerate(inst_list):
            route_tile_idx = num_tiles_unit * (idx + 1) - 1
            for en_idx in range(num_en):
                en_vm = inst.get_pin(f'en<{en_idx}>')
                enb_vm = inst.get_pin(f'enb<{en_idx}>')
                en_tid = self.get_hm_track_id(xm_layer, 'sig', en_idx, tile_idx=route_tile_idx)
                enb_tid = self.get_hm_track_id(hm_layer, 'sig', en_idx, tile_idx=route_tile_idx)
                en_warr = self.connect_to_tracks(en_vm, en_tid, track_lower=self.bound_box.xl)
                enb_warr = self.connect_to_tracks(enb_vm, enb_tid, track_lower=self.bound_box.xl)
                _idx = num_en // 2 * idx + en_idx
                if en_idx >= num_en // 2:
                    _idx += (num_en // 2) * (num_stages - 1)
                self.add_pin(f'en<{_idx}>', en_warr, mode=PinMode.LOWER)
                self.add_pin(f'enb<{_idx}>', enb_warr, mode=PinMode.LOWER)

        # route tails
        tails = ['tail_aa', 'tail_injb', 'tail_inj', 'taile_injb', 'taile_inj']
        for idx, inst in enumerate(inst_list):
            for tail in tails:
                tail_pin = inst.get_pin(tail)
                tail_pin = self.connect_via_stack(self.tr_manager, tail_pin, xxm_layer, 'sup')
                self.add_pin(f'{tail}<{idx}>', tail_pin)

        # route supplies
        # Connect shared vm on dummy sides
        vss_vm = []
        for inst in inst_list:
            vss_vm.extend(inst.get_all_port_pins('VSS', vm_layer))
        self.connect_wires(vss_vm)
        # xm layer is already routed for each inst
        # via up to xxm
        vss_xm = []
        for inst in inst_list:
            _xm = inst.get_all_port_pins('VSS', xm_layer)[0]
            vss_xm.append(_xm[0])
            vss_xm.append(_xm[1])
        ret_warr_dict = {}
        vss_xxm = []
        for _xm in vss_xm:
            vss_xxm.append(self.connect_via_stack(self.tr_manager, _xm, xxm_layer, 'sup', ret_warr_dict=ret_warr_dict))
        self.add_pin('VSS', vss_xxm, connect=True)

        for inst in inst_list:
            _xm = inst.get_pin('VDD')
            _xxm = self.connect_via_stack(self.tr_manager, _xm, xxm_layer, 'sup')
            self.add_pin('VDD', _xxm, connect=True)

        # Shared bias. Route together at higher hierarchy
        nbias_list = []
        for inst in inst_list:
            _xm = inst.get_pin('NBIAS')
            nbias_list.append(self.connect_via_stack(self.tr_manager, _xm, xxm_layer, 'sup'))
        self.add_pin('NBIAS', self.connect_wires(nbias_list), connect=True)

        # sch params
        self._sch_params = dict(
            num_stages=num_stages,
            **unit_template.sch_params
        )


class IMIRRArray(MOSBase):
    """Array of IDACUnits"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
    
    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag_ilropr__idac_array

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            num_stages='',
            pinfo='The MOSBasePlaceInfo object.',
            seg_tail='Segments for the unit device. Value shared by the CS and switch',
            cs_stack='Current source stack height',
            num_diode='Number of units in the diode connected device',
            num_en='Number of en / enb bits.',
            num_aa='Number of always on units.',
            num_dum='Number of dummy units.',
            connect_out_lr="True to connect the left and right outs",
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            num_stages=4,
            cs_stack=1,
            num_diode=0,
            num_aa=0,
            num_dum=2,
            connect_out_lr=False
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        num_stages: int = self.params['num_stages']
        seg_tail: int = self.params['seg_tail']
        cs_stack: int = self.params['cs_stack']
        num_diode: int = self.params['num_diode']
        num_en: int = self.params['num_en']
        num_aa: int = self.params['num_aa']
        num_dummy: int = self.params['num_dum']
        connect_out_lr: bool = self.params['connect_out_lr']

        """
        Pinfo is expected to be in tile format, and include:
        - ptap bottom row
        - 2 rows of nch switches, followed by...
        - ...`cs_stack` rows of nch current source
        - ptap top row ONLY on the top row (manually added at this level)
        The last step is for added compactness
        """
        num_tiles_unit = 3 + cs_stack
        num_tiles = num_tiles_unit * num_stages + 1

        # ========= Placement =========

        # Create IDAC unit templates
        unit_params = self.params.copy(append=dict(has_top_sub=False, remove=['num_stages']))
        unit_template = self.new_template(IDACUnit, params=unit_params)
        inst_list = []
        for idx in range(num_stages):
            tile_idx = idx * num_tiles_unit
            inst_list.append(self.add_tile(unit_template, tile_idx, 0))

        # Add top tap
        top_vss = self.add_substrate_contact(row_idx=0, col_idx=0, tile_idx=num_tiles-1, seg=unit_template.num_cols)

        hm_layer = top_vss.layer_id + 1
        top_vss_hm_tid = self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=num_tiles-1)
        # This should already overlap with the templates
        top_vss_hm = self.connect_to_tracks(top_vss, top_vss_hm_tid)  

        self.set_mos_size(unit_template.num_cols, num_tiles)

        # ========= Routing =========
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xxm_layer = ym_layer + 1

        # route tails
        tails = ['tail_aa']
        for idx, inst in enumerate(inst_list):
            for tail in tails:
                tail_pin = inst.get_pin(tail)
                tail_pin = self.connect_via_stack(self.tr_manager, tail_pin, xxm_layer, 'sup')
                self.add_pin(f'{tail}<{idx}>', tail_pin)

        # route supplies
        # Connect shared vm on dummy sides
        vss_vm = []
        for inst in inst_list:
            vss_vm.extend(inst.get_all_port_pins('VSS', vm_layer))
        self.connect_wires(vss_vm)
        self.connect_to_track_wires(vss_vm, top_vss_hm)  # Extra
        
        # xm layer is already routed for each inst
        # via up to xxm
        vss_xm = []
        for inst in inst_list:
            _xm = inst.get_all_port_pins('VSS', xm_layer)[0]
            vss_xm.append(_xm[0])
            vss_xm.append(_xm[1])
        ret_warr_dict = {}
        vss_xxm = []
        for _xm in vss_xm:
            vss_xxm.append(self.connect_via_stack(self.tr_manager, _xm, xxm_layer, 'sup', ret_warr_dict=ret_warr_dict))
        self.add_pin('VSS', vss_xxm, connect=True)

        for inst in inst_list:
            _xm = inst.get_pin('VDD')
            _xxm = self.connect_via_stack(self.tr_manager, _xm, xxm_layer, 'sup')
            self.add_pin('VDD', _xxm, connect=True)

        # Shared bias. Route together at higher hierarchy
        for inst in inst_list:
            _xm = inst.get_pin('NBIAS')
            _xxm = self.connect_via_stack(self.tr_manager, _xm, xxm_layer, 'sup')
            self.add_pin('NBIAS', _xxm, connect=True)

        # sch params
        self._sch_params = dict(
            num_stages=num_stages,
            **unit_template.sch_params
        )
