module {
  aie.device(npu2) @op0_PostAttnFusedMLIR {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %silu_mem_0_cons_buff_0 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "silu_mem_0_cons_buff_0"} : memref<8192xbf16> 
    %silu_mem_0_cons_prod_lock_0 = aie.lock(%tile_2_3, 4) {init = 1 : i32, sym_name = "silu_mem_0_cons_prod_lock_0"}
    %silu_mem_0_cons_cons_lock_0 = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "silu_mem_0_cons_cons_lock_0"}
    %silu_mem_1_cons_buff_0 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "silu_mem_1_cons_buff_0"} : memref<8192xbf16> 
    %silu_mem_1_cons_prod_lock_0 = aie.lock(%tile_2_4, 4) {init = 1 : i32, sym_name = "silu_mem_1_cons_prod_lock_0"}
    %silu_mem_1_cons_cons_lock_0 = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "silu_mem_1_cons_cons_lock_0"}
    %silu_mem_2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "silu_mem_2_cons_buff_0"} : memref<8192xbf16> 
    %silu_mem_2_cons_prod_lock_0 = aie.lock(%tile_2_5, 4) {init = 1 : i32, sym_name = "silu_mem_2_cons_prod_lock_0"}
    %silu_mem_2_cons_cons_lock_0 = aie.lock(%tile_2_5, 5) {init = 0 : i32, sym_name = "silu_mem_2_cons_cons_lock_0"}
    %silu_mem_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "silu_mem_3_cons_buff_0"} : memref<8192xbf16> 
    %silu_mem_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "silu_mem_3_cons_prod_lock_0"}
    %silu_mem_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "silu_mem_3_cons_cons_lock_0"}
    %silu_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "silu_L3L2_cons_buff_0"} : memref<8192xbf16> 
    %silu_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "silu_L3L2_cons_prod_lock_0"}
    %silu_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "silu_L3L2_cons_cons_lock_0"}
    %silu_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "silu_L3L2_prod_lock_0"}
    %silu_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "silu_L3L2_cons_lock_0"}
    %kqv_mem_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "kqv_mem_0_cons_buff_0"} : memref<2048xbf16> 
    %kqv_mem_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "kqv_mem_0_cons_prod_lock_0"}
    %kqv_mem_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "kqv_mem_0_cons_cons_lock_0"}
    %kqv_mem_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "kqv_mem_1_cons_buff_0"} : memref<2048xbf16> 
    %kqv_mem_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 1 : i32, sym_name = "kqv_mem_1_cons_prod_lock_0"}
    %kqv_mem_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "kqv_mem_1_cons_cons_lock_0"}
    %kqv_mem_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "kqv_mem_2_cons_buff_0"} : memref<2048xbf16> 
    %kqv_mem_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 1 : i32, sym_name = "kqv_mem_2_cons_prod_lock_0"}
    %kqv_mem_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "kqv_mem_2_cons_cons_lock_0"}
    %kqv_mem_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "kqv_mem_3_cons_buff_0"} : memref<2048xbf16> 
    %kqv_mem_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 1 : i32, sym_name = "kqv_mem_3_cons_prod_lock_0"}
    %kqv_mem_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "kqv_mem_3_cons_cons_lock_0"}
    %kqv_L3L2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "kqv_L3L2_cons_buff_0"} : memref<2048xbf16> 
    %kqv_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "kqv_L3L2_cons_prod_lock_0"}
    %kqv_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "kqv_L3L2_cons_cons_lock_0"}
    %kqv_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "kqv_L3L2_prod_lock_0"}
    %kqv_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "kqv_L3L2_cons_lock_0"}
    %anm_inpff_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 4) {init = 0 : i32, sym_name = "anm_inpff_cons_prod_lock_0"}
    %anm_inpff_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 5) {init = 0 : i32, sym_name = "anm_inpff_cons_cons_lock_0"}
    %anm_inpff_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "anm_inpff_buff_0"} : memref<2048xbf16> 
    %anm_inpff_prod_lock_0 = aie.lock(%tile_1_2, 6) {init = 1 : i32, sym_name = "anm_inpff_prod_lock_0"}
    %anm_inpff_cons_lock_0 = aie.lock(%tile_1_2, 7) {init = 0 : i32, sym_name = "anm_inpff_cons_lock_0"}
    %anm_in_cons_buff_0 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "anm_in_cons_buff_0"} : memref<2048xbf16> 
    %anm_in_cons_buff_1 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "anm_in_cons_buff_1"} : memref<2048xbf16> 
    %anm_in_cons_buff_2 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "anm_in_cons_buff_2"} : memref<2048xbf16> 
    %anm_in_cons_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 3 : i32, sym_name = "anm_in_cons_prod_lock_0"}
    %anm_in_cons_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "anm_in_cons_cons_lock_0"}
    %anm_in_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "anm_in_prod_lock_0"}
    %anm_in_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "anm_in_cons_lock_0"}
    %ffi_mem_0_cons_buff_0 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "ffi_mem_0_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_0_cons_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 1 : i32, sym_name = "ffi_mem_0_cons_prod_lock_0"}
    %ffi_mem_0_cons_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "ffi_mem_0_cons_cons_lock_0"}
    %ffi_mem_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "ffi_mem_1_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 1 : i32, sym_name = "ffi_mem_1_cons_prod_lock_0"}
    %ffi_mem_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "ffi_mem_1_cons_cons_lock_0"}
    %ffi_mem_2_cons_buff_0 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "ffi_mem_2_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_2_cons_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 1 : i32, sym_name = "ffi_mem_2_cons_prod_lock_0"}
    %ffi_mem_2_cons_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "ffi_mem_2_cons_cons_lock_0"}
    %ffi_mem_3_cons_buff_0 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "ffi_mem_3_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_3_cons_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "ffi_mem_3_cons_prod_lock_0"}
    %ffi_mem_3_cons_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "ffi_mem_3_cons_cons_lock_0"}
    %anm_ffi_l1l2_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "anm_ffi_l1l2_cons_buff_0"} : memref<2048xbf16> 
    %anm_ffi_l1l2_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "anm_ffi_l1l2_cons_prod_lock_0"}
    %anm_ffi_l1l2_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "anm_ffi_l1l2_cons_cons_lock_0"}
    %anm_ffi_l1l2_buff_0 = aie.buffer(%tile_1_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "anm_ffi_l1l2_buff_0"} : memref<2048xbf16> 
    %anm_ffi_l1l2_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 1 : i32, sym_name = "anm_ffi_l1l2_prod_lock_0"}
    %anm_ffi_l1l2_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "anm_ffi_l1l2_cons_lock_0"}
    %o_out_joined_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "o_out_joined_0_cons_buff_0"} : memref<8xbf16> 
    %o_out_joined_0_cons_buff_1 = aie.buffer(%tile_1_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "o_out_joined_0_cons_buff_1"} : memref<8xbf16> 
    %o_out_joined_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "o_out_joined_0_cons_prod_lock_0"}
    %o_out_joined_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "o_out_joined_0_cons_cons_lock_0"}
    %o_out_joined_0_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "o_out_joined_0_buff_0"} : memref<8xbf16> 
    %o_out_joined_0_buff_1 = aie.buffer(%mem_tile_1_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "o_out_joined_0_buff_1"} : memref<8xbf16> 
    %o_out_joined_0_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_0"}
    %o_out_joined_0_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_0"}
    %o_out_joined_0_prod_lock_1 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_1"}
    %o_out_joined_0_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_1"}
    %o_out_joined_0_prod_lock_2 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_2"}
    %o_out_joined_0_cons_lock_2 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_2"}
    %o_out_joined_0_prod_lock_3 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_3"}
    %o_out_joined_0_cons_lock_3 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_3"}
    %Co_3_buff_0 = aie.buffer(%tile_0_5) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_3_buff_0"} : memref<2xbf16> 
    %Co_3_buff_1 = aie.buffer(%tile_0_5) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_3_buff_1"} : memref<2xbf16> 
    %Co_3_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "Co_3_prod_lock_0"}
    %Co_3_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "Co_3_cons_lock_0"}
    %Co_2_buff_0 = aie.buffer(%tile_0_4) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_2_buff_0"} : memref<2xbf16> 
    %Co_2_buff_1 = aie.buffer(%tile_0_4) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_2_buff_1"} : memref<2xbf16> 
    %Co_2_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "Co_2_prod_lock_0"}
    %Co_2_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "Co_2_cons_lock_0"}
    %Co_1_buff_0 = aie.buffer(%tile_0_3) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_1_buff_0"} : memref<2xbf16> 
    %Co_1_buff_1 = aie.buffer(%tile_0_3) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_1_buff_1"} : memref<2xbf16> 
    %Co_1_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "Co_1_prod_lock_0"}
    %Co_1_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "Co_1_cons_lock_0"}
    %Co_0_buff_0 = aie.buffer(%tile_0_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_0_buff_0"} : memref<2xbf16> 
    %Co_0_buff_1 = aie.buffer(%tile_0_2) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_0_buff_1"} : memref<2xbf16> 
    %Co_0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "Co_0_prod_lock_0"}
    %Co_0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "Co_0_cons_lock_0"}
    %Cgu_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 6) {init = 0 : i32, sym_name = "Cgu_3_cons_prod_lock_0"}
    %Cgu_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 7) {init = 0 : i32, sym_name = "Cgu_3_cons_cons_lock_0"}
    %Cgu_3_buff_0 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Cgu_3_buff_0"} : memref<8xbf16> 
    %Cgu_3_buff_1 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cgu_3_buff_1"} : memref<8xbf16> 
    %Cgu_3_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "Cgu_3_prod_lock_0"}
    %Cgu_3_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "Cgu_3_cons_lock_0"}
    %Cgu_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 4) {init = 0 : i32, sym_name = "Cgu_2_cons_prod_lock_0"}
    %Cgu_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 5) {init = 0 : i32, sym_name = "Cgu_2_cons_cons_lock_0"}
    %Cgu_2_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Cgu_2_buff_0"} : memref<8xbf16> 
    %Cgu_2_buff_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cgu_2_buff_1"} : memref<8xbf16> 
    %Cgu_2_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "Cgu_2_prod_lock_0"}
    %Cgu_2_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "Cgu_2_cons_lock_0"}
    %Cgu_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "Cgu_1_cons_prod_lock_0"}
    %Cgu_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "Cgu_1_cons_cons_lock_0"}
    %Cgu_1_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Cgu_1_buff_0"} : memref<8xbf16> 
    %Cgu_1_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cgu_1_buff_1"} : memref<8xbf16> 
    %Cgu_1_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "Cgu_1_prod_lock_0"}
    %Cgu_1_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "Cgu_1_cons_lock_0"}
    %Cgu_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "Cgu_0_cons_prod_lock_0"}
    %Cgu_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "Cgu_0_cons_cons_lock_0"}
    %Cgu_0_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Cgu_0_buff_0"} : memref<8xbf16> 
    %Cgu_0_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cgu_0_buff_1"} : memref<8xbf16> 
    %Cgu_0_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "Cgu_0_prod_lock_0"}
    %Cgu_0_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "Cgu_0_cons_lock_0"}
    %Cd_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "Cd_3_cons_prod_lock_0"}
    %Cd_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "Cd_3_cons_cons_lock_0"}
    %Cd_3_buff_0 = aie.buffer(%tile_3_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Cd_3_buff_0"} : memref<2xbf16> 
    %Cd_3_buff_1 = aie.buffer(%tile_3_2) {address = 41984 : i32, mem_bank = 2 : i32, sym_name = "Cd_3_buff_1"} : memref<2xbf16> 
    %Cd_3_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "Cd_3_prod_lock_0"}
    %Cd_3_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "Cd_3_cons_lock_0"}
    %Cd_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "Cd_2_cons_prod_lock_0"}
    %Cd_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "Cd_2_cons_cons_lock_0"}
    %Cd_2_buff_0 = aie.buffer(%tile_2_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Cd_2_buff_0"} : memref<2xbf16> 
    %Cd_2_buff_1 = aie.buffer(%tile_2_5) {address = 41984 : i32, mem_bank = 2 : i32, sym_name = "Cd_2_buff_1"} : memref<2xbf16> 
    %Cd_2_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 2 : i32, sym_name = "Cd_2_prod_lock_0"}
    %Cd_2_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "Cd_2_cons_lock_0"}
    %Cd_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "Cd_1_cons_prod_lock_0"}
    %Cd_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "Cd_1_cons_cons_lock_0"}
    %Cd_1_buff_0 = aie.buffer(%tile_2_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Cd_1_buff_0"} : memref<2xbf16> 
    %Cd_1_buff_1 = aie.buffer(%tile_2_4) {address = 41984 : i32, mem_bank = 2 : i32, sym_name = "Cd_1_buff_1"} : memref<2xbf16> 
    %Cd_1_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 2 : i32, sym_name = "Cd_1_prod_lock_0"}
    %Cd_1_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "Cd_1_cons_lock_0"}
    %Cd_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "Cd_0_cons_prod_lock_0"}
    %Cd_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "Cd_0_cons_cons_lock_0"}
    %Cd_0_buff_0 = aie.buffer(%tile_2_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Cd_0_buff_0"} : memref<2xbf16> 
    %Cd_0_buff_1 = aie.buffer(%tile_2_3) {address = 41984 : i32, mem_bank = 2 : i32, sym_name = "Cd_0_buff_1"} : memref<2xbf16> 
    %Cd_0_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "Cd_0_prod_lock_0"}
    %Cd_0_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "Cd_0_cons_lock_0"}
    %Ao_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_3_cons_buff_0"} : memref<2304xui8> 
    %Ao_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_3_cons_buff_1"} : memref<2304xui8> 
    %Ao_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "Ao_3_cons_prod_lock_0"}
    %Ao_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "Ao_3_cons_cons_lock_0"}
    %Ao_3_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "Ao_3_prod_lock_0"}
    %Ao_3_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "Ao_3_cons_lock_0"}
    %Ao_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_2_cons_buff_0"} : memref<2304xui8> 
    %Ao_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_2_cons_buff_1"} : memref<2304xui8> 
    %Ao_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "Ao_2_cons_prod_lock_0"}
    %Ao_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "Ao_2_cons_cons_lock_0"}
    %Ao_2_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "Ao_2_prod_lock_0"}
    %Ao_2_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "Ao_2_cons_lock_0"}
    %Ao_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_1_cons_buff_0"} : memref<2304xui8> 
    %Ao_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_1_cons_buff_1"} : memref<2304xui8> 
    %Ao_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "Ao_1_cons_prod_lock_0"}
    %Ao_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "Ao_1_cons_cons_lock_0"}
    %Ao_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "Ao_1_prod_lock_0"}
    %Ao_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "Ao_1_cons_lock_0"}
    %Ao_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_0_cons_buff_0"} : memref<2304xui8> 
    %Ao_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_0_cons_buff_1"} : memref<2304xui8> 
    %Ao_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "Ao_0_cons_prod_lock_0"}
    %Ao_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "Ao_0_cons_cons_lock_0"}
    %Ao_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "Ao_0_prod_lock_0"}
    %Ao_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "Ao_0_cons_lock_0"}
    %Agu_3_cons_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_3_cons_buff_0"} : memref<9216xui8> 
    %Agu_3_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 1 : i32, sym_name = "Agu_3_cons_prod_lock_0"}
    %Agu_3_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "Agu_3_cons_cons_lock_0"}
    %Agu_3_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "Agu_3_prod_lock_0"}
    %Agu_3_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "Agu_3_cons_lock_0"}
    %Agu_2_cons_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_2_cons_buff_0"} : memref<9216xui8> 
    %Agu_2_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 1 : i32, sym_name = "Agu_2_cons_prod_lock_0"}
    %Agu_2_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "Agu_2_cons_cons_lock_0"}
    %Agu_2_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "Agu_2_prod_lock_0"}
    %Agu_2_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "Agu_2_cons_lock_0"}
    %Agu_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_1_cons_buff_0"} : memref<9216xui8> 
    %Agu_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 1 : i32, sym_name = "Agu_1_cons_prod_lock_0"}
    %Agu_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "Agu_1_cons_cons_lock_0"}
    %Agu_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "Agu_1_prod_lock_0"}
    %Agu_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "Agu_1_cons_lock_0"}
    %Agu_0_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_0_cons_buff_0"} : memref<9216xui8> 
    %Agu_0_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 1 : i32, sym_name = "Agu_0_cons_prod_lock_0"}
    %Agu_0_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "Agu_0_cons_cons_lock_0"}
    %Agu_0_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "Agu_0_prod_lock_0"}
    %Agu_0_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "Agu_0_cons_lock_0"}
    %Ad_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ad_3_cons_buff_0"} : memref<9216xui8> 
    %Ad_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 1 : i32, sym_name = "Ad_3_cons_prod_lock_0"}
    %Ad_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "Ad_3_cons_cons_lock_0"}
    %Ad_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "Ad_3_prod_lock_0"}
    %Ad_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "Ad_3_cons_lock_0"}
    %Ad_2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ad_2_cons_buff_0"} : memref<9216xui8> 
    %Ad_2_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 1 : i32, sym_name = "Ad_2_cons_prod_lock_0"}
    %Ad_2_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "Ad_2_cons_cons_lock_0"}
    %Ad_2_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "Ad_2_prod_lock_0"}
    %Ad_2_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "Ad_2_cons_lock_0"}
    %Ad_1_cons_buff_0 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ad_1_cons_buff_0"} : memref<9216xui8> 
    %Ad_1_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 1 : i32, sym_name = "Ad_1_cons_prod_lock_0"}
    %Ad_1_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "Ad_1_cons_cons_lock_0"}
    %Ad_1_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "Ad_1_prod_lock_0"}
    %Ad_1_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "Ad_1_cons_lock_0"}
    %Ad_0_cons_buff_0 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ad_0_cons_buff_0"} : memref<9216xui8> 
    %Ad_0_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 1 : i32, sym_name = "Ad_0_cons_prod_lock_0"}
    %Ad_0_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "Ad_0_cons_cons_lock_0"}
    %Ad_0_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "Ad_0_prod_lock_0"}
    %Ad_0_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "Ad_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_2_4, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_1_4, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_1_5, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%tile_2_3, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_2_4, DMA : 0, %shim_noc_tile_2_0, DMA : 1)
    aie.flow(%tile_2_5, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %shim_noc_tile_3_0, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_1_4, DMA : 0, %shim_noc_tile_1_0, DMA : 1)
    aie.flow(%tile_1_5, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %shim_noc_tile_4_0, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_1_1, DMA : 3)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_3, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%tile_1_2, DMA : 1, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_2_3, DMA : 1)
    func.func private @op0_fused_dequant_matvec_v2_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    %anm_oout_buf = aie.buffer(%tile_1_2) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "anm_oout_buf"} : memref<2048xbf16> 
    func.func private @op0_post_attn_o_out_assemble_bf16(memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_post_attn_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_post_attn_rms_norm_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_dual_fused_dequant_gemv_bf16(i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_dual_fused_dequant_gemv_silu_mul_bf16(memref<8xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_fused_dequant_matvec_down_bf16(i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%kqv_mem_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Ao_0_cons_buff_0 : memref<2304xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Ao_0_cons_buff_1 : memref<2304xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Ao_0_cons_buff_0 : memref<2304xui8>)
    ^bb10(%9: memref<2304xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%Co_0_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Co_0_buff_0 : memref<2xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Co_0_buff_1 : memref<2xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Co_0_buff_0 : memref<2xbf16>)
    ^bb14(%13: memref<2xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %9, %kqv_mem_0_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_0_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%Co_0_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous0[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%kqv_mem_0_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous0[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%kqv_mem_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Ao_1_cons_buff_0 : memref<2304xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Ao_1_cons_buff_1 : memref<2304xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Ao_1_cons_buff_0 : memref<2304xui8>)
    ^bb10(%9: memref<2304xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%Co_1_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Co_1_buff_0 : memref<2xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Co_1_buff_1 : memref<2xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Co_1_buff_0 : memref<2xbf16>)
    ^bb14(%13: memref<2xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %9, %kqv_mem_1_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_1_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%Co_1_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous1[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%kqv_mem_1_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous1[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous2 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%kqv_mem_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Ao_2_cons_buff_0 : memref<2304xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Ao_2_cons_buff_1 : memref<2304xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Ao_2_cons_buff_0 : memref<2304xui8>)
    ^bb10(%9: memref<2304xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%Co_2_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Co_2_buff_0 : memref<2xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Co_2_buff_1 : memref<2xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Co_2_buff_0 : memref<2xbf16>)
    ^bb14(%13: memref<2xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %9, %kqv_mem_2_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_2_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%Co_2_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous2[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%kqv_mem_2_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous2[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous3 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb15
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      aie.use_lock(%kqv_mem_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Ao_3_cons_buff_0 : memref<2304xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Ao_3_cons_buff_1 : memref<2304xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Ao_3_cons_buff_0 : memref<2304xui8>)
    ^bb10(%9: memref<2304xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%Co_3_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Co_3_buff_0 : memref<2xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Co_3_buff_1 : memref<2xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Co_3_buff_0 : memref<2xbf16>)
    ^bb14(%13: memref<2xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %9, %kqv_mem_3_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_3_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%Co_3_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous3[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%kqv_mem_3_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous3[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous4 = aie.buffer(%tile_1_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "_anonymous4"} : memref<4xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c512_i32 = arith.constant 512 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c3 = arith.constant 3 : index
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous4[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous4[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb22
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb23
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb21
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb22
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      %6 = arith.index_cast %4 : index to i32
      aie.use_lock(%o_out_joined_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %7 = memref.load %_anonymous4[%c0] : memref<4xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = arith.index_cast %8 : index to i32
      cf.switch %9 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%o_out_joined_0_cons_buff_0 : memref<8xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%o_out_joined_0_cons_buff_1 : memref<8xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%o_out_joined_0_cons_buff_0 : memref<8xbf16>)
    ^bb10(%10: memref<8xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_post_attn_o_out_assemble_bf16(%10, %anm_oout_buf, %6, %c0_i32, %c4_i32, %c2_i32, %c512_i32) : (memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%o_out_joined_0_cons_prod_lock_0, Release, 1)
      %11 = memref.load %_anonymous4[%c0] : memref<4xi32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c2_i32 : i32
      %14 = arith.subi %12, %c2_i32 : i32
      %15 = arith.select %13, %14, %12 : i32
      memref.store %15, %_anonymous4[%c0] : memref<4xi32>
      %16 = arith.addi %4, %c1 : index
      cf.br ^bb5(%16 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%anm_in_cons_cons_lock_0, AcquireGreaterEqual, 2)
      %17 = memref.load %_anonymous4[%c1] : memref<4xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb12,
        1: ^bb13,
        2: ^bb14
      ]
    ^bb12:  // pred: ^bb11
      cf.br ^bb16(%anm_in_cons_buff_0 : memref<2048xbf16>)
    ^bb13:  // pred: ^bb11
      cf.br ^bb16(%anm_in_cons_buff_1 : memref<2048xbf16>)
    ^bb14:  // pred: ^bb11
      cf.br ^bb16(%anm_in_cons_buff_2 : memref<2048xbf16>)
    ^bb15:  // pred: ^bb11
      cf.br ^bb16(%anm_in_cons_buff_0 : memref<2048xbf16>)
    ^bb16(%20: memref<2048xbf16>):  // 4 preds: ^bb12, ^bb13, ^bb14, ^bb15
      %21 = memref.load %_anonymous4[%c1] : memref<4xi32>
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.index_cast %22 : index to i32
      cf.switch %23 : i32, [
        default: ^bb20,
        0: ^bb17,
        1: ^bb18,
        2: ^bb19
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb21(%anm_in_cons_buff_1 : memref<2048xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb21(%anm_in_cons_buff_2 : memref<2048xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb21(%anm_in_cons_buff_0 : memref<2048xbf16>)
    ^bb20:  // pred: ^bb16
      cf.br ^bb21(%anm_in_cons_buff_1 : memref<2048xbf16>)
    ^bb21(%24: memref<2048xbf16>):  // 4 preds: ^bb17, ^bb18, ^bb19, ^bb20
      aie.use_lock(%anm_inpff_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%anm_ffi_l1l2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_post_attn_add_bf16(%anm_oout_buf, %20, %anm_inpff_buff_0, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
      func.call @op0_post_attn_rms_norm_bf16(%anm_inpff_buff_0, %24, %anm_ffi_l1l2_buff_0, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%anm_in_cons_prod_lock_0, Release, 2)
      %25 = memref.load %_anonymous4[%c1] : memref<4xi32>
      %26 = arith.addi %25, %c2_i32 : i32
      %27 = arith.cmpi sge, %26, %c3_i32 : i32
      %28 = arith.subi %26, %c3_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous4[%c1] : memref<4xi32>
      aie.use_lock(%anm_inpff_cons_lock_0, Release, 1)
      %30 = memref.load %_anonymous4[%c2] : memref<4xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c1_i32 : i32
      %33 = arith.select %32, %30, %31 : i32
      memref.store %33, %_anonymous4[%c2] : memref<4xi32>
      aie.use_lock(%anm_ffi_l1l2_cons_lock_0, Release, 1)
      %34 = memref.load %_anonymous4[%c3] : memref<4xi32>
      %35 = arith.addi %34, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c1_i32 : i32
      %37 = arith.select %36, %34, %35 : i32
      memref.store %37, %_anonymous4[%c3] : memref<4xi32>
      %38 = arith.addi %2, %c1 : index
      cf.br ^bb3(%38 : index)
    ^bb22:  // pred: ^bb3
      %39 = arith.addi %0, %c1 : index
      cf.br ^bb1(%39 : index)
    ^bb23:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous5 = aie.buffer(%tile_1_3) {address = 10240 : i32, mem_bank = 0 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c8_i32 = arith.constant 8 : i32
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_0_cons_buff_0, %ffi_mem_0_cons_buff_0, %c0_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_0_cons_prod_lock_0, Release, 1)
      %6 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %7 = arith.addi %6, %c1_i32 : i32
      %8 = arith.cmpi sge, %7, %c1_i32 : i32
      %9 = arith.select %8, %6, %7 : i32
      memref.store %9, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%Agu_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_0_cons_buff_0, %ffi_mem_0_cons_buff_0, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_0_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%Cgu_0_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cgu_0_buff_0 : memref<8xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cgu_0_buff_1 : memref<8xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cgu_0_buff_0 : memref<8xbf16>)
    ^bb10(%17: memref<8xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%17, %c8_i32) : (memref<8xbf16>, i32) -> ()
      aie.use_lock(%Cgu_0_cons_lock_0, Release, 1)
      %18 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous5[%c2] : memref<3xi32>
      %23 = arith.addi %4, %c1 : index
      cf.br ^bb5(%23 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%ffi_mem_0_cons_prod_lock_0, Release, 1)
      %24 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c1_i32 : i32
      %27 = arith.select %26, %24, %25 : i32
      memref.store %27, %_anonymous5[%c0] : memref<3xi32>
      %28 = arith.addi %2, %c1 : index
      cf.br ^bb3(%28 : index)
    ^bb12:  // pred: ^bb3
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous6 = aie.buffer(%tile_1_4) {address = 10240 : i32, mem_bank = 0 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c8_i32 = arith.constant 8 : i32
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_1_cons_buff_0, %ffi_mem_1_cons_buff_0, %c0_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_1_cons_prod_lock_0, Release, 1)
      %6 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %7 = arith.addi %6, %c1_i32 : i32
      %8 = arith.cmpi sge, %7, %c1_i32 : i32
      %9 = arith.select %8, %6, %7 : i32
      memref.store %9, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%Agu_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_1_cons_buff_0, %ffi_mem_1_cons_buff_0, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_1_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%Cgu_1_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cgu_1_buff_0 : memref<8xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cgu_1_buff_1 : memref<8xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cgu_1_buff_0 : memref<8xbf16>)
    ^bb10(%17: memref<8xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%17, %c8_i32) : (memref<8xbf16>, i32) -> ()
      aie.use_lock(%Cgu_1_cons_lock_0, Release, 1)
      %18 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous6[%c2] : memref<3xi32>
      %23 = arith.addi %4, %c1 : index
      cf.br ^bb5(%23 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%ffi_mem_1_cons_prod_lock_0, Release, 1)
      %24 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c1_i32 : i32
      %27 = arith.select %26, %24, %25 : i32
      memref.store %27, %_anonymous6[%c0] : memref<3xi32>
      %28 = arith.addi %2, %c1 : index
      cf.br ^bb3(%28 : index)
    ^bb12:  // pred: ^bb3
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous7 = aie.buffer(%tile_1_5) {address = 10240 : i32, mem_bank = 0 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c8_i32 = arith.constant 8 : i32
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_2_cons_buff_0, %ffi_mem_2_cons_buff_0, %c0_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_2_cons_prod_lock_0, Release, 1)
      %6 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %7 = arith.addi %6, %c1_i32 : i32
      %8 = arith.cmpi sge, %7, %c1_i32 : i32
      %9 = arith.select %8, %6, %7 : i32
      memref.store %9, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%Agu_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_2_cons_buff_0, %ffi_mem_2_cons_buff_0, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_2_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%Cgu_2_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cgu_2_buff_0 : memref<8xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cgu_2_buff_1 : memref<8xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cgu_2_buff_0 : memref<8xbf16>)
    ^bb10(%17: memref<8xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%17, %c8_i32) : (memref<8xbf16>, i32) -> ()
      aie.use_lock(%Cgu_2_cons_lock_0, Release, 1)
      %18 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous7[%c2] : memref<3xi32>
      %23 = arith.addi %4, %c1 : index
      cf.br ^bb5(%23 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%ffi_mem_2_cons_prod_lock_0, Release, 1)
      %24 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c1_i32 : i32
      %27 = arith.select %26, %24, %25 : i32
      memref.store %27, %_anonymous7[%c0] : memref<3xi32>
      %28 = arith.addi %2, %c1 : index
      cf.br ^bb3(%28 : index)
    ^bb12:  // pred: ^bb3
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous8 = aie.buffer(%tile_2_2) {address = 10240 : i32, mem_bank = 0 : i32, sym_name = "_anonymous8"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c8_i32 = arith.constant 8 : i32
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous8[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_3_cons_buff_0, %ffi_mem_3_cons_buff_0, %c0_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_3_cons_prod_lock_0, Release, 1)
      %6 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %7 = arith.addi %6, %c1_i32 : i32
      %8 = arith.cmpi sge, %7, %c1_i32 : i32
      %9 = arith.select %8, %6, %7 : i32
      memref.store %9, %_anonymous8[%c1] : memref<3xi32>
      aie.use_lock(%Agu_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %Agu_3_cons_buff_0, %ffi_mem_3_cons_buff_0, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_3_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous8[%c1] : memref<3xi32>
      aie.use_lock(%Cgu_3_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cgu_3_buff_0 : memref<8xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cgu_3_buff_1 : memref<8xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cgu_3_buff_0 : memref<8xbf16>)
    ^bb10(%17: memref<8xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%17, %c8_i32) : (memref<8xbf16>, i32) -> ()
      aie.use_lock(%Cgu_3_cons_lock_0, Release, 1)
      %18 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous8[%c2] : memref<3xi32>
      %23 = arith.addi %4, %c1 : index
      cf.br ^bb5(%23 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%ffi_mem_3_cons_prod_lock_0, Release, 1)
      %24 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %25 = arith.addi %24, %c1_i32 : i32
      %26 = arith.cmpi sge, %25, %c1_i32 : i32
      %27 = arith.select %26, %24, %25 : i32
      memref.store %27, %_anonymous8[%c0] : memref<3xi32>
      %28 = arith.addi %2, %c1 : index
      cf.br ^bb3(%28 : index)
    ^bb12:  // pred: ^bb3
      %29 = arith.addi %0, %c1 : index
      cf.br ^bb1(%29 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous9 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous9"} : memref<3xi32> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous9[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%silu_mem_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ad_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%Cd_0_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cd_0_buff_0 : memref<2xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cd_0_buff_1 : memref<2xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cd_0_buff_0 : memref<2xbf16>)
    ^bb10(%9: memref<2xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %Ad_0_cons_buff_0, %silu_mem_0_cons_buff_0, %9) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ad_0_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous9[%c1] : memref<3xi32>
      aie.use_lock(%Cd_0_cons_lock_0, Release, 1)
      %14 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous9[%c2] : memref<3xi32>
      %19 = arith.addi %4, %c1 : index
      cf.br ^bb5(%19 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%silu_mem_0_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c1_i32 : i32
      %23 = arith.select %22, %20, %21 : i32
      memref.store %23, %_anonymous9[%c0] : memref<3xi32>
      %24 = arith.addi %2, %c1 : index
      cf.br ^bb3(%24 : index)
    ^bb12:  // pred: ^bb3
      %25 = arith.addi %0, %c1 : index
      cf.br ^bb1(%25 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous10 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous10"} : memref<3xi32> 
    %core_2_4 = aie.core(%tile_2_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous10[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%silu_mem_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ad_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%Cd_1_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cd_1_buff_0 : memref<2xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cd_1_buff_1 : memref<2xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cd_1_buff_0 : memref<2xbf16>)
    ^bb10(%9: memref<2xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %Ad_1_cons_buff_0, %silu_mem_1_cons_buff_0, %9) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ad_1_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous10[%c1] : memref<3xi32>
      aie.use_lock(%Cd_1_cons_lock_0, Release, 1)
      %14 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous10[%c2] : memref<3xi32>
      %19 = arith.addi %4, %c1 : index
      cf.br ^bb5(%19 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%silu_mem_1_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c1_i32 : i32
      %23 = arith.select %22, %20, %21 : i32
      memref.store %23, %_anonymous10[%c0] : memref<3xi32>
      %24 = arith.addi %2, %c1 : index
      cf.br ^bb3(%24 : index)
    ^bb12:  // pred: ^bb3
      %25 = arith.addi %0, %c1 : index
      cf.br ^bb1(%25 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous11 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous11"} : memref<3xi32> 
    %core_2_5 = aie.core(%tile_2_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous11[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%silu_mem_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ad_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%Cd_2_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cd_2_buff_0 : memref<2xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cd_2_buff_1 : memref<2xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cd_2_buff_0 : memref<2xbf16>)
    ^bb10(%9: memref<2xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %Ad_2_cons_buff_0, %silu_mem_2_cons_buff_0, %9) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ad_2_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous11[%c1] : memref<3xi32>
      aie.use_lock(%Cd_2_cons_lock_0, Release, 1)
      %14 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous11[%c2] : memref<3xi32>
      %19 = arith.addi %4, %c1 : index
      cf.br ^bb5(%19 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%silu_mem_2_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c1_i32 : i32
      %23 = arith.select %22, %20, %21 : i32
      memref.store %23, %_anonymous11[%c0] : memref<3xi32>
      %24 = arith.addi %2, %c1 : index
      cf.br ^bb3(%24 : index)
    ^bb12:  // pred: ^bb3
      %25 = arith.addi %0, %c1 : index
      cf.br ^bb1(%25 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    %_anonymous12 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous12"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous12[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb12
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb13
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb11
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb12
    ^bb4:  // pred: ^bb3
      aie.use_lock(%silu_mem_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb10
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb11
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ad_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%Cd_3_prod_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Cd_3_buff_0 : memref<2xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Cd_3_buff_1 : memref<2xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Cd_3_buff_0 : memref<2xbf16>)
    ^bb10(%9: memref<2xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %Ad_3_cons_buff_0, %silu_mem_3_cons_buff_0, %9) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ad_3_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c1_i32 : i32
      %13 = arith.select %12, %10, %11 : i32
      memref.store %13, %_anonymous12[%c1] : memref<3xi32>
      aie.use_lock(%Cd_3_cons_lock_0, Release, 1)
      %14 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous12[%c2] : memref<3xi32>
      %19 = arith.addi %4, %c1 : index
      cf.br ^bb5(%19 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%silu_mem_3_cons_prod_lock_0, Release, 1)
      %20 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %21 = arith.addi %20, %c1_i32 : i32
      %22 = arith.cmpi sge, %21, %c1_i32 : i32
      %23 = arith.select %22, %20, %21 : i32
      memref.store %23, %_anonymous12[%c0] : memref<3xi32>
      %24 = arith.addi %2, %c1 : index
      cf.br ^bb3(%24 : index)
    ^bb12:  // pred: ^bb3
      %25 = arith.addi %0, %c1 : index
      cf.br ^bb1(%25 : index)
    ^bb13:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_post_attn_fused_2048k_g32.o"]}
    aie.runtime_sequence(%arg0: memref<1179648xbf16>, %arg1: memref<9437184xbf16>, %arg2: memref<4718592xbf16>, %arg3: memref<6144xbf16>, %arg4: memref<14336xbf16>) {
      %0 = aiex.dma_configure_task_for @Ao_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 0, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @Ao_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 589824, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @Ao_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 1179648, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @Ao_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 1769472, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @kqv_L3L2_shim_alloc {
        aie.dma_bd(%arg3 : memref<6144xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      %5 = aiex.dma_configure_task_for @anm_in_shim_alloc {
        aie.dma_bd(%arg3 : memref<6144xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @anm_in_shim_alloc {
        aie.dma_bd(%arg3 : memref<6144xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @anm_inpff_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 12288, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      aiex.dma_await_task(%7)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      %8 = aiex.dma_configure_task_for @Agu_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 0, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @Agu_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 4718592, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @Agu_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 9437184, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @Agu_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 14155776, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @Cgu_0_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 8>, <size = 8, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @Cgu_1_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 8>, <size = 8, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @Cgu_2_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 6144, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 8>, <size = 8, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @Cgu_3_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 8192, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 8>, <size = 8, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      %16 = aiex.dma_configure_task_for @Ad_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 0, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @Ad_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 2359296, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @Ad_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 4718592, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @Ad_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 7077888, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @silu_L3L2_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 2048, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @Cd_0_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 10240, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 2>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @Cd_1_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 10752, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 2>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @Cd_2_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 11264, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 2>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @Cd_3_shim_alloc {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 11776, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 2>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%24)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%24)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
    }
    aie.shim_dma_allocation @Ad_0_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Ad_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ad_0_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ad_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cd_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_0_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cd_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cd_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_0_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cd_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%silu_mem_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%silu_mem_0_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%silu_mem_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Ad_1_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Ad_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ad_1_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ad_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cd_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_1_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cd_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cd_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_1_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cd_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%silu_mem_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%silu_mem_1_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%silu_mem_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Ad_2_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Ad_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ad_2_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ad_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cd_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_2_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cd_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cd_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_2_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cd_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%silu_mem_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%silu_mem_2_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%silu_mem_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Ad_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Ad_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ad_3_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ad_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cd_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_3_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cd_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cd_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cd_3_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cd_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%silu_mem_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%silu_mem_3_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%silu_mem_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Agu_0_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Agu_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_0_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cgu_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_0_buff_0 : memref<8xbf16>, 0, 8) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cgu_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cgu_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_0_buff_1 : memref<8xbf16>, 0, 8) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cgu_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%ffi_mem_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_0_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ffi_mem_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Agu_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Agu_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_1_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cgu_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_1_buff_0 : memref<8xbf16>, 0, 8) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cgu_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cgu_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_1_buff_1 : memref<8xbf16>, 0, 8) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cgu_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%ffi_mem_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_1_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ffi_mem_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Agu_2_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Agu_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_2_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cgu_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_2_buff_0 : memref<8xbf16>, 0, 8) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cgu_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cgu_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_2_buff_1 : memref<8xbf16>, 0, 8) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cgu_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%ffi_mem_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ffi_mem_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Agu_3_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%Agu_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_3_cons_buff_0 : memref<9216xui8>, 0, 9216) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%Cgu_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_3_buff_0 : memref<8xbf16>, 0, 8) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cgu_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Cgu_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cgu_3_buff_1 : memref<8xbf16>, 0, 8) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Cgu_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%ffi_mem_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_3_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ffi_mem_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @Ao_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ao_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_0_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ao_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ao_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_0_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ao_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Co_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_0_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Co_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Co_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_0_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Co_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%kqv_mem_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%kqv_mem_0_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%kqv_mem_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @Ao_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ao_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_1_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ao_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ao_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_1_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ao_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Co_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_1_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Co_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Co_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_1_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Co_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%kqv_mem_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%kqv_mem_1_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%kqv_mem_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @Ao_2_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ao_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_2_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ao_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ao_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_2_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ao_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Co_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_2_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Co_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Co_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_2_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Co_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%kqv_mem_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%kqv_mem_2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%kqv_mem_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @Ao_3_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ao_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_3_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ao_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ao_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_3_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ao_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Co_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_3_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Co_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Co_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Co_3_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Co_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%kqv_mem_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%kqv_mem_3_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%kqv_mem_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @Cd_0_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @Cd_1_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @Cd_2_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @Cd_3_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
    aie.shim_dma_allocation @Cgu_0_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @Cgu_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @Cgu_2_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @Cgu_3_shim_alloc(%shim_noc_tile_4_0, S2MM, 1)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%o_out_joined_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 0, 2) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%o_out_joined_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 0, 2) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%o_out_joined_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 2, 2) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%o_out_joined_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 2, 2) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%o_out_joined_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 4, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%o_out_joined_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 4, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 3, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%o_out_joined_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 6, 2) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%o_out_joined_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 6, 2) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 0, ^bb13, ^bb21)
    ^bb13:  // 2 preds: ^bb12, ^bb20
      aie.use_lock(%o_out_joined_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 0, 2) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%o_out_joined_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 2, 2) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      aie.use_lock(%o_out_joined_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 4, 2) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%o_out_joined_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 6, 2) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%o_out_joined_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 0, 2) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      aie.use_lock(%o_out_joined_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 2, 2) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%o_out_joined_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 4, 2) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%o_out_joined_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 6, 2) {bd_id = 11 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb13
    ^bb21:  // pred: ^bb12
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%o_out_joined_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_cons_buff_0 : memref<8xbf16>, 0, 8) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%o_out_joined_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%o_out_joined_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_cons_buff_1 : memref<8xbf16>, 0, 8) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%o_out_joined_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%anm_ffi_l1l2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_ffi_l1l2_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%anm_ffi_l1l2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb9)
    ^bb6:  // 2 preds: ^bb5, ^bb8
      aie.use_lock(%anm_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_in_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%anm_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%anm_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_in_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%anm_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%anm_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_in_cons_buff_2 : memref<2048xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%anm_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb9:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%anm_inpff_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_inpff_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 6 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%anm_inpff_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%anm_ffi_l1l2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_ffi_l1l2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%anm_ffi_l1l2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%anm_ffi_l1l2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_ffi_l1l2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%anm_ffi_l1l2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%kqv_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%kqv_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%kqv_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%kqv_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%kqv_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%kqv_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(S2MM, 2, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%silu_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%silu_L3L2_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%silu_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %5 = aie.dma_start(MM2S, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%silu_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%silu_L3L2_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%silu_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    aie.shim_dma_allocation @anm_in_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @anm_inpff_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @kqv_L3L2_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @silu_L3L2_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
