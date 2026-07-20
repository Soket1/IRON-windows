module {
  aie.device(npu2) @op0_LayerFusedMLIR {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_4_4 = aie.tile(4, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_7_1 = aie.tile(7, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_6_1 = aie.tile(6, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %rms_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "rms_in_cons_buff_0"} : memref<2048xbf16> 
    %rms_in_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "rms_in_cons_buff_1"} : memref<2048xbf16> 
    %rms_in_cons_buff_2 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "rms_in_cons_buff_2"} : memref<2048xbf16> 
    %rms_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 6) {init = 3 : i32, sym_name = "rms_in_cons_prod_lock_0"}
    %rms_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 7) {init = 0 : i32, sym_name = "rms_in_cons_cons_lock_0"}
    %rms_in_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "rms_in_prod_lock_0"}
    %rms_in_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "rms_in_cons_lock_0"}
    %inter_3_buff_0 = aie.buffer(%tile_3_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "inter_3_buff_0"} : memref<2048xbf16> 
    %inter_3_buff_1 = aie.buffer(%tile_3_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "inter_3_buff_1"} : memref<2048xbf16> 
    %inter_3_prod_lock_0 = aie.lock(%tile_3_4, 4) {init = 2 : i32, sym_name = "inter_3_prod_lock_0"}
    %inter_3_cons_lock_0 = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "inter_3_cons_lock_0"}
    %inter_2_buff_0 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "inter_2_buff_0"} : memref<2048xbf16> 
    %inter_2_buff_1 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "inter_2_buff_1"} : memref<2048xbf16> 
    %inter_2_prod_lock_0 = aie.lock(%tile_2_4, 4) {init = 2 : i32, sym_name = "inter_2_prod_lock_0"}
    %inter_2_cons_lock_0 = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "inter_2_cons_lock_0"}
    %inter_1_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "inter_1_buff_0"} : memref<2048xbf16> 
    %inter_1_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "inter_1_buff_1"} : memref<2048xbf16> 
    %inter_1_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 2 : i32, sym_name = "inter_1_prod_lock_0"}
    %inter_1_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "inter_1_cons_lock_0"}
    %inter_0_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "inter_0_buff_0"} : memref<2048xbf16> 
    %inter_0_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "inter_0_buff_1"} : memref<2048xbf16> 
    %inter_0_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "inter_0_prod_lock_0"}
    %inter_0_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "inter_0_cons_lock_0"}
    %ffi_mem_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 5632 : i32, mem_bank = 0 : i32, sym_name = "ffi_mem_0_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "ffi_mem_0_cons_prod_lock_0"}
    %ffi_mem_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "ffi_mem_0_cons_cons_lock_0"}
    %ffi_mem_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 5632 : i32, mem_bank = 0 : i32, sym_name = "ffi_mem_1_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 1 : i32, sym_name = "ffi_mem_1_cons_prod_lock_0"}
    %ffi_mem_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "ffi_mem_1_cons_cons_lock_0"}
    %ffi_mem_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 5632 : i32, mem_bank = 0 : i32, sym_name = "ffi_mem_2_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 1 : i32, sym_name = "ffi_mem_2_cons_prod_lock_0"}
    %ffi_mem_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "ffi_mem_2_cons_cons_lock_0"}
    %ffi_mem_3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 5632 : i32, mem_bank = 0 : i32, sym_name = "ffi_mem_3_cons_buff_0"} : memref<2048xbf16> 
    %ffi_mem_3_cons_prod_lock_0 = aie.lock(%tile_3_4, 2) {init = 1 : i32, sym_name = "ffi_mem_3_cons_prod_lock_0"}
    %ffi_mem_3_cons_cons_lock_0 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "ffi_mem_3_cons_cons_lock_0"}
    %ffi_L3L2_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "ffi_L3L2_cons_buff_0"} : memref<2048xbf16> 
    %ffi_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 2) {init = 1 : i32, sym_name = "ffi_L3L2_cons_prod_lock_0"}
    %ffi_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "ffi_L3L2_cons_cons_lock_0"}
    %ffi_L3L2_buff_0 = aie.buffer(%tile_4_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "ffi_L3L2_buff_0"} : memref<2048xbf16> 
    %ffi_L3L2_prod_lock_0 = aie.lock(%tile_4_4, 4) {init = 1 : i32, sym_name = "ffi_L3L2_prod_lock_0"}
    %ffi_L3L2_cons_lock_0 = aie.lock(%tile_4_4, 5) {init = 0 : i32, sym_name = "ffi_L3L2_cons_lock_0"}
    %bq_mem_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "bq_mem_0_cons_buff_0"} : memref<2048xbf16> 
    %bq_mem_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "bq_mem_0_cons_prod_lock_0"}
    %bq_mem_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "bq_mem_0_cons_cons_lock_0"}
    %bq_mem_1_cons_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "bq_mem_1_cons_buff_0"} : memref<2048xbf16> 
    %bq_mem_1_cons_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "bq_mem_1_cons_prod_lock_0"}
    %bq_mem_1_cons_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "bq_mem_1_cons_cons_lock_0"}
    %bq_mem_2_cons_buff_0 = aie.buffer(%tile_3_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "bq_mem_2_cons_buff_0"} : memref<2048xbf16> 
    %bq_mem_2_cons_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "bq_mem_2_cons_prod_lock_0"}
    %bq_mem_2_cons_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "bq_mem_2_cons_cons_lock_0"}
    %bq_L3L2_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "bq_L3L2_cons_buff_0"} : memref<2048xbf16> 
    %bq_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 8) {init = 1 : i32, sym_name = "bq_L3L2_cons_prod_lock_0"}
    %bq_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 9) {init = 0 : i32, sym_name = "bq_L3L2_cons_cons_lock_0"}
    %bq_L3L2_buff_0 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bq_L3L2_buff_0"} : memref<2048xbf16> 
    %bq_L3L2_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "bq_L3L2_prod_lock_0"}
    %bq_L3L2_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "bq_L3L2_cons_lock_0"}
    %bo_mem_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "bo_mem_0_cons_buff_0"} : memref<2048xbf16> 
    %bo_mem_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 1 : i32, sym_name = "bo_mem_0_cons_prod_lock_0"}
    %bo_mem_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "bo_mem_0_cons_cons_lock_0"}
    %bo_mem_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "bo_mem_1_cons_buff_0"} : memref<2048xbf16> 
    %bo_mem_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 1 : i32, sym_name = "bo_mem_1_cons_prod_lock_0"}
    %bo_mem_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "bo_mem_1_cons_cons_lock_0"}
    %bo_mem_2_cons_buff_0 = aie.buffer(%tile_2_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "bo_mem_2_cons_buff_0"} : memref<2048xbf16> 
    %bo_mem_2_cons_prod_lock_0 = aie.lock(%tile_2_3, 4) {init = 1 : i32, sym_name = "bo_mem_2_cons_prod_lock_0"}
    %bo_mem_2_cons_cons_lock_0 = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "bo_mem_2_cons_cons_lock_0"}
    %bo_mem_3_cons_buff_0 = aie.buffer(%tile_3_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "bo_mem_3_cons_buff_0"} : memref<2048xbf16> 
    %bo_mem_3_cons_prod_lock_0 = aie.lock(%tile_3_3, 4) {init = 1 : i32, sym_name = "bo_mem_3_cons_prod_lock_0"}
    %bo_mem_3_cons_cons_lock_0 = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "bo_mem_3_cons_cons_lock_0"}
    %bo_L3L2_cons_buff_0 = aie.buffer(%mem_tile_6_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "bo_L3L2_cons_buff_0"} : memref<2048xbf16> 
    %bo_L3L2_cons_prod_lock_0 = aie.lock(%mem_tile_6_1, 0) {init = 1 : i32, sym_name = "bo_L3L2_cons_prod_lock_0"}
    %bo_L3L2_cons_cons_lock_0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "bo_L3L2_cons_cons_lock_0"}
    %bo_L3L2_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "bo_L3L2_prod_lock_0"}
    %bo_L3L2_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "bo_L3L2_cons_lock_0"}
    %anm_mem_cons_buff_0 = aie.buffer(%tile_4_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "anm_mem_cons_buff_0"} : memref<2048xbf16> 
    %anm_mem_cons_buff_1 = aie.buffer(%tile_4_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "anm_mem_cons_buff_1"} : memref<2048xbf16> 
    %anm_mem_cons_buff_2 = aie.buffer(%tile_4_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "anm_mem_cons_buff_2"} : memref<2048xbf16> 
    %anm_mem_cons_prod_lock_0 = aie.lock(%tile_4_4, 2) {init = 3 : i32, sym_name = "anm_mem_cons_prod_lock_0"}
    %anm_mem_cons_cons_lock_0 = aie.lock(%tile_4_4, 3) {init = 0 : i32, sym_name = "anm_mem_cons_cons_lock_0"}
    %anm_in_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "anm_in_cons_buff_0"} : memref<2048xbf16> 
    %anm_in_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "anm_in_cons_buff_1"} : memref<2048xbf16> 
    %anm_in_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 12) {init = 2 : i32, sym_name = "anm_in_cons_prod_lock_0"}
    %anm_in_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 13) {init = 0 : i32, sym_name = "anm_in_cons_cons_lock_0"}
    %anm_in_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "anm_in_prod_lock_0"}
    %anm_in_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "anm_in_cons_lock_0"}
    %Cqkv_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "Cqkv_3_cons_prod_lock_0"}
    %Cqkv_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "Cqkv_3_cons_cons_lock_0"}
    %Cqkv_3_buff_0 = aie.buffer(%tile_3_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Cqkv_3_buff_0"} : memref<2xbf16> 
    %Cqkv_3_buff_1 = aie.buffer(%tile_3_2) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Cqkv_3_buff_1"} : memref<2xbf16> 
    %Cqkv_3_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "Cqkv_3_prod_lock_0"}
    %Cqkv_3_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "Cqkv_3_cons_lock_0"}
    %Cqkv_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "Cqkv_2_cons_prod_lock_0"}
    %Cqkv_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "Cqkv_2_cons_cons_lock_0"}
    %Cqkv_2_buff_0 = aie.buffer(%tile_2_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Cqkv_2_buff_0"} : memref<2xbf16> 
    %Cqkv_2_buff_1 = aie.buffer(%tile_2_2) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Cqkv_2_buff_1"} : memref<2xbf16> 
    %Cqkv_2_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "Cqkv_2_prod_lock_0"}
    %Cqkv_2_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "Cqkv_2_cons_lock_0"}
    %Cqkv_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "Cqkv_1_cons_prod_lock_0"}
    %Cqkv_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "Cqkv_1_cons_cons_lock_0"}
    %Cqkv_1_buff_0 = aie.buffer(%tile_1_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Cqkv_1_buff_0"} : memref<2xbf16> 
    %Cqkv_1_buff_1 = aie.buffer(%tile_1_2) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Cqkv_1_buff_1"} : memref<2xbf16> 
    %Cqkv_1_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "Cqkv_1_prod_lock_0"}
    %Cqkv_1_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "Cqkv_1_cons_lock_0"}
    %Cqkv_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "Cqkv_0_cons_prod_lock_0"}
    %Cqkv_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "Cqkv_0_cons_cons_lock_0"}
    %Cqkv_0_buff_0 = aie.buffer(%tile_0_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "Cqkv_0_buff_0"} : memref<2xbf16> 
    %Cqkv_0_buff_1 = aie.buffer(%tile_0_2) {address = 7424 : i32, mem_bank = 0 : i32, sym_name = "Cqkv_0_buff_1"} : memref<2xbf16> 
    %Cqkv_0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "Cqkv_0_prod_lock_0"}
    %Cqkv_0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "Cqkv_0_cons_lock_0"}
    %o_out_joined_0_cons_buff_0 = aie.buffer(%tile_4_4) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "o_out_joined_0_cons_buff_0"} : memref<8xbf16> 
    %o_out_joined_0_cons_buff_1 = aie.buffer(%tile_4_4) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "o_out_joined_0_cons_buff_1"} : memref<8xbf16> 
    %o_out_joined_0_cons_prod_lock_0 = aie.lock(%tile_4_4, 0) {init = 2 : i32, sym_name = "o_out_joined_0_cons_prod_lock_0"}
    %o_out_joined_0_cons_cons_lock_0 = aie.lock(%tile_4_4, 1) {init = 0 : i32, sym_name = "o_out_joined_0_cons_cons_lock_0"}
    %o_out_joined_0_buff_0 = aie.buffer(%mem_tile_1_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "o_out_joined_0_buff_0"} : memref<8xbf16> 
    %o_out_joined_0_buff_1 = aie.buffer(%mem_tile_1_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "o_out_joined_0_buff_1"} : memref<8xbf16> 
    %o_out_joined_0_prod_lock_0 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_0"}
    %o_out_joined_0_cons_lock_0 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_0"}
    %o_out_joined_0_prod_lock_1 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_1"}
    %o_out_joined_0_cons_lock_1 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_1"}
    %o_out_joined_0_prod_lock_2 = aie.lock(%mem_tile_1_1, 8) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_2"}
    %o_out_joined_0_cons_lock_2 = aie.lock(%mem_tile_1_1, 9) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_2"}
    %o_out_joined_0_prod_lock_3 = aie.lock(%mem_tile_1_1, 10) {init = 2 : i32, sym_name = "o_out_joined_0_prod_lock_3"}
    %o_out_joined_0_cons_lock_3 = aie.lock(%mem_tile_1_1, 11) {init = 0 : i32, sym_name = "o_out_joined_0_cons_lock_3"}
    %Co_3_buff_0 = aie.buffer(%tile_3_3) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_3_buff_0"} : memref<2xbf16> 
    %Co_3_buff_1 = aie.buffer(%tile_3_3) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_3_buff_1"} : memref<2xbf16> 
    %Co_3_prod_lock_0 = aie.lock(%tile_3_3, 2) {init = 2 : i32, sym_name = "Co_3_prod_lock_0"}
    %Co_3_cons_lock_0 = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "Co_3_cons_lock_0"}
    %Co_2_buff_0 = aie.buffer(%tile_2_3) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_2_buff_0"} : memref<2xbf16> 
    %Co_2_buff_1 = aie.buffer(%tile_2_3) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_2_buff_1"} : memref<2xbf16> 
    %Co_2_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "Co_2_prod_lock_0"}
    %Co_2_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "Co_2_cons_lock_0"}
    %Co_1_buff_0 = aie.buffer(%tile_1_3) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_1_buff_0"} : memref<2xbf16> 
    %Co_1_buff_1 = aie.buffer(%tile_1_3) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_1_buff_1"} : memref<2xbf16> 
    %Co_1_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "Co_1_prod_lock_0"}
    %Co_1_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "Co_1_cons_lock_0"}
    %Co_0_buff_0 = aie.buffer(%tile_0_3) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Co_0_buff_0"} : memref<2xbf16> 
    %Co_0_buff_1 = aie.buffer(%tile_0_3) {address = 18688 : i32, mem_bank = 1 : i32, sym_name = "Co_0_buff_1"} : memref<2xbf16> 
    %Co_0_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "Co_0_prod_lock_0"}
    %Co_0_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "Co_0_cons_lock_0"}
    %ffn_out_joined_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "ffn_out_joined_1_cons_prod_lock_0"}
    %ffn_out_joined_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "ffn_out_joined_1_cons_cons_lock_0"}
    %ffn_out_joined_1_buff_0 = aie.buffer(%mem_tile_7_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "ffn_out_joined_1_buff_0"} : memref<4xbf16> 
    %ffn_out_joined_1_buff_1 = aie.buffer(%mem_tile_7_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "ffn_out_joined_1_buff_1"} : memref<4xbf16> 
    %ffn_out_joined_1_prod_lock_0 = aie.lock(%mem_tile_7_1, 0) {init = 2 : i32, sym_name = "ffn_out_joined_1_prod_lock_0"}
    %ffn_out_joined_1_cons_lock_0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "ffn_out_joined_1_cons_lock_0"}
    %ffn_out_joined_1_prod_lock_1 = aie.lock(%mem_tile_7_1, 2) {init = 2 : i32, sym_name = "ffn_out_joined_1_prod_lock_1"}
    %ffn_out_joined_1_cons_lock_1 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "ffn_out_joined_1_cons_lock_1"}
    %Cdp_3_buff_0 = aie.buffer(%tile_3_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cdp_3_buff_0"} : memref<2xbf16> 
    %Cdp_3_buff_1 = aie.buffer(%tile_3_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "Cdp_3_buff_1"} : memref<2xbf16> 
    %Cdp_3_prod_lock_0 = aie.lock(%tile_3_5, 2) {init = 2 : i32, sym_name = "Cdp_3_prod_lock_0"}
    %Cdp_3_cons_lock_0 = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "Cdp_3_cons_lock_0"}
    %Cdp_2_buff_0 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cdp_2_buff_0"} : memref<2xbf16> 
    %Cdp_2_buff_1 = aie.buffer(%tile_2_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "Cdp_2_buff_1"} : memref<2xbf16> 
    %Cdp_2_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 2 : i32, sym_name = "Cdp_2_prod_lock_0"}
    %Cdp_2_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "Cdp_2_cons_lock_0"}
    %ffn_out_joined_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "ffn_out_joined_0_cons_prod_lock_0"}
    %ffn_out_joined_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "ffn_out_joined_0_cons_cons_lock_0"}
    %ffn_out_joined_0_buff_0 = aie.buffer(%mem_tile_3_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "ffn_out_joined_0_buff_0"} : memref<4xbf16> 
    %ffn_out_joined_0_buff_1 = aie.buffer(%mem_tile_3_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "ffn_out_joined_0_buff_1"} : memref<4xbf16> 
    %ffn_out_joined_0_prod_lock_0 = aie.lock(%mem_tile_3_1, 4) {init = 2 : i32, sym_name = "ffn_out_joined_0_prod_lock_0"}
    %ffn_out_joined_0_cons_lock_0 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "ffn_out_joined_0_cons_lock_0"}
    %ffn_out_joined_0_prod_lock_1 = aie.lock(%mem_tile_3_1, 6) {init = 2 : i32, sym_name = "ffn_out_joined_0_prod_lock_1"}
    %ffn_out_joined_0_cons_lock_1 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "ffn_out_joined_0_cons_lock_1"}
    %Cdp_1_buff_0 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cdp_1_buff_0"} : memref<2xbf16> 
    %Cdp_1_buff_1 = aie.buffer(%tile_1_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "Cdp_1_buff_1"} : memref<2xbf16> 
    %Cdp_1_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "Cdp_1_prod_lock_0"}
    %Cdp_1_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "Cdp_1_cons_lock_0"}
    %Cdp_0_buff_0 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "Cdp_0_buff_0"} : memref<2xbf16> 
    %Cdp_0_buff_1 = aie.buffer(%tile_0_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "Cdp_0_buff_1"} : memref<2xbf16> 
    %Cdp_0_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "Cdp_0_prod_lock_0"}
    %Cdp_0_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "Cdp_0_cons_lock_0"}
    %Aqkv_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Aqkv_3_cons_buff_0"} : memref<2304xui8> 
    %Aqkv_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Aqkv_3_cons_buff_1"} : memref<2304xui8> 
    %Aqkv_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "Aqkv_3_cons_prod_lock_0"}
    %Aqkv_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "Aqkv_3_cons_cons_lock_0"}
    %Aqkv_src_1_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "Aqkv_src_1_cons_buff_0"} : memref<4608xui8> 
    %Aqkv_src_1_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "Aqkv_src_1_cons_buff_1"} : memref<4608xui8> 
    %Aqkv_src_1_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 8) {init = 2 : i32, sym_name = "Aqkv_src_1_cons_prod_lock_0"}
    %Aqkv_src_1_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 9) {init = 0 : i32, sym_name = "Aqkv_src_1_cons_cons_lock_0"}
    %Aqkv_src_1_cons_prod_lock_1 = aie.lock(%mem_tile_2_1, 10) {init = 2 : i32, sym_name = "Aqkv_src_1_cons_prod_lock_1"}
    %Aqkv_src_1_cons_cons_lock_1 = aie.lock(%mem_tile_2_1, 11) {init = 0 : i32, sym_name = "Aqkv_src_1_cons_cons_lock_1"}
    %Aqkv_src_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "Aqkv_src_1_prod_lock_0"}
    %Aqkv_src_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "Aqkv_src_1_cons_lock_0"}
    %Aqkv_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Aqkv_2_cons_buff_0"} : memref<2304xui8> 
    %Aqkv_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Aqkv_2_cons_buff_1"} : memref<2304xui8> 
    %Aqkv_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "Aqkv_2_cons_prod_lock_0"}
    %Aqkv_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "Aqkv_2_cons_cons_lock_0"}
    %Aqkv_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Aqkv_1_cons_buff_0"} : memref<2304xui8> 
    %Aqkv_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Aqkv_1_cons_buff_1"} : memref<2304xui8> 
    %Aqkv_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "Aqkv_1_cons_prod_lock_0"}
    %Aqkv_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "Aqkv_1_cons_cons_lock_0"}
    %Aqkv_src_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "Aqkv_src_0_cons_buff_0"} : memref<4608xui8> 
    %Aqkv_src_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "Aqkv_src_0_cons_buff_1"} : memref<4608xui8> 
    %Aqkv_src_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "Aqkv_src_0_cons_prod_lock_0"}
    %Aqkv_src_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "Aqkv_src_0_cons_cons_lock_0"}
    %Aqkv_src_0_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "Aqkv_src_0_cons_prod_lock_1"}
    %Aqkv_src_0_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "Aqkv_src_0_cons_cons_lock_1"}
    %Aqkv_src_0_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "Aqkv_src_0_prod_lock_0"}
    %Aqkv_src_0_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "Aqkv_src_0_cons_lock_0"}
    %Aqkv_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "Aqkv_0_cons_buff_0"} : memref<2304xui8> 
    %Aqkv_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "Aqkv_0_cons_buff_1"} : memref<2304xui8> 
    %Aqkv_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "Aqkv_0_cons_prod_lock_0"}
    %Aqkv_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "Aqkv_0_cons_cons_lock_0"}
    %Ao_3_cons_buff_0 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_3_cons_buff_0"} : memref<2304xui8> 
    %Ao_3_cons_buff_1 = aie.buffer(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_3_cons_buff_1"} : memref<2304xui8> 
    %Ao_3_cons_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "Ao_3_cons_prod_lock_0"}
    %Ao_3_cons_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "Ao_3_cons_cons_lock_0"}
    %Ao_src_1_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "Ao_src_1_cons_buff_0"} : memref<4608xui8> 
    %Ao_src_1_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "Ao_src_1_cons_buff_1"} : memref<4608xui8> 
    %Ao_src_1_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "Ao_src_1_cons_prod_lock_0"}
    %Ao_src_1_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "Ao_src_1_cons_cons_lock_0"}
    %Ao_src_1_cons_prod_lock_1 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "Ao_src_1_cons_prod_lock_1"}
    %Ao_src_1_cons_cons_lock_1 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "Ao_src_1_cons_cons_lock_1"}
    %Ao_src_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "Ao_src_1_prod_lock_0"}
    %Ao_src_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "Ao_src_1_cons_lock_0"}
    %Ao_2_cons_buff_0 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_2_cons_buff_0"} : memref<2304xui8> 
    %Ao_2_cons_buff_1 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_2_cons_buff_1"} : memref<2304xui8> 
    %Ao_2_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "Ao_2_cons_prod_lock_0"}
    %Ao_2_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "Ao_2_cons_cons_lock_0"}
    %Ao_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_1_cons_buff_0"} : memref<2304xui8> 
    %Ao_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_1_cons_buff_1"} : memref<2304xui8> 
    %Ao_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "Ao_1_cons_prod_lock_0"}
    %Ao_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "Ao_1_cons_cons_lock_0"}
    %Ao_src_0_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "Ao_src_0_cons_buff_0"} : memref<4608xui8> 
    %Ao_src_0_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "Ao_src_0_cons_buff_1"} : memref<4608xui8> 
    %Ao_src_0_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "Ao_src_0_cons_prod_lock_0"}
    %Ao_src_0_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "Ao_src_0_cons_cons_lock_0"}
    %Ao_src_0_cons_prod_lock_1 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "Ao_src_0_cons_prod_lock_1"}
    %Ao_src_0_cons_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "Ao_src_0_cons_cons_lock_1"}
    %Ao_src_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "Ao_src_0_prod_lock_0"}
    %Ao_src_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "Ao_src_0_cons_lock_0"}
    %Ao_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Ao_0_cons_buff_0"} : memref<2304xui8> 
    %Ao_0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "Ao_0_cons_buff_1"} : memref<2304xui8> 
    %Ao_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "Ao_0_cons_prod_lock_0"}
    %Ao_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "Ao_0_cons_cons_lock_0"}
    %Adp_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Adp_3_cons_buff_0"} : memref<2304xui8> 
    %Adp_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Adp_3_cons_buff_1"} : memref<2304xui8> 
    %Adp_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 0) {init = 2 : i32, sym_name = "Adp_3_cons_prod_lock_0"}
    %Adp_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "Adp_3_cons_cons_lock_0"}
    %Agu_3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_3_cons_buff_0"} : memref<4608xui8> 
    %Agu_3_cons_buff_1 = aie.buffer(%tile_3_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Agu_3_cons_buff_1"} : memref<4608xui8> 
    %Agu_3_cons_prod_lock_0 = aie.lock(%tile_3_4, 0) {init = 2 : i32, sym_name = "Agu_3_cons_prod_lock_0"}
    %Agu_3_cons_cons_lock_0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "Agu_3_cons_cons_lock_0"}
    %Agu_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_2_cons_buff_0"} : memref<4608xui8> 
    %Agu_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Agu_2_cons_buff_1"} : memref<4608xui8> 
    %Agu_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "Agu_2_cons_prod_lock_0"}
    %Agu_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "Agu_2_cons_cons_lock_0"}
    %Ffn_src_1_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "Ffn_src_1_cons_buff_0"} : memref<13824xui8> 
    %Ffn_src_1_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "Ffn_src_1_cons_buff_1"} : memref<13824xui8> 
    %Ffn_src_1_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "Ffn_src_1_cons_prod_lock_0"}
    %Ffn_src_1_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "Ffn_src_1_cons_cons_lock_0"}
    %Ffn_src_1_cons_prod_lock_1 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "Ffn_src_1_cons_prod_lock_1"}
    %Ffn_src_1_cons_cons_lock_1 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "Ffn_src_1_cons_cons_lock_1"}
    %Ffn_src_1_cons_prod_lock_2 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "Ffn_src_1_cons_prod_lock_2"}
    %Ffn_src_1_cons_cons_lock_2 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "Ffn_src_1_cons_cons_lock_2"}
    %Ffn_src_1_cons_prod_lock_3 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "Ffn_src_1_cons_prod_lock_3"}
    %Ffn_src_1_cons_cons_lock_3 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "Ffn_src_1_cons_cons_lock_3"}
    %Ffn_src_1_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "Ffn_src_1_prod_lock_0"}
    %Ffn_src_1_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "Ffn_src_1_cons_lock_0"}
    %Adp_2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Adp_2_cons_buff_0"} : memref<2304xui8> 
    %Adp_2_cons_buff_1 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Adp_2_cons_buff_1"} : memref<2304xui8> 
    %Adp_2_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 2 : i32, sym_name = "Adp_2_cons_prod_lock_0"}
    %Adp_2_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "Adp_2_cons_cons_lock_0"}
    %Adp_1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Adp_1_cons_buff_0"} : memref<2304xui8> 
    %Adp_1_cons_buff_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Adp_1_cons_buff_1"} : memref<2304xui8> 
    %Adp_1_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "Adp_1_cons_prod_lock_0"}
    %Adp_1_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "Adp_1_cons_cons_lock_0"}
    %Agu_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_1_cons_buff_0"} : memref<4608xui8> 
    %Agu_1_cons_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Agu_1_cons_buff_1"} : memref<4608xui8> 
    %Agu_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "Agu_1_cons_prod_lock_0"}
    %Agu_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "Agu_1_cons_cons_lock_0"}
    %Agu_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Agu_0_cons_buff_0"} : memref<4608xui8> 
    %Agu_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Agu_0_cons_buff_1"} : memref<4608xui8> 
    %Agu_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "Agu_0_cons_prod_lock_0"}
    %Agu_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "Agu_0_cons_cons_lock_0"}
    %Ffn_src_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "Ffn_src_0_cons_buff_0"} : memref<13824xui8> 
    %Ffn_src_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "Ffn_src_0_cons_buff_1"} : memref<13824xui8> 
    %Ffn_src_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "Ffn_src_0_cons_prod_lock_0"}
    %Ffn_src_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "Ffn_src_0_cons_cons_lock_0"}
    %Ffn_src_0_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "Ffn_src_0_cons_prod_lock_1"}
    %Ffn_src_0_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "Ffn_src_0_cons_cons_lock_1"}
    %Ffn_src_0_cons_prod_lock_2 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "Ffn_src_0_cons_prod_lock_2"}
    %Ffn_src_0_cons_cons_lock_2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "Ffn_src_0_cons_cons_lock_2"}
    %Ffn_src_0_cons_prod_lock_3 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "Ffn_src_0_cons_prod_lock_3"}
    %Ffn_src_0_cons_cons_lock_3 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "Ffn_src_0_cons_cons_lock_3"}
    %Ffn_src_0_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "Ffn_src_0_prod_lock_0"}
    %Ffn_src_0_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "Ffn_src_0_cons_lock_0"}
    %Adp_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "Adp_0_cons_buff_0"} : memref<2304xui8> 
    %Adp_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "Adp_0_cons_buff_1"} : memref<2304xui8> 
    %Adp_0_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "Adp_0_cons_prod_lock_0"}
    %Adp_0_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "Adp_0_cons_cons_lock_0"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 3, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 4, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 5, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 4, %tile_2_2, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_2_1, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 5, %tile_3_2, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_1_5, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.flow(%mem_tile_3_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_2_5, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%tile_3_5, DMA : 0, %mem_tile_7_1, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_1_1, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_2_3, DMA : 0, %mem_tile_1_1, DMA : 3)
    aie.flow(%tile_3_3, DMA : 0, %mem_tile_1_1, DMA : 4)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_4_4, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_1_1, DMA : 5)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_4_4, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_6_1, DMA : 0)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%tile_0_2, DMA : 1, %mem_tile_3_1, DMA : 3)
    aie.flow(%mem_tile_3_1, DMA : 3, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 3, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 3, %tile_1_2, DMA : 1)
    aie.flow(%tile_4_4, DMA : 0, %mem_tile_6_1, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_0_2, DMA : 1)
    func.func private @op0_layer_fused_pre_rms_col0_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_qkv_gemv_static_bf16(i32, i32, memref<2304xui8>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_qkv_gemv_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_o_proj_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_gate_up_bf16(i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_silu_mul_bf16(memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_down_partial_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    %anm_oout_buf = aie.buffer(%tile_4_4) {address = 5120 : i32, mem_bank = 0 : i32, sym_name = "anm_oout_buf"} : memref<2048xbf16> 
    func.func private @op0_layer_fused_o_out_assemble_bf16(memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    %anm_inpff_buf = aie.buffer(%tile_4_4) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "anm_inpff_buf"} : memref<2048xbf16> 
    func.func private @op0_layer_fused_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_rms_norm2_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 36864 : i32, mem_bank = 2 : i32, sym_name = "_anonymous0"} : memref<4xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %c3 = arith.constant 3 : index
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c3_i32 = arith.constant 3 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<4xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<4xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<4xi32>
      memref.store %c0_i32, %_anonymous0[%c3] : memref<4xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb48
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb49
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb47
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb48
    ^bb4:  // pred: ^bb3
      aie.use_lock(%rms_in_cons_cons_lock_0, AcquireGreaterEqual, 2)
      %4 = memref.load %_anonymous0[%c0] : memref<4xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb8,
        0: ^bb5,
        1: ^bb6,
        2: ^bb7
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb9(%rms_in_cons_buff_0 : memref<2048xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb9(%rms_in_cons_buff_1 : memref<2048xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb9(%rms_in_cons_buff_2 : memref<2048xbf16>)
    ^bb8:  // pred: ^bb4
      cf.br ^bb9(%rms_in_cons_buff_0 : memref<2048xbf16>)
    ^bb9(%7: memref<2048xbf16>):  // 4 preds: ^bb5, ^bb6, ^bb7, ^bb8
      %8 = memref.load %_anonymous0[%c0] : memref<4xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i32
      cf.switch %10 : i32, [
        default: ^bb13,
        0: ^bb10,
        1: ^bb11,
        2: ^bb12
      ]
    ^bb10:  // pred: ^bb9
      cf.br ^bb14(%rms_in_cons_buff_1 : memref<2048xbf16>)
    ^bb11:  // pred: ^bb9
      cf.br ^bb14(%rms_in_cons_buff_2 : memref<2048xbf16>)
    ^bb12:  // pred: ^bb9
      cf.br ^bb14(%rms_in_cons_buff_0 : memref<2048xbf16>)
    ^bb13:  // pred: ^bb9
      cf.br ^bb14(%rms_in_cons_buff_1 : memref<2048xbf16>)
    ^bb14(%11: memref<2048xbf16>):  // 4 preds: ^bb10, ^bb11, ^bb12, ^bb13
      aie.use_lock(%bq_L3L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_layer_fused_pre_rms_col0_bf16(%7, %11, %bq_L3L2_buff_0, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%rms_in_cons_prod_lock_0, Release, 2)
      %12 = memref.load %_anonymous0[%c0] : memref<4xi32>
      %13 = arith.addi %12, %c2_i32 : i32
      %14 = arith.cmpi sge, %13, %c3_i32 : i32
      %15 = arith.subi %13, %c3_i32 : i32
      %16 = arith.select %14, %15, %13 : i32
      memref.store %16, %_anonymous0[%c0] : memref<4xi32>
      aie.use_lock(%bq_L3L2_cons_lock_0, Release, 1)
      %17 = memref.load %_anonymous0[%c1] : memref<4xi32>
      %18 = arith.addi %17, %c1_i32 : i32
      %19 = arith.cmpi sge, %18, %c1_i32 : i32
      %20 = arith.select %19, %17, %18 : i32
      memref.store %20, %_anonymous0[%c1] : memref<4xi32>
      cf.br ^bb15(%c0 : index)
    ^bb15(%21: index):  // 2 preds: ^bb14, ^bb24
      %22 = arith.cmpi slt, %21, %c256 : index
      cf.cond_br %22, ^bb16, ^bb25
    ^bb16:  // pred: ^bb15
      aie.use_lock(%Aqkv_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %23 = memref.load %_anonymous0[%c2] : memref<4xi32>
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %24 : index to i32
      cf.switch %25 : i32, [
        default: ^bb19,
        0: ^bb17,
        1: ^bb18
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb20(%Aqkv_0_cons_buff_0 : memref<2304xui8>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb20(%Aqkv_0_cons_buff_1 : memref<2304xui8>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb20(%Aqkv_0_cons_buff_0 : memref<2304xui8>)
    ^bb20(%26: memref<2304xui8>):  // 3 preds: ^bb17, ^bb18, ^bb19
      aie.use_lock(%Cqkv_0_prod_lock_0, AcquireGreaterEqual, 1)
      %27 = memref.load %_anonymous0[%c3] : memref<4xi32>
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %28 : index to i32
      cf.switch %29 : i32, [
        default: ^bb23,
        0: ^bb21,
        1: ^bb22
      ]
    ^bb21:  // pred: ^bb20
      cf.br ^bb24(%Cqkv_0_buff_0 : memref<2xbf16>)
    ^bb22:  // pred: ^bb20
      cf.br ^bb24(%Cqkv_0_buff_1 : memref<2xbf16>)
    ^bb23:  // pred: ^bb20
      cf.br ^bb24(%Cqkv_0_buff_0 : memref<2xbf16>)
    ^bb24(%30: memref<2xbf16>):  // 3 preds: ^bb21, ^bb22, ^bb23
      func.call @op0_layer_fused_qkv_gemv_static_bf16(%c2_i32, %c0_i32, %26, %30) : (i32, i32, memref<2304xui8>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_0_cons_prod_lock_0, Release, 1)
      %31 = memref.load %_anonymous0[%c2] : memref<4xi32>
      %32 = arith.addi %31, %c1_i32 : i32
      %33 = arith.cmpi sge, %32, %c2_i32 : i32
      %34 = arith.subi %32, %c2_i32 : i32
      %35 = arith.select %33, %34, %32 : i32
      memref.store %35, %_anonymous0[%c2] : memref<4xi32>
      aie.use_lock(%Cqkv_0_cons_lock_0, Release, 1)
      %36 = memref.load %_anonymous0[%c3] : memref<4xi32>
      %37 = arith.addi %36, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.subi %37, %c2_i32 : i32
      %40 = arith.select %38, %39, %37 : i32
      memref.store %40, %_anonymous0[%c3] : memref<4xi32>
      %41 = arith.addi %21, %c1 : index
      cf.br ^bb15(%41 : index)
    ^bb25:  // pred: ^bb15
      cf.br ^bb26(%c0 : index)
    ^bb26(%42: index):  // 2 preds: ^bb25, ^bb35
      %43 = arith.cmpi slt, %42, %c64 : index
      cf.cond_br %43, ^bb27, ^bb36
    ^bb27:  // pred: ^bb26
      aie.use_lock(%Aqkv_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %44 = memref.load %_anonymous0[%c2] : memref<4xi32>
      %45 = arith.index_cast %44 : i32 to index
      %46 = arith.index_cast %45 : index to i32
      cf.switch %46 : i32, [
        default: ^bb30,
        0: ^bb28,
        1: ^bb29
      ]
    ^bb28:  // pred: ^bb27
      cf.br ^bb31(%Aqkv_0_cons_buff_0 : memref<2304xui8>)
    ^bb29:  // pred: ^bb27
      cf.br ^bb31(%Aqkv_0_cons_buff_1 : memref<2304xui8>)
    ^bb30:  // pred: ^bb27
      cf.br ^bb31(%Aqkv_0_cons_buff_0 : memref<2304xui8>)
    ^bb31(%47: memref<2304xui8>):  // 3 preds: ^bb28, ^bb29, ^bb30
      aie.use_lock(%Cqkv_0_prod_lock_0, AcquireGreaterEqual, 1)
      %48 = memref.load %_anonymous0[%c3] : memref<4xi32>
      %49 = arith.index_cast %48 : i32 to index
      %50 = arith.index_cast %49 : index to i32
      cf.switch %50 : i32, [
        default: ^bb34,
        0: ^bb32,
        1: ^bb33
      ]
    ^bb32:  // pred: ^bb31
      cf.br ^bb35(%Cqkv_0_buff_0 : memref<2xbf16>)
    ^bb33:  // pred: ^bb31
      cf.br ^bb35(%Cqkv_0_buff_1 : memref<2xbf16>)
    ^bb34:  // pred: ^bb31
      cf.br ^bb35(%Cqkv_0_buff_0 : memref<2xbf16>)
    ^bb35(%51: memref<2xbf16>):  // 3 preds: ^bb32, ^bb33, ^bb34
      func.call @op0_layer_fused_qkv_gemv_static_bf16(%c2_i32, %c0_i32, %47, %51) : (i32, i32, memref<2304xui8>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_0_cons_prod_lock_0, Release, 1)
      %52 = memref.load %_anonymous0[%c2] : memref<4xi32>
      %53 = arith.addi %52, %c1_i32 : i32
      %54 = arith.cmpi sge, %53, %c2_i32 : i32
      %55 = arith.subi %53, %c2_i32 : i32
      %56 = arith.select %54, %55, %53 : i32
      memref.store %56, %_anonymous0[%c2] : memref<4xi32>
      aie.use_lock(%Cqkv_0_cons_lock_0, Release, 1)
      %57 = memref.load %_anonymous0[%c3] : memref<4xi32>
      %58 = arith.addi %57, %c1_i32 : i32
      %59 = arith.cmpi sge, %58, %c2_i32 : i32
      %60 = arith.subi %58, %c2_i32 : i32
      %61 = arith.select %59, %60, %58 : i32
      memref.store %61, %_anonymous0[%c3] : memref<4xi32>
      %62 = arith.addi %42, %c1 : index
      cf.br ^bb26(%62 : index)
    ^bb36:  // pred: ^bb26
      cf.br ^bb37(%c0 : index)
    ^bb37(%63: index):  // 2 preds: ^bb36, ^bb46
      %64 = arith.cmpi slt, %63, %c64 : index
      cf.cond_br %64, ^bb38, ^bb47
    ^bb38:  // pred: ^bb37
      aie.use_lock(%Aqkv_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %65 = memref.load %_anonymous0[%c2] : memref<4xi32>
      %66 = arith.index_cast %65 : i32 to index
      %67 = arith.index_cast %66 : index to i32
      cf.switch %67 : i32, [
        default: ^bb41,
        0: ^bb39,
        1: ^bb40
      ]
    ^bb39:  // pred: ^bb38
      cf.br ^bb42(%Aqkv_0_cons_buff_0 : memref<2304xui8>)
    ^bb40:  // pred: ^bb38
      cf.br ^bb42(%Aqkv_0_cons_buff_1 : memref<2304xui8>)
    ^bb41:  // pred: ^bb38
      cf.br ^bb42(%Aqkv_0_cons_buff_0 : memref<2304xui8>)
    ^bb42(%68: memref<2304xui8>):  // 3 preds: ^bb39, ^bb40, ^bb41
      aie.use_lock(%Cqkv_0_prod_lock_0, AcquireGreaterEqual, 1)
      %69 = memref.load %_anonymous0[%c3] : memref<4xi32>
      %70 = arith.index_cast %69 : i32 to index
      %71 = arith.index_cast %70 : index to i32
      cf.switch %71 : i32, [
        default: ^bb45,
        0: ^bb43,
        1: ^bb44
      ]
    ^bb43:  // pred: ^bb42
      cf.br ^bb46(%Cqkv_0_buff_0 : memref<2xbf16>)
    ^bb44:  // pred: ^bb42
      cf.br ^bb46(%Cqkv_0_buff_1 : memref<2xbf16>)
    ^bb45:  // pred: ^bb42
      cf.br ^bb46(%Cqkv_0_buff_0 : memref<2xbf16>)
    ^bb46(%72: memref<2xbf16>):  // 3 preds: ^bb43, ^bb44, ^bb45
      func.call @op0_layer_fused_qkv_gemv_static_bf16(%c2_i32, %c0_i32, %68, %72) : (i32, i32, memref<2304xui8>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_0_cons_prod_lock_0, Release, 1)
      %73 = memref.load %_anonymous0[%c2] : memref<4xi32>
      %74 = arith.addi %73, %c1_i32 : i32
      %75 = arith.cmpi sge, %74, %c2_i32 : i32
      %76 = arith.subi %74, %c2_i32 : i32
      %77 = arith.select %75, %76, %74 : i32
      memref.store %77, %_anonymous0[%c2] : memref<4xi32>
      aie.use_lock(%Cqkv_0_cons_lock_0, Release, 1)
      %78 = memref.load %_anonymous0[%c3] : memref<4xi32>
      %79 = arith.addi %78, %c1_i32 : i32
      %80 = arith.cmpi sge, %79, %c2_i32 : i32
      %81 = arith.subi %79, %c2_i32 : i32
      %82 = arith.select %80, %81, %79 : i32
      memref.store %82, %_anonymous0[%c3] : memref<4xi32>
      %83 = arith.addi %63, %c1 : index
      cf.br ^bb37(%83 : index)
    ^bb47:  // pred: ^bb37
      %84 = arith.addi %2, %c1 : index
      cf.br ^bb3(%84 : index)
    ^bb48:  // pred: ^bb3
      %85 = arith.addi %0, %c1 : index
      cf.br ^bb1(%85 : index)
    ^bb49:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb38
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb39
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb37
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb38
    ^bb4:  // pred: ^bb3
      aie.use_lock(%bq_mem_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Aqkv_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_1_cons_buff_0 : memref<2304xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_1_cons_buff_1 : memref<2304xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_1_cons_buff_0 : memref<2304xui8>)
    ^bb10(%9: memref<2304xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%Cqkv_1_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_1_buff_0 : memref<2xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_1_buff_1 : memref<2xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_1_buff_0 : memref<2xbf16>)
    ^bb14(%13: memref<2xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %9, %bq_mem_0_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_1_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_1_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous1[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      cf.br ^bb16(%c0 : index)
    ^bb16(%25: index):  // 2 preds: ^bb15, ^bb25
      %26 = arith.cmpi slt, %25, %c64 : index
      cf.cond_br %26, ^bb17, ^bb26
    ^bb17:  // pred: ^bb16
      aie.use_lock(%Aqkv_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %27 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %28 : index to i32
      cf.switch %29 : i32, [
        default: ^bb20,
        0: ^bb18,
        1: ^bb19
      ]
    ^bb18:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_1_cons_buff_0 : memref<2304xui8>)
    ^bb19:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_1_cons_buff_1 : memref<2304xui8>)
    ^bb20:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_1_cons_buff_0 : memref<2304xui8>)
    ^bb21(%30: memref<2304xui8>):  // 3 preds: ^bb18, ^bb19, ^bb20
      aie.use_lock(%Cqkv_1_prod_lock_0, AcquireGreaterEqual, 1)
      %31 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.index_cast %32 : index to i32
      cf.switch %33 : i32, [
        default: ^bb24,
        0: ^bb22,
        1: ^bb23
      ]
    ^bb22:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_1_buff_0 : memref<2xbf16>)
    ^bb23:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_1_buff_1 : memref<2xbf16>)
    ^bb24:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_1_buff_0 : memref<2xbf16>)
    ^bb25(%34: memref<2xbf16>):  // 3 preds: ^bb22, ^bb23, ^bb24
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %30, %bq_mem_0_cons_buff_0, %34) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_1_cons_prod_lock_0, Release, 1)
      %35 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %36 = arith.addi %35, %c1_i32 : i32
      %37 = arith.cmpi sge, %36, %c2_i32 : i32
      %38 = arith.subi %36, %c2_i32 : i32
      %39 = arith.select %37, %38, %36 : i32
      memref.store %39, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_1_cons_lock_0, Release, 1)
      %40 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %41 = arith.addi %40, %c1_i32 : i32
      %42 = arith.cmpi sge, %41, %c2_i32 : i32
      %43 = arith.subi %41, %c2_i32 : i32
      %44 = arith.select %42, %43, %41 : i32
      memref.store %44, %_anonymous1[%c2] : memref<3xi32>
      %45 = arith.addi %25, %c1 : index
      cf.br ^bb16(%45 : index)
    ^bb26:  // pred: ^bb16
      cf.br ^bb27(%c0 : index)
    ^bb27(%46: index):  // 2 preds: ^bb26, ^bb36
      %47 = arith.cmpi slt, %46, %c64 : index
      cf.cond_br %47, ^bb28, ^bb37
    ^bb28:  // pred: ^bb27
      aie.use_lock(%Aqkv_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %48 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %49 = arith.index_cast %48 : i32 to index
      %50 = arith.index_cast %49 : index to i32
      cf.switch %50 : i32, [
        default: ^bb31,
        0: ^bb29,
        1: ^bb30
      ]
    ^bb29:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_1_cons_buff_0 : memref<2304xui8>)
    ^bb30:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_1_cons_buff_1 : memref<2304xui8>)
    ^bb31:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_1_cons_buff_0 : memref<2304xui8>)
    ^bb32(%51: memref<2304xui8>):  // 3 preds: ^bb29, ^bb30, ^bb31
      aie.use_lock(%Cqkv_1_prod_lock_0, AcquireGreaterEqual, 1)
      %52 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %53 = arith.index_cast %52 : i32 to index
      %54 = arith.index_cast %53 : index to i32
      cf.switch %54 : i32, [
        default: ^bb35,
        0: ^bb33,
        1: ^bb34
      ]
    ^bb33:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_1_buff_0 : memref<2xbf16>)
    ^bb34:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_1_buff_1 : memref<2xbf16>)
    ^bb35:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_1_buff_0 : memref<2xbf16>)
    ^bb36(%55: memref<2xbf16>):  // 3 preds: ^bb33, ^bb34, ^bb35
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %51, %bq_mem_0_cons_buff_0, %55) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_1_cons_prod_lock_0, Release, 1)
      %56 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %57 = arith.addi %56, %c1_i32 : i32
      %58 = arith.cmpi sge, %57, %c2_i32 : i32
      %59 = arith.subi %57, %c2_i32 : i32
      %60 = arith.select %58, %59, %57 : i32
      memref.store %60, %_anonymous1[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_1_cons_lock_0, Release, 1)
      %61 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous1[%c2] : memref<3xi32>
      %66 = arith.addi %46, %c1 : index
      cf.br ^bb27(%66 : index)
    ^bb37:  // pred: ^bb27
      aie.use_lock(%bq_mem_0_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c1_i32 : i32
      %70 = arith.select %69, %67, %68 : i32
      memref.store %70, %_anonymous1[%c0] : memref<3xi32>
      %71 = arith.addi %2, %c1 : index
      cf.br ^bb3(%71 : index)
    ^bb38:  // pred: ^bb3
      %72 = arith.addi %0, %c1 : index
      cf.br ^bb1(%72 : index)
    ^bb39:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb38
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb39
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb37
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb38
    ^bb4:  // pred: ^bb3
      aie.use_lock(%bq_mem_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Aqkv_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_2_cons_buff_0 : memref<2304xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_2_cons_buff_1 : memref<2304xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_2_cons_buff_0 : memref<2304xui8>)
    ^bb10(%9: memref<2304xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%Cqkv_2_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_2_buff_0 : memref<2xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_2_buff_1 : memref<2xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_2_buff_0 : memref<2xbf16>)
    ^bb14(%13: memref<2xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %9, %bq_mem_1_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_2_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_2_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous2[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      cf.br ^bb16(%c0 : index)
    ^bb16(%25: index):  // 2 preds: ^bb15, ^bb25
      %26 = arith.cmpi slt, %25, %c64 : index
      cf.cond_br %26, ^bb17, ^bb26
    ^bb17:  // pred: ^bb16
      aie.use_lock(%Aqkv_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %27 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %28 : index to i32
      cf.switch %29 : i32, [
        default: ^bb20,
        0: ^bb18,
        1: ^bb19
      ]
    ^bb18:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_2_cons_buff_0 : memref<2304xui8>)
    ^bb19:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_2_cons_buff_1 : memref<2304xui8>)
    ^bb20:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_2_cons_buff_0 : memref<2304xui8>)
    ^bb21(%30: memref<2304xui8>):  // 3 preds: ^bb18, ^bb19, ^bb20
      aie.use_lock(%Cqkv_2_prod_lock_0, AcquireGreaterEqual, 1)
      %31 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.index_cast %32 : index to i32
      cf.switch %33 : i32, [
        default: ^bb24,
        0: ^bb22,
        1: ^bb23
      ]
    ^bb22:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_2_buff_0 : memref<2xbf16>)
    ^bb23:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_2_buff_1 : memref<2xbf16>)
    ^bb24:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_2_buff_0 : memref<2xbf16>)
    ^bb25(%34: memref<2xbf16>):  // 3 preds: ^bb22, ^bb23, ^bb24
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %30, %bq_mem_1_cons_buff_0, %34) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_2_cons_prod_lock_0, Release, 1)
      %35 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %36 = arith.addi %35, %c1_i32 : i32
      %37 = arith.cmpi sge, %36, %c2_i32 : i32
      %38 = arith.subi %36, %c2_i32 : i32
      %39 = arith.select %37, %38, %36 : i32
      memref.store %39, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_2_cons_lock_0, Release, 1)
      %40 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %41 = arith.addi %40, %c1_i32 : i32
      %42 = arith.cmpi sge, %41, %c2_i32 : i32
      %43 = arith.subi %41, %c2_i32 : i32
      %44 = arith.select %42, %43, %41 : i32
      memref.store %44, %_anonymous2[%c2] : memref<3xi32>
      %45 = arith.addi %25, %c1 : index
      cf.br ^bb16(%45 : index)
    ^bb26:  // pred: ^bb16
      cf.br ^bb27(%c0 : index)
    ^bb27(%46: index):  // 2 preds: ^bb26, ^bb36
      %47 = arith.cmpi slt, %46, %c64 : index
      cf.cond_br %47, ^bb28, ^bb37
    ^bb28:  // pred: ^bb27
      aie.use_lock(%Aqkv_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %48 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %49 = arith.index_cast %48 : i32 to index
      %50 = arith.index_cast %49 : index to i32
      cf.switch %50 : i32, [
        default: ^bb31,
        0: ^bb29,
        1: ^bb30
      ]
    ^bb29:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_2_cons_buff_0 : memref<2304xui8>)
    ^bb30:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_2_cons_buff_1 : memref<2304xui8>)
    ^bb31:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_2_cons_buff_0 : memref<2304xui8>)
    ^bb32(%51: memref<2304xui8>):  // 3 preds: ^bb29, ^bb30, ^bb31
      aie.use_lock(%Cqkv_2_prod_lock_0, AcquireGreaterEqual, 1)
      %52 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %53 = arith.index_cast %52 : i32 to index
      %54 = arith.index_cast %53 : index to i32
      cf.switch %54 : i32, [
        default: ^bb35,
        0: ^bb33,
        1: ^bb34
      ]
    ^bb33:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_2_buff_0 : memref<2xbf16>)
    ^bb34:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_2_buff_1 : memref<2xbf16>)
    ^bb35:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_2_buff_0 : memref<2xbf16>)
    ^bb36(%55: memref<2xbf16>):  // 3 preds: ^bb33, ^bb34, ^bb35
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %51, %bq_mem_1_cons_buff_0, %55) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_2_cons_prod_lock_0, Release, 1)
      %56 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %57 = arith.addi %56, %c1_i32 : i32
      %58 = arith.cmpi sge, %57, %c2_i32 : i32
      %59 = arith.subi %57, %c2_i32 : i32
      %60 = arith.select %58, %59, %57 : i32
      memref.store %60, %_anonymous2[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_2_cons_lock_0, Release, 1)
      %61 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous2[%c2] : memref<3xi32>
      %66 = arith.addi %46, %c1 : index
      cf.br ^bb27(%66 : index)
    ^bb37:  // pred: ^bb27
      aie.use_lock(%bq_mem_1_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c1_i32 : i32
      %70 = arith.select %69, %67, %68 : i32
      memref.store %70, %_anonymous2[%c0] : memref<3xi32>
      %71 = arith.addi %2, %c1 : index
      cf.br ^bb3(%71 : index)
    ^bb38:  // pred: ^bb3
      %72 = arith.addi %0, %c1 : index
      cf.br ^bb1(%72 : index)
    ^bb39:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
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
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb38
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb39
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb37
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb38
    ^bb4:  // pred: ^bb3
      aie.use_lock(%bq_mem_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Aqkv_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_3_cons_buff_0 : memref<2304xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_3_cons_buff_1 : memref<2304xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Aqkv_3_cons_buff_0 : memref<2304xui8>)
    ^bb10(%9: memref<2304xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%Cqkv_3_prod_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_3_buff_0 : memref<2xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_3_buff_1 : memref<2xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Cqkv_3_buff_0 : memref<2xbf16>)
    ^bb14(%13: memref<2xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %9, %bq_mem_2_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_3_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_3_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous3[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      cf.br ^bb16(%c0 : index)
    ^bb16(%25: index):  // 2 preds: ^bb15, ^bb25
      %26 = arith.cmpi slt, %25, %c64 : index
      cf.cond_br %26, ^bb17, ^bb26
    ^bb17:  // pred: ^bb16
      aie.use_lock(%Aqkv_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %27 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %28 : index to i32
      cf.switch %29 : i32, [
        default: ^bb20,
        0: ^bb18,
        1: ^bb19
      ]
    ^bb18:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_3_cons_buff_0 : memref<2304xui8>)
    ^bb19:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_3_cons_buff_1 : memref<2304xui8>)
    ^bb20:  // pred: ^bb17
      cf.br ^bb21(%Aqkv_3_cons_buff_0 : memref<2304xui8>)
    ^bb21(%30: memref<2304xui8>):  // 3 preds: ^bb18, ^bb19, ^bb20
      aie.use_lock(%Cqkv_3_prod_lock_0, AcquireGreaterEqual, 1)
      %31 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.index_cast %32 : index to i32
      cf.switch %33 : i32, [
        default: ^bb24,
        0: ^bb22,
        1: ^bb23
      ]
    ^bb22:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_3_buff_0 : memref<2xbf16>)
    ^bb23:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_3_buff_1 : memref<2xbf16>)
    ^bb24:  // pred: ^bb21
      cf.br ^bb25(%Cqkv_3_buff_0 : memref<2xbf16>)
    ^bb25(%34: memref<2xbf16>):  // 3 preds: ^bb22, ^bb23, ^bb24
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %30, %bq_mem_2_cons_buff_0, %34) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_3_cons_prod_lock_0, Release, 1)
      %35 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %36 = arith.addi %35, %c1_i32 : i32
      %37 = arith.cmpi sge, %36, %c2_i32 : i32
      %38 = arith.subi %36, %c2_i32 : i32
      %39 = arith.select %37, %38, %36 : i32
      memref.store %39, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_3_cons_lock_0, Release, 1)
      %40 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %41 = arith.addi %40, %c1_i32 : i32
      %42 = arith.cmpi sge, %41, %c2_i32 : i32
      %43 = arith.subi %41, %c2_i32 : i32
      %44 = arith.select %42, %43, %41 : i32
      memref.store %44, %_anonymous3[%c2] : memref<3xi32>
      %45 = arith.addi %25, %c1 : index
      cf.br ^bb16(%45 : index)
    ^bb26:  // pred: ^bb16
      cf.br ^bb27(%c0 : index)
    ^bb27(%46: index):  // 2 preds: ^bb26, ^bb36
      %47 = arith.cmpi slt, %46, %c64 : index
      cf.cond_br %47, ^bb28, ^bb37
    ^bb28:  // pred: ^bb27
      aie.use_lock(%Aqkv_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %48 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %49 = arith.index_cast %48 : i32 to index
      %50 = arith.index_cast %49 : index to i32
      cf.switch %50 : i32, [
        default: ^bb31,
        0: ^bb29,
        1: ^bb30
      ]
    ^bb29:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_3_cons_buff_0 : memref<2304xui8>)
    ^bb30:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_3_cons_buff_1 : memref<2304xui8>)
    ^bb31:  // pred: ^bb28
      cf.br ^bb32(%Aqkv_3_cons_buff_0 : memref<2304xui8>)
    ^bb32(%51: memref<2304xui8>):  // 3 preds: ^bb29, ^bb30, ^bb31
      aie.use_lock(%Cqkv_3_prod_lock_0, AcquireGreaterEqual, 1)
      %52 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %53 = arith.index_cast %52 : i32 to index
      %54 = arith.index_cast %53 : index to i32
      cf.switch %54 : i32, [
        default: ^bb35,
        0: ^bb33,
        1: ^bb34
      ]
    ^bb33:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_3_buff_0 : memref<2xbf16>)
    ^bb34:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_3_buff_1 : memref<2xbf16>)
    ^bb35:  // pred: ^bb32
      cf.br ^bb36(%Cqkv_3_buff_0 : memref<2xbf16>)
    ^bb36(%55: memref<2xbf16>):  // 3 preds: ^bb33, ^bb34, ^bb35
      func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %51, %bq_mem_2_cons_buff_0, %55) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Aqkv_3_cons_prod_lock_0, Release, 1)
      %56 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %57 = arith.addi %56, %c1_i32 : i32
      %58 = arith.cmpi sge, %57, %c2_i32 : i32
      %59 = arith.subi %57, %c2_i32 : i32
      %60 = arith.select %58, %59, %57 : i32
      memref.store %60, %_anonymous3[%c1] : memref<3xi32>
      aie.use_lock(%Cqkv_3_cons_lock_0, Release, 1)
      %61 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %62 = arith.addi %61, %c1_i32 : i32
      %63 = arith.cmpi sge, %62, %c2_i32 : i32
      %64 = arith.subi %62, %c2_i32 : i32
      %65 = arith.select %63, %64, %62 : i32
      memref.store %65, %_anonymous3[%c2] : memref<3xi32>
      %66 = arith.addi %46, %c1 : index
      cf.br ^bb27(%66 : index)
    ^bb37:  // pred: ^bb27
      aie.use_lock(%bq_mem_2_cons_prod_lock_0, Release, 1)
      %67 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %68 = arith.addi %67, %c1_i32 : i32
      %69 = arith.cmpi sge, %68, %c1_i32 : i32
      %70 = arith.select %69, %67, %68 : i32
      memref.store %70, %_anonymous3[%c0] : memref<3xi32>
      %71 = arith.addi %2, %c1 : index
      cf.br ^bb3(%71 : index)
    ^bb38:  // pred: ^bb3
      %72 = arith.addi %0, %c1 : index
      cf.br ^bb1(%72 : index)
    ^bb39:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous4 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
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
      memref.store %c0_i32, %_anonymous4[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<3xi32>
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
      aie.use_lock(%bo_mem_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous4[%c1] : memref<3xi32>
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
      %10 = memref.load %_anonymous4[%c2] : memref<3xi32>
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
      func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %9, %bo_mem_0_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_0_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous4[%c1] : memref<3xi32>
      aie.use_lock(%Co_0_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous4[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%bo_mem_0_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous4[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous5 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
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
      aie.use_lock(%bo_mem_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous5[%c1] : memref<3xi32>
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
      %10 = memref.load %_anonymous5[%c2] : memref<3xi32>
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
      func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %9, %bo_mem_1_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_1_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous5[%c1] : memref<3xi32>
      aie.use_lock(%Co_1_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous5[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%bo_mem_1_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous5[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous6 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<3xi32>
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
      aie.use_lock(%bo_mem_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous6[%c1] : memref<3xi32>
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
      %10 = memref.load %_anonymous6[%c2] : memref<3xi32>
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
      func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %9, %bo_mem_2_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_2_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous6[%c1] : memref<3xi32>
      aie.use_lock(%Co_2_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous6[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%bo_mem_2_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous6[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous7 = aie.buffer(%tile_3_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_3_3 = aie.core(%tile_3_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<3xi32>
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
      aie.use_lock(%bo_mem_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous7[%c1] : memref<3xi32>
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
      %10 = memref.load %_anonymous7[%c2] : memref<3xi32>
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
      func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %9, %bo_mem_3_cons_buff_0, %13) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Ao_3_cons_prod_lock_0, Release, 1)
      %14 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %15 = arith.addi %14, %c1_i32 : i32
      %16 = arith.cmpi sge, %15, %c2_i32 : i32
      %17 = arith.subi %15, %c2_i32 : i32
      %18 = arith.select %16, %17, %15 : i32
      memref.store %18, %_anonymous7[%c1] : memref<3xi32>
      aie.use_lock(%Co_3_cons_lock_0, Release, 1)
      %19 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous7[%c2] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%bo_mem_3_cons_prod_lock_0, Release, 1)
      %25 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %26 = arith.addi %25, %c1_i32 : i32
      %27 = arith.cmpi sge, %26, %c1_i32 : i32
      %28 = arith.select %27, %25, %26 : i32
      memref.store %28, %_anonymous7[%c0] : memref<3xi32>
      %29 = arith.addi %2, %c1 : index
      cf.br ^bb3(%29 : index)
    ^bb16:  // pred: ^bb3
      %30 = arith.addi %0, %c1 : index
      cf.br ^bb1(%30 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous8 = aie.buffer(%tile_0_4) {address = 20992 : i32, mem_bank = 1 : i32, sym_name = "_anonymous8"} : memref<3xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c512 = arith.constant 512 : index
      %c4_i32 = arith.constant 4 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous8[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c512 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Agu_0_cons_buff_0 : memref<4608xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Agu_0_cons_buff_1 : memref<4608xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Agu_0_cons_buff_0 : memref<4608xui8>)
    ^bb10(%9: memref<4608xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %9, %ffi_mem_0_cons_buff_0, %c0_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_0_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c2_i32 : i32
      %13 = arith.subi %11, %c2_i32 : i32
      %14 = arith.select %12, %13, %11 : i32
      memref.store %14, %_anonymous8[%c1] : memref<3xi32>
      aie.use_lock(%Agu_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %15 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.index_cast %16 : index to i32
      cf.switch %17 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Agu_0_cons_buff_0 : memref<4608xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Agu_0_cons_buff_1 : memref<4608xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Agu_0_cons_buff_0 : memref<4608xui8>)
    ^bb14(%18: memref<4608xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %18, %ffi_mem_0_cons_buff_0, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_0_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous8[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%inter_0_prod_lock_0, AcquireGreaterEqual, 1)
      %25 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %26 = arith.index_cast %25 : i32 to index
      %27 = arith.index_cast %26 : index to i32
      cf.switch %27 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%inter_0_buff_0 : memref<2048xbf16>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%inter_0_buff_1 : memref<2048xbf16>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%inter_0_buff_0 : memref<2048xbf16>)
    ^bb19(%28: memref<2048xbf16>):  // 3 preds: ^bb16, ^bb17, ^bb18
      func.call @op0_layer_fused_silu_mul_bf16(%28, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
      aie.use_lock(%inter_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous8[%c2] : memref<3xi32>
      aie.use_lock(%ffi_mem_0_cons_prod_lock_0, Release, 1)
      %34 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %35 = arith.addi %34, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c1_i32 : i32
      %37 = arith.select %36, %34, %35 : i32
      memref.store %37, %_anonymous8[%c0] : memref<3xi32>
      %38 = arith.addi %2, %c1 : index
      cf.br ^bb3(%38 : index)
    ^bb20:  // pred: ^bb3
      %39 = arith.addi %0, %c1 : index
      cf.br ^bb1(%39 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous9 = aie.buffer(%tile_1_4) {address = 20992 : i32, mem_bank = 1 : i32, sym_name = "_anonymous9"} : memref<3xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c512 = arith.constant 512 : index
      %c4_i32 = arith.constant 4 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous9[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c512 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Agu_1_cons_buff_0 : memref<4608xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Agu_1_cons_buff_1 : memref<4608xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Agu_1_cons_buff_0 : memref<4608xui8>)
    ^bb10(%9: memref<4608xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %9, %ffi_mem_1_cons_buff_0, %c0_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_1_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c2_i32 : i32
      %13 = arith.subi %11, %c2_i32 : i32
      %14 = arith.select %12, %13, %11 : i32
      memref.store %14, %_anonymous9[%c1] : memref<3xi32>
      aie.use_lock(%Agu_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %15 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.index_cast %16 : index to i32
      cf.switch %17 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Agu_1_cons_buff_0 : memref<4608xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Agu_1_cons_buff_1 : memref<4608xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Agu_1_cons_buff_0 : memref<4608xui8>)
    ^bb14(%18: memref<4608xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %18, %ffi_mem_1_cons_buff_0, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_1_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous9[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%inter_1_prod_lock_0, AcquireGreaterEqual, 1)
      %25 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %26 = arith.index_cast %25 : i32 to index
      %27 = arith.index_cast %26 : index to i32
      cf.switch %27 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%inter_1_buff_0 : memref<2048xbf16>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%inter_1_buff_1 : memref<2048xbf16>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%inter_1_buff_0 : memref<2048xbf16>)
    ^bb19(%28: memref<2048xbf16>):  // 3 preds: ^bb16, ^bb17, ^bb18
      func.call @op0_layer_fused_silu_mul_bf16(%28, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
      aie.use_lock(%inter_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous9[%c2] : memref<3xi32>
      aie.use_lock(%ffi_mem_1_cons_prod_lock_0, Release, 1)
      %34 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %35 = arith.addi %34, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c1_i32 : i32
      %37 = arith.select %36, %34, %35 : i32
      memref.store %37, %_anonymous9[%c0] : memref<3xi32>
      %38 = arith.addi %2, %c1 : index
      cf.br ^bb3(%38 : index)
    ^bb20:  // pred: ^bb3
      %39 = arith.addi %0, %c1 : index
      cf.br ^bb1(%39 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous10 = aie.buffer(%tile_2_4) {address = 20992 : i32, mem_bank = 1 : i32, sym_name = "_anonymous10"} : memref<3xi32> 
    %core_2_4 = aie.core(%tile_2_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c512 = arith.constant 512 : index
      %c4_i32 = arith.constant 4 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous10[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c512 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Agu_2_cons_buff_0 : memref<4608xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Agu_2_cons_buff_1 : memref<4608xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Agu_2_cons_buff_0 : memref<4608xui8>)
    ^bb10(%9: memref<4608xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %9, %ffi_mem_2_cons_buff_0, %c0_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_2_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c2_i32 : i32
      %13 = arith.subi %11, %c2_i32 : i32
      %14 = arith.select %12, %13, %11 : i32
      memref.store %14, %_anonymous10[%c1] : memref<3xi32>
      aie.use_lock(%Agu_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %15 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.index_cast %16 : index to i32
      cf.switch %17 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Agu_2_cons_buff_0 : memref<4608xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Agu_2_cons_buff_1 : memref<4608xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Agu_2_cons_buff_0 : memref<4608xui8>)
    ^bb14(%18: memref<4608xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %18, %ffi_mem_2_cons_buff_0, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_2_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous10[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%inter_2_prod_lock_0, AcquireGreaterEqual, 1)
      %25 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %26 = arith.index_cast %25 : i32 to index
      %27 = arith.index_cast %26 : index to i32
      cf.switch %27 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%inter_2_buff_0 : memref<2048xbf16>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%inter_2_buff_1 : memref<2048xbf16>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%inter_2_buff_0 : memref<2048xbf16>)
    ^bb19(%28: memref<2048xbf16>):  // 3 preds: ^bb16, ^bb17, ^bb18
      func.call @op0_layer_fused_silu_mul_bf16(%28, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
      aie.use_lock(%inter_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous10[%c2] : memref<3xi32>
      aie.use_lock(%ffi_mem_2_cons_prod_lock_0, Release, 1)
      %34 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %35 = arith.addi %34, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c1_i32 : i32
      %37 = arith.select %36, %34, %35 : i32
      memref.store %37, %_anonymous10[%c0] : memref<3xi32>
      %38 = arith.addi %2, %c1 : index
      cf.br ^bb3(%38 : index)
    ^bb20:  // pred: ^bb3
      %39 = arith.addi %0, %c1 : index
      cf.br ^bb1(%39 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous11 = aie.buffer(%tile_3_4) {address = 20992 : i32, mem_bank = 1 : i32, sym_name = "_anonymous11"} : memref<3xi32> 
    %core_3_4 = aie.core(%tile_3_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c512 = arith.constant 512 : index
      %c4_i32 = arith.constant 4 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      memref.store %c0_i32, %_anonymous11[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%ffi_mem_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb14
      %5 = arith.cmpi slt, %4, %c512 : index
      cf.cond_br %5, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Agu_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.index_cast %7 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%Agu_3_cons_buff_0 : memref<4608xui8>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%Agu_3_cons_buff_1 : memref<4608xui8>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%Agu_3_cons_buff_0 : memref<4608xui8>)
    ^bb10(%9: memref<4608xui8>):  // 3 preds: ^bb7, ^bb8, ^bb9
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %9, %ffi_mem_3_cons_buff_0, %c0_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_3_cons_prod_lock_0, Release, 1)
      %10 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %11 = arith.addi %10, %c1_i32 : i32
      %12 = arith.cmpi sge, %11, %c2_i32 : i32
      %13 = arith.subi %11, %c2_i32 : i32
      %14 = arith.select %12, %13, %11 : i32
      memref.store %14, %_anonymous11[%c1] : memref<3xi32>
      aie.use_lock(%Agu_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %15 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.index_cast %16 : index to i32
      cf.switch %17 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Agu_3_cons_buff_0 : memref<4608xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Agu_3_cons_buff_1 : memref<4608xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Agu_3_cons_buff_0 : memref<4608xui8>)
    ^bb14(%18: memref<4608xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %18, %ffi_mem_3_cons_buff_0, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%Agu_3_cons_prod_lock_0, Release, 1)
      %19 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c2_i32 : i32
      %22 = arith.subi %20, %c2_i32 : i32
      %23 = arith.select %21, %22, %20 : i32
      memref.store %23, %_anonymous11[%c1] : memref<3xi32>
      %24 = arith.addi %4, %c1 : index
      cf.br ^bb5(%24 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%inter_3_prod_lock_0, AcquireGreaterEqual, 1)
      %25 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %26 = arith.index_cast %25 : i32 to index
      %27 = arith.index_cast %26 : index to i32
      cf.switch %27 : i32, [
        default: ^bb18,
        0: ^bb16,
        1: ^bb17
      ]
    ^bb16:  // pred: ^bb15
      cf.br ^bb19(%inter_3_buff_0 : memref<2048xbf16>)
    ^bb17:  // pred: ^bb15
      cf.br ^bb19(%inter_3_buff_1 : memref<2048xbf16>)
    ^bb18:  // pred: ^bb15
      cf.br ^bb19(%inter_3_buff_0 : memref<2048xbf16>)
    ^bb19(%28: memref<2048xbf16>):  // 3 preds: ^bb16, ^bb17, ^bb18
      func.call @op0_layer_fused_silu_mul_bf16(%28, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
      aie.use_lock(%inter_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous11[%c2] : memref<3xi32>
      aie.use_lock(%ffi_mem_3_cons_prod_lock_0, Release, 1)
      %34 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %35 = arith.addi %34, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c1_i32 : i32
      %37 = arith.select %36, %34, %35 : i32
      memref.store %37, %_anonymous11[%c0] : memref<3xi32>
      %38 = arith.addi %2, %c1 : index
      cf.br ^bb3(%38 : index)
    ^bb20:  // pred: ^bb3
      %39 = arith.addi %0, %c1 : index
      cf.br ^bb1(%39 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous12 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "_anonymous12"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1024 = arith.constant 1024 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous12[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inter_0_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%inter_0_buff_0 : memref<2048xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%inter_0_buff_1 : memref<2048xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%inter_0_buff_0 : memref<2048xbf16>)
    ^bb8(%7: memref<2048xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c1024 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Adp_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Adp_0_cons_buff_0 : memref<2304xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Adp_0_cons_buff_1 : memref<2304xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Adp_0_cons_buff_0 : memref<2304xui8>)
    ^bb14(%13: memref<2304xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%Cdp_0_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%Cdp_0_buff_0 : memref<2xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%Cdp_0_buff_1 : memref<2xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%Cdp_0_buff_0 : memref<2xbf16>)
    ^bb18(%17: memref<2xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %13, %7, %17) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Adp_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous12[%c1] : memref<3xi32>
      aie.use_lock(%Cdp_0_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous12[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      aie.use_lock(%inter_0_prod_lock_0, Release, 1)
      %29 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous12[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous13 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "_anonymous13"} : memref<3xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1024 = arith.constant 1024 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous13[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inter_1_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%inter_1_buff_0 : memref<2048xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%inter_1_buff_1 : memref<2048xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%inter_1_buff_0 : memref<2048xbf16>)
    ^bb8(%7: memref<2048xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c1024 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Adp_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Adp_1_cons_buff_0 : memref<2304xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Adp_1_cons_buff_1 : memref<2304xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Adp_1_cons_buff_0 : memref<2304xui8>)
    ^bb14(%13: memref<2304xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%Cdp_1_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%Cdp_1_buff_0 : memref<2xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%Cdp_1_buff_1 : memref<2xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%Cdp_1_buff_0 : memref<2xbf16>)
    ^bb18(%17: memref<2xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %13, %7, %17) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Adp_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous13[%c1] : memref<3xi32>
      aie.use_lock(%Cdp_1_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous13[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      aie.use_lock(%inter_1_prod_lock_0, Release, 1)
      %29 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous13[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous14 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "_anonymous14"} : memref<3xi32> 
    %core_2_5 = aie.core(%tile_2_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1024 = arith.constant 1024 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous14[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inter_2_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%inter_2_buff_0 : memref<2048xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%inter_2_buff_1 : memref<2048xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%inter_2_buff_0 : memref<2048xbf16>)
    ^bb8(%7: memref<2048xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c1024 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Adp_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Adp_2_cons_buff_0 : memref<2304xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Adp_2_cons_buff_1 : memref<2304xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Adp_2_cons_buff_0 : memref<2304xui8>)
    ^bb14(%13: memref<2304xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%Cdp_2_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%Cdp_2_buff_0 : memref<2xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%Cdp_2_buff_1 : memref<2xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%Cdp_2_buff_0 : memref<2xbf16>)
    ^bb18(%17: memref<2xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %13, %7, %17) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Adp_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous14[%c1] : memref<3xi32>
      aie.use_lock(%Cdp_2_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous14[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      aie.use_lock(%inter_2_prod_lock_0, Release, 1)
      %29 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous14[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous15 = aie.buffer(%tile_3_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "_anonymous15"} : memref<3xi32> 
    %core_3_5 = aie.core(%tile_3_5) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1024 = arith.constant 1024 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous15[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb20
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb21
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb19
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb20
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inter_3_cons_lock_0, AcquireGreaterEqual, 1)
      %4 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb7,
        0: ^bb5,
        1: ^bb6
      ]
    ^bb5:  // pred: ^bb4
      cf.br ^bb8(%inter_3_buff_0 : memref<2048xbf16>)
    ^bb6:  // pred: ^bb4
      cf.br ^bb8(%inter_3_buff_1 : memref<2048xbf16>)
    ^bb7:  // pred: ^bb4
      cf.br ^bb8(%inter_3_buff_0 : memref<2048xbf16>)
    ^bb8(%7: memref<2048xbf16>):  // 3 preds: ^bb5, ^bb6, ^bb7
      cf.br ^bb9(%c0 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb18
      %9 = arith.cmpi slt, %8, %c1024 : index
      cf.cond_br %9, ^bb10, ^bb19
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Adp_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%Adp_3_cons_buff_0 : memref<2304xui8>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%Adp_3_cons_buff_1 : memref<2304xui8>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%Adp_3_cons_buff_0 : memref<2304xui8>)
    ^bb14(%13: memref<2304xui8>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%Cdp_3_prod_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb17,
        0: ^bb15,
        1: ^bb16
      ]
    ^bb15:  // pred: ^bb14
      cf.br ^bb18(%Cdp_3_buff_0 : memref<2xbf16>)
    ^bb16:  // pred: ^bb14
      cf.br ^bb18(%Cdp_3_buff_1 : memref<2xbf16>)
    ^bb17:  // pred: ^bb14
      cf.br ^bb18(%Cdp_3_buff_0 : memref<2xbf16>)
    ^bb18(%17: memref<2xbf16>):  // 3 preds: ^bb15, ^bb16, ^bb17
      func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %13, %7, %17) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
      aie.use_lock(%Adp_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous15[%c1] : memref<3xi32>
      aie.use_lock(%Cdp_3_cons_lock_0, Release, 1)
      %23 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous15[%c2] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb19:  // pred: ^bb9
      aie.use_lock(%inter_3_prod_lock_0, Release, 1)
      %29 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.subi %30, %c2_i32 : i32
      %33 = arith.select %31, %32, %30 : i32
      memref.store %33, %_anonymous15[%c0] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb20:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb21:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    %_anonymous16 = aie.buffer(%tile_4_4) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "_anonymous16"} : memref<3xi32> 
    %core_4_4 = aie.core(%tile_4_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c512_i32 = arith.constant 512 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous16[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous16[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous16[%c2] : memref<3xi32>
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
      %7 = memref.load %_anonymous16[%c0] : memref<3xi32>
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
      func.call @op0_layer_fused_o_out_assemble_bf16(%10, %anm_oout_buf, %6, %c0_i32, %c4_i32, %c2_i32, %c512_i32) : (memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%o_out_joined_0_cons_prod_lock_0, Release, 1)
      %11 = memref.load %_anonymous16[%c0] : memref<3xi32>
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c2_i32 : i32
      %14 = arith.subi %12, %c2_i32 : i32
      %15 = arith.select %13, %14, %12 : i32
      memref.store %15, %_anonymous16[%c0] : memref<3xi32>
      %16 = arith.addi %4, %c1 : index
      cf.br ^bb5(%16 : index)
    ^bb11:  // pred: ^bb5
      aie.use_lock(%anm_mem_cons_cons_lock_0, AcquireGreaterEqual, 2)
      %17 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.index_cast %18 : index to i32
      cf.switch %19 : i32, [
        default: ^bb15,
        0: ^bb12,
        1: ^bb13,
        2: ^bb14
      ]
    ^bb12:  // pred: ^bb11
      cf.br ^bb16(%anm_mem_cons_buff_0 : memref<2048xbf16>)
    ^bb13:  // pred: ^bb11
      cf.br ^bb16(%anm_mem_cons_buff_1 : memref<2048xbf16>)
    ^bb14:  // pred: ^bb11
      cf.br ^bb16(%anm_mem_cons_buff_2 : memref<2048xbf16>)
    ^bb15:  // pred: ^bb11
      cf.br ^bb16(%anm_mem_cons_buff_0 : memref<2048xbf16>)
    ^bb16(%20: memref<2048xbf16>):  // 4 preds: ^bb12, ^bb13, ^bb14, ^bb15
      %21 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.index_cast %22 : index to i32
      cf.switch %23 : i32, [
        default: ^bb20,
        0: ^bb17,
        1: ^bb18,
        2: ^bb19
      ]
    ^bb17:  // pred: ^bb16
      cf.br ^bb21(%anm_mem_cons_buff_1 : memref<2048xbf16>)
    ^bb18:  // pred: ^bb16
      cf.br ^bb21(%anm_mem_cons_buff_2 : memref<2048xbf16>)
    ^bb19:  // pred: ^bb16
      cf.br ^bb21(%anm_mem_cons_buff_0 : memref<2048xbf16>)
    ^bb20:  // pred: ^bb16
      cf.br ^bb21(%anm_mem_cons_buff_1 : memref<2048xbf16>)
    ^bb21(%24: memref<2048xbf16>):  // 4 preds: ^bb17, ^bb18, ^bb19, ^bb20
      aie.use_lock(%ffi_L3L2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_layer_fused_add_bf16(%anm_oout_buf, %20, %anm_inpff_buf, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
      func.call @op0_layer_fused_rms_norm2_bf16(%anm_inpff_buf, %24, %ffi_L3L2_buff_0, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
      aie.use_lock(%anm_mem_cons_prod_lock_0, Release, 2)
      %25 = memref.load %_anonymous16[%c1] : memref<3xi32>
      %26 = arith.addi %25, %c2_i32 : i32
      %27 = arith.cmpi sge, %26, %c3_i32 : i32
      %28 = arith.subi %26, %c3_i32 : i32
      %29 = arith.select %27, %28, %26 : i32
      memref.store %29, %_anonymous16[%c1] : memref<3xi32>
      aie.use_lock(%ffi_L3L2_cons_lock_0, Release, 1)
      %30 = memref.load %_anonymous16[%c2] : memref<3xi32>
      %31 = arith.addi %30, %c1_i32 : i32
      %32 = arith.cmpi sge, %31, %c1_i32 : i32
      %33 = arith.select %32, %30, %31 : i32
      memref.store %33, %_anonymous16[%c2] : memref<3xi32>
      %34 = arith.addi %2, %c1 : index
      cf.br ^bb3(%34 : index)
    ^bb22:  // pred: ^bb3
      %35 = arith.addi %0, %c1 : index
      cf.br ^bb1(%35 : index)
    ^bb23:  // pred: ^bb1
      aie.end
    } {link_files = ["op0_layer_fused_2048_8192_g32.o"]}
    aie.runtime_sequence(%arg0: memref<1771520xbf16>, %arg1: memref<1179648xbf16>, %arg2: memref<14157824xbf16>, %arg3: memref<2097152xbf16>, %arg4: memref<300032xbf16>) {
      %0 = aiex.dma_configure_task_for @rms_in_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @rms_in_shim_alloc {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      %2 = aiex.dma_configure_task_for @Aqkv_src_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 2048, 589824, [<size = 1, stride = 0>, <size = 256, stride = 1152>, <size = 2, stride = 294912>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      %3 = aiex.dma_configure_task_for @Ao_src_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 0, 36864, [<size = 16, stride = 36864>, <size = 64, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      %4 = aiex.dma_configure_task_for @Ffn_src_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 2048, 147456, [<size = 32, stride = 2304>, <size = 64, stride = 73728>, <size = 72, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      aiex.dma_await_task(%4)
      %5 = aiex.dma_configure_task_for @Aqkv_src_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 591872, 589824, [<size = 1, stride = 0>, <size = 256, stride = 1152>, <size = 2, stride = 294912>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      aiex.dma_await_task(%5)
      %6 = aiex.dma_configure_task_for @Ao_src_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 589824, 36864, [<size = 16, stride = 36864>, <size = 64, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%6)
      %7 = aiex.dma_configure_task_for @Ffn_src_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 4720640, 147456, [<size = 32, stride = 2304>, <size = 64, stride = 73728>, <size = 72, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      aiex.dma_await_task(%7)
      %8 = aiex.dma_configure_task_for @Aqkv_src_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @Ao_src_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @Ffn_src_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 0, 6912, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 6912, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @Aqkv_src_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @Ao_src_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @Ffn_src_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 0, 6912, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 6912, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @Cqkv_0_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266240, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @Cqkv_1_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266752, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @Cqkv_2_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267264, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @Cqkv_3_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267776, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @Cqkv_0_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266240, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      aiex.dma_await_task(%18)
      %19 = aiex.dma_configure_task_for @Cqkv_1_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266752, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      aiex.dma_await_task(%19)
      %20 = aiex.dma_configure_task_for @Cqkv_2_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267264, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      aiex.dma_await_task(%20)
      %21 = aiex.dma_configure_task_for @Cqkv_3_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267776, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      aiex.dma_await_task(%21)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      %22 = aiex.dma_configure_task_for @bo_L3L2_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 269312, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      aiex.dma_free_task(%22)
      %23 = aiex.dma_configure_task_for @anm_in_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @anm_in_shim_alloc {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      %25 = aiex.dma_configure_task_for @ffn_out_joined_0_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 291840, 2048, [<size = 2, stride = 1024>, <size = 512, stride = 2>, <size = 2, stride = 2048>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @ffn_out_joined_1_shim_alloc {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 295936, 2048, [<size = 2, stride = 1024>, <size = 512, stride = 2>, <size = 2, stride = 2048>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%26)
      aiex.dma_await_task(%26)
      aiex.dma_free_task(%25)
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ffn_src_0_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 9216, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ffn_src_0_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 9216, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb12)
    ^bb4:  // 2 preds: ^bb3, ^bb11
      aie.use_lock(%Ffn_src_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 0, 4608) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Ffn_src_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 4608, 4608) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ffn_src_0_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 9216, 2304) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%Ffn_src_0_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 11520, 2304) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%Ffn_src_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 0, 4608) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%Ffn_src_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 4608, 4608) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Ffn_src_0_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 9216, 2304) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%Ffn_src_0_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 11520, 2304) {bd_id = 9 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Ffn_src_0_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb4
    ^bb12:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%Ffn_src_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 0, 4608) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%Ffn_src_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 0, 4608) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%Ffn_src_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 4608, 4608) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%Ffn_src_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 4608, 4608) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%Ffn_src_0_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_0 : memref<13824xui8>, 11520, 2304) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%Ffn_src_0_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_0_cons_buff_1 : memref<13824xui8>, 11520, 2304) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%Ffn_src_0_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(MM2S, 4, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 1, ^bb25, ^bb29)
    ^bb25:  // 2 preds: ^bb24, ^bb28
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 31 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb29:  // pred: ^bb24
      %7 = aie.dma_start(MM2S, 5, ^bb30, ^bb32)
    ^bb30:  // 2 preds: ^bb29, ^bb31
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%Aqkv_src_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_0_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 33 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%Aqkv_src_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb30
    ^bb32:  // pred: ^bb29
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Adp_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_0_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Adp_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Adp_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_0_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Adp_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cdp_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_0_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cdp_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cdp_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_0_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cdp_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @Ffn_src_0_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Agu_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_0_cons_buff_0 : memref<4608xui8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Agu_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Agu_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_0_cons_buff_1 : memref<4608xui8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%ffi_mem_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_0_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ffi_mem_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Agu_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_1_cons_buff_0 : memref<4608xui8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Agu_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Agu_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_1_cons_buff_1 : memref<4608xui8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%ffi_mem_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_1_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ffi_mem_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Adp_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_1_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Adp_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Adp_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_1_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Adp_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cdp_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_1_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cdp_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cdp_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_1_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cdp_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ffn_src_1_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 9216, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ffn_src_1_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 9216, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb12)
    ^bb4:  // 2 preds: ^bb3, ^bb11
      aie.use_lock(%Ffn_src_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 0, 4608) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Ffn_src_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 4608, 4608) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ffn_src_1_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 9216, 2304) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%Ffn_src_1_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 11520, 2304) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%Ffn_src_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 0, 4608) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%Ffn_src_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 4608, 4608) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Ffn_src_1_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 9216, 2304) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%Ffn_src_1_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 11520, 2304) {bd_id = 9 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Ffn_src_1_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb4
    ^bb12:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%Ffn_src_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 0, 4608) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%Ffn_src_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 0, 4608) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %3 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%Ffn_src_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 4608, 4608) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%Ffn_src_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 4608, 4608) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %4 = aie.dma_start(MM2S, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%Ffn_src_1_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_0 : memref<13824xui8>, 11520, 2304) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%Ffn_src_1_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ffn_src_1_cons_buff_1 : memref<13824xui8>, 11520, 2304) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%Ffn_src_1_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %5 = aie.dma_start(MM2S, 4, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 13 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %6 = aie.dma_start(S2MM, 1, ^bb25, ^bb29)
    ^bb25:  // 2 preds: ^bb24, ^bb28
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 31 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb25
    ^bb29:  // pred: ^bb24
      %7 = aie.dma_start(MM2S, 5, ^bb30, ^bb32)
    ^bb30:  // 2 preds: ^bb29, ^bb31
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%Aqkv_src_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_src_1_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 33 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%Aqkv_src_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb30
    ^bb32:  // pred: ^bb29
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Adp_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_2_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Adp_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Adp_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_2_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Adp_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cdp_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_2_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cdp_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cdp_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_2_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cdp_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @Ffn_src_1_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Agu_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_2_cons_buff_0 : memref<4608xui8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Agu_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Agu_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_2_cons_buff_1 : memref<4608xui8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%ffi_mem_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ffi_mem_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Agu_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_3_cons_buff_0 : memref<4608xui8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Agu_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Agu_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Agu_3_cons_buff_1 : memref<4608xui8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Agu_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%ffi_mem_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_mem_3_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ffi_mem_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Adp_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_3_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Adp_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Adp_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Adp_3_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Adp_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cdp_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_3_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cdp_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cdp_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cdp_3_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cdp_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ao_src_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ao_src_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ao_src_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ao_src_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // 2 preds: ^bb3, ^bb7
      aie.use_lock(%Ao_src_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Ao_src_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Ao_src_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Ao_src_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_src_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Ao_src_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%Ao_src_0_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Ao_src_0_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
    ^bb9:  // 2 preds: ^bb8, ^bb10
      aie.use_lock(%Ao_src_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Ao_src_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Ao_src_0_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_0_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Ao_src_0_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb11:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb12, ^bb14)
    ^bb12:  // 2 preds: ^bb11, ^bb13
      aie.use_lock(%o_out_joined_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 0, 2) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%o_out_joined_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 0, 2) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb14:  // pred: ^bb11
      %4 = aie.dma_start(S2MM, 2, ^bb15, ^bb17)
    ^bb15:  // 2 preds: ^bb14, ^bb16
      aie.use_lock(%o_out_joined_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 2, 2) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%o_out_joined_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 2, 2) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb15
    ^bb17:  // pred: ^bb14
      %5 = aie.dma_start(S2MM, 3, ^bb18, ^bb20)
    ^bb18:  // 2 preds: ^bb17, ^bb19
      aie.use_lock(%o_out_joined_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 4, 2) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%o_out_joined_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 4, 2) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb18
    ^bb20:  // pred: ^bb17
      %6 = aie.dma_start(S2MM, 4, ^bb21, ^bb23)
    ^bb21:  // 2 preds: ^bb20, ^bb22
      aie.use_lock(%o_out_joined_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 6, 2) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%o_out_joined_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 6, 2) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%o_out_joined_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb21
    ^bb23:  // pred: ^bb20
      %7 = aie.dma_start(MM2S, 2, ^bb24, ^bb32)
    ^bb24:  // 2 preds: ^bb23, ^bb31
      aie.use_lock(%o_out_joined_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 0, 2) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%o_out_joined_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 2, 2) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%o_out_joined_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 4, 2) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%o_out_joined_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_0 : memref<8xbf16>, 6, 2) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%o_out_joined_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 0, 2) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%o_out_joined_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 2, 2) {bd_id = 15 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%o_out_joined_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 4, 2) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%o_out_joined_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%o_out_joined_0_buff_1 : memref<8xbf16>, 6, 2) {bd_id = 17 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%o_out_joined_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb24
    ^bb32:  // pred: ^bb23
      %8 = aie.dma_start(S2MM, 5, ^bb33, ^bb35)
    ^bb33:  // 2 preds: ^bb32, ^bb34
      aie.use_lock(%anm_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_in_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%anm_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb34
    ^bb34:  // pred: ^bb33
      aie.use_lock(%anm_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_in_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%anm_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb33
    ^bb35:  // pred: ^bb32
      %9 = aie.dma_start(MM2S, 3, ^bb36, ^bb38)
    ^bb36:  // 2 preds: ^bb35, ^bb37
      aie.use_lock(%anm_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_in_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.use_lock(%anm_in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb37
    ^bb37:  // pred: ^bb36
      aie.use_lock(%anm_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_in_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 33 : i32, next_bd_id = 32 : i32}
      aie.use_lock(%anm_in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb36
    ^bb38:  // pred: ^bb35
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
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
      aie.use_lock(%bo_mem_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bo_mem_0_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bo_mem_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @Ao_src_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_1_3 = aie.mem(%tile_1_3) {
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
      aie.use_lock(%bo_mem_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bo_mem_1_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bo_mem_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Ao_src_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Ao_src_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Ao_src_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Ao_src_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb8)
    ^bb4:  // 2 preds: ^bb3, ^bb7
      aie.use_lock(%Ao_src_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_0 : memref<4608xui8>, 0, 2304) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Ao_src_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Ao_src_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Ao_src_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%Ao_src_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_1 : memref<4608xui8>, 0, 2304) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Ao_src_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%Ao_src_1_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Ao_src_1_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb8:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
    ^bb9:  // 2 preds: ^bb8, ^bb10
      aie.use_lock(%Ao_src_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_0 : memref<4608xui8>, 2304, 2304) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Ao_src_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%Ao_src_1_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Ao_src_1_cons_buff_1 : memref<4608xui8>, 2304, 2304) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Ao_src_1_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb11:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 1, ^bb12, ^bb14)
    ^bb12:  // 2 preds: ^bb11, ^bb13
      aie.use_lock(%ffn_out_joined_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_0 : memref<4xbf16>, 0, 2) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%ffn_out_joined_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%ffn_out_joined_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_1 : memref<4xbf16>, 0, 2) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%ffn_out_joined_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb14:  // pred: ^bb11
      %4 = aie.dma_start(S2MM, 2, ^bb15, ^bb17)
    ^bb15:  // 2 preds: ^bb14, ^bb16
      aie.use_lock(%ffn_out_joined_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_0 : memref<4xbf16>, 2, 2) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%ffn_out_joined_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%ffn_out_joined_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_1 : memref<4xbf16>, 2, 2) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%ffn_out_joined_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb15
    ^bb17:  // pred: ^bb14
      %5 = aie.dma_start(MM2S, 2, ^bb18, ^bb22)
    ^bb18:  // 2 preds: ^bb17, ^bb21
      aie.use_lock(%ffn_out_joined_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_0 : memref<4xbf16>, 0, 2) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%ffn_out_joined_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%ffn_out_joined_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_0 : memref<4xbf16>, 2, 2) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%ffn_out_joined_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%ffn_out_joined_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_1 : memref<4xbf16>, 0, 2) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%ffn_out_joined_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%ffn_out_joined_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_0_buff_1 : memref<4xbf16>, 2, 2) {bd_id = 11 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%ffn_out_joined_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb18
    ^bb22:  // pred: ^bb17
      %6 = aie.dma_start(S2MM, 3, ^bb23, ^bb24)
    ^bb23:  // 2 preds: ^bb22, ^bb23
      aie.use_lock(%bq_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bq_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 28 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%bq_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb24:  // pred: ^bb22
      %7 = aie.dma_start(MM2S, 3, ^bb25, ^bb26)
    ^bb25:  // 2 preds: ^bb24, ^bb25
      aie.use_lock(%bq_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bq_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 29 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%bq_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb25
    ^bb26:  // pred: ^bb24
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
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
      aie.use_lock(%bo_mem_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bo_mem_2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bo_mem_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @Ao_src_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_3_3 = aie.mem(%tile_3_3) {
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
      aie.use_lock(%bo_mem_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bo_mem_3_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bo_mem_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Aqkv_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_0_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Aqkv_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Aqkv_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_0_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Aqkv_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cqkv_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_0_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cqkv_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cqkv_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_0_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cqkv_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%bq_L3L2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bq_L3L2_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bq_L3L2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb9, ^bb12)
    ^bb9:  // 2 preds: ^bb8, ^bb11
      aie.use_lock(%rms_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%rms_in_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%rms_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%rms_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%rms_in_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%rms_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%rms_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%rms_in_cons_buff_2 : memref<2048xbf16>, 0, 2048) {bd_id = 7 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%rms_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb12:  // pred: ^bb8
      aie.end
    }
    aie.shim_dma_allocation @Aqkv_src_0_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Aqkv_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_1_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Aqkv_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Aqkv_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_1_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Aqkv_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cqkv_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_1_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cqkv_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cqkv_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_1_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cqkv_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%bq_mem_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bq_mem_0_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bq_mem_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Aqkv_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_2_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Aqkv_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Aqkv_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_2_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Aqkv_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cqkv_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_2_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cqkv_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cqkv_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_2_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cqkv_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%bq_mem_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bq_mem_1_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bq_mem_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @Aqkv_src_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%Aqkv_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_3_cons_buff_0 : memref<2304xui8>, 0, 2304) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Aqkv_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Aqkv_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Aqkv_3_cons_buff_1 : memref<2304xui8>, 0, 2304) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Aqkv_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%Cqkv_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_3_buff_0 : memref<2xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Cqkv_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%Cqkv_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Cqkv_3_buff_1 : memref<2xbf16>, 0, 2) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Cqkv_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%bq_mem_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bq_mem_2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%bq_mem_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @ffn_out_joined_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%ffn_out_joined_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_0 : memref<4xbf16>, 0, 2) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%ffn_out_joined_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%ffn_out_joined_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_1 : memref<4xbf16>, 0, 2) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%ffn_out_joined_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%ffn_out_joined_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_0 : memref<4xbf16>, 2, 2) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%ffn_out_joined_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%ffn_out_joined_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_1 : memref<4xbf16>, 2, 2) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%ffn_out_joined_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb11)
    ^bb7:  // 2 preds: ^bb6, ^bb10
      aie.use_lock(%ffn_out_joined_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_0 : memref<4xbf16>, 0, 2) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%ffn_out_joined_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%ffn_out_joined_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_0 : memref<4xbf16>, 2, 2) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%ffn_out_joined_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%ffn_out_joined_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_1 : memref<4xbf16>, 0, 2) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%ffn_out_joined_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%ffn_out_joined_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffn_out_joined_1_buff_1 : memref<4xbf16>, 2, 2) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%ffn_out_joined_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb11:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @ffn_out_joined_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    %mem_4_4 = aie.mem(%tile_4_4) {
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
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%anm_mem_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_mem_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%anm_mem_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%anm_mem_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_mem_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%anm_mem_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%anm_mem_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%anm_mem_cons_buff_2 : memref<2048xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%anm_mem_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%ffi_L3L2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_L3L2_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%ffi_L3L2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    aie.shim_dma_allocation @Cqkv_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @Cqkv_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @Cqkv_2_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @Cqkv_3_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @anm_in_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @bo_L3L2_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%bo_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bo_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%bo_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%bo_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%bo_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%bo_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%ffi_L3L2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%ffi_L3L2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%ffi_L3L2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ffi_L3L2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%ffi_L3L2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @rms_in_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
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
  }
}
