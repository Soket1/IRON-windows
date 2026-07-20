module {
  aie.device(npu2) @op0_RMSNorm {
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
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %out2_7_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "out2_7_0_cons_prod_lock_0"}
    %out2_7_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "out2_7_0_cons_cons_lock_0"}
    %out2_7_0_buff_0 = aie.buffer(%tile_3_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_7_0_buff_0"} : memref<256xbf16> 
    %out2_7_0_buff_1 = aie.buffer(%tile_3_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_7_0_buff_1"} : memref<256xbf16> 
    %out2_7_0_prod_lock_0 = aie.lock(%tile_3_5, 4) {init = 2 : i32, sym_name = "out2_7_0_prod_lock_0"}
    %out2_7_0_cons_lock_0 = aie.lock(%tile_3_5, 5) {init = 0 : i32, sym_name = "out2_7_0_cons_lock_0"}
    %out2_6_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "out2_6_0_cons_prod_lock_0"}
    %out2_6_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "out2_6_0_cons_cons_lock_0"}
    %out2_6_0_buff_0 = aie.buffer(%tile_3_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_6_0_buff_0"} : memref<256xbf16> 
    %out2_6_0_buff_1 = aie.buffer(%tile_3_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_6_0_buff_1"} : memref<256xbf16> 
    %out2_6_0_prod_lock_0 = aie.lock(%tile_3_4, 4) {init = 2 : i32, sym_name = "out2_6_0_prod_lock_0"}
    %out2_6_0_cons_lock_0 = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "out2_6_0_cons_lock_0"}
    %out2_5_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 4) {init = 0 : i32, sym_name = "out2_5_0_cons_prod_lock_0"}
    %out2_5_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 5) {init = 0 : i32, sym_name = "out2_5_0_cons_cons_lock_0"}
    %out2_5_0_buff_0 = aie.buffer(%tile_3_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_5_0_buff_0"} : memref<256xbf16> 
    %out2_5_0_buff_1 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_5_0_buff_1"} : memref<256xbf16> 
    %out2_5_0_prod_lock_0 = aie.lock(%tile_3_3, 4) {init = 2 : i32, sym_name = "out2_5_0_prod_lock_0"}
    %out2_5_0_cons_lock_0 = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "out2_5_0_cons_lock_0"}
    %out2_4_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "out2_4_0_cons_prod_lock_0"}
    %out2_4_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "out2_4_0_cons_cons_lock_0"}
    %out2_4_0_buff_0 = aie.buffer(%tile_3_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_4_0_buff_0"} : memref<256xbf16> 
    %out2_4_0_buff_1 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_4_0_buff_1"} : memref<256xbf16> 
    %out2_4_0_prod_lock_0 = aie.lock(%tile_3_2, 4) {init = 2 : i32, sym_name = "out2_4_0_prod_lock_0"}
    %out2_4_0_cons_lock_0 = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "out2_4_0_cons_lock_0"}
    %out2_3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "out2_3_0_cons_prod_lock_0"}
    %out2_3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "out2_3_0_cons_cons_lock_0"}
    %out2_3_0_buff_0 = aie.buffer(%tile_2_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_3_0_buff_0"} : memref<256xbf16> 
    %out2_3_0_buff_1 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_3_0_buff_1"} : memref<256xbf16> 
    %out2_3_0_prod_lock_0 = aie.lock(%tile_2_5, 4) {init = 2 : i32, sym_name = "out2_3_0_prod_lock_0"}
    %out2_3_0_cons_lock_0 = aie.lock(%tile_2_5, 5) {init = 0 : i32, sym_name = "out2_3_0_cons_lock_0"}
    %out2_2_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "out2_2_0_cons_prod_lock_0"}
    %out2_2_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "out2_2_0_cons_cons_lock_0"}
    %out2_2_0_buff_0 = aie.buffer(%tile_2_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_2_0_buff_0"} : memref<256xbf16> 
    %out2_2_0_buff_1 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_2_0_buff_1"} : memref<256xbf16> 
    %out2_2_0_prod_lock_0 = aie.lock(%tile_2_4, 4) {init = 2 : i32, sym_name = "out2_2_0_prod_lock_0"}
    %out2_2_0_cons_lock_0 = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "out2_2_0_cons_lock_0"}
    %out2_1_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "out2_1_0_cons_prod_lock_0"}
    %out2_1_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "out2_1_0_cons_cons_lock_0"}
    %out2_1_0_buff_0 = aie.buffer(%tile_2_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_1_0_buff_0"} : memref<256xbf16> 
    %out2_1_0_buff_1 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_1_0_buff_1"} : memref<256xbf16> 
    %out2_1_0_prod_lock_0 = aie.lock(%tile_2_3, 4) {init = 2 : i32, sym_name = "out2_1_0_prod_lock_0"}
    %out2_1_0_cons_lock_0 = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "out2_1_0_cons_lock_0"}
    %out2_0_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "out2_0_0_cons_prod_lock_0"}
    %out2_0_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "out2_0_0_cons_cons_lock_0"}
    %out2_0_0_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_0_0_buff_0"} : memref<256xbf16> 
    %out2_0_0_buff_1 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_0_0_buff_1"} : memref<256xbf16> 
    %out2_0_0_prod_lock_0 = aie.lock(%tile_2_2, 4) {init = 2 : i32, sym_name = "out2_0_0_prod_lock_0"}
    %out2_0_0_cons_lock_0 = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "out2_0_0_cons_lock_0"}
    %out1_7_0_cons_buff_0 = aie.buffer(%tile_3_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_7_0_cons_buff_0"} : memref<256xbf16> 
    %out1_7_0_cons_buff_1 = aie.buffer(%tile_3_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_7_0_cons_buff_1"} : memref<256xbf16> 
    %out1_7_0_cons_prod_lock_0 = aie.lock(%tile_3_5, 2) {init = 2 : i32, sym_name = "out1_7_0_cons_prod_lock_0"}
    %out1_7_0_cons_cons_lock_0 = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "out1_7_0_cons_cons_lock_0"}
    %out1_7_0_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_7_0_buff_0"} : memref<256xbf16> 
    %out1_7_0_buff_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_7_0_buff_1"} : memref<256xbf16> 
    %out1_7_0_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "out1_7_0_prod_lock_0"}
    %out1_7_0_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "out1_7_0_cons_lock_0"}
    %out1_6_0_cons_buff_0 = aie.buffer(%tile_3_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_6_0_cons_buff_0"} : memref<256xbf16> 
    %out1_6_0_cons_buff_1 = aie.buffer(%tile_3_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_6_0_cons_buff_1"} : memref<256xbf16> 
    %out1_6_0_cons_prod_lock_0 = aie.lock(%tile_3_4, 2) {init = 2 : i32, sym_name = "out1_6_0_cons_prod_lock_0"}
    %out1_6_0_cons_cons_lock_0 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "out1_6_0_cons_cons_lock_0"}
    %out1_6_0_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_6_0_buff_0"} : memref<256xbf16> 
    %out1_6_0_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_6_0_buff_1"} : memref<256xbf16> 
    %out1_6_0_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "out1_6_0_prod_lock_0"}
    %out1_6_0_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "out1_6_0_cons_lock_0"}
    %out1_5_0_cons_buff_0 = aie.buffer(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_5_0_cons_buff_0"} : memref<256xbf16> 
    %out1_5_0_cons_buff_1 = aie.buffer(%tile_3_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_5_0_cons_buff_1"} : memref<256xbf16> 
    %out1_5_0_cons_prod_lock_0 = aie.lock(%tile_3_3, 2) {init = 2 : i32, sym_name = "out1_5_0_cons_prod_lock_0"}
    %out1_5_0_cons_cons_lock_0 = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "out1_5_0_cons_cons_lock_0"}
    %out1_5_0_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_5_0_buff_0"} : memref<256xbf16> 
    %out1_5_0_buff_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_5_0_buff_1"} : memref<256xbf16> 
    %out1_5_0_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "out1_5_0_prod_lock_0"}
    %out1_5_0_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "out1_5_0_cons_lock_0"}
    %out1_4_0_cons_buff_0 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_4_0_cons_buff_0"} : memref<256xbf16> 
    %out1_4_0_cons_buff_1 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_4_0_cons_buff_1"} : memref<256xbf16> 
    %out1_4_0_cons_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "out1_4_0_cons_prod_lock_0"}
    %out1_4_0_cons_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "out1_4_0_cons_cons_lock_0"}
    %out1_4_0_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_4_0_buff_0"} : memref<256xbf16> 
    %out1_4_0_buff_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_4_0_buff_1"} : memref<256xbf16> 
    %out1_4_0_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "out1_4_0_prod_lock_0"}
    %out1_4_0_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "out1_4_0_cons_lock_0"}
    %out1_3_0_cons_buff_0 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_3_0_cons_buff_0"} : memref<256xbf16> 
    %out1_3_0_cons_buff_1 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_3_0_cons_buff_1"} : memref<256xbf16> 
    %out1_3_0_cons_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 2 : i32, sym_name = "out1_3_0_cons_prod_lock_0"}
    %out1_3_0_cons_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "out1_3_0_cons_cons_lock_0"}
    %out1_3_0_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_3_0_buff_0"} : memref<256xbf16> 
    %out1_3_0_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_3_0_buff_1"} : memref<256xbf16> 
    %out1_3_0_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "out1_3_0_prod_lock_0"}
    %out1_3_0_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "out1_3_0_cons_lock_0"}
    %out1_2_0_cons_buff_0 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_2_0_cons_buff_0"} : memref<256xbf16> 
    %out1_2_0_cons_buff_1 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_2_0_cons_buff_1"} : memref<256xbf16> 
    %out1_2_0_cons_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 2 : i32, sym_name = "out1_2_0_cons_prod_lock_0"}
    %out1_2_0_cons_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "out1_2_0_cons_cons_lock_0"}
    %out1_2_0_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_2_0_buff_0"} : memref<256xbf16> 
    %out1_2_0_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_2_0_buff_1"} : memref<256xbf16> 
    %out1_2_0_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "out1_2_0_prod_lock_0"}
    %out1_2_0_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "out1_2_0_cons_lock_0"}
    %out1_1_0_cons_buff_0 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_1_0_cons_buff_0"} : memref<256xbf16> 
    %out1_1_0_cons_buff_1 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_1_0_cons_buff_1"} : memref<256xbf16> 
    %out1_1_0_cons_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "out1_1_0_cons_prod_lock_0"}
    %out1_1_0_cons_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "out1_1_0_cons_cons_lock_0"}
    %out1_1_0_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_1_0_buff_0"} : memref<256xbf16> 
    %out1_1_0_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_1_0_buff_1"} : memref<256xbf16> 
    %out1_1_0_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "out1_1_0_prod_lock_0"}
    %out1_1_0_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "out1_1_0_cons_lock_0"}
    %out1_0_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out1_0_0_cons_buff_0"} : memref<256xbf16> 
    %out1_0_0_cons_buff_1 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "out1_0_0_cons_buff_1"} : memref<256xbf16> 
    %out1_0_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "out1_0_0_cons_prod_lock_0"}
    %out1_0_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "out1_0_0_cons_cons_lock_0"}
    %out1_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_0_0_buff_0"} : memref<256xbf16> 
    %out1_0_0_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_0_0_buff_1"} : memref<256xbf16> 
    %out1_0_0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out1_0_0_prod_lock_0"}
    %out1_0_0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out1_0_0_cons_lock_0"}
    %in2_weights_0_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_0_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_0_cons_buff_1 = aie.buffer(%tile_2_2) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_0_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "in2_weights_0_0_cons_prod_lock_0"}
    %in2_weights_0_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "in2_weights_0_0_cons_cons_lock_0"}
    %in2_weights_0_1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_1_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_1_cons_buff_1 = aie.buffer(%tile_2_3) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_1_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_1_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "in2_weights_0_1_cons_prod_lock_0"}
    %in2_weights_0_1_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "in2_weights_0_1_cons_cons_lock_0"}
    %in2_weights_0_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_2_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_2_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "in2_weights_0_2_cons_prod_lock_0"}
    %in2_weights_0_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "in2_weights_0_2_cons_cons_lock_0"}
    %in2_weights_0_3_cons_buff_0 = aie.buffer(%tile_2_5) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_3_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_3_cons_buff_1 = aie.buffer(%tile_2_5) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_3_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_3_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 2 : i32, sym_name = "in2_weights_0_3_cons_prod_lock_0"}
    %in2_weights_0_3_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "in2_weights_0_3_cons_cons_lock_0"}
    %in2_weights_0_4_cons_buff_0 = aie.buffer(%tile_3_2) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_4_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_4_cons_buff_1 = aie.buffer(%tile_3_2) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_4_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_4_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "in2_weights_0_4_cons_prod_lock_0"}
    %in2_weights_0_4_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "in2_weights_0_4_cons_cons_lock_0"}
    %in2_weights_0_5_cons_buff_0 = aie.buffer(%tile_3_3) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_5_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_5_cons_buff_1 = aie.buffer(%tile_3_3) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_5_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_5_cons_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "in2_weights_0_5_cons_prod_lock_0"}
    %in2_weights_0_5_cons_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "in2_weights_0_5_cons_cons_lock_0"}
    %in2_weights_0_6_cons_buff_0 = aie.buffer(%tile_3_4) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_6_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_6_cons_buff_1 = aie.buffer(%tile_3_4) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_6_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_6_cons_prod_lock_0 = aie.lock(%tile_3_4, 0) {init = 2 : i32, sym_name = "in2_weights_0_6_cons_prod_lock_0"}
    %in2_weights_0_6_cons_cons_lock_0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "in2_weights_0_6_cons_cons_lock_0"}
    %in2_weights_0_7_cons_buff_0 = aie.buffer(%tile_3_5) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in2_weights_0_7_cons_buff_0"} : memref<256xbf16> 
    %in2_weights_0_7_cons_buff_1 = aie.buffer(%tile_3_5) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in2_weights_0_7_cons_buff_1"} : memref<256xbf16> 
    %in2_weights_0_7_cons_prod_lock_0 = aie.lock(%tile_3_5, 0) {init = 2 : i32, sym_name = "in2_weights_0_7_cons_prod_lock_0"}
    %in2_weights_0_7_cons_cons_lock_0 = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "in2_weights_0_7_cons_cons_lock_0"}
    %in2_weights_0_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "in2_weights_0_prod_lock_0"}
    %in2_weights_0_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "in2_weights_0_cons_lock_0"}
    %in1_7_0_cons_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_7_0_cons_buff_0"} : memref<256xbf16> 
    %in1_7_0_cons_buff_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_7_0_cons_buff_1"} : memref<256xbf16> 
    %in1_7_0_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "in1_7_0_cons_prod_lock_0"}
    %in1_7_0_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "in1_7_0_cons_cons_lock_0"}
    %in1_7_0_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "in1_7_0_prod_lock_0"}
    %in1_7_0_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "in1_7_0_cons_lock_0"}
    %in1_6_0_cons_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_6_0_cons_buff_0"} : memref<256xbf16> 
    %in1_6_0_cons_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_6_0_cons_buff_1"} : memref<256xbf16> 
    %in1_6_0_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "in1_6_0_cons_prod_lock_0"}
    %in1_6_0_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "in1_6_0_cons_cons_lock_0"}
    %in1_6_0_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "in1_6_0_prod_lock_0"}
    %in1_6_0_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "in1_6_0_cons_lock_0"}
    %in1_5_0_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_5_0_cons_buff_0"} : memref<256xbf16> 
    %in1_5_0_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_5_0_cons_buff_1"} : memref<256xbf16> 
    %in1_5_0_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "in1_5_0_cons_prod_lock_0"}
    %in1_5_0_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "in1_5_0_cons_cons_lock_0"}
    %in1_5_0_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "in1_5_0_prod_lock_0"}
    %in1_5_0_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "in1_5_0_cons_lock_0"}
    %in1_4_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_4_0_cons_buff_0"} : memref<256xbf16> 
    %in1_4_0_cons_buff_1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_4_0_cons_buff_1"} : memref<256xbf16> 
    %in1_4_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "in1_4_0_cons_prod_lock_0"}
    %in1_4_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "in1_4_0_cons_cons_lock_0"}
    %in1_4_0_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "in1_4_0_prod_lock_0"}
    %in1_4_0_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "in1_4_0_cons_lock_0"}
    %in1_3_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_3_0_cons_buff_0"} : memref<256xbf16> 
    %in1_3_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_3_0_cons_buff_1"} : memref<256xbf16> 
    %in1_3_0_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "in1_3_0_cons_prod_lock_0"}
    %in1_3_0_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "in1_3_0_cons_cons_lock_0"}
    %in1_3_0_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "in1_3_0_prod_lock_0"}
    %in1_3_0_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "in1_3_0_cons_lock_0"}
    %in1_2_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_2_0_cons_buff_0"} : memref<256xbf16> 
    %in1_2_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_2_0_cons_buff_1"} : memref<256xbf16> 
    %in1_2_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "in1_2_0_cons_prod_lock_0"}
    %in1_2_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "in1_2_0_cons_cons_lock_0"}
    %in1_2_0_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "in1_2_0_prod_lock_0"}
    %in1_2_0_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "in1_2_0_cons_lock_0"}
    %in1_1_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_1_0_cons_buff_0"} : memref<256xbf16> 
    %in1_1_0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_1_0_cons_buff_1"} : memref<256xbf16> 
    %in1_1_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "in1_1_0_cons_prod_lock_0"}
    %in1_1_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in1_1_0_cons_cons_lock_0"}
    %in1_1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "in1_1_0_prod_lock_0"}
    %in1_1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "in1_1_0_cons_lock_0"}
    %in1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_0_0_cons_buff_0"} : memref<256xbf16> 
    %in1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_0_0_cons_buff_1"} : memref<256xbf16> 
    %in1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in1_0_0_cons_prod_lock_0"}
    %in1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in1_0_0_cons_cons_lock_0"}
    %in1_0_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in1_0_0_prod_lock_0"}
    %in1_0_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in1_0_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_3_5, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_3_3, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_2_2, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_2_3, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%tile_0_5, DMA : 0, %tile_2_5, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %tile_3_2, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %tile_3_3, DMA : 1)
    aie.flow(%tile_1_4, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%tile_1_5, DMA : 0, %tile_3_5, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_2_3, DMA : 0, %shim_noc_tile_2_0, DMA : 1)
    aie.flow(%tile_2_4, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_2_5, DMA : 0, %shim_noc_tile_3_0, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_3_3, DMA : 0, %shim_noc_tile_4_0, DMA : 1)
    aie.flow(%tile_3_4, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%tile_3_5, DMA : 0, %shim_noc_tile_5_0, DMA : 1)
    func.func private @op0_rms_norm_bf16_vector(memref<256xbf16>, memref<256xbf16>, i32) attributes {link_with = "op0_rms_norm.o"}
    func.func private @op0_eltwise_mul_bf16_vector(memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) attributes {link_with = "op0_mul.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_0_0_cons_buff_0, %out1_0_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_0_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_0_0_cons_buff_1, %out1_0_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_0_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_0_0_cons_buff_0, %out1_0_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_0_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_0_3 = aie.core(%tile_0_3) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_1_0_cons_buff_0, %out1_1_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_1_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_1_0_cons_buff_1, %out1_1_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_1_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_1_0_cons_buff_0, %out1_1_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_1_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_0_4 = aie.core(%tile_0_4) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_2_0_cons_buff_0, %out1_2_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_2_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_2_0_cons_buff_1, %out1_2_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_2_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_2_0_cons_buff_0, %out1_2_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_2_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_0_5 = aie.core(%tile_0_5) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_3_0_cons_buff_0, %out1_3_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_3_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_3_0_cons_buff_1, %out1_3_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_3_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_3_0_cons_buff_0, %out1_3_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_3_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_1_2 = aie.core(%tile_1_2) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_4_0_cons_buff_0, %out1_4_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_4_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_4_0_cons_buff_1, %out1_4_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_4_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_4_0_cons_buff_0, %out1_4_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_4_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_1_3 = aie.core(%tile_1_3) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_5_0_cons_buff_0, %out1_5_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_5_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_5_0_cons_buff_1, %out1_5_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_5_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_5_0_cons_buff_0, %out1_5_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_5_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_1_4 = aie.core(%tile_1_4) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_6_0_cons_buff_0, %out1_6_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_6_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_6_0_cons_buff_1, %out1_6_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_6_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_6_0_cons_buff_0, %out1_6_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_6_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_1_5 = aie.core(%tile_1_5) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_7_0_cons_buff_0, %out1_7_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_7_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_7_0_cons_buff_1, %out1_7_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_7_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_rms_norm_bf16_vector(%in1_7_0_cons_buff_0, %out1_7_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_7_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_rms_norm.o"]}
    %core_2_2 = aie.core(%tile_2_2) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_0_0_cons_buff_0, %in2_weights_0_0_cons_buff_0, %out2_0_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_0_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_0_0_cons_buff_1, %in2_weights_0_0_cons_buff_1, %out2_0_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_0_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_0_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_0_0_cons_buff_0, %in2_weights_0_0_cons_buff_0, %out2_0_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_0_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_0_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    %core_2_3 = aie.core(%tile_2_3) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_1_0_cons_buff_0, %in2_weights_0_1_cons_buff_0, %out2_1_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_1_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_1_0_cons_buff_1, %in2_weights_0_1_cons_buff_1, %out2_1_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_1_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_1_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_1_0_cons_buff_0, %in2_weights_0_1_cons_buff_0, %out2_1_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_1_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_1_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    %core_2_4 = aie.core(%tile_2_4) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_2_0_cons_buff_0, %in2_weights_0_2_cons_buff_0, %out2_2_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_2_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_2_0_cons_buff_1, %in2_weights_0_2_cons_buff_1, %out2_2_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_2_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_2_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_2_0_cons_buff_0, %in2_weights_0_2_cons_buff_0, %out2_2_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_2_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_2_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    %core_2_5 = aie.core(%tile_2_5) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_3_0_cons_buff_0, %in2_weights_0_3_cons_buff_0, %out2_3_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_3_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_3_0_cons_buff_1, %in2_weights_0_3_cons_buff_1, %out2_3_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_3_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_3_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_3_0_cons_buff_0, %in2_weights_0_3_cons_buff_0, %out2_3_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_3_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_3_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    %core_3_2 = aie.core(%tile_3_2) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_4_0_cons_buff_0, %in2_weights_0_4_cons_buff_0, %out2_4_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_4_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_4_0_cons_buff_1, %in2_weights_0_4_cons_buff_1, %out2_4_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_4_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_4_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_4_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_4_0_cons_buff_0, %in2_weights_0_4_cons_buff_0, %out2_4_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_4_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_4_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    %core_3_3 = aie.core(%tile_3_3) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_5_0_cons_buff_0, %in2_weights_0_5_cons_buff_0, %out2_5_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_5_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_5_0_cons_buff_1, %in2_weights_0_5_cons_buff_1, %out2_5_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_5_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_5_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_5_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_5_0_cons_buff_0, %in2_weights_0_5_cons_buff_0, %out2_5_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_5_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_5_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    %core_3_4 = aie.core(%tile_3_4) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_6_0_cons_buff_0, %in2_weights_0_6_cons_buff_0, %out2_6_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_6_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_6_0_cons_buff_1, %in2_weights_0_6_cons_buff_1, %out2_6_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_6_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_6_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_6_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_6_0_cons_buff_0, %in2_weights_0_6_cons_buff_0, %out2_6_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_6_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_6_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    %core_3_5 = aie.core(%tile_3_5) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_7_0_cons_buff_0, %in2_weights_0_7_cons_buff_0, %out2_7_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_7_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_7_0_cons_buff_1, %in2_weights_0_7_cons_buff_1, %out2_7_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_7_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_7_cons_prod_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_7_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_eltwise_mul_bf16_vector(%out1_7_0_cons_buff_0, %in2_weights_0_7_cons_buff_0, %out2_7_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%out1_7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_7_0_cons_lock_0, Release, 1)
      aie.use_lock(%in2_weights_0_7_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_mul.o"]}
    aie.runtime_sequence(%arg0: memref<2048xbf16>, %arg1: memref<256xbf16>, %arg2: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @in1_0_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in1_1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @in1_2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @in1_3_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @in1_4_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @in1_5_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @in1_6_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @in1_7_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @in2_weights_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<256xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @out2_0_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @out2_1_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @out2_2_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @out2_3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @out2_4_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @out2_5_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @out2_6_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @out2_7_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%16)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
    }
    aie.shim_dma_allocation @in1_0_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_0_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_0_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_1_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_1_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_2_0_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_2_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_2_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_3_0_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_3_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_3_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_4_0_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_4_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_4_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_5_0_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_5_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_5_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_6_0_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_6_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_6_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_7_0_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_7_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_7_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in2_weights_0_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_0_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_0_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_0_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_0_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_1_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_1_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_1_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_1_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_1_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_1_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_2_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_2_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_2_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_2_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_2_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_2_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_3_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_3_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_3_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_3_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_3_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_3_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_4_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_4_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_4_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_4_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_4_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_4_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_5_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_5_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_5_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_5_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_5_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_5_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_6_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_6_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_6_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_6_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_6_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_6_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_weights_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_7_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_weights_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_weights_0_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_weights_0_7_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_weights_0_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_7_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_7_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out2_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_7_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out2_7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out2_7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_7_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out2_7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @out2_0_0_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @out2_1_0_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @out2_2_0_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @out2_3_0_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
    aie.shim_dma_allocation @out2_4_0_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @out2_5_0_shim_alloc(%shim_noc_tile_4_0, S2MM, 1)
    aie.shim_dma_allocation @out2_6_0_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @out2_7_0_shim_alloc(%shim_noc_tile_5_0, S2MM, 1)
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
  }
  aie.device(npu2) @op1_ElementwiseMul {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %out_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "out_7_cons_prod_lock_0"}
    %out_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "out_7_cons_cons_lock_0"}
    %out_7_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_7_buff_0"} : memref<256xbf16> 
    %out_7_buff_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_7_buff_1"} : memref<256xbf16> 
    %out_7_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 2 : i32, sym_name = "out_7_prod_lock_0"}
    %out_7_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "out_7_cons_lock_0"}
    %out_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "out_6_cons_prod_lock_0"}
    %out_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "out_6_cons_cons_lock_0"}
    %out_6_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_6_buff_0"} : memref<256xbf16> 
    %out_6_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_6_buff_1"} : memref<256xbf16> 
    %out_6_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 2 : i32, sym_name = "out_6_prod_lock_0"}
    %out_6_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "out_6_cons_lock_0"}
    %out_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "out_5_cons_prod_lock_0"}
    %out_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "out_5_cons_cons_lock_0"}
    %out_5_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_5_buff_0"} : memref<256xbf16> 
    %out_5_buff_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_5_buff_1"} : memref<256xbf16> 
    %out_5_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 2 : i32, sym_name = "out_5_prod_lock_0"}
    %out_5_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "out_5_cons_lock_0"}
    %out_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "out_4_cons_prod_lock_0"}
    %out_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "out_4_cons_cons_lock_0"}
    %out_4_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_4_buff_0"} : memref<256xbf16> 
    %out_4_buff_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_4_buff_1"} : memref<256xbf16> 
    %out_4_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "out_4_prod_lock_0"}
    %out_4_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "out_4_cons_lock_0"}
    %out_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "out_3_cons_prod_lock_0"}
    %out_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "out_3_cons_cons_lock_0"}
    %out_3_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_3_buff_0"} : memref<256xbf16> 
    %out_3_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_3_buff_1"} : memref<256xbf16> 
    %out_3_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "out_3_prod_lock_0"}
    %out_3_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "out_3_cons_lock_0"}
    %out_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "out_2_cons_prod_lock_0"}
    %out_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "out_2_cons_cons_lock_0"}
    %out_2_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_2_buff_0"} : memref<256xbf16> 
    %out_2_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_2_buff_1"} : memref<256xbf16> 
    %out_2_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "out_2_prod_lock_0"}
    %out_2_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "out_2_cons_lock_0"}
    %out_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "out_1_cons_prod_lock_0"}
    %out_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "out_1_cons_cons_lock_0"}
    %out_1_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_1_buff_0"} : memref<256xbf16> 
    %out_1_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_1_buff_1"} : memref<256xbf16> 
    %out_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "out_1_prod_lock_0"}
    %out_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "out_1_cons_lock_0"}
    %out_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "out_0_cons_prod_lock_0"}
    %out_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "out_0_cons_cons_lock_0"}
    %out_0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_0_buff_0"} : memref<256xbf16> 
    %out_0_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_0_buff_1"} : memref<256xbf16> 
    %out_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "out_0_prod_lock_0"}
    %out_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "out_0_cons_lock_0"}
    %in2_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_7_cons_buff_0"} : memref<256xbf16> 
    %in2_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_7_cons_buff_1"} : memref<256xbf16> 
    %in2_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "in2_7_cons_prod_lock_0"}
    %in2_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "in2_7_cons_cons_lock_0"}
    %in2_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "in2_7_prod_lock_0"}
    %in2_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "in2_7_cons_lock_0"}
    %in2_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_6_cons_buff_0"} : memref<256xbf16> 
    %in2_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_6_cons_buff_1"} : memref<256xbf16> 
    %in2_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "in2_6_cons_prod_lock_0"}
    %in2_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "in2_6_cons_cons_lock_0"}
    %in2_6_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "in2_6_prod_lock_0"}
    %in2_6_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "in2_6_cons_lock_0"}
    %in2_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_5_cons_buff_0"} : memref<256xbf16> 
    %in2_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_5_cons_buff_1"} : memref<256xbf16> 
    %in2_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "in2_5_cons_prod_lock_0"}
    %in2_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "in2_5_cons_cons_lock_0"}
    %in2_5_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "in2_5_prod_lock_0"}
    %in2_5_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "in2_5_cons_lock_0"}
    %in2_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_4_cons_buff_0"} : memref<256xbf16> 
    %in2_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_4_cons_buff_1"} : memref<256xbf16> 
    %in2_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "in2_4_cons_prod_lock_0"}
    %in2_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "in2_4_cons_cons_lock_0"}
    %in2_4_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "in2_4_prod_lock_0"}
    %in2_4_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "in2_4_cons_lock_0"}
    %in2_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_3_cons_buff_0"} : memref<256xbf16> 
    %in2_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_3_cons_buff_1"} : memref<256xbf16> 
    %in2_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "in2_3_cons_prod_lock_0"}
    %in2_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "in2_3_cons_cons_lock_0"}
    %in2_3_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "in2_3_prod_lock_0"}
    %in2_3_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "in2_3_cons_lock_0"}
    %in2_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_2_cons_buff_0"} : memref<256xbf16> 
    %in2_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_2_cons_buff_1"} : memref<256xbf16> 
    %in2_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "in2_2_cons_prod_lock_0"}
    %in2_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "in2_2_cons_cons_lock_0"}
    %in2_2_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "in2_2_prod_lock_0"}
    %in2_2_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "in2_2_cons_lock_0"}
    %in2_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_1_cons_buff_0"} : memref<256xbf16> 
    %in2_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_1_cons_buff_1"} : memref<256xbf16> 
    %in2_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "in2_1_cons_prod_lock_0"}
    %in2_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "in2_1_cons_cons_lock_0"}
    %in2_1_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "in2_1_prod_lock_0"}
    %in2_1_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "in2_1_cons_lock_0"}
    %in2_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_0_cons_buff_0"} : memref<256xbf16> 
    %in2_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_0_cons_buff_1"} : memref<256xbf16> 
    %in2_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "in2_0_cons_prod_lock_0"}
    %in2_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "in2_0_cons_cons_lock_0"}
    %in2_0_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "in2_0_prod_lock_0"}
    %in2_0_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "in2_0_cons_lock_0"}
    %in1_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_7_cons_buff_0"} : memref<256xbf16> 
    %in1_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_7_cons_buff_1"} : memref<256xbf16> 
    %in1_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "in1_7_cons_prod_lock_0"}
    %in1_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "in1_7_cons_cons_lock_0"}
    %in1_7_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "in1_7_prod_lock_0"}
    %in1_7_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "in1_7_cons_lock_0"}
    %in1_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_6_cons_buff_0"} : memref<256xbf16> 
    %in1_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_6_cons_buff_1"} : memref<256xbf16> 
    %in1_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "in1_6_cons_prod_lock_0"}
    %in1_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "in1_6_cons_cons_lock_0"}
    %in1_6_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "in1_6_prod_lock_0"}
    %in1_6_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "in1_6_cons_lock_0"}
    %in1_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_5_cons_buff_0"} : memref<256xbf16> 
    %in1_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_5_cons_buff_1"} : memref<256xbf16> 
    %in1_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "in1_5_cons_prod_lock_0"}
    %in1_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "in1_5_cons_cons_lock_0"}
    %in1_5_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "in1_5_prod_lock_0"}
    %in1_5_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "in1_5_cons_lock_0"}
    %in1_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_4_cons_buff_0"} : memref<256xbf16> 
    %in1_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_4_cons_buff_1"} : memref<256xbf16> 
    %in1_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "in1_4_cons_prod_lock_0"}
    %in1_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "in1_4_cons_cons_lock_0"}
    %in1_4_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "in1_4_prod_lock_0"}
    %in1_4_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "in1_4_cons_lock_0"}
    %in1_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_3_cons_buff_0"} : memref<256xbf16> 
    %in1_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_3_cons_buff_1"} : memref<256xbf16> 
    %in1_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "in1_3_cons_prod_lock_0"}
    %in1_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "in1_3_cons_cons_lock_0"}
    %in1_3_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "in1_3_prod_lock_0"}
    %in1_3_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "in1_3_cons_lock_0"}
    %in1_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_2_cons_buff_0"} : memref<256xbf16> 
    %in1_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_2_cons_buff_1"} : memref<256xbf16> 
    %in1_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "in1_2_cons_prod_lock_0"}
    %in1_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "in1_2_cons_cons_lock_0"}
    %in1_2_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "in1_2_prod_lock_0"}
    %in1_2_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "in1_2_cons_lock_0"}
    %in1_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_1_cons_buff_0"} : memref<256xbf16> 
    %in1_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_1_cons_buff_1"} : memref<256xbf16> 
    %in1_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "in1_1_cons_prod_lock_0"}
    %in1_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in1_1_cons_cons_lock_0"}
    %in1_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "in1_1_prod_lock_0"}
    %in1_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "in1_1_cons_lock_0"}
    %in1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1536 : i32, mem_bank = 0 : i32, sym_name = "in1_0_cons_buff_0"} : memref<256xbf16> 
    %in1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16896 : i32, mem_bank = 1 : i32, sym_name = "in1_0_cons_buff_1"} : memref<256xbf16> 
    %in1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in1_0_cons_prod_lock_0"}
    %in1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in1_0_cons_cons_lock_0"}
    %in1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in1_0_prod_lock_0"}
    %in1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in1_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %tile_1_5, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %tile_0_4, DMA : 1)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %tile_1_3, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%shim_noc_tile_7_0, DMA : 1, %tile_1_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %shim_noc_tile_0_0, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %shim_noc_tile_1_0, DMA : 1)
    aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_1_3, DMA : 0, %shim_noc_tile_2_0, DMA : 1)
    aie.flow(%tile_1_4, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_1_5, DMA : 0, %shim_noc_tile_3_0, DMA : 1)
    func.func private @op1_eltwise_mul_bf16_vector(memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) attributes {link_with = "op1_mul.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_0_cons_buff_0, %in2_0_cons_buff_0, %out_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_0_cons_buff_1, %in2_0_cons_buff_1, %out_0_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_0_cons_buff_0, %in2_0_cons_buff_0, %out_0_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_0_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    %core_0_3 = aie.core(%tile_0_3) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_1_cons_buff_0, %in2_1_cons_buff_0, %out_1_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_1_cons_lock_0, Release, 1)
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_1_cons_buff_1, %in2_1_cons_buff_1, %out_1_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_1_cons_buff_0, %in2_1_cons_buff_0, %out_1_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_1_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    %core_0_4 = aie.core(%tile_0_4) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_2_cons_buff_0, %in2_2_cons_buff_0, %out_2_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_2_cons_lock_0, Release, 1)
      aie.use_lock(%in1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_2_cons_buff_1, %in2_2_cons_buff_1, %out_2_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_2_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_2_cons_buff_0, %in2_2_cons_buff_0, %out_2_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_2_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    %core_0_5 = aie.core(%tile_0_5) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_3_cons_buff_0, %in2_3_cons_buff_0, %out_3_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_3_cons_lock_0, Release, 1)
      aie.use_lock(%in1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_3_cons_buff_1, %in2_3_cons_buff_1, %out_3_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_3_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_3_cons_buff_0, %in2_3_cons_buff_0, %out_3_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_3_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    %core_1_2 = aie.core(%tile_1_2) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_4_cons_buff_0, %in2_4_cons_buff_0, %out_4_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_4_cons_lock_0, Release, 1)
      aie.use_lock(%in1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_4_cons_buff_1, %in2_4_cons_buff_1, %out_4_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_4_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_4_cons_buff_0, %in2_4_cons_buff_0, %out_4_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_4_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    %core_1_3 = aie.core(%tile_1_3) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_5_cons_buff_0, %in2_5_cons_buff_0, %out_5_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_5_cons_lock_0, Release, 1)
      aie.use_lock(%in1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_5_cons_buff_1, %in2_5_cons_buff_1, %out_5_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_5_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_5_cons_buff_0, %in2_5_cons_buff_0, %out_5_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_5_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    %core_1_4 = aie.core(%tile_1_4) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_6_cons_buff_0, %in2_6_cons_buff_0, %out_6_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_6_cons_lock_0, Release, 1)
      aie.use_lock(%in1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_6_cons_buff_1, %in2_6_cons_buff_1, %out_6_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_6_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_6_cons_buff_0, %in2_6_cons_buff_0, %out_6_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_6_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    %core_1_5 = aie.core(%tile_1_5) {
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_7_cons_buff_0, %in2_7_cons_buff_0, %out_7_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_7_cons_lock_0, Release, 1)
      aie.use_lock(%in1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_7_cons_buff_1, %in2_7_cons_buff_1, %out_7_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_7_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_eltwise_mul_bf16_vector(%in1_7_cons_buff_0, %in2_7_cons_buff_0, %out_7_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_7_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_mul.o"]}
    aie.runtime_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @in1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @in1_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @in2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @in1_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @in2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @in1_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @in2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @in1_4_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @in2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @in1_5_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @in2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @in1_6_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @in2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @in1_7_shim_alloc {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @in2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @out_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @out_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @out_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @out_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @out_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @out_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @out_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @out_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
    }
    aie.shim_dma_allocation @in1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_0_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_1_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_1_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_2_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_2_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_2_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_2_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_2_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_3_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_3_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_3_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_3_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_3_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_4_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_4_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_4_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_4_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_4_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_5_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_5_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_5_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_5_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_5_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_6_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_6_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_6_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_6_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_6_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_7_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_7_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_7_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_7_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_7_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in2_0_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @in2_1_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @in2_2_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @in2_3_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @in2_4_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @in2_5_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @in2_6_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @in2_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 1)
    aie.shim_dma_allocation @out_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out_1_shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @out_2_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @out_3_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @out_4_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @out_5_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @out_6_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @out_7_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
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
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2304xbf16>) {
      aiex.configure @op0_RMSNorm {
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [256], strides: [1] : memref<2304xbf16> to memref<256xbf16>
        aiex.run @sequence(%arg0, %reinterpret_cast, %arg0) : (memref<2048xbf16>, memref<256xbf16>, memref<2048xbf16>)
      }
      aiex.configure @op1_ElementwiseMul {
        %subview = memref.subview %arg2[256] [2048] [1] : memref<2304xbf16> to memref<2048xbf16, strided<[1], offset: 256>>
        %reinterpret_cast = memref.reinterpret_cast %subview to offset: [0], sizes: [2048], strides: [1] : memref<2048xbf16, strided<[1], offset: 256>> to memref<2048xbf16>
        aiex.run @sequence(%arg0, %reinterpret_cast, %arg1) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>)
      }
    }
  }
}
