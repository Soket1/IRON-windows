module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %out_fifo_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "out_fifo_cons_prod_lock_0"}
    %out_fifo_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "out_fifo_cons_cons_lock_0"}
    %out_fifo_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_fifo_buff_0"} : memref<256xbf16> 
    %out_fifo_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_fifo_buff_1"} : memref<256xbf16> 
    %out_fifo_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out_fifo_prod_lock_0"}
    %out_fifo_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_fifo_cons_lock_0"}
    %in_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in_fifo_cons_buff_0"} : memref<256xbf16> 
    %in_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in_fifo_cons_buff_1"} : memref<256xbf16> 
    %in_fifo_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in_fifo_cons_prod_lock_0"}
    %in_fifo_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_fifo_cons_cons_lock_0"}
    %in_fifo_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in_fifo_prod_lock_0"}
    %in_fifo_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in_fifo_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    func.func private @echo_copy_bf16(memref<256xbf16>, memref<256xbf16>, i32) attributes {link_with = "echo.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c4294967294 = arith.constant 4294967294 : index
      %c256_i32 = arith.constant 256 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c4294967294 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_0, %out_fifo_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_1, %out_fifo_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_0, %out_fifo_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c4294967294 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_1, %out_fifo_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_0, %out_fifo_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      %7 = arith.addi %5, %c2 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_1, %out_fifo_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      %8 = arith.addi %0, %c2 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      cf.br ^bb10(%c0 : index)
    ^bb10(%9: index):  // 2 preds: ^bb9, ^bb11
      %10 = arith.cmpi slt, %9, %c4294967294 : index
      cf.cond_br %10, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_0, %out_fifo_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_1, %out_fifo_buff_1, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      %11 = arith.addi %9, %c2 : index
      cf.br ^bb10(%11 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%in_fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_fifo_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @echo_copy_bf16(%in_fifo_cons_buff_0, %out_fifo_buff_0, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
      aie.use_lock(%in_fifo_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_fifo_cons_lock_0, Release, 1)
      aie.end
    } {link_files = ["echo.o"]}
    aie.runtime_sequence(%arg0: memref<256xbf16>, %arg1: memref<256xbf16>) {
      %0 = aiex.dma_configure_task_for @in_fifo_shim_alloc {
        aie.dma_bd(%arg0 : memref<256xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out_fifo_shim_alloc {
        aie.dma_bd(%arg1 : memref<256xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
    aie.shim_dma_allocation @in_fifo_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_fifo_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_fifo_cons_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in_fifo_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in_fifo_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_fifo_cons_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_fifo_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out_fifo_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_fifo_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out_fifo_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out_fifo_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_fifo_buff_1 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out_fifo_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @out_fifo_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
