module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in_fifo(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<256xbf16>> 
    aie.objectfifo @out_fifo(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>> 
    func.func private @echo_copy_bf16(memref<256xbf16>, memref<256xbf16>, i32) attributes {link_with = "echo.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_fifo(Consume, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          %2 = aie.objectfifo.acquire @out_fifo(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          %c256_i32 = arith.constant 256 : i32
          func.call @echo_copy_bf16(%1, %3, %c256_i32) : (memref<256xbf16>, memref<256xbf16>, i32) -> ()
          aie.objectfifo.release @in_fifo(Consume, 1)
          aie.objectfifo.release @out_fifo(Produce, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<256xbf16>, %arg1: memref<256xbf16>) {
      %0 = aiex.dma_configure_task_for @in_fifo {
        aie.dma_bd(%arg0 : memref<256xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out_fifo {
        aie.dma_bd(%arg1 : memref<256xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}
