module {
  aie.device(npu2) @op0_LayerFusedMLIR {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_2 = aie.tile(4, 2)
    %tile_4_3 = aie.tile(4, 3)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    aie.objectfifo @Adp_0(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ffn_src_0(%shim_noc_tile_2_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<13824xui8>> 
    aie.objectfifo @Agu_0(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_1(%mem_tile_0_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Adp_1(%mem_tile_0_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Ffn_src_0] -> [@Agu_0, @Agu_1, @Adp_0, @Adp_1]([] [0, 4608, 9216, 11520])
    aie.objectfifo @Adp_2(%mem_tile_2_1, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ffn_src_1(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<13824xui8>> 
    aie.objectfifo @Agu_2(%mem_tile_2_1, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_3(%mem_tile_2_1, {%tile_3_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Adp_3(%mem_tile_2_1, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Ffn_src_1] -> [@Agu_2, @Agu_3, @Adp_2, @Adp_3]([] [0, 4608, 9216, 11520])
    aie.objectfifo @Ao_0(%mem_tile_1_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_src_0(%shim_noc_tile_0_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Ao_1(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Ao_src_0] -> [@Ao_0, @Ao_1]([] [0, 2304])
    aie.objectfifo @Ao_2(%mem_tile_3_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_src_1(%shim_noc_tile_0_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Ao_3(%mem_tile_3_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Ao_src_1] -> [@Ao_2, @Ao_3]([] [0, 2304])
    aie.objectfifo @Aqkv_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_src_0(%shim_noc_tile_1_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Aqkv_1(%mem_tile_0_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Aqkv_src_0] -> [@Aqkv_0, @Aqkv_1]([] [0, 2304])
    aie.objectfifo @Aqkv_2(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_src_1(%shim_noc_tile_1_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Aqkv_3(%mem_tile_2_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Aqkv_src_1] -> [@Aqkv_2, @Aqkv_3]([] [0, 2304])
    aie.objectfifo @Cdp_0(%tile_0_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_1(%tile_1_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_2(%tile_2_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_3(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @ffn_out_joined_0(%mem_tile_3_1, {%tile_4_3}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo.link [@Cdp_0, @Cdp_1, @Cdp_2, @Cdp_3] -> [@ffn_out_joined_0]([0, 2, 4, 6] [])
    aie.objectfifo @Co_0(%tile_0_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_1(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_2(%tile_2_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_3(%tile_3_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @o_out_joined_0(%mem_tile_1_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo.link [@Co_0, @Co_1, @Co_2, @Co_3] -> [@o_out_joined_0]([0, 2, 4, 6] [])
    aie.objectfifo @Cqkv_0(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_1(%tile_1_2, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_2(%tile_2_2, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_3(%tile_3_2, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @anm_in(%shim_noc_tile_3_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @anm_mem(%mem_tile_1_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@anm_in] -> [@anm_mem]([] [0])
    aie.objectfifo @bo_L3L2(%shim_noc_tile_3_0, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @bo_mem(%mem_tile_2_1, {%tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@bo_L3L2] -> [@bo_mem]([] [0])
    aie.objectfifo @bq_L3L2(%tile_0_2, {%mem_tile_3_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @bq_mem(%mem_tile_3_1, {%tile_1_2, %tile_2_2, %tile_3_2}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@bq_L3L2] -> [@bq_mem]([] [0])
    aie.objectfifo @ffi_L3L2(%tile_4_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @ffi_mem(%mem_tile_2_1, {%tile_0_4, %tile_1_4, %tile_2_4, %tile_3_4}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@ffi_L3L2] -> [@ffi_mem]([] [0])
    aie.objectfifo @inpff_fifo(%tile_4_2, {%tile_4_3}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @inter_0(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @inter_1(%tile_1_4, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @inter_2(%tile_2_4, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @inter_3(%tile_3_4, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @outL_fifo(%tile_4_3, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @rms_in(%shim_noc_tile_4_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    func.func private @op0_layer_fused_pre_rms_col0_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_qkv_gemv_static_bf16(i32, i32, memref<2304xui8>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_qkv_gemv_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_o_proj_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_gate_up_bf16(i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_silu_mul_bf16(memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_down_partial_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    %anm_oout_buf = aie.buffer(%tile_4_2) {sym_name = "anm_oout_buf"} : memref<2048xbf16> 
    func.func private @op0_layer_fused_o_out_assemble_bf16(memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_rms_norm2_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_sum_partials_add_bf16(memref<8xbf16>, memref<2048xbf16>, memref<2xbf16>, i32, i32, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @rms_in(Consume, 2) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %3 = aie.objectfifo.acquire @bq_L3L2(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_layer_fused_pre_rms_col0_bf16(%1, %2, %4, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @rms_in(Consume, 2)
          aie.objectfifo.release @bq_L3L2(Produce, 1)
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %5 = aie.objectfifo.acquire @Aqkv_0(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %7 = aie.objectfifo.acquire @Cqkv_0(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_static_bf16(%c2_i32, %c0_i32, %6, %8) : (i32, i32, memref<2304xui8>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_0(Consume, 1)
            aie.objectfifo.release @Cqkv_0(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c64 step %c1_5 {
            %5 = aie.objectfifo.acquire @Aqkv_0(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %7 = aie.objectfifo.acquire @Cqkv_0(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_static_bf16(%c2_i32, %c0_i32, %6, %8) : (i32, i32, memref<2304xui8>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_0(Consume, 1)
            aie.objectfifo.release @Cqkv_0(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c64_7 = arith.constant 64 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c64_7 step %c1_8 {
            %5 = aie.objectfifo.acquire @Aqkv_0(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %7 = aie.objectfifo.acquire @Cqkv_0(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_static_bf16(%c2_i32, %c0_i32, %6, %8) : (i32, i32, memref<2304xui8>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_0(Consume, 1)
            aie.objectfifo.release @Cqkv_0(Produce, 1)
          }
        }
      }
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @bq_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Aqkv_1(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_1(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_1(Consume, 1)
            aie.objectfifo.release @Cqkv_1(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c64 step %c1_5 {
            %2 = aie.objectfifo.acquire @Aqkv_1(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_1(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_1(Consume, 1)
            aie.objectfifo.release @Cqkv_1(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c64_7 = arith.constant 64 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c64_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @Aqkv_1(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_1(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_1(Consume, 1)
            aie.objectfifo.release @Cqkv_1(Produce, 1)
          }
          aie.objectfifo.release @bq_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @bq_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Aqkv_2(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_2(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_2(Consume, 1)
            aie.objectfifo.release @Cqkv_2(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c64 step %c1_5 {
            %2 = aie.objectfifo.acquire @Aqkv_2(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_2(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_2(Consume, 1)
            aie.objectfifo.release @Cqkv_2(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c64_7 = arith.constant 64 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c64_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @Aqkv_2(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_2(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_2(Consume, 1)
            aie.objectfifo.release @Cqkv_2(Produce, 1)
          }
          aie.objectfifo.release @bq_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @bq_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Aqkv_3(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_3(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_3(Consume, 1)
            aie.objectfifo.release @Cqkv_3(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c64 step %c1_5 {
            %2 = aie.objectfifo.acquire @Aqkv_3(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_3(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_3(Consume, 1)
            aie.objectfifo.release @Cqkv_3(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c64_7 = arith.constant 64 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c64_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @Aqkv_3(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_3(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_3(Consume, 1)
            aie.objectfifo.release @Cqkv_3(Produce, 1)
          }
          aie.objectfifo.release @bq_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @bo_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_0(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_0(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_0(Consume, 1)
            aie.objectfifo.release @Co_0(Produce, 1)
          }
          aie.objectfifo.release @bo_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @bo_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_1(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_1(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_1(Consume, 1)
            aie.objectfifo.release @Co_1(Produce, 1)
          }
          aie.objectfifo.release @bo_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @bo_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_2(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_2(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_2(Consume, 1)
            aie.objectfifo.release @Co_2(Produce, 1)
          }
          aie.objectfifo.release @bo_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @bo_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_3(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_3(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_3(Consume, 1)
            aie.objectfifo.release @Co_3(Produce, 1)
          }
          aie.objectfifo.release @bo_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @ffi_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c512 step %c1_3 {
            %4 = aie.objectfifo.acquire @Agu_0(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_0(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_0(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_0(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_0(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @inter_0(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @ffi_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c512 step %c1_3 {
            %4 = aie.objectfifo.acquire @Agu_1(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_1(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_1(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_1(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_1(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @inter_1(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @ffi_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c512 step %c1_3 {
            %4 = aie.objectfifo.acquire @Agu_2(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_2(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_2(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_2(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_2(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @inter_2(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @ffi_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c512 step %c1_3 {
            %4 = aie.objectfifo.acquire @Agu_3(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_3(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_3(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_3(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_3(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c2048_i32) : (memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @inter_3(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_0(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_0(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cdp_0(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_0(Consume, 1)
            aie.objectfifo.release @Cdp_0(Produce, 1)
          }
          aie.objectfifo.release @inter_0(Consume, 1)
        }
      }
      aie.end
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_1(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_1(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cdp_1(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_1(Consume, 1)
            aie.objectfifo.release @Cdp_1(Produce, 1)
          }
          aie.objectfifo.release @inter_1(Consume, 1)
        }
      }
      aie.end
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_2(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_2(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cdp_2(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_2(Consume, 1)
            aie.objectfifo.release @Cdp_2(Produce, 1)
          }
          aie.objectfifo.release @inter_2(Consume, 1)
        }
      }
      aie.end
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_3(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_3(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cdp_3(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_3(Consume, 1)
            aie.objectfifo.release @Cdp_3(Produce, 1)
          }
          aie.objectfifo.release @inter_3(Consume, 1)
        }
      }
      aie.end
    }
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %7 = arith.index_cast %arg2 : index to i32
            %8 = aie.objectfifo.acquire @o_out_joined_0(Consume, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            %c512_i32 = arith.constant 512 : i32
            func.call @op0_layer_fused_o_out_assemble_bf16(%9, %anm_oout_buf, %7, %c0_i32, %c4_i32, %c2_i32, %c512_i32) : (memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @o_out_joined_0(Consume, 1)
          }
          %0 = aie.objectfifo.acquire @anm_mem(Consume, 2) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %3 = aie.objectfifo.acquire @ffi_L3L2(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %5 = aie.objectfifo.acquire @inpff_fifo(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_layer_fused_add_bf16(%anm_oout_buf, %1, %6, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
          %c2048_i32_4 = arith.constant 2048 : i32
          func.call @op0_layer_fused_rms_norm2_bf16(%6, %2, %4, %c2048_i32_4) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @anm_mem(Consume, 2)
          aie.objectfifo.release @ffi_L3L2(Produce, 1)
          aie.objectfifo.release @inpff_fifo(Produce, 1)
        }
      }
      aie.end
    }
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inpff_fifo(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = arith.index_cast %arg2 : index to i32
            %3 = aie.objectfifo.acquire @outL_fifo(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %5 = aie.objectfifo.acquire @ffn_out_joined_0(Consume, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            func.call @op0_layer_fused_sum_partials_add_bf16(%6, %1, %4, %2, %c4_i32, %c2_i32) : (memref<8xbf16>, memref<2048xbf16>, memref<2xbf16>, i32, i32, i32) -> ()
            aie.objectfifo.release @ffn_out_joined_0(Consume, 1)
            aie.objectfifo.release @outL_fifo(Produce, 1)
          }
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<1771520xbf16>, %arg1: memref<1179648xbf16>, %arg2: memref<14157824xbf16>, %arg3: memref<2097152xbf16>, %arg4: memref<300032xbf16>) {
      %0 = aiex.dma_configure_task_for @rms_in {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @rms_in {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      %2 = aiex.dma_configure_task_for @Aqkv_src_0 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 2048, 589824, [<size = 1, stride = 0>, <size = 256, stride = 1152>, <size = 2, stride = 294912>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1181696, 147456, [<size = 1, stride = 0>, <size = 64, stride = 1152>, <size = 2, stride = 73728>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1476608, 147456, [<size = 1, stride = 0>, <size = 64, stride = 1152>, <size = 2, stride = 73728>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      %3 = aiex.dma_configure_task_for @Ao_src_0 {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 0, 36864, [<size = 16, stride = 36864>, <size = 64, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      %4 = aiex.dma_configure_task_for @Ffn_src_0 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 2048, 147456, [<size = 32, stride = 2304>, <size = 64, stride = 73728>, <size = 72, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 9439232, 294912, [<size = 4, stride = 589824>, <size = 512, stride = 1152>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      aiex.dma_await_task(%4)
      %5 = aiex.dma_configure_task_for @Aqkv_src_1 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 591872, 589824, [<size = 1, stride = 0>, <size = 256, stride = 1152>, <size = 2, stride = 294912>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1329152, 147456, [<size = 1, stride = 0>, <size = 64, stride = 1152>, <size = 2, stride = 73728>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1624064, 147456, [<size = 1, stride = 0>, <size = 64, stride = 1152>, <size = 2, stride = 73728>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      aiex.dma_await_task(%5)
      %6 = aiex.dma_configure_task_for @Ao_src_1 {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 589824, 36864, [<size = 16, stride = 36864>, <size = 64, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%6)
      %7 = aiex.dma_configure_task_for @Ffn_src_1 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 4720640, 147456, [<size = 32, stride = 2304>, <size = 64, stride = 73728>, <size = 72, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 11798528, 294912, [<size = 4, stride = 589824>, <size = 512, stride = 1152>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      aiex.dma_await_task(%7)
      %8 = aiex.dma_configure_task_for @Aqkv_src_0 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @Ao_src_0 {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @Ffn_src_0 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 0, 6912, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 6912, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @Aqkv_src_1 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @Ao_src_1 {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2304, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @Ffn_src_1 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 0, 6912, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 6912, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @Cqkv_0 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266240, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @Cqkv_1 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266752, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @Cqkv_2 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267264, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @Cqkv_3 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267776, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @Cqkv_0 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266240, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 268288, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 268800, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      aiex.dma_await_task(%18)
      %19 = aiex.dma_configure_task_for @Cqkv_1 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 266752, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 268416, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 268928, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      aiex.dma_await_task(%19)
      %20 = aiex.dma_configure_task_for @Cqkv_2 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267264, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 268544, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 269056, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      aiex.dma_await_task(%20)
      %21 = aiex.dma_configure_task_for @Cqkv_3 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 267776, 510, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 510, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 268672, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<300032xbf16>, 269184, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
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
      %22 = aiex.dma_configure_task_for @bo_L3L2 {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 269312, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      aiex.dma_free_task(%22)
      %23 = aiex.dma_configure_task_for @anm_in {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @anm_in {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      %25 = aiex.dma_configure_task_for @outL_fifo {
        aie.dma_bd(%arg4 : memref<300032xbf16>, 289792, 2048, [<size = 1, stride = 0>, <size = 1024, stride = 2>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%25)
      aiex.dma_await_task(%25)
    }
  }
}
