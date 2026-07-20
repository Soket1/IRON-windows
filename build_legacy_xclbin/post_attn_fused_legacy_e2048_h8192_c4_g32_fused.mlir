module {
  aie.device(npu2) @op0_PostAttnFusedMLIR {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_2 = aie.tile(3, 2)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    aie.objectfifo @Ad_0(%shim_noc_tile_2_0, {%tile_2_3}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Ad_1(%shim_noc_tile_2_0, {%tile_2_4}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Ad_2(%shim_noc_tile_3_0, {%tile_2_5}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Ad_3(%shim_noc_tile_3_0, {%tile_3_2}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Agu_0(%shim_noc_tile_1_0, {%tile_1_3}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Agu_1(%shim_noc_tile_1_0, {%tile_1_4}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Agu_2(%shim_noc_tile_4_0, {%tile_1_5}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Agu_3(%shim_noc_tile_4_0, {%tile_2_2}, 1 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Ao_0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_1(%shim_noc_tile_0_0, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_2(%shim_noc_tile_5_0, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_3(%shim_noc_tile_5_0, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Cd_0(%tile_2_3, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cd_1(%tile_2_4, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cd_2(%tile_2_5, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cd_3(%tile_3_2, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cgu_0(%tile_1_3, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo @Cgu_1(%tile_1_4, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo @Cgu_2(%tile_1_5, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo @Cgu_3(%tile_2_2, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo @Co_0(%tile_0_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_1(%tile_0_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_2(%tile_0_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_3(%tile_0_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @o_out_joined_0(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo.link [@Co_0, @Co_1, @Co_2, @Co_3] -> [@o_out_joined_0]([0, 2, 4, 6] [])
    aie.objectfifo @anm_ffi_l1l2(%tile_1_2, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @ffi_mem(%mem_tile_0_1, {%tile_1_3, %tile_1_4, %tile_1_5, %tile_2_2}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@anm_ffi_l1l2] -> [@ffi_mem]([] [0])
    aie.objectfifo @anm_in(%shim_noc_tile_6_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @anm_inpff(%tile_1_2, {%shim_noc_tile_5_0}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @kqv_L3L2(%shim_noc_tile_6_0, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @kqv_mem(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@kqv_L3L2] -> [@kqv_mem]([] [0])
    aie.objectfifo @silu_L3L2(%shim_noc_tile_7_0, {%mem_tile_0_1}, 1 : i32) : !aie.objectfifo<memref<8192xbf16>> 
    aie.objectfifo @silu_mem(%mem_tile_0_1, {%tile_2_3, %tile_2_4, %tile_2_5, %tile_3_2}, 1 : i32) : !aie.objectfifo<memref<8192xbf16>> 
    aie.objectfifo.link [@silu_L3L2] -> [@silu_mem]([] [0])
    func.func private @op0_fused_dequant_matvec_v2_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    %anm_oout_buf = aie.buffer(%tile_1_2) {sym_name = "anm_oout_buf"} : memref<2048xbf16> 
    func.func private @op0_post_attn_o_out_assemble_bf16(memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_post_attn_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_post_attn_rms_norm_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_dual_fused_dequant_gemv_bf16(i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_dual_fused_dequant_gemv_silu_mul_bf16(memref<8xbf16>, i32) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    func.func private @op0_fused_dequant_matvec_down_bf16(i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) attributes {link_with = "op0_post_attn_fused_2048k_g32.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @kqv_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
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
            func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_0(Consume, 1)
            aie.objectfifo.release @Co_0(Produce, 1)
          }
          aie.objectfifo.release @kqv_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @kqv_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
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
            func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_1(Consume, 1)
            aie.objectfifo.release @Co_1(Produce, 1)
          }
          aie.objectfifo.release @kqv_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @kqv_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
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
            func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_2(Consume, 1)
            aie.objectfifo.release @Co_2(Produce, 1)
          }
          aie.objectfifo.release @kqv_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @kqv_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
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
            func.call @op0_fused_dequant_matvec_v2_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_3(Consume, 1)
            aie.objectfifo.release @Co_3(Produce, 1)
          }
          aie.objectfifo.release @kqv_mem(Consume, 1)
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
            func.call @op0_post_attn_o_out_assemble_bf16(%9, %anm_oout_buf, %7, %c0_i32, %c4_i32, %c2_i32, %c512_i32) : (memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @o_out_joined_0(Consume, 1)
          }
          %0 = aie.objectfifo.acquire @anm_in(Consume, 2) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %3 = aie.objectfifo.acquire @anm_inpff(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %5 = aie.objectfifo.acquire @anm_ffi_l1l2(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_post_attn_add_bf16(%anm_oout_buf, %1, %4, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
          %c2048_i32_4 = arith.constant 2048 : i32
          func.call @op0_post_attn_rms_norm_bf16(%4, %2, %6, %c2048_i32_4) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @anm_in(Consume, 2)
          aie.objectfifo.release @anm_inpff(Produce, 1)
          aie.objectfifo.release @anm_ffi_l1l2(Produce, 1)
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
          %0 = aie.objectfifo.acquire @ffi_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Agu_0(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32 = arith.constant 8 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %3, %1, %c0_i32_4) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_0(Consume, 1)
            %4 = aie.objectfifo.acquire @Agu_0(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32_5 = arith.constant 8 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32_5, %c0_i32_6, %5, %1, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_0(Consume, 1)
            %6 = aie.objectfifo.acquire @Cgu_0(Produce, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c8_i32_7 = arith.constant 8 : i32
            func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%7, %c8_i32_7) : (memref<8xbf16>, i32) -> ()
            aie.objectfifo.release @Cgu_0(Produce, 1)
          }
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
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Agu_1(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32 = arith.constant 8 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %3, %1, %c0_i32_4) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_1(Consume, 1)
            %4 = aie.objectfifo.acquire @Agu_1(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32_5 = arith.constant 8 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32_5, %c0_i32_6, %5, %1, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_1(Consume, 1)
            %6 = aie.objectfifo.acquire @Cgu_1(Produce, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c8_i32_7 = arith.constant 8 : i32
            func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%7, %c8_i32_7) : (memref<8xbf16>, i32) -> ()
            aie.objectfifo.release @Cgu_1(Produce, 1)
          }
          aie.objectfifo.release @ffi_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @ffi_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Agu_2(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32 = arith.constant 8 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %3, %1, %c0_i32_4) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_2(Consume, 1)
            %4 = aie.objectfifo.acquire @Agu_2(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32_5 = arith.constant 8 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32_5, %c0_i32_6, %5, %1, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_2(Consume, 1)
            %6 = aie.objectfifo.acquire @Cgu_2(Produce, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c8_i32_7 = arith.constant 8 : i32
            func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%7, %c8_i32_7) : (memref<8xbf16>, i32) -> ()
            aie.objectfifo.release @Cgu_2(Produce, 1)
          }
          aie.objectfifo.release @ffi_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @ffi_mem(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Agu_3(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32 = arith.constant 8 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32, %c0_i32, %3, %1, %c0_i32_4) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_3(Consume, 1)
            %4 = aie.objectfifo.acquire @Agu_3(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %c8_i32_5 = arith.constant 8 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_dual_fused_dequant_gemv_bf16(%c8_i32_5, %c0_i32_6, %5, %1, %c1_i32) : (i32, i32, memref<9216xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_3(Consume, 1)
            %6 = aie.objectfifo.acquire @Cgu_3(Produce, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c8_i32_7 = arith.constant 8 : i32
            func.call @op0_dual_fused_dequant_gemv_silu_mul_bf16(%7, %c8_i32_7) : (memref<8xbf16>, i32) -> ()
            aie.objectfifo.release @Cgu_3(Produce, 1)
          }
          aie.objectfifo.release @ffi_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @silu_mem(Consume, 1) : !aie.objectfifosubview<memref<8192xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ad_0(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %4 = aie.objectfifo.acquire @Cd_0(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ad_0(Consume, 1)
            aie.objectfifo.release @Cd_0(Produce, 1)
          }
          aie.objectfifo.release @silu_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @silu_mem(Consume, 1) : !aie.objectfifosubview<memref<8192xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ad_1(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %4 = aie.objectfifo.acquire @Cd_1(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ad_1(Consume, 1)
            aie.objectfifo.release @Cd_1(Produce, 1)
          }
          aie.objectfifo.release @silu_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @silu_mem(Consume, 1) : !aie.objectfifosubview<memref<8192xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ad_2(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %4 = aie.objectfifo.acquire @Cd_2(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ad_2(Consume, 1)
            aie.objectfifo.release @Cd_2(Produce, 1)
          }
          aie.objectfifo.release @silu_mem(Consume, 1)
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
          %0 = aie.objectfifo.acquire @silu_mem(Consume, 1) : !aie.objectfifosubview<memref<8192xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
          %c0_2 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ad_3(Consume, 1) : !aie.objectfifosubview<memref<9216xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<9216xui8>> -> memref<9216xui8>
            %4 = aie.objectfifo.acquire @Cd_3(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_fused_dequant_matvec_down_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<9216xui8>, memref<8192xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ad_3(Consume, 1)
            aie.objectfifo.release @Cd_3(Produce, 1)
          }
          aie.objectfifo.release @silu_mem(Consume, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<1179648xbf16>, %arg1: memref<9437184xbf16>, %arg2: memref<4718592xbf16>, %arg3: memref<6144xbf16>, %arg4: memref<14336xbf16>) {
      %0 = aiex.dma_configure_task_for @Ao_0 {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 0, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @Ao_1 {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 589824, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @Ao_2 {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 1179648, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @Ao_3 {
        aie.dma_bd(%arg0 : memref<1179648xbf16>, 1769472, 589824, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 589824, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @kqv_L3L2 {
        aie.dma_bd(%arg3 : memref<6144xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      %5 = aiex.dma_configure_task_for @anm_in {
        aie.dma_bd(%arg3 : memref<6144xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @anm_in {
        aie.dma_bd(%arg3 : memref<6144xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @anm_inpff {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 12288, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      aiex.dma_await_task(%7)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      %8 = aiex.dma_configure_task_for @Agu_0 {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 0, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @Agu_1 {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 4718592, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @Agu_2 {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 9437184, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @Agu_3 {
        aie.dma_bd(%arg1 : memref<9437184xbf16>, 14155776, 4718592, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4718592, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @Cgu_0 {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 8>, <size = 8, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @Cgu_1 {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 8>, <size = 8, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @Cgu_2 {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 6144, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 8>, <size = 8, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @Cgu_3 {
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
      %16 = aiex.dma_configure_task_for @Ad_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 0, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @Ad_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 2359296, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @Ad_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 4718592, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @Ad_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16>, 7077888, 2359296, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2359296, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @silu_L3L2 {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 2048, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @Cd_0 {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 10240, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 2>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @Cd_1 {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 10752, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 2>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @Cd_2 {
        aie.dma_bd(%arg4 : memref<14336xbf16>, 11264, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 2>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @Cd_3 {
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
  }
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<15341568xbf16>, %arg1: memref<14336xbf16>, %arg2: memref<0xbf16>) {
      aiex.configure @op0_PostAttnFusedMLIR {
        %subview = memref.subview %arg0[0] [1179648] [1] : memref<15341568xbf16> to memref<1179648xbf16>
        %reinterpret_cast = memref.reinterpret_cast %subview to offset: [0], sizes: [1179648], strides: [1] : memref<1179648xbf16> to memref<1179648xbf16>
        %subview_0 = memref.subview %arg0[1179648] [9437184] [1] : memref<15341568xbf16> to memref<9437184xbf16, strided<[1], offset: 1179648>>
        %reinterpret_cast_1 = memref.reinterpret_cast %subview_0 to offset: [0], sizes: [9437184], strides: [1] : memref<9437184xbf16, strided<[1], offset: 1179648>> to memref<9437184xbf16>
        %subview_2 = memref.subview %arg0[10616832] [4718592] [1] : memref<15341568xbf16> to memref<4718592xbf16, strided<[1], offset: 10616832>>
        %reinterpret_cast_3 = memref.reinterpret_cast %subview_2 to offset: [0], sizes: [4718592], strides: [1] : memref<4718592xbf16, strided<[1], offset: 10616832>> to memref<4718592xbf16>
        %subview_4 = memref.subview %arg0[15335424] [6144] [1] : memref<15341568xbf16> to memref<6144xbf16, strided<[1], offset: 15335424>>
        %reinterpret_cast_5 = memref.reinterpret_cast %subview_4 to offset: [0], sizes: [6144], strides: [1] : memref<6144xbf16, strided<[1], offset: 15335424>> to memref<6144xbf16>
        %subview_6 = memref.subview %arg1[0] [14336] [1] : memref<14336xbf16> to memref<14336xbf16>
        %reinterpret_cast_7 = memref.reinterpret_cast %subview_6 to offset: [0], sizes: [14336], strides: [1] : memref<14336xbf16> to memref<14336xbf16>
        aiex.run @sequence(%reinterpret_cast, %reinterpret_cast_1, %reinterpret_cast_3, %reinterpret_cast_5, %reinterpret_cast_7) : (memref<1179648xbf16>, memref<9437184xbf16>, memref<4718592xbf16>, memref<6144xbf16>, memref<14336xbf16>)
      }
    }
  }
}
