module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    aie.objectfifo @Adp_0(%mem_tile_3_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo @Adp_src_0(%shim_noc_tile_0_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Adp_1(%mem_tile_3_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo @Adp_2(%mem_tile_3_1, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo @Adp_3(%mem_tile_3_1, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo.link [@Adp_src_0] -> [@Adp_0, @Adp_1, @Adp_2, @Adp_3]([] [0, 1152, 2304, 3456])
    aie.objectfifo @Adp_4(%mem_tile_7_1, {%tile_4_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo @Adp_src_1(%shim_noc_tile_0_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Adp_5(%mem_tile_7_1, {%tile_5_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo @Adp_6(%mem_tile_7_1, {%tile_6_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo @Adp_7(%mem_tile_7_1, {%tile_7_5}, 2 : i32) : !aie.objectfifo<memref<1152xui8>> 
    aie.objectfifo.link [@Adp_src_1] -> [@Adp_4, @Adp_5, @Adp_6, @Adp_7]([] [0, 1152, 2304, 3456])
    aie.objectfifo @Agu_0(%mem_tile_2_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_src_0(%shim_noc_tile_1_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<18432xui8>> 
    aie.objectfifo @Agu_1(%mem_tile_2_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_2(%mem_tile_2_1, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_3(%mem_tile_2_1, {%tile_3_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo.link [@Agu_src_0] -> [@Agu_0, @Agu_1, @Agu_2, @Agu_3]([] [0, 4608, 9216, 13824])
    aie.objectfifo @Agu_4(%mem_tile_6_1, {%tile_4_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_src_1(%shim_noc_tile_1_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<18432xui8>> 
    aie.objectfifo @Agu_5(%mem_tile_6_1, {%tile_5_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_6(%mem_tile_6_1, {%tile_6_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo @Agu_7(%mem_tile_6_1, {%tile_7_4}, 2 : i32) : !aie.objectfifo<memref<4608xui8>> 
    aie.objectfifo.link [@Agu_src_1] -> [@Agu_4, @Agu_5, @Agu_6, @Agu_7]([] [0, 4608, 9216, 13824])
    aie.objectfifo @Ao_0(%mem_tile_1_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_src_0(%shim_noc_tile_2_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Ao_1(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_2(%mem_tile_1_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_3(%mem_tile_1_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Ao_src_0] -> [@Ao_0, @Ao_1, @Ao_2, @Ao_3]([] [0, 2304, 4608, 6912])
    aie.objectfifo @Ao_4(%mem_tile_5_1, {%tile_4_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_src_1(%shim_noc_tile_2_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Ao_5(%mem_tile_5_1, {%tile_5_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_6(%mem_tile_5_1, {%tile_6_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Ao_7(%mem_tile_5_1, {%tile_7_3}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Ao_src_1] -> [@Ao_4, @Ao_5, @Ao_6, @Ao_7]([] [0, 2304, 4608, 6912])
    aie.objectfifo @Aqkv_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_src_0(%shim_noc_tile_3_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Aqkv_1(%mem_tile_0_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_2(%mem_tile_0_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_3(%mem_tile_0_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Aqkv_src_0] -> [@Aqkv_0, @Aqkv_1, @Aqkv_2, @Aqkv_3]([] [0, 2304, 4608, 6912])
    aie.objectfifo @Aqkv_4(%mem_tile_4_1, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_src_1(%shim_noc_tile_3_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<9216xui8>> 
    aie.objectfifo @Aqkv_5(%mem_tile_4_1, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_6(%mem_tile_4_1, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo @Aqkv_7(%mem_tile_4_1, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<2304xui8>> 
    aie.objectfifo.link [@Aqkv_src_1] -> [@Aqkv_4, @Aqkv_5, @Aqkv_6, @Aqkv_7]([] [0, 2304, 4608, 6912])
    aie.objectfifo @Cdp_0(%tile_0_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_1(%tile_1_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_2(%tile_2_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_3(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @ffn_out_joined_0(%mem_tile_3_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo.link [@Cdp_0, @Cdp_1, @Cdp_2, @Cdp_3] -> [@ffn_out_joined_0]([0, 2, 4, 6] [])
    aie.objectfifo @Cdp_4(%tile_4_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_5(%tile_5_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_6(%tile_6_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cdp_7(%tile_7_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @ffn_out_joined_1(%mem_tile_7_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo.link [@Cdp_4, @Cdp_5, @Cdp_6, @Cdp_7] -> [@ffn_out_joined_1]([0, 2, 4, 6] [])
    aie.objectfifo @Co_0(%tile_0_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_1(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_2(%tile_2_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_3(%tile_3_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @o_out_joined_0(%mem_tile_1_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo.link [@Co_0, @Co_1, @Co_2, @Co_3] -> [@o_out_joined_0]([0, 2, 4, 6] [])
    aie.objectfifo @Co_4(%tile_4_3, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_5(%tile_5_3, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_6(%tile_6_3, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Co_7(%tile_7_3, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @o_out_joined_1(%mem_tile_5_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<8xbf16>> 
    aie.objectfifo.link [@Co_4, @Co_5, @Co_6, @Co_7] -> [@o_out_joined_1]([0, 2, 4, 6] [])
    aie.objectfifo @Cqkv_0(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_1(%tile_1_2, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_2(%tile_2_2, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_3(%tile_3_2, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_4(%tile_4_2, {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_5(%tile_5_2, {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_6(%tile_6_2, {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @Cqkv_7(%tile_7_2, {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<2xbf16>> 
    aie.objectfifo @anm_in(%shim_noc_tile_4_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @anm_mem(%mem_tile_1_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@anm_in] -> [@anm_mem]([] [0])
    aie.objectfifo @bo_L3L2(%shim_noc_tile_4_0, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @bo_mem(%mem_tile_6_1, {%tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3, %tile_4_3, %tile_5_3, %tile_6_3, %tile_7_3}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@bo_L3L2] -> [@bo_mem]([] [0])
    aie.objectfifo @bq_L3L2(%tile_0_2, {%mem_tile_2_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @bq_mem(%mem_tile_2_1, {%tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@bq_L3L2] -> [@bq_mem]([] [0])
    aie.objectfifo @ffi_L3L2(%tile_0_4, {%mem_tile_6_1}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @ffi_mem(%mem_tile_6_1, {%tile_1_4, %tile_2_4, %tile_3_4, %tile_4_4, %tile_5_4, %tile_6_4, %tile_7_4}, 1 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@ffi_L3L2] -> [@ffi_mem]([] [0])
    aie.objectfifo @inter_0(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @inter_1(%tile_1_4, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @inter_2(%tile_2_4, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @inter_3(%tile_3_4, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @inter_4(%tile_4_4, {%tile_4_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @inter_5(%tile_5_4, {%tile_5_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @inter_6(%tile_6_4, {%tile_6_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @inter_7(%tile_7_4, {%tile_7_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @rms_in(%shim_noc_tile_5_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    func.func private @op0_layer_fused_pre_rms_col0_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_qkv_gemv_static_bf16(i32, i32, memref<2304xui8>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_qkv_gemv_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_o_proj_bf16(i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    %anm_oout_buf = aie.buffer(%tile_0_4) {sym_name = "anm_oout_buf"} : memref<2048xbf16> 
    func.func private @op0_layer_fused_o_out_assemble_bf16(memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    %anm_inpff_buf = aie.buffer(%tile_0_4) {sym_name = "anm_inpff_buf"} : memref<2048xbf16> 
    func.func private @op0_layer_fused_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_rms_norm2_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_gate_up_bf16(i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_silu_mul_bf16(memref<1024xbf16>, i32) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
    func.func private @op0_layer_fused_down_partial_bf16(i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) attributes {link_with = "op0_layer_fused_2048_8192_g32.o"}
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
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
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
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
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
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
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
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
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
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
    %core_4_2 = aie.core(%tile_4_2) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Aqkv_4(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_4(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_4(Consume, 1)
            aie.objectfifo.release @Cqkv_4(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
            %2 = aie.objectfifo.acquire @Aqkv_4(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_4(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_4(Consume, 1)
            aie.objectfifo.release @Cqkv_4(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @Aqkv_4(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_4(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_4(Consume, 1)
            aie.objectfifo.release @Cqkv_4(Produce, 1)
          }
          aie.objectfifo.release @bq_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_5_2 = aie.core(%tile_5_2) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Aqkv_5(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_5(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_5(Consume, 1)
            aie.objectfifo.release @Cqkv_5(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
            %2 = aie.objectfifo.acquire @Aqkv_5(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_5(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_5(Consume, 1)
            aie.objectfifo.release @Cqkv_5(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @Aqkv_5(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_5(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_5(Consume, 1)
            aie.objectfifo.release @Cqkv_5(Produce, 1)
          }
          aie.objectfifo.release @bq_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_6_2 = aie.core(%tile_6_2) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Aqkv_6(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_6(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_6(Consume, 1)
            aie.objectfifo.release @Cqkv_6(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
            %2 = aie.objectfifo.acquire @Aqkv_6(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_6(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_6(Consume, 1)
            aie.objectfifo.release @Cqkv_6(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @Aqkv_6(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_6(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_6(Consume, 1)
            aie.objectfifo.release @Cqkv_6(Produce, 1)
          }
          aie.objectfifo.release @bq_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_7_2 = aie.core(%tile_7_2) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Aqkv_7(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_7(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_7(Consume, 1)
            aie.objectfifo.release @Cqkv_7(Produce, 1)
          }
          %c0_4 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %c32 step %c1_5 {
            %2 = aie.objectfifo.acquire @Aqkv_7(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_7(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_7(Consume, 1)
            aie.objectfifo.release @Cqkv_7(Produce, 1)
          }
          %c0_6 = arith.constant 0 : index
          %c32_7 = arith.constant 32 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c32_7 step %c1_8 {
            %2 = aie.objectfifo.acquire @Aqkv_7(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Cqkv_7(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_qkv_gemv_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Aqkv_7(Consume, 1)
            aie.objectfifo.release @Cqkv_7(Produce, 1)
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
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
    %core_4_3 = aie.core(%tile_4_3) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_4(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_4(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_4(Consume, 1)
            aie.objectfifo.release @Co_4(Produce, 1)
          }
          aie.objectfifo.release @bo_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_5_3 = aie.core(%tile_5_3) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_5(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_5(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_5(Consume, 1)
            aie.objectfifo.release @Co_5(Produce, 1)
          }
          aie.objectfifo.release @bo_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_6_3 = aie.core(%tile_6_3) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_6(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_6(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_6(Consume, 1)
            aie.objectfifo.release @Co_6(Produce, 1)
          }
          aie.objectfifo.release @bo_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_7_3 = aie.core(%tile_7_3) {
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
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %2 = aie.objectfifo.acquire @Ao_7(Consume, 1) : !aie.objectfifosubview<memref<2304xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2304xui8>> -> memref<2304xui8>
            %4 = aie.objectfifo.acquire @Co_7(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_o_proj_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<2304xui8>, memref<2048xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Ao_7(Consume, 1)
            aie.objectfifo.release @Co_7(Produce, 1)
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
          %c0_2 = arith.constant 0 : index
          %c128 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c128 step %c1_3 {
            %7 = arith.index_cast %arg2 : index to i32
            %8 = aie.objectfifo.acquire @o_out_joined_0(Consume, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            %c256_i32 = arith.constant 256 : i32
            func.call @op0_layer_fused_o_out_assemble_bf16(%9, %anm_oout_buf, %7, %c0_i32, %c4_i32, %c2_i32, %c256_i32) : (memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @o_out_joined_0(Consume, 1)
            %10 = aie.objectfifo.acquire @o_out_joined_1(Consume, 1) : !aie.objectfifosubview<memref<8xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
            %c4_i32_7 = arith.constant 4 : i32
            %c4_i32_8 = arith.constant 4 : i32
            %c2_i32_9 = arith.constant 2 : i32
            %c256_i32_10 = arith.constant 256 : i32
            func.call @op0_layer_fused_o_out_assemble_bf16(%11, %anm_oout_buf, %7, %c4_i32_7, %c4_i32_8, %c2_i32_9, %c256_i32_10) : (memref<8xbf16>, memref<2048xbf16>, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @o_out_joined_1(Consume, 1)
          }
          %0 = aie.objectfifo.acquire @anm_mem(Consume, 2) : !aie.objectfifosubview<memref<2048xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %3 = aie.objectfifo.acquire @ffi_L3L2(Produce, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c2048_i32 = arith.constant 2048 : i32
          func.call @op0_layer_fused_add_bf16(%anm_oout_buf, %1, %anm_inpff_buf, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
          %c2048_i32_4 = arith.constant 2048 : i32
          func.call @op0_layer_fused_rms_norm2_bf16(%anm_inpff_buf, %2, %4, %c2048_i32_4) : (memref<2048xbf16>, memref<2048xbf16>, memref<2048xbf16>, i32) -> ()
          aie.objectfifo.release @anm_mem(Consume, 2)
          %c0_5 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_6 = arith.constant 1 : index
          scf.for %arg2 = %c0_5 to %c256 step %c1_6 {
            %7 = aie.objectfifo.acquire @Agu_0(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_7 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %8, %4, %c0_i32_7) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_0(Consume, 1)
            %9 = aie.objectfifo.acquire @Agu_0(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_8 = arith.constant 4 : i32
            %c0_i32_9 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_8, %c0_i32_9, %10, %4, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_0(Consume, 1)
          }
          %5 = aie.objectfifo.acquire @inter_0(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%6, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @inter_0(Produce, 1)
          aie.objectfifo.release @ffi_L3L2(Produce, 1)
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
          %2 = aie.objectfifo.acquire @inter_1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
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
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
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
          %2 = aie.objectfifo.acquire @inter_2(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
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
          %c256 = arith.constant 256 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c256 step %c1_3 {
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
          %2 = aie.objectfifo.acquire @inter_3(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @inter_3(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_4_4 = aie.core(%tile_4_4) {
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
            %4 = aie.objectfifo.acquire @Agu_4(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_4(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_4(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_4(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_4(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @inter_4(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_5_4 = aie.core(%tile_5_4) {
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
            %4 = aie.objectfifo.acquire @Agu_5(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_5(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_5(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_5(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_5(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @inter_5(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_6_4 = aie.core(%tile_6_4) {
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
            %4 = aie.objectfifo.acquire @Agu_6(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_6(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_6(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_6(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_6(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @inter_6(Produce, 1)
          aie.objectfifo.release @ffi_mem(Consume, 1)
        }
      }
      aie.end
    }
    %core_7_4 = aie.core(%tile_7_4) {
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
            %4 = aie.objectfifo.acquire @Agu_7(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32 = arith.constant 4 : i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_4 = arith.constant 0 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32, %c0_i32, %5, %1, %c0_i32_4) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_7(Consume, 1)
            %6 = aie.objectfifo.acquire @Agu_7(Consume, 1) : !aie.objectfifosubview<memref<4608xui8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<4608xui8>> -> memref<4608xui8>
            %c4_i32_5 = arith.constant 4 : i32
            %c0_i32_6 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            func.call @op0_layer_fused_gate_up_bf16(%c4_i32_5, %c0_i32_6, %7, %1, %c1_i32) : (i32, i32, memref<4608xui8>, memref<2048xbf16>, i32) -> ()
            aie.objectfifo.release @Agu_7(Consume, 1)
          }
          %2 = aie.objectfifo.acquire @inter_7(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @op0_layer_fused_silu_mul_bf16(%3, %c1024_i32) : (memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @inter_7(Produce, 1)
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
          %0 = aie.objectfifo.acquire @inter_0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_0(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_0(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
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
          %0 = aie.objectfifo.acquire @inter_1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_1(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_1(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
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
          %0 = aie.objectfifo.acquire @inter_2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_2(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_2(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
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
          %0 = aie.objectfifo.acquire @inter_3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_3(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_3(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_3(Consume, 1)
            aie.objectfifo.release @Cdp_3(Produce, 1)
          }
          aie.objectfifo.release @inter_3(Consume, 1)
        }
      }
      aie.end
    }
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_4(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_4(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_4(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_4(Consume, 1)
            aie.objectfifo.release @Cdp_4(Produce, 1)
          }
          aie.objectfifo.release @inter_4(Consume, 1)
        }
      }
      aie.end
    }
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_5(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_5(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_5(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_5(Consume, 1)
            aie.objectfifo.release @Cdp_5(Produce, 1)
          }
          aie.objectfifo.release @inter_5(Consume, 1)
        }
      }
      aie.end
    }
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_6(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_6(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_6(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_6(Consume, 1)
            aie.objectfifo.release @Cdp_6(Produce, 1)
          }
          aie.objectfifo.release @inter_6(Consume, 1)
        }
      }
      aie.end
    }
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c4294967295 = arith.constant 4294967295 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4294967295 step %c1_1 {
          %0 = aie.objectfifo.acquire @inter_7(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c0_2 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1024 step %c1_3 {
            %2 = aie.objectfifo.acquire @Adp_7(Consume, 1) : !aie.objectfifosubview<memref<1152xui8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1152xui8>> -> memref<1152xui8>
            %4 = aie.objectfifo.acquire @Cdp_7(Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            %c2_i32 = arith.constant 2 : i32
            %c0_i32 = arith.constant 0 : i32
            func.call @op0_layer_fused_down_partial_bf16(%c2_i32, %c0_i32, %3, %1, %5) : (i32, i32, memref<1152xui8>, memref<1024xbf16>, memref<2xbf16>) -> ()
            aie.objectfifo.release @Adp_7(Consume, 1)
            aie.objectfifo.release @Cdp_7(Produce, 1)
          }
          aie.objectfifo.release @inter_7(Consume, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<1771520xbf16>, %arg1: memref<1179648xbf16>, %arg2: memref<14157824xbf16>, %arg3: memref<2097152xbf16>, %arg4: memref<308224xbf16>) {
      %0 = aiex.dma_configure_task_for @rms_in {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
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
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 2048, 589824, [<size = 1, stride = 0>, <size = 128, stride = 1152>, <size = 4, stride = 147456>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @Aqkv_src_1 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 591872, 589824, [<size = 1, stride = 0>, <size = 128, stride = 1152>, <size = 4, stride = 147456>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @Cqkv_0 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 266240, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @Cqkv_1 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 266496, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @Cqkv_2 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 266752, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @Cqkv_3 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267008, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @Cqkv_4 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267264, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @Cqkv_5 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267520, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @Cqkv_6 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267776, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @Cqkv_7 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268032, 2, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @Aqkv_src_0 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1181696, 147456, [<size = 1, stride = 0>, <size = 32, stride = 1152>, <size = 4, stride = 36864>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @Aqkv_src_1 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1329152, 147456, [<size = 1, stride = 0>, <size = 32, stride = 1152>, <size = 4, stride = 36864>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @Aqkv_src_0 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1476608, 147456, [<size = 1, stride = 0>, <size = 32, stride = 1152>, <size = 4, stride = 36864>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @Aqkv_src_1 {
        aie.dma_bd(%arg0 : memref<1771520xbf16>, 1624064, 147456, [<size = 1, stride = 0>, <size = 32, stride = 1152>, <size = 4, stride = 36864>, <size = 1152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @Cqkv_0 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 266240, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268288, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268800, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      aiex.dma_await_task(%16)
      %17 = aiex.dma_configure_task_for @Cqkv_1 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 266496, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268352, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268864, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      aiex.dma_await_task(%17)
      %18 = aiex.dma_configure_task_for @Cqkv_2 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 266752, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268416, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268928, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      aiex.dma_await_task(%18)
      %19 = aiex.dma_configure_task_for @Cqkv_3 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267008, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268480, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268992, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      aiex.dma_await_task(%19)
      %20 = aiex.dma_configure_task_for @Cqkv_4 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267264, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268544, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 269056, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      aiex.dma_await_task(%20)
      %21 = aiex.dma_configure_task_for @Cqkv_5 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267520, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268608, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 269120, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      aiex.dma_await_task(%21)
      %22 = aiex.dma_configure_task_for @Cqkv_6 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 267776, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268672, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 269184, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      aiex.dma_await_task(%22)
      %23 = aiex.dma_configure_task_for @Cqkv_7 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268032, 254, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 254, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb1:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 268736, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      ^bb2:  // no predecessors
        aie.dma_bd(%arg4 : memref<308224xbf16>, 269248, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%23)
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
      %24 = aiex.dma_configure_task_for @Ao_src_0 {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 0, 36864, [<size = 16, stride = 36864>, <size = 64, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 15 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @Ao_src_1 {
        aie.dma_bd(%arg1 : memref<1179648xbf16>, 589824, 36864, [<size = 16, stride = 36864>, <size = 64, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 15 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @bo_L3L2 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 269312, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%26)
      %27 = aiex.dma_configure_task_for @anm_in {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @anm_in {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      %29 = aiex.dma_configure_task_for @Agu_src_0 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 2048, 147456, [<size = 32, stride = 2304>, <size = 64, stride = 73728>, <size = 72, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @Agu_src_1 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 4720640, 147456, [<size = 32, stride = 2304>, <size = 64, stride = 73728>, <size = 72, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 31 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @Adp_src_0 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 9439232, 294912, [<size = 8, stride = 294912>, <size = 512, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @Adp_src_1 {
        aie.dma_bd(%arg2 : memref<14157824xbf16>, 11798528, 294912, [<size = 8, stride = 294912>, <size = 512, stride = 576>, <size = 18, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @ffn_out_joined_0 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 291840, 4096, [<size = 2, stride = 1024>, <size = 512, stride = 2>, <size = 4, stride = 2048>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @ffn_out_joined_1 {
        aie.dma_bd(%arg4 : memref<308224xbf16>, 300032, 4096, [<size = 2, stride = 1024>, <size = 512, stride = 2>, <size = 4, stride = 2048>, <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%34)
      aiex.dma_await_task(%34)
      aiex.dma_free_task(%29)
      aiex.dma_free_task(%30)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
    }
  }
}
