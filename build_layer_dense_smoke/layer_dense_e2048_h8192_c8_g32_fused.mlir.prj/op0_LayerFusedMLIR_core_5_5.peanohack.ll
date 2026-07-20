; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous31 = external global [3 x i32]
@_anonymous30 = external global [3 x i32]
@_anonymous29 = external global [3 x i32]
@_anonymous28 = external global [3 x i32]
@_anonymous27 = external global [3 x i32]
@_anonymous26 = external global [3 x i32]
@_anonymous25 = external global [3 x i32]
@_anonymous24 = external global [3 x i32]
@_anonymous23 = external global [3 x i32]
@_anonymous22 = external global [3 x i32]
@_anonymous21 = external global [3 x i32]
@_anonymous20 = external global [3 x i32]
@_anonymous19 = external global [3 x i32]
@_anonymous18 = external global [3 x i32]
@_anonymous17 = external global [3 x i32]
@_anonymous16 = external global [4 x i32]
@_anonymous15 = external global [3 x i32]
@_anonymous14 = external global [3 x i32]
@_anonymous13 = external global [3 x i32]
@_anonymous12 = external global [3 x i32]
@_anonymous11 = external global [3 x i32]
@_anonymous10 = external global [3 x i32]
@_anonymous9 = external global [3 x i32]
@_anonymous8 = external global [3 x i32]
@_anonymous7 = external global [3 x i32]
@_anonymous6 = external global [3 x i32]
@_anonymous5 = external global [3 x i32]
@_anonymous4 = external global [3 x i32]
@_anonymous3 = external global [3 x i32]
@_anonymous2 = external global [3 x i32]
@_anonymous1 = external global [3 x i32]
@_anonymous0 = external global [4 x i32]
@anm_inpff_buf = external global [2048 x bfloat]
@Adp_0_cons_buff_1 = external global [1152 x i8]
@Adp_0_cons_buff_0 = external global [1152 x i8]
@Adp_src_0_cons_buff_1 = external global [4608 x i8]
@Adp_src_0_cons_buff_0 = external global [4608 x i8]
@Adp_1_cons_buff_1 = external global [1152 x i8]
@Adp_1_cons_buff_0 = external global [1152 x i8]
@Adp_2_cons_buff_1 = external global [1152 x i8]
@Adp_2_cons_buff_0 = external global [1152 x i8]
@Adp_3_cons_buff_1 = external global [1152 x i8]
@Adp_3_cons_buff_0 = external global [1152 x i8]
@Adp_4_cons_buff_1 = external global [1152 x i8]
@Adp_4_cons_buff_0 = external global [1152 x i8]
@Adp_src_1_cons_buff_1 = external global [4608 x i8]
@Adp_src_1_cons_buff_0 = external global [4608 x i8]
@Adp_5_cons_buff_1 = external global [1152 x i8]
@Adp_5_cons_buff_0 = external global [1152 x i8]
@Adp_6_cons_buff_1 = external global [1152 x i8]
@Adp_6_cons_buff_0 = external global [1152 x i8]
@Adp_7_cons_buff_1 = external global [1152 x i8]
@Adp_7_cons_buff_0 = external global [1152 x i8]
@Agu_0_cons_buff_1 = external global [4608 x i8]
@Agu_0_cons_buff_0 = external global [4608 x i8]
@Agu_src_0_cons_buff_1 = external global [18432 x i8]
@Agu_src_0_cons_buff_0 = external global [18432 x i8]
@Agu_1_cons_buff_1 = external global [4608 x i8]
@Agu_1_cons_buff_0 = external global [4608 x i8]
@Agu_2_cons_buff_1 = external global [4608 x i8]
@Agu_2_cons_buff_0 = external global [4608 x i8]
@Agu_3_cons_buff_1 = external global [4608 x i8]
@Agu_3_cons_buff_0 = external global [4608 x i8]
@Agu_4_cons_buff_1 = external global [4608 x i8]
@Agu_4_cons_buff_0 = external global [4608 x i8]
@Agu_src_1_cons_buff_1 = external global [18432 x i8]
@Agu_src_1_cons_buff_0 = external global [18432 x i8]
@Agu_5_cons_buff_1 = external global [4608 x i8]
@Agu_5_cons_buff_0 = external global [4608 x i8]
@Agu_6_cons_buff_1 = external global [4608 x i8]
@Agu_6_cons_buff_0 = external global [4608 x i8]
@Agu_7_cons_buff_1 = external global [4608 x i8]
@Agu_7_cons_buff_0 = external global [4608 x i8]
@Ao_0_cons_buff_1 = external global [2304 x i8]
@Ao_0_cons_buff_0 = external global [2304 x i8]
@Ao_src_0_cons_buff_1 = external global [9216 x i8]
@Ao_src_0_cons_buff_0 = external global [9216 x i8]
@Ao_1_cons_buff_1 = external global [2304 x i8]
@Ao_1_cons_buff_0 = external global [2304 x i8]
@Ao_2_cons_buff_1 = external global [2304 x i8]
@Ao_2_cons_buff_0 = external global [2304 x i8]
@Ao_3_cons_buff_1 = external global [2304 x i8]
@Ao_3_cons_buff_0 = external global [2304 x i8]
@Ao_4_cons_buff_1 = external global [2304 x i8]
@Ao_4_cons_buff_0 = external global [2304 x i8]
@Ao_src_1_cons_buff_1 = external global [9216 x i8]
@Ao_src_1_cons_buff_0 = external global [9216 x i8]
@Ao_5_cons_buff_1 = external global [2304 x i8]
@Ao_5_cons_buff_0 = external global [2304 x i8]
@Ao_6_cons_buff_1 = external global [2304 x i8]
@Ao_6_cons_buff_0 = external global [2304 x i8]
@Ao_7_cons_buff_1 = external global [2304 x i8]
@Ao_7_cons_buff_0 = external global [2304 x i8]
@Aqkv_0_cons_buff_1 = external global [2304 x i8]
@Aqkv_0_cons_buff_0 = external global [2304 x i8]
@Aqkv_src_0_cons_buff_1 = external global [9216 x i8]
@Aqkv_src_0_cons_buff_0 = external global [9216 x i8]
@Aqkv_1_cons_buff_1 = external global [2304 x i8]
@Aqkv_1_cons_buff_0 = external global [2304 x i8]
@Aqkv_2_cons_buff_1 = external global [2304 x i8]
@Aqkv_2_cons_buff_0 = external global [2304 x i8]
@Aqkv_3_cons_buff_1 = external global [2304 x i8]
@Aqkv_3_cons_buff_0 = external global [2304 x i8]
@Aqkv_4_cons_buff_1 = external global [2304 x i8]
@Aqkv_4_cons_buff_0 = external global [2304 x i8]
@Aqkv_src_1_cons_buff_1 = external global [9216 x i8]
@Aqkv_src_1_cons_buff_0 = external global [9216 x i8]
@Aqkv_5_cons_buff_1 = external global [2304 x i8]
@Aqkv_5_cons_buff_0 = external global [2304 x i8]
@Aqkv_6_cons_buff_1 = external global [2304 x i8]
@Aqkv_6_cons_buff_0 = external global [2304 x i8]
@Aqkv_7_cons_buff_1 = external global [2304 x i8]
@Aqkv_7_cons_buff_0 = external global [2304 x i8]
@Cdp_0_buff_1 = external global [2 x bfloat]
@Cdp_0_buff_0 = external global [2 x bfloat]
@Cdp_1_buff_1 = external global [2 x bfloat]
@Cdp_1_buff_0 = external global [2 x bfloat]
@Cdp_2_buff_1 = external global [2 x bfloat]
@Cdp_2_buff_0 = external global [2 x bfloat]
@Cdp_3_buff_1 = external global [2 x bfloat]
@Cdp_3_buff_0 = external global [2 x bfloat]
@ffn_out_joined_0_buff_1 = external global [8 x bfloat]
@ffn_out_joined_0_buff_0 = external global [8 x bfloat]
@Cdp_4_buff_1 = external global [2 x bfloat]
@Cdp_4_buff_0 = external global [2 x bfloat]
@Cdp_5_buff_1 = external global [2 x bfloat]
@Cdp_5_buff_0 = external global [2 x bfloat]
@Cdp_6_buff_1 = external global [2 x bfloat]
@Cdp_6_buff_0 = external global [2 x bfloat]
@Cdp_7_buff_1 = external global [2 x bfloat]
@Cdp_7_buff_0 = external global [2 x bfloat]
@ffn_out_joined_1_buff_1 = external global [8 x bfloat]
@ffn_out_joined_1_buff_0 = external global [8 x bfloat]
@Co_0_buff_1 = external global [2 x bfloat]
@Co_0_buff_0 = external global [2 x bfloat]
@Co_1_buff_1 = external global [2 x bfloat]
@Co_1_buff_0 = external global [2 x bfloat]
@Co_2_buff_1 = external global [2 x bfloat]
@Co_2_buff_0 = external global [2 x bfloat]
@Co_3_buff_1 = external global [2 x bfloat]
@Co_3_buff_0 = external global [2 x bfloat]
@o_out_joined_0_buff_1 = external global [8 x bfloat]
@o_out_joined_0_buff_0 = external global [8 x bfloat]
@Co_4_buff_1 = external global [2 x bfloat]
@Co_4_buff_0 = external global [2 x bfloat]
@Co_5_buff_1 = external global [2 x bfloat]
@Co_5_buff_0 = external global [2 x bfloat]
@Co_6_buff_1 = external global [2 x bfloat]
@Co_6_buff_0 = external global [2 x bfloat]
@Co_7_buff_1 = external global [2 x bfloat]
@Co_7_buff_0 = external global [2 x bfloat]
@o_out_joined_1_buff_1 = external global [8 x bfloat]
@o_out_joined_1_buff_0 = external global [8 x bfloat]
@Cqkv_0_buff_1 = external global [2 x bfloat]
@Cqkv_0_buff_0 = external global [2 x bfloat]
@Cqkv_1_buff_1 = external global [2 x bfloat]
@Cqkv_1_buff_0 = external global [2 x bfloat]
@Cqkv_2_buff_1 = external global [2 x bfloat]
@Cqkv_2_buff_0 = external global [2 x bfloat]
@Cqkv_3_buff_1 = external global [2 x bfloat]
@Cqkv_3_buff_0 = external global [2 x bfloat]
@Cqkv_4_buff_1 = external global [2 x bfloat]
@Cqkv_4_buff_0 = external global [2 x bfloat]
@Cqkv_5_buff_1 = external global [2 x bfloat]
@Cqkv_5_buff_0 = external global [2 x bfloat]
@Cqkv_6_buff_1 = external global [2 x bfloat]
@Cqkv_6_buff_0 = external global [2 x bfloat]
@Cqkv_7_buff_1 = external global [2 x bfloat]
@Cqkv_7_buff_0 = external global [2 x bfloat]
@anm_in_cons_buff_2 = external global [2048 x bfloat]
@anm_in_cons_buff_1 = external global [2048 x bfloat]
@anm_in_cons_buff_0 = external global [2048 x bfloat]
@anm_mem_cons_buff_3 = external global [2048 x bfloat]
@anm_mem_cons_buff_2 = external global [2048 x bfloat]
@anm_mem_cons_buff_1 = external global [2048 x bfloat]
@anm_mem_cons_buff_0 = external global [2048 x bfloat]
@bo_L3L2_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_7_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_6_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_5_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_4_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_3_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_2_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_1_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_0_cons_buff_0 = external global [2048 x bfloat]
@bq_L3L2_buff_0 = external global [2048 x bfloat]
@bq_L3L2_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_6_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_5_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_4_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_3_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_2_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_1_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_0_cons_buff_0 = external global [2048 x bfloat]
@ffi_L3L2_buff_0 = external global [2048 x bfloat]
@ffi_L3L2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_6_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_5_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_4_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_3_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_1_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_0_cons_buff_0 = external global [2048 x bfloat]
@inter_0_buff_1 = external global [1024 x bfloat]
@inter_0_buff_0 = external global [1024 x bfloat]
@inter_1_buff_1 = external global [1024 x bfloat]
@inter_1_buff_0 = external global [1024 x bfloat]
@inter_2_buff_1 = external global [1024 x bfloat]
@inter_2_buff_0 = external global [1024 x bfloat]
@inter_3_buff_1 = external global [1024 x bfloat]
@inter_3_buff_0 = external global [1024 x bfloat]
@inter_4_buff_1 = external global [1024 x bfloat]
@inter_4_buff_0 = external global [1024 x bfloat]
@inter_5_buff_1 = external global [1024 x bfloat]
@inter_5_buff_0 = external global [1024 x bfloat]
@inter_6_buff_1 = external global [1024 x bfloat]
@inter_6_buff_0 = external global [1024 x bfloat]
@inter_7_buff_1 = external global [1024 x bfloat]
@inter_7_buff_0 = external global [1024 x bfloat]
@rms_in_cons_buff_2 = external global [2048 x bfloat]
@rms_in_cons_buff_1 = external global [2048 x bfloat]
@rms_in_cons_buff_0 = external global [2048 x bfloat]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @op0_layer_fused_pre_rms_col0_bf16(ptr, ptr, ptr, i32)

declare void @op0_layer_fused_qkv_gemv_static_bf16(i32, i32, ptr, ptr)

declare void @op0_layer_fused_qkv_gemv_bf16(i32, i32, ptr, ptr, ptr)

declare void @op0_layer_fused_o_proj_bf16(i32, i32, ptr, ptr, ptr)

declare void @op0_layer_fused_add_bf16(ptr, ptr, ptr, i32)

declare void @op0_layer_fused_rms_norm2_bf16(ptr, ptr, ptr, i32)

declare void @op0_layer_fused_gate_up_bf16(i32, i32, ptr, ptr, i32)

declare void @op0_layer_fused_silu_mul_bf16(ptr, i32)

declare void @op0_layer_fused_down_partial_bf16(i32, i32, ptr, ptr, ptr)

define void @core_5_5() {
  store i32 0, ptr @_anonymous29
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 8)
  br label %1

1:                                                ; preds = %45, %0
  %2 = phi i64 [ %46, %45 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %47

4:                                                ; preds = %38, %1
  %5 = phi i64 [ %44, %38 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %45

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 5, i32 -1)
  %8 = load i32, ptr @_anonymous29
  switch i32 %8, label %9 [
    i32 0, label %48
    i32 1, label %50
  ]

9:                                                ; preds = %48, %50, %7
  %10 = phi ptr [ %51, %50 ], [ %49, %48 ], [ @inter_5_buff_0, %7 ]
  %11 = getelementptr [1024 x bfloat], ptr %10, i32 0, i32 0
  br label %12

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %26, %12
  %14 = phi i64 [ %37, %26 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 1024
  br i1 %15, label %16, label %38

16:                                               ; preds = %13
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 4)
  switch i32 %17, label %18 [
    i32 0, label %52
    i32 1, label %54
  ]

18:                                               ; preds = %52, %54, %16
  %19 = phi ptr [ %55, %54 ], [ %53, %52 ], [ @Adp_5_cons_buff_0, %16 ]
  %20 = getelementptr [1152 x i8], ptr %19, i32 0, i32 0
  br label %21

21:                                               ; preds = %18
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %22 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 8)
  switch i32 %22, label %23 [
    i32 0, label %56
    i32 1, label %58
  ]

23:                                               ; preds = %56, %58, %21
  %24 = phi ptr [ %59, %58 ], [ %57, %56 ], [ @Cdp_5_buff_0, %21 ]
  %25 = getelementptr [2 x bfloat], ptr %24, i32 0, i32 0
  br label %26

26:                                               ; preds = %23
  call void @op0_layer_fused_down_partial_bf16(i32 2, i32 0, ptr %20, ptr %11, ptr %25)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 4)
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 2
  %30 = add i32 %27, -1
  %31 = select i1 %29, i32 %30, i32 %28
  store i32 %31, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 4)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %32 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 8)
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds (i8, ptr @_anonymous29, i64 8)
  %37 = add i64 %14, 1
  br label %13

38:                                               ; preds = %13
  call void @llvm.aie2p.release(i32 4, i32 1)
  %39 = load i32, ptr @_anonymous29
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 2
  %42 = add i32 %39, -1
  %43 = select i1 %41, i32 %42, i32 %40
  store i32 %43, ptr @_anonymous29
  %44 = add i64 %5, 1
  br label %4

45:                                               ; preds = %4
  %46 = add i64 %2, 1
  br label %1

47:                                               ; preds = %1
  ret void

48:                                               ; preds = %7
  %49 = phi ptr [ @inter_5_buff_0, %7 ]
  br label %9

50:                                               ; preds = %7
  %51 = phi ptr [ @inter_5_buff_1, %7 ]
  br label %9

52:                                               ; preds = %16
  %53 = phi ptr [ @Adp_5_cons_buff_0, %16 ]
  br label %18

54:                                               ; preds = %16
  %55 = phi ptr [ @Adp_5_cons_buff_1, %16 ]
  br label %18

56:                                               ; preds = %21
  %57 = phi ptr [ @Cdp_5_buff_0, %21 ]
  br label %23

58:                                               ; preds = %21
  %59 = phi ptr [ @Cdp_5_buff_1, %21 ]
  br label %23
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
