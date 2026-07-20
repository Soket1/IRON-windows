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

define void @core_4_2() {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  br label %1

1:                                                ; preds = %89, %0
  %2 = phi i64 [ %90, %89 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %91

4:                                                ; preds = %83, %1
  %5 = phi i64 [ %88, %83 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %89

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  br label %8

8:                                                ; preds = %21, %7
  %9 = phi i64 [ %32, %21 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 128
  br i1 %10, label %11, label %33

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %92
    i32 1, label %94
  ]

13:                                               ; preds = %92, %94, %11
  %14 = phi ptr [ %95, %94 ], [ %93, %92 ], [ @Aqkv_4_cons_buff_0, %11 ]
  %15 = getelementptr [2304 x i8], ptr %14, i32 0, i32 0
  br label %16

16:                                               ; preds = %13
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %17 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %17, label %18 [
    i32 0, label %96
    i32 1, label %98
  ]

18:                                               ; preds = %96, %98, %16
  %19 = phi ptr [ %99, %98 ], [ %97, %96 ], [ @Cqkv_4_buff_0, %16 ]
  %20 = getelementptr [2 x bfloat], ptr %19, i32 0, i32 0
  br label %21

21:                                               ; preds = %18
  call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr %15, ptr @bq_mem_3_cons_buff_0, ptr %20)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %22 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  %23 = add i32 %22, 1
  %24 = icmp sge i32 %23, 2
  %25 = add i32 %22, -1
  %26 = select i1 %24, i32 %25, i32 %23
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  call void @llvm.aie2p.release(i32 51, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 2
  %30 = add i32 %27, -1
  %31 = select i1 %29, i32 %30, i32 %28
  store i32 %31, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %32 = add i64 %9, 1
  br label %8

33:                                               ; preds = %46, %8
  %34 = phi i64 [ %57, %46 ], [ 0, %8 ]
  %35 = icmp slt i64 %34, 32
  br i1 %35, label %36, label %58

36:                                               ; preds = %33
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %37, label %38 [
    i32 0, label %100
    i32 1, label %102
  ]

38:                                               ; preds = %100, %102, %36
  %39 = phi ptr [ %103, %102 ], [ %101, %100 ], [ @Aqkv_4_cons_buff_0, %36 ]
  %40 = getelementptr [2304 x i8], ptr %39, i32 0, i32 0
  br label %41

41:                                               ; preds = %38
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %42 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %42, label %43 [
    i32 0, label %104
    i32 1, label %106
  ]

43:                                               ; preds = %104, %106, %41
  %44 = phi ptr [ %107, %106 ], [ %105, %104 ], [ @Cqkv_4_buff_0, %41 ]
  %45 = getelementptr [2 x bfloat], ptr %44, i32 0, i32 0
  br label %46

46:                                               ; preds = %43
  call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr %40, ptr @bq_mem_3_cons_buff_0, ptr %45)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  %48 = add i32 %47, 1
  %49 = icmp sge i32 %48, 2
  %50 = add i32 %47, -1
  %51 = select i1 %49, i32 %50, i32 %48
  store i32 %51, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  call void @llvm.aie2p.release(i32 51, i32 1)
  %52 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %53 = add i32 %52, 1
  %54 = icmp sge i32 %53, 2
  %55 = add i32 %52, -1
  %56 = select i1 %54, i32 %55, i32 %53
  store i32 %56, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %57 = add i64 %34, 1
  br label %33

58:                                               ; preds = %71, %33
  %59 = phi i64 [ %82, %71 ], [ 0, %33 ]
  %60 = icmp slt i64 %59, 32
  br i1 %60, label %61, label %83

61:                                               ; preds = %58
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %62 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %62, label %63 [
    i32 0, label %108
    i32 1, label %110
  ]

63:                                               ; preds = %108, %110, %61
  %64 = phi ptr [ %111, %110 ], [ %109, %108 ], [ @Aqkv_4_cons_buff_0, %61 ]
  %65 = getelementptr [2304 x i8], ptr %64, i32 0, i32 0
  br label %66

66:                                               ; preds = %63
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %67 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  switch i32 %67, label %68 [
    i32 0, label %112
    i32 1, label %114
  ]

68:                                               ; preds = %112, %114, %66
  %69 = phi ptr [ %115, %114 ], [ %113, %112 ], [ @Cqkv_4_buff_0, %66 ]
  %70 = getelementptr [2 x bfloat], ptr %69, i32 0, i32 0
  br label %71

71:                                               ; preds = %68
  call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr %65, ptr @bq_mem_3_cons_buff_0, ptr %70)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %72 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  %73 = add i32 %72, 1
  %74 = icmp sge i32 %73, 2
  %75 = add i32 %72, -1
  %76 = select i1 %74, i32 %75, i32 %73
  store i32 %76, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  call void @llvm.aie2p.release(i32 51, i32 1)
  %77 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %78 = add i32 %77, 1
  %79 = icmp sge i32 %78, 2
  %80 = add i32 %77, -1
  %81 = select i1 %79, i32 %80, i32 %78
  store i32 %81, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %82 = add i64 %59, 1
  br label %58

83:                                               ; preds = %58
  call void @llvm.aie2p.release(i32 52, i32 1)
  %84 = load i32, ptr @_anonymous4, align 4
  %85 = add i32 %84, 1
  %86 = icmp sge i32 %85, 1
  %87 = select i1 %86, i32 %84, i32 %85
  store i32 %87, ptr @_anonymous4, align 4
  %88 = add i64 %5, 1
  br label %4

89:                                               ; preds = %4
  %90 = add i64 %2, 1
  br label %1

91:                                               ; preds = %1
  ret void

92:                                               ; preds = %11
  %93 = phi ptr [ @Aqkv_4_cons_buff_0, %11 ]
  br label %13

94:                                               ; preds = %11
  %95 = phi ptr [ @Aqkv_4_cons_buff_1, %11 ]
  br label %13

96:                                               ; preds = %16
  %97 = phi ptr [ @Cqkv_4_buff_0, %16 ]
  br label %18

98:                                               ; preds = %16
  %99 = phi ptr [ @Cqkv_4_buff_1, %16 ]
  br label %18

100:                                              ; preds = %36
  %101 = phi ptr [ @Aqkv_4_cons_buff_0, %36 ]
  br label %38

102:                                              ; preds = %36
  %103 = phi ptr [ @Aqkv_4_cons_buff_1, %36 ]
  br label %38

104:                                              ; preds = %41
  %105 = phi ptr [ @Cqkv_4_buff_0, %41 ]
  br label %43

106:                                              ; preds = %41
  %107 = phi ptr [ @Cqkv_4_buff_1, %41 ]
  br label %43

108:                                              ; preds = %61
  %109 = phi ptr [ @Aqkv_4_cons_buff_0, %61 ]
  br label %63

110:                                              ; preds = %61
  %111 = phi ptr [ @Aqkv_4_cons_buff_1, %61 ]
  br label %63

112:                                              ; preds = %66
  %113 = phi ptr [ @Cqkv_4_buff_0, %66 ]
  br label %68

114:                                              ; preds = %66
  %115 = phi ptr [ @Cqkv_4_buff_1, %66 ]
  br label %68
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
