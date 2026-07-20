; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous16 = external global [3 x i32]
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
@anm_oout_buf = external global [2048 x bfloat]
@Adp_0_cons_buff_1 = external global [2304 x i8]
@Adp_0_cons_buff_0 = external global [2304 x i8]
@Ffn_src_0_cons_buff_1 = external global [13824 x i8]
@Ffn_src_0_cons_buff_0 = external global [13824 x i8]
@Agu_0_cons_buff_1 = external global [4608 x i8]
@Agu_0_cons_buff_0 = external global [4608 x i8]
@Agu_1_cons_buff_1 = external global [4608 x i8]
@Agu_1_cons_buff_0 = external global [4608 x i8]
@Adp_1_cons_buff_1 = external global [2304 x i8]
@Adp_1_cons_buff_0 = external global [2304 x i8]
@Adp_2_cons_buff_1 = external global [2304 x i8]
@Adp_2_cons_buff_0 = external global [2304 x i8]
@Ffn_src_1_cons_buff_1 = external global [13824 x i8]
@Ffn_src_1_cons_buff_0 = external global [13824 x i8]
@Agu_2_cons_buff_1 = external global [4608 x i8]
@Agu_2_cons_buff_0 = external global [4608 x i8]
@Agu_3_cons_buff_1 = external global [4608 x i8]
@Agu_3_cons_buff_0 = external global [4608 x i8]
@Adp_3_cons_buff_1 = external global [2304 x i8]
@Adp_3_cons_buff_0 = external global [2304 x i8]
@Ao_0_cons_buff_1 = external global [2304 x i8]
@Ao_0_cons_buff_0 = external global [2304 x i8]
@Ao_src_0_cons_buff_1 = external global [4608 x i8]
@Ao_src_0_cons_buff_0 = external global [4608 x i8]
@Ao_1_cons_buff_1 = external global [2304 x i8]
@Ao_1_cons_buff_0 = external global [2304 x i8]
@Ao_2_cons_buff_1 = external global [2304 x i8]
@Ao_2_cons_buff_0 = external global [2304 x i8]
@Ao_src_1_cons_buff_1 = external global [4608 x i8]
@Ao_src_1_cons_buff_0 = external global [4608 x i8]
@Ao_3_cons_buff_1 = external global [2304 x i8]
@Ao_3_cons_buff_0 = external global [2304 x i8]
@Aqkv_0_cons_buff_1 = external global [2304 x i8]
@Aqkv_0_cons_buff_0 = external global [2304 x i8]
@Aqkv_src_0_cons_buff_1 = external global [4608 x i8]
@Aqkv_src_0_cons_buff_0 = external global [4608 x i8]
@Aqkv_1_cons_buff_1 = external global [2304 x i8]
@Aqkv_1_cons_buff_0 = external global [2304 x i8]
@Aqkv_2_cons_buff_1 = external global [2304 x i8]
@Aqkv_2_cons_buff_0 = external global [2304 x i8]
@Aqkv_src_1_cons_buff_1 = external global [4608 x i8]
@Aqkv_src_1_cons_buff_0 = external global [4608 x i8]
@Aqkv_3_cons_buff_1 = external global [2304 x i8]
@Aqkv_3_cons_buff_0 = external global [2304 x i8]
@Cdp_0_buff_1 = external global [2 x bfloat]
@Cdp_0_buff_0 = external global [2 x bfloat]
@Cdp_1_buff_1 = external global [2 x bfloat]
@Cdp_1_buff_0 = external global [2 x bfloat]
@ffn_out_joined_0_buff_1 = external global [4 x bfloat]
@ffn_out_joined_0_buff_0 = external global [4 x bfloat]
@Cdp_2_buff_1 = external global [2 x bfloat]
@Cdp_2_buff_0 = external global [2 x bfloat]
@Cdp_3_buff_1 = external global [2 x bfloat]
@Cdp_3_buff_0 = external global [2 x bfloat]
@ffn_out_joined_1_buff_1 = external global [4 x bfloat]
@ffn_out_joined_1_buff_0 = external global [4 x bfloat]
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
@o_out_joined_0_cons_buff_1 = external global [8 x bfloat]
@o_out_joined_0_cons_buff_0 = external global [8 x bfloat]
@Cqkv_0_buff_1 = external global [2 x bfloat]
@Cqkv_0_buff_0 = external global [2 x bfloat]
@Cqkv_1_buff_1 = external global [2 x bfloat]
@Cqkv_1_buff_0 = external global [2 x bfloat]
@Cqkv_2_buff_1 = external global [2 x bfloat]
@Cqkv_2_buff_0 = external global [2 x bfloat]
@Cqkv_3_buff_1 = external global [2 x bfloat]
@Cqkv_3_buff_0 = external global [2 x bfloat]
@anm_in_cons_buff_1 = external global [2048 x bfloat]
@anm_in_cons_buff_0 = external global [2048 x bfloat]
@anm_mem_cons_buff_2 = external global [2048 x bfloat]
@anm_mem_cons_buff_1 = external global [2048 x bfloat]
@anm_mem_cons_buff_0 = external global [2048 x bfloat]
@bo_L3L2_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_3_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_2_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_1_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_0_cons_buff_0 = external global [2048 x bfloat]
@bq_L3L2_buff_0 = external global [2048 x bfloat]
@bq_L3L2_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_2_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_1_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_0_cons_buff_0 = external global [2048 x bfloat]
@ffi_L3L2_buff_0 = external global [2048 x bfloat]
@ffi_L3L2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_3_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_1_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_0_cons_buff_0 = external global [2048 x bfloat]
@inter_0_buff_1 = external global [2048 x bfloat]
@inter_0_buff_0 = external global [2048 x bfloat]
@inter_1_buff_1 = external global [2048 x bfloat]
@inter_1_buff_0 = external global [2048 x bfloat]
@inter_2_buff_1 = external global [2048 x bfloat]
@inter_2_buff_0 = external global [2048 x bfloat]
@inter_3_buff_1 = external global [2048 x bfloat]
@inter_3_buff_0 = external global [2048 x bfloat]
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

declare void @op0_layer_fused_gate_up_bf16(i32, i32, ptr, ptr, i32)

declare void @op0_layer_fused_silu_mul_bf16(ptr, i32)

declare void @op0_layer_fused_down_partial_bf16(i32, i32, ptr, ptr, ptr)

declare void @op0_layer_fused_o_out_assemble_bf16(ptr, ptr, i32, i32, i32, i32, i32)

declare void @op0_layer_fused_add_bf16(ptr, ptr, ptr, i32)

declare void @op0_layer_fused_rms_norm2_bf16(ptr, ptr, ptr, i32)

define void @core_3_4() {
  store i32 0, ptr @_anonymous11, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 8), align 4
  br label %1

1:                                                ; preds = %49, %0
  %2 = phi i64 [ %50, %49 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %51

4:                                                ; preds = %38, %1
  %5 = phi i64 [ %48, %38 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %49

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  br label %8

8:                                                ; preds = %26, %7
  %9 = phi i64 [ %32, %26 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 512
  br i1 %10, label %11, label %33

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 4), align 4
  switch i32 %12, label %13 [
    i32 0, label %52
    i32 1, label %54
  ]

13:                                               ; preds = %52, %54, %11
  %14 = phi ptr [ %55, %54 ], [ %53, %52 ], [ @Agu_3_cons_buff_0, %11 ]
  %15 = getelementptr [4608 x i8], ptr %14, i32 0, i32 0
  br label %16

16:                                               ; preds = %13
  call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr %15, ptr @ffi_mem_3_cons_buff_0, i32 0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 4), align 4
  %18 = add i32 %17, 1
  %19 = icmp sge i32 %18, 2
  %20 = add i32 %17, -1
  %21 = select i1 %19, i32 %20, i32 %18
  store i32 %21, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 4), align 4
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %22 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 4), align 4
  switch i32 %22, label %23 [
    i32 0, label %56
    i32 1, label %58
  ]

23:                                               ; preds = %56, %58, %16
  %24 = phi ptr [ %59, %58 ], [ %57, %56 ], [ @Agu_3_cons_buff_0, %16 ]
  %25 = getelementptr [4608 x i8], ptr %24, i32 0, i32 0
  br label %26

26:                                               ; preds = %23
  call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr %25, ptr @ffi_mem_3_cons_buff_0, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 2
  %30 = add i32 %27, -1
  %31 = select i1 %29, i32 %30, i32 %28
  store i32 %31, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 4), align 4
  %32 = add i64 %9, 1
  br label %8

33:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 8), align 4
  switch i32 %34, label %35 [
    i32 0, label %60
    i32 1, label %62
  ]

35:                                               ; preds = %60, %62, %33
  %36 = phi ptr [ %63, %62 ], [ %61, %60 ], [ @inter_3_buff_0, %33 ]
  %37 = getelementptr [2048 x bfloat], ptr %36, i32 0, i32 0
  br label %38

38:                                               ; preds = %35
  call void @op0_layer_fused_silu_mul_bf16(ptr %37, i32 2048)
  call void @llvm.aie2p.release(i32 53, i32 1)
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 8), align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 2
  %42 = add i32 %39, -1
  %43 = select i1 %41, i32 %42, i32 %40
  store i32 %43, ptr getelementptr inbounds nuw (i8, ptr @_anonymous11, i64 8), align 4
  call void @llvm.aie2p.release(i32 50, i32 1)
  %44 = load i32, ptr @_anonymous11, align 4
  %45 = add i32 %44, 1
  %46 = icmp sge i32 %45, 1
  %47 = select i1 %46, i32 %44, i32 %45
  store i32 %47, ptr @_anonymous11, align 4
  %48 = add i64 %5, 1
  br label %4

49:                                               ; preds = %4
  %50 = add i64 %2, 1
  br label %1

51:                                               ; preds = %1
  ret void

52:                                               ; preds = %11
  %53 = phi ptr [ @Agu_3_cons_buff_0, %11 ]
  br label %13

54:                                               ; preds = %11
  %55 = phi ptr [ @Agu_3_cons_buff_1, %11 ]
  br label %13

56:                                               ; preds = %16
  %57 = phi ptr [ @Agu_3_cons_buff_0, %16 ]
  br label %23

58:                                               ; preds = %16
  %59 = phi ptr [ @Agu_3_cons_buff_1, %16 ]
  br label %23

60:                                               ; preds = %33
  %61 = phi ptr [ @inter_3_buff_0, %33 ]
  br label %35

62:                                               ; preds = %33
  %63 = phi ptr [ @inter_3_buff_1, %33 ]
  br label %35
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
