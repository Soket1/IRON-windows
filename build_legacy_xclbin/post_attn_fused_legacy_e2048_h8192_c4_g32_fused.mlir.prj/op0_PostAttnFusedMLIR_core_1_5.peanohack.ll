; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous12 = external global [3 x i32]
@_anonymous11 = external global [3 x i32]
@_anonymous10 = external global [3 x i32]
@_anonymous9 = external global [3 x i32]
@_anonymous8 = external global [3 x i32]
@_anonymous7 = external global [3 x i32]
@_anonymous6 = external global [3 x i32]
@_anonymous5 = external global [3 x i32]
@_anonymous4 = external global [4 x i32]
@_anonymous3 = external global [3 x i32]
@_anonymous2 = external global [3 x i32]
@_anonymous1 = external global [3 x i32]
@_anonymous0 = external global [3 x i32]
@anm_oout_buf = external global [2048 x bfloat]
@Ad_0_cons_buff_0 = external global [9216 x i8]
@Ad_1_cons_buff_0 = external global [9216 x i8]
@Ad_2_cons_buff_0 = external global [9216 x i8]
@Ad_3_cons_buff_0 = external global [9216 x i8]
@Agu_0_cons_buff_0 = external global [9216 x i8]
@Agu_1_cons_buff_0 = external global [9216 x i8]
@Agu_2_cons_buff_0 = external global [9216 x i8]
@Agu_3_cons_buff_0 = external global [9216 x i8]
@Ao_0_cons_buff_1 = external global [2304 x i8]
@Ao_0_cons_buff_0 = external global [2304 x i8]
@Ao_1_cons_buff_1 = external global [2304 x i8]
@Ao_1_cons_buff_0 = external global [2304 x i8]
@Ao_2_cons_buff_1 = external global [2304 x i8]
@Ao_2_cons_buff_0 = external global [2304 x i8]
@Ao_3_cons_buff_1 = external global [2304 x i8]
@Ao_3_cons_buff_0 = external global [2304 x i8]
@Cd_0_buff_1 = external global [2 x bfloat]
@Cd_0_buff_0 = external global [2 x bfloat]
@Cd_1_buff_1 = external global [2 x bfloat]
@Cd_1_buff_0 = external global [2 x bfloat]
@Cd_2_buff_1 = external global [2 x bfloat]
@Cd_2_buff_0 = external global [2 x bfloat]
@Cd_3_buff_1 = external global [2 x bfloat]
@Cd_3_buff_0 = external global [2 x bfloat]
@Cgu_0_buff_1 = external global [8 x bfloat]
@Cgu_0_buff_0 = external global [8 x bfloat]
@Cgu_1_buff_1 = external global [8 x bfloat]
@Cgu_1_buff_0 = external global [8 x bfloat]
@Cgu_2_buff_1 = external global [8 x bfloat]
@Cgu_2_buff_0 = external global [8 x bfloat]
@Cgu_3_buff_1 = external global [8 x bfloat]
@Cgu_3_buff_0 = external global [8 x bfloat]
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
@anm_ffi_l1l2_buff_0 = external global [2048 x bfloat]
@anm_ffi_l1l2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_3_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_1_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_0_cons_buff_0 = external global [2048 x bfloat]
@anm_in_cons_buff_2 = external global [2048 x bfloat]
@anm_in_cons_buff_1 = external global [2048 x bfloat]
@anm_in_cons_buff_0 = external global [2048 x bfloat]
@anm_inpff_buff_0 = external global [2048 x bfloat]
@kqv_L3L2_cons_buff_0 = external global [2048 x bfloat]
@kqv_mem_3_cons_buff_0 = external global [2048 x bfloat]
@kqv_mem_2_cons_buff_0 = external global [2048 x bfloat]
@kqv_mem_1_cons_buff_0 = external global [2048 x bfloat]
@kqv_mem_0_cons_buff_0 = external global [2048 x bfloat]
@silu_L3L2_cons_buff_0 = external global [8192 x bfloat]
@silu_mem_3_cons_buff_0 = external global [8192 x bfloat]
@silu_mem_2_cons_buff_0 = external global [8192 x bfloat]
@silu_mem_1_cons_buff_0 = external global [8192 x bfloat]
@silu_mem_0_cons_buff_0 = external global [8192 x bfloat]

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

declare void @op0_fused_dequant_matvec_v2_bf16(i32, i32, ptr, ptr, ptr)

declare void @op0_post_attn_o_out_assemble_bf16(ptr, ptr, i32, i32, i32, i32, i32)

declare void @op0_post_attn_add_bf16(ptr, ptr, ptr, i32)

declare void @op0_post_attn_rms_norm_bf16(ptr, ptr, ptr, i32)

declare void @op0_dual_fused_dequant_gemv_bf16(i32, i32, ptr, ptr, i32)

declare void @op0_dual_fused_dequant_gemv_silu_mul_bf16(ptr, i32)

declare void @op0_fused_dequant_matvec_down_bf16(i32, i32, ptr, ptr, ptr)

define void @core_1_5() {
  store i32 0, ptr @_anonymous7
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8)
  br label %1

1:                                                ; preds = %37, %0
  %2 = phi i64 [ %38, %37 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %39

4:                                                ; preds = %31, %1
  %5 = phi i64 [ %36, %31 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %37

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  br label %8

8:                                                ; preds = %24, %7
  %9 = phi i64 [ %30, %24 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 256
  br i1 %10, label %11, label %31

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_dual_fused_dequant_gemv_bf16(i32 8, i32 0, ptr @Agu_2_cons_buff_0, ptr @ffi_mem_2_cons_buff_0, i32 0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4)
  %13 = add i32 %12, 1
  %14 = icmp sge i32 %13, 1
  %15 = select i1 %14, i32 %12, i32 %13
  store i32 %15, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_dual_fused_dequant_gemv_bf16(i32 8, i32 0, ptr @Agu_2_cons_buff_0, ptr @ffi_mem_2_cons_buff_0, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %16 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4)
  %17 = add i32 %16, 1
  %18 = icmp sge i32 %17, 1
  %19 = select i1 %18, i32 %16, i32 %17
  store i32 %19, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 4)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %20 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8)
  switch i32 %20, label %21 [
    i32 0, label %40
    i32 1, label %42
  ]

21:                                               ; preds = %40, %42, %11
  %22 = phi ptr [ %43, %42 ], [ %41, %40 ], [ @Cgu_2_buff_0, %11 ]
  %23 = getelementptr [8 x bfloat], ptr %22, i32 0, i32 0
  br label %24

24:                                               ; preds = %21
  call void @op0_dual_fused_dequant_gemv_silu_mul_bf16(ptr %23, i32 8)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %25 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8)
  %26 = add i32 %25, 1
  %27 = icmp sge i32 %26, 2
  %28 = add i32 %25, -1
  %29 = select i1 %27, i32 %28, i32 %26
  store i32 %29, ptr getelementptr inbounds (i8, ptr @_anonymous7, i64 8)
  %30 = add i64 %9, 1
  br label %8

31:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 52, i32 1)
  %32 = load i32, ptr @_anonymous7
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 1
  %35 = select i1 %34, i32 %32, i32 %33
  store i32 %35, ptr @_anonymous7
  %36 = add i64 %5, 1
  br label %4

37:                                               ; preds = %4
  %38 = add i64 %2, 1
  br label %1

39:                                               ; preds = %1
  ret void

40:                                               ; preds = %11
  %41 = phi ptr [ @Cgu_2_buff_0, %11 ]
  br label %21

42:                                               ; preds = %11
  %43 = phi ptr [ @Cgu_2_buff_1, %11 ]
  br label %21
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
