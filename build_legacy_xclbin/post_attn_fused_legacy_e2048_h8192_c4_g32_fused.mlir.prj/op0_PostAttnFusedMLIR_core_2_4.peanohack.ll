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

define void @core_2_4() {
  store i32 0, ptr @_anonymous10
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous10, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous10, i64 8)
  br label %1

1:                                                ; preds = %33, %0
  %2 = phi i64 [ %34, %33 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %35

4:                                                ; preds = %27, %1
  %5 = phi i64 [ %32, %27 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %33

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  br label %8

8:                                                ; preds = %16, %7
  %9 = phi i64 [ %26, %16 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 256
  br i1 %10, label %11, label %27

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %12 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous10, i64 8)
  switch i32 %12, label %13 [
    i32 0, label %36
    i32 1, label %38
  ]

13:                                               ; preds = %36, %38, %11
  %14 = phi ptr [ %39, %38 ], [ %37, %36 ], [ @Cd_1_buff_0, %11 ]
  %15 = getelementptr [2 x bfloat], ptr %14, i32 0, i32 0
  br label %16

16:                                               ; preds = %13
  call void @op0_fused_dequant_matvec_down_bf16(i32 2, i32 0, ptr @Ad_1_cons_buff_0, ptr @silu_mem_1_cons_buff_0, ptr %15)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous10, i64 4)
  %18 = add i32 %17, 1
  %19 = icmp sge i32 %18, 1
  %20 = select i1 %19, i32 %17, i32 %18
  store i32 %20, ptr getelementptr inbounds (i8, ptr @_anonymous10, i64 4)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous10, i64 8)
  %22 = add i32 %21, 1
  %23 = icmp sge i32 %22, 2
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds (i8, ptr @_anonymous10, i64 8)
  %26 = add i64 %9, 1
  br label %8

27:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 52, i32 1)
  %28 = load i32, ptr @_anonymous10
  %29 = add i32 %28, 1
  %30 = icmp sge i32 %29, 1
  %31 = select i1 %30, i32 %28, i32 %29
  store i32 %31, ptr @_anonymous10
  %32 = add i64 %5, 1
  br label %4

33:                                               ; preds = %4
  %34 = add i64 %2, 1
  br label %1

35:                                               ; preds = %1
  ret void

36:                                               ; preds = %11
  %37 = phi ptr [ @Cd_1_buff_0, %11 ]
  br label %13

38:                                               ; preds = %11
  %39 = phi ptr [ @Cd_1_buff_1, %11 ]
  br label %13
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
