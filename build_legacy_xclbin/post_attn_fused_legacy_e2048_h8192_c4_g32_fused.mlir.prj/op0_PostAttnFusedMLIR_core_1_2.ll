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

define void @core_1_2() {
  store i32 0, ptr @_anonymous4, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 12), align 4
  br label %1

1:                                                ; preds = %48, %0
  %2 = phi i64 [ %49, %48 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %50

4:                                                ; preds = %33, %1
  %5 = phi i64 [ %47, %33 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %48

7:                                                ; preds = %16, %4
  %8 = phi i64 [ %22, %16 ], [ 0, %4 ]
  %9 = icmp slt i64 %8, 256
  br i1 %9, label %10, label %23

10:                                               ; preds = %7
  %11 = trunc i64 %8 to i32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = load i32, ptr @_anonymous4, align 4
  switch i32 %12, label %13 [
    i32 0, label %51
    i32 1, label %53
  ]

13:                                               ; preds = %51, %53, %10
  %14 = phi ptr [ %54, %53 ], [ %52, %51 ], [ @o_out_joined_0_cons_buff_0, %10 ]
  %15 = getelementptr [8 x bfloat], ptr %14, i32 0, i32 0
  br label %16

16:                                               ; preds = %13
  call void @op0_post_attn_o_out_assemble_bf16(ptr %15, ptr @anm_oout_buf, i32 %11, i32 0, i32 4, i32 2, i32 512)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = load i32, ptr @_anonymous4, align 4
  %18 = add i32 %17, 1
  %19 = icmp sge i32 %18, 2
  %20 = add i32 %17, -1
  %21 = select i1 %19, i32 %20, i32 %18
  store i32 %21, ptr @_anonymous4, align 4
  %22 = add i64 %8, 1
  br label %7

23:                                               ; preds = %7
  call void @llvm.aie2p.acquire(i32 53, i32 -2)
  %24 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %24, label %25 [
    i32 0, label %55
    i32 1, label %57
    i32 2, label %59
  ]

25:                                               ; preds = %55, %57, %59, %23
  %26 = phi ptr [ %60, %59 ], [ %58, %57 ], [ %56, %55 ], [ @anm_in_cons_buff_0, %23 ]
  %27 = getelementptr [2048 x bfloat], ptr %26, i32 0, i32 0
  br label %28

28:                                               ; preds = %25
  %29 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  switch i32 %29, label %30 [
    i32 0, label %61
    i32 1, label %63
    i32 2, label %65
  ]

30:                                               ; preds = %61, %63, %65, %28
  %31 = phi ptr [ %66, %65 ], [ %64, %63 ], [ %62, %61 ], [ @anm_in_cons_buff_1, %28 ]
  %32 = getelementptr [2048 x bfloat], ptr %31, i32 0, i32 0
  br label %33

33:                                               ; preds = %30
  call void @llvm.aie2p.acquire(i32 54, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @op0_post_attn_add_bf16(ptr @anm_oout_buf, ptr %27, ptr @anm_inpff_buff_0, i32 2048)
  call void @op0_post_attn_rms_norm_bf16(ptr @anm_inpff_buff_0, ptr %32, ptr @anm_ffi_l1l2_buff_0, i32 2048)
  call void @llvm.aie2p.release(i32 52, i32 2)
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  %35 = add i32 %34, 2
  %36 = icmp sge i32 %35, 3
  %37 = add i32 %34, -1
  %38 = select i1 %36, i32 %37, i32 %35
  store i32 %38, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 4), align 4
  call void @llvm.aie2p.release(i32 55, i32 1)
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 1
  %42 = select i1 %41, i32 %39, i32 %40
  store i32 %42, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 8), align 4
  call void @llvm.aie2p.release(i32 51, i32 1)
  %43 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 12), align 4
  %44 = add i32 %43, 1
  %45 = icmp sge i32 %44, 1
  %46 = select i1 %45, i32 %43, i32 %44
  store i32 %46, ptr getelementptr inbounds nuw (i8, ptr @_anonymous4, i64 12), align 4
  %47 = add i64 %5, 1
  br label %4

48:                                               ; preds = %4
  %49 = add i64 %2, 1
  br label %1

50:                                               ; preds = %1
  ret void

51:                                               ; preds = %10
  %52 = phi ptr [ @o_out_joined_0_cons_buff_0, %10 ]
  br label %13

53:                                               ; preds = %10
  %54 = phi ptr [ @o_out_joined_0_cons_buff_1, %10 ]
  br label %13

55:                                               ; preds = %23
  %56 = phi ptr [ @anm_in_cons_buff_0, %23 ]
  br label %25

57:                                               ; preds = %23
  %58 = phi ptr [ @anm_in_cons_buff_1, %23 ]
  br label %25

59:                                               ; preds = %23
  %60 = phi ptr [ @anm_in_cons_buff_2, %23 ]
  br label %25

61:                                               ; preds = %28
  %62 = phi ptr [ @anm_in_cons_buff_1, %28 ]
  br label %30

63:                                               ; preds = %28
  %64 = phi ptr [ @anm_in_cons_buff_2, %28 ]
  br label %30

65:                                               ; preds = %28
  %66 = phi ptr [ @anm_in_cons_buff_0, %28 ]
  br label %30
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
