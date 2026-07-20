; ModuleID = 'post_attn_fused_legacy_e2048_h8192_c4_g32_fused.mlir.prj\op0_PostAttnFusedMLIR_core_1_3.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous5 = external local_unnamed_addr global [3 x i32]
@Agu_0_cons_buff_0 = external global [9216 x i8]
@Cgu_0_buff_1 = external global [8 x bfloat]
@Cgu_0_buff_0 = external global [8 x bfloat]
@ffi_mem_0_cons_buff_0 = external global [2048 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_dual_fused_dequant_gemv_bf16(i32, i32, ptr, ptr, i32) local_unnamed_addr

declare void @op0_dual_fused_dequant_gemv_silu_mul_bf16(ptr, i32) local_unnamed_addr

define void @core_1_3() local_unnamed_addr {
  store i32 0, ptr @_anonymous5, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %44
  %1 = phi i64 [ 0, %0 ], [ %45, %44 ]
  br label %2

2:                                                ; preds = %.preheader, %37
  %3 = phi i64 [ 0, %.preheader ], [ %42, %37 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %35, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op0_dual_fused_dequant_gemv_bf16(i32 8, i32 0, ptr nonnull @Agu_0_cons_buff_0, ptr nonnull @ffi_mem_0_cons_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  %7 = icmp ugt i32 %6, 2147483646
  %8 = zext i1 %7 to i32
  %9 = add i32 %6, %8
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op0_dual_fused_dequant_gemv_bf16(i32 8, i32 0, ptr nonnull @Agu_0_cons_buff_0, ptr nonnull @ffi_mem_0_cons_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %10 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  %11 = icmp ugt i32 %10, 2147483646
  %12 = zext i1 %11 to i32
  %13 = add i32 %10, %12
  store i32 %13, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %14 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 8), align 4
  %cond = icmp eq i32 %14, 1
  %spec.select = select i1 %cond, ptr @Cgu_0_buff_1, ptr @Cgu_0_buff_0
  tail call void @op0_dual_fused_dequant_gemv_silu_mul_bf16(ptr nonnull %spec.select, i32 8)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 8), align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 8), align 4
  %20 = or disjoint i64 %5, 1
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op0_dual_fused_dequant_gemv_bf16(i32 8, i32 0, ptr nonnull @Agu_0_cons_buff_0, ptr nonnull @ffi_mem_0_cons_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  %22 = icmp ugt i32 %21, 2147483646
  %23 = zext i1 %22 to i32
  %24 = add i32 %21, %23
  store i32 %24, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op0_dual_fused_dequant_gemv_bf16(i32 8, i32 0, ptr nonnull @Agu_0_cons_buff_0, ptr nonnull @ffi_mem_0_cons_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  %26 = icmp ugt i32 %25, 2147483646
  %27 = zext i1 %26 to i32
  %28 = add i32 %25, %27
  store i32 %28, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %29 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 8), align 4
  %cond.1 = icmp eq i32 %29, 1
  %spec.select.1 = select i1 %cond.1, ptr @Cgu_0_buff_1, ptr @Cgu_0_buff_0
  tail call void @op0_dual_fused_dequant_gemv_silu_mul_bf16(ptr nonnull %spec.select.1, i32 8)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 8), align 4
  %31 = add i32 %30, 1
  %32 = icmp sgt i32 %31, 1
  %33 = add i32 %30, -1
  %34 = select i1 %32, i32 %33, i32 %31
  store i32 %34, ptr getelementptr inbounds nuw (i8, ptr @_anonymous5, i20 8), align 4
  %35 = add nuw nsw i64 %5, 2
  %36 = icmp samesign ult i64 %20, 255
  br i1 %36, label %4, label %37

37:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %38 = load i32, ptr @_anonymous5, align 4
  %39 = icmp ugt i32 %38, 2147483646
  %40 = zext i1 %39 to i32
  %41 = add i32 %38, %40
  store i32 %41, ptr @_anonymous5, align 4
  %42 = add nuw nsw i64 %3, 1
  %43 = icmp samesign ult i64 %3, 4294967294
  br i1 %43, label %2, label %44

44:                                               ; preds = %37
  %45 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %45, 9223372036854775807
  br i1 %.not, label %46, label %.preheader

46:                                               ; preds = %44
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
