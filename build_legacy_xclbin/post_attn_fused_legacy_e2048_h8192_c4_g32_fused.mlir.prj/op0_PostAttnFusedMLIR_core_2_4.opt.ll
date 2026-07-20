; ModuleID = 'post_attn_fused_legacy_e2048_h8192_c4_g32_fused.mlir.prj\op0_PostAttnFusedMLIR_core_2_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous10 = external local_unnamed_addr global [3 x i32]
@Ad_1_cons_buff_0 = external global [9216 x i8]
@Cd_1_buff_1 = external global [2 x bfloat]
@Cd_1_buff_0 = external global [2 x bfloat]
@silu_mem_1_cons_buff_0 = external global [8192 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_fused_dequant_matvec_down_bf16(i32, i32, ptr, ptr, ptr) local_unnamed_addr

define void @core_2_4() local_unnamed_addr {
  store i32 0, ptr @_anonymous10, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %56
  %1 = phi i64 [ 0, %0 ], [ %57, %56 ]
  br label %2

2:                                                ; preds = %.preheader, %49
  %3 = phi i64 [ 0, %.preheader ], [ %54, %49 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %47, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %cond = icmp eq i32 %6, 1
  %spec.select = select i1 %cond, ptr @Cd_1_buff_1, ptr @Cd_1_buff_0
  tail call void @op0_fused_dequant_matvec_down_bf16(i32 2, i32 0, ptr nonnull @Ad_1_cons_buff_0, ptr nonnull @silu_mem_1_cons_buff_0, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  %8 = icmp ugt i32 %7, 2147483646
  %9 = zext i1 %8 to i32
  %10 = add i32 %7, %9
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %11 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %12 = add i32 %11, 1
  %13 = icmp sgt i32 %12, 1
  %14 = add i32 %11, -1
  %15 = select i1 %13, i32 %14, i32 %12
  store i32 %15, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %cond.1 = icmp eq i32 %16, 1
  %spec.select.1 = select i1 %cond.1, ptr @Cd_1_buff_1, ptr @Cd_1_buff_0
  tail call void @op0_fused_dequant_matvec_down_bf16(i32 2, i32 0, ptr nonnull @Ad_1_cons_buff_0, ptr nonnull @silu_mem_1_cons_buff_0, ptr nonnull %spec.select.1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  %18 = icmp ugt i32 %17, 2147483646
  %19 = zext i1 %18 to i32
  %20 = add i32 %17, %19
  store i32 %20, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sgt i32 %22, 1
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %26 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %cond.2 = icmp eq i32 %26, 1
  %spec.select.2 = select i1 %cond.2, ptr @Cd_1_buff_1, ptr @Cd_1_buff_0
  tail call void @op0_fused_dequant_matvec_down_bf16(i32 2, i32 0, ptr nonnull @Ad_1_cons_buff_0, ptr nonnull @silu_mem_1_cons_buff_0, ptr nonnull %spec.select.2)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  %28 = icmp ugt i32 %27, 2147483646
  %29 = zext i1 %28 to i32
  %30 = add i32 %27, %29
  store i32 %30, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %32 = add i32 %31, 1
  %33 = icmp sgt i32 %32, 1
  %34 = add i32 %31, -1
  %35 = select i1 %33, i32 %34, i32 %32
  store i32 %35, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %36 = or disjoint i64 %5, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %cond.3 = icmp eq i32 %37, 1
  %spec.select.3 = select i1 %cond.3, ptr @Cd_1_buff_1, ptr @Cd_1_buff_0
  tail call void @op0_fused_dequant_matvec_down_bf16(i32 2, i32 0, ptr nonnull @Ad_1_cons_buff_0, ptr nonnull @silu_mem_1_cons_buff_0, ptr nonnull %spec.select.3)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  %39 = icmp ugt i32 %38, 2147483646
  %40 = zext i1 %39 to i32
  %41 = add i32 %38, %40
  store i32 %41, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %42 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %43 = add i32 %42, 1
  %44 = icmp sgt i32 %43, 1
  %45 = add i32 %42, -1
  %46 = select i1 %44, i32 %45, i32 %43
  store i32 %46, ptr getelementptr inbounds nuw (i8, ptr @_anonymous10, i20 8), align 4
  %47 = add nuw nsw i64 %5, 4
  %48 = icmp samesign ult i64 %36, 255
  br i1 %48, label %4, label %49

49:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %50 = load i32, ptr @_anonymous10, align 4
  %51 = icmp ugt i32 %50, 2147483646
  %52 = zext i1 %51 to i32
  %53 = add i32 %50, %52
  store i32 %53, ptr @_anonymous10, align 4
  %54 = add nuw nsw i64 %3, 1
  %55 = icmp samesign ult i64 %3, 4294967294
  br i1 %55, label %2, label %56

56:                                               ; preds = %49
  %57 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %57, 9223372036854775807
  br i1 %.not, label %58, label %.preheader

58:                                               ; preds = %56
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
