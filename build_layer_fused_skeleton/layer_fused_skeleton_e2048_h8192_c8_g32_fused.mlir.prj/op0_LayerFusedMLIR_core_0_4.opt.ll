; ModuleID = 'layer_fused_skeleton_e2048_h8192_c8_g32_fused.mlir.prj\op0_LayerFusedMLIR_core_0_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous8 = external local_unnamed_addr global [3 x i32]
@Agu_0_cons_buff_1 = external global [4608 x i8]
@Agu_0_cons_buff_0 = external global [4608 x i8]
@ffi_mem_0_cons_buff_0 = external global [2048 x bfloat]
@inter_0_buff_1 = external global [2048 x bfloat]
@inter_0_buff_0 = external global [2048 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_layer_fused_gate_up_bf16(i32, i32, ptr, ptr, i32) local_unnamed_addr

declare void @op0_layer_fused_silu_mul_bf16(ptr, i32) local_unnamed_addr

define void @core_0_4() local_unnamed_addr {
  store i32 0, ptr @_anonymous8, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %74
  %1 = phi i64 [ 0, %0 ], [ %75, %74 ]
  br label %2

2:                                                ; preds = %.preheader, %61
  %3 = phi i64 [ 0, %.preheader ], [ %72, %61 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %59, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond = icmp eq i32 %6, 1
  %spec.select = select i1 %cond, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %spec.select, ptr nonnull @ffi_mem_0_cons_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %8 = add i32 %7, 1
  %9 = icmp sgt i32 %8, 1
  %10 = add i32 %7, -1
  %11 = select i1 %9, i32 %10, i32 %8
  store i32 %11, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond1 = icmp eq i32 %12, 1
  %13 = select i1 %cond1, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %13, ptr nonnull @ffi_mem_0_cons_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %15 = add i32 %14, 1
  %16 = icmp sgt i32 %15, 1
  %17 = add i32 %14, -1
  %18 = select i1 %16, i32 %17, i32 %15
  store i32 %18, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %19 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond.1 = icmp eq i32 %19, 1
  %spec.select.1 = select i1 %cond.1, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %spec.select.1, ptr nonnull @ffi_mem_0_cons_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %21 = add i32 %20, 1
  %22 = icmp sgt i32 %21, 1
  %23 = add i32 %20, -1
  %24 = select i1 %22, i32 %23, i32 %21
  store i32 %24, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond1.1 = icmp eq i32 %25, 1
  %26 = select i1 %cond1.1, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %26, ptr nonnull @ffi_mem_0_cons_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sgt i32 %28, 1
  %30 = add i32 %27, -1
  %31 = select i1 %29, i32 %30, i32 %28
  store i32 %31, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond.2 = icmp eq i32 %32, 1
  %spec.select.2 = select i1 %cond.2, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %spec.select.2, ptr nonnull @ffi_mem_0_cons_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %34 = add i32 %33, 1
  %35 = icmp sgt i32 %34, 1
  %36 = add i32 %33, -1
  %37 = select i1 %35, i32 %36, i32 %34
  store i32 %37, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond1.2 = icmp eq i32 %38, 1
  %39 = select i1 %cond1.2, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %39, ptr nonnull @ffi_mem_0_cons_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %40 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %41 = add i32 %40, 1
  %42 = icmp sgt i32 %41, 1
  %43 = add i32 %40, -1
  %44 = select i1 %42, i32 %43, i32 %41
  store i32 %44, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %45 = or disjoint i64 %5, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond.3 = icmp eq i32 %46, 1
  %spec.select.3 = select i1 %cond.3, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %spec.select.3, ptr nonnull @ffi_mem_0_cons_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %48 = add i32 %47, 1
  %49 = icmp sgt i32 %48, 1
  %50 = add i32 %47, -1
  %51 = select i1 %49, i32 %50, i32 %48
  store i32 %51, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %52 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %cond1.3 = icmp eq i32 %52, 1
  %53 = select i1 %cond1.3, ptr @Agu_0_cons_buff_1, ptr @Agu_0_cons_buff_0
  tail call void @op0_layer_fused_gate_up_bf16(i32 4, i32 0, ptr nonnull %53, ptr nonnull @ffi_mem_0_cons_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %54 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %55 = add i32 %54, 1
  %56 = icmp sgt i32 %55, 1
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 4), align 4
  %59 = add nuw nsw i64 %5, 4
  %60 = icmp samesign ult i64 %45, 511
  br i1 %60, label %4, label %61

61:                                               ; preds = %4
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %62 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 8), align 4
  %cond2 = icmp eq i32 %62, 1
  %spec.select4 = select i1 %cond2, ptr @inter_0_buff_1, ptr @inter_0_buff_0
  tail call void @op0_layer_fused_silu_mul_bf16(ptr nonnull %spec.select4, i32 2048)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %63 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 8), align 4
  %64 = add i32 %63, 1
  %65 = icmp sgt i32 %64, 1
  %66 = add i32 %63, -1
  %67 = select i1 %65, i32 %66, i32 %64
  store i32 %67, ptr getelementptr inbounds nuw (i8, ptr @_anonymous8, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %68 = load i32, ptr @_anonymous8, align 4
  %69 = icmp ugt i32 %68, 2147483646
  %70 = zext i1 %69 to i32
  %71 = add i32 %68, %70
  store i32 %71, ptr @_anonymous8, align 4
  %72 = add nuw nsw i64 %3, 1
  %73 = icmp samesign ult i64 %3, 4294967294
  br i1 %73, label %2, label %74

74:                                               ; preds = %61
  %75 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %75, 9223372036854775807
  br i1 %.not, label %76, label %.preheader

76:                                               ; preds = %74
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
