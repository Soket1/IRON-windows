; ModuleID = 'layer_fused_skeleton_e2048_h8192_c8_g32_fused.mlir.prj\op0_LayerFusedMLIR_core_4_3.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous17 = external local_unnamed_addr global [3 x i32]
@ffn_out_joined_0_cons_buff_1 = external global [8 x bfloat]
@ffn_out_joined_0_cons_buff_0 = external global [8 x bfloat]
@inpff_fifo_buff_1 = external global [2048 x bfloat]
@inpff_fifo_buff_0 = external global [2048 x bfloat]
@outL_fifo_buff_1 = external global [2 x bfloat]
@outL_fifo_buff_0 = external global [2 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_layer_fused_sum_partials_add_bf16(ptr, ptr, ptr, i32, i32, i32) local_unnamed_addr

define void @core_4_3() local_unnamed_addr {
  store i32 0, ptr @_anonymous17, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  br label %.preheader

.preheader:                                       ; preds = %0, %76
  %1 = phi i64 [ 0, %0 ], [ %77, %76 ]
  br label %2

2:                                                ; preds = %.preheader, %68
  %3 = phi i64 [ 0, %.preheader ], [ %74, %68 ]
  tail call void @llvm.aie2p.acquire(i32 7, i32 -1)
  %4 = load i32, ptr @_anonymous17, align 4
  %cond = icmp eq i32 %4, 1
  %spec.select = select i1 %cond, ptr @inpff_fifo_buff_1, ptr @inpff_fifo_buff_0
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i64 [ 0, %2 ], [ %66, %5 ]
  %7 = trunc nuw i64 %6 to i32
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %8 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %cond1 = icmp eq i32 %8, 1
  %spec.select4 = select i1 %cond1, ptr @outL_fifo_buff_1, ptr @outL_fifo_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %cond2 = icmp eq i32 %9, 1
  %10 = select i1 %cond2, ptr @ffn_out_joined_0_cons_buff_1, ptr @ffn_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_sum_partials_add_bf16(ptr nonnull %10, ptr nonnull %spec.select, ptr nonnull %spec.select4, i32 %7, i32 4, i32 2)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %11 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %12 = add i32 %11, 1
  %13 = icmp sgt i32 %12, 1
  %14 = add i32 %11, -1
  %15 = select i1 %13, i32 %14, i32 %12
  store i32 %15, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %17 = add i32 %16, 1
  %18 = icmp sgt i32 %17, 1
  %19 = add i32 %16, -1
  %20 = select i1 %18, i32 %19, i32 %17
  store i32 %20, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %21 = trunc i64 %6 to i32
  %22 = or disjoint i32 %21, 1
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %cond1.1 = icmp eq i32 %23, 1
  %spec.select4.1 = select i1 %cond1.1, ptr @outL_fifo_buff_1, ptr @outL_fifo_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %24 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %cond2.1 = icmp eq i32 %24, 1
  %25 = select i1 %cond2.1, ptr @ffn_out_joined_0_cons_buff_1, ptr @ffn_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_sum_partials_add_bf16(ptr nonnull %25, ptr nonnull %spec.select, ptr nonnull %spec.select4.1, i32 %22, i32 4, i32 2)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %26 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %27 = add i32 %26, 1
  %28 = icmp sgt i32 %27, 1
  %29 = add i32 %26, -1
  %30 = select i1 %28, i32 %29, i32 %27
  store i32 %30, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %32 = add i32 %31, 1
  %33 = icmp sgt i32 %32, 1
  %34 = add i32 %31, -1
  %35 = select i1 %33, i32 %34, i32 %32
  store i32 %35, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %36 = trunc i64 %6 to i32
  %37 = or disjoint i32 %36, 2
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %cond1.2 = icmp eq i32 %38, 1
  %spec.select4.2 = select i1 %cond1.2, ptr @outL_fifo_buff_1, ptr @outL_fifo_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %cond2.2 = icmp eq i32 %39, 1
  %40 = select i1 %cond2.2, ptr @ffn_out_joined_0_cons_buff_1, ptr @ffn_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_sum_partials_add_bf16(ptr nonnull %40, ptr nonnull %spec.select, ptr nonnull %spec.select4.2, i32 %37, i32 4, i32 2)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %42 = add i32 %41, 1
  %43 = icmp sgt i32 %42, 1
  %44 = add i32 %41, -1
  %45 = select i1 %43, i32 %44, i32 %42
  store i32 %45, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %46 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %47 = add i32 %46, 1
  %48 = icmp sgt i32 %47, 1
  %49 = add i32 %46, -1
  %50 = select i1 %48, i32 %49, i32 %47
  store i32 %50, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %51 = or disjoint i64 %6, 3
  %52 = trunc nuw i64 %51 to i32
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %53 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %cond1.3 = icmp eq i32 %53, 1
  %spec.select4.3 = select i1 %cond1.3, ptr @outL_fifo_buff_1, ptr @outL_fifo_buff_0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %54 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %cond2.3 = icmp eq i32 %54, 1
  %55 = select i1 %cond2.3, ptr @ffn_out_joined_0_cons_buff_1, ptr @ffn_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_sum_partials_add_bf16(ptr nonnull %55, ptr nonnull %spec.select, ptr nonnull %spec.select4.3, i32 %52, i32 4, i32 2)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %56 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  %57 = add i32 %56, 1
  %58 = icmp sgt i32 %57, 1
  %59 = add i32 %56, -1
  %60 = select i1 %58, i32 %59, i32 %57
  store i32 %60, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %61 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %62 = add i32 %61, 1
  %63 = icmp sgt i32 %62, 1
  %64 = add i32 %61, -1
  %65 = select i1 %63, i32 %64, i32 %62
  store i32 %65, ptr getelementptr inbounds nuw (i8, ptr @_anonymous17, i20 4), align 4
  %66 = add nuw nsw i64 %6, 4
  %67 = icmp samesign ult i64 %51, 1023
  br i1 %67, label %5, label %68

68:                                               ; preds = %5
  tail call void @llvm.aie2p.release(i32 6, i32 1)
  %69 = load i32, ptr @_anonymous17, align 4
  %70 = add i32 %69, 1
  %71 = icmp sgt i32 %70, 1
  %72 = add i32 %69, -1
  %73 = select i1 %71, i32 %72, i32 %70
  store i32 %73, ptr @_anonymous17, align 4
  %74 = add nuw nsw i64 %3, 1
  %75 = icmp samesign ult i64 %3, 4294967294
  br i1 %75, label %2, label %76

76:                                               ; preds = %68
  %77 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %77, 9223372036854775807
  br i1 %.not, label %78, label %.preheader

78:                                               ; preds = %76
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
