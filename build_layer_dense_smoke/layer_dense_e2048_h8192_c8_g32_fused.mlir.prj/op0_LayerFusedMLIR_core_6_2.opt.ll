; ModuleID = 'layer_dense_e2048_h8192_c8_g32_fused.mlir.prj\op0_LayerFusedMLIR_core_6_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous6 = external local_unnamed_addr global [3 x i32]
@Aqkv_6_cons_buff_1 = external global [2304 x i8]
@Aqkv_6_cons_buff_0 = external global [2304 x i8]
@Cqkv_6_buff_1 = external global [2 x bfloat]
@Cqkv_6_buff_0 = external global [2 x bfloat]
@bq_mem_5_cons_buff_0 = external global [2048 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_layer_fused_qkv_gemv_bf16(i32, i32, ptr, ptr, ptr) local_unnamed_addr

define void @core_6_2() local_unnamed_addr {
  store i32 0, ptr @_anonymous6, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  br label %.preheader12

.preheader12:                                     ; preds = %0, %180
  %1 = phi i64 [ 0, %0 ], [ %181, %180 ]
  br label %2

2:                                                ; preds = %.preheader12, %173
  %3 = phi i64 [ 0, %.preheader12 ], [ %178, %173 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %59, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond = icmp eq i32 %6, 1
  %spec.select = select i1 %cond, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond1 = icmp eq i32 %7, 1
  %8 = select i1 %cond1, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %8)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sgt i32 %10, 1
  %12 = add i32 %9, -1
  %13 = select i1 %11, i32 %12, i32 %10
  store i32 %13, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %14 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %15 = add i32 %14, 1
  %16 = icmp sgt i32 %15, 1
  %17 = add i32 %14, -1
  %18 = select i1 %16, i32 %17, i32 %15
  store i32 %18, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %19 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond.1 = icmp eq i32 %19, 1
  %spec.select.1 = select i1 %cond.1, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond1.1 = icmp eq i32 %20, 1
  %21 = select i1 %cond1.1, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select.1, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %21)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %22 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %23 = add i32 %22, 1
  %24 = icmp sgt i32 %23, 1
  %25 = add i32 %22, -1
  %26 = select i1 %24, i32 %25, i32 %23
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %28 = add i32 %27, 1
  %29 = icmp sgt i32 %28, 1
  %30 = add i32 %27, -1
  %31 = select i1 %29, i32 %30, i32 %28
  store i32 %31, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond.2 = icmp eq i32 %32, 1
  %spec.select.2 = select i1 %cond.2, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond1.2 = icmp eq i32 %33, 1
  %34 = select i1 %cond1.2, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select.2, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %34)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %35 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %36 = add i32 %35, 1
  %37 = icmp sgt i32 %36, 1
  %38 = add i32 %35, -1
  %39 = select i1 %37, i32 %38, i32 %36
  store i32 %39, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %40 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %41 = add i32 %40, 1
  %42 = icmp sgt i32 %41, 1
  %43 = add i32 %40, -1
  %44 = select i1 %42, i32 %43, i32 %41
  store i32 %44, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %45 = or disjoint i64 %5, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond.3 = icmp eq i32 %46, 1
  %spec.select.3 = select i1 %cond.3, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond1.3 = icmp eq i32 %47, 1
  %48 = select i1 %cond1.3, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select.3, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %48)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %49 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %50 = add i32 %49, 1
  %51 = icmp sgt i32 %50, 1
  %52 = add i32 %49, -1
  %53 = select i1 %51, i32 %52, i32 %50
  store i32 %53, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %54 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sgt i32 %55, 1
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %59 = add nuw nsw i64 %5, 4
  %60 = icmp samesign ult i64 %45, 127
  br i1 %60, label %4, label %.preheader11

.preheader11:                                     ; preds = %4, %.preheader11
  %61 = phi i64 [ %115, %.preheader11 ], [ 0, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %62 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond2 = icmp eq i32 %62, 1
  %spec.select9 = select i1 %cond2, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %63 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond3 = icmp eq i32 %63, 1
  %64 = select i1 %cond3, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select9, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %64)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %65 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %66 = add i32 %65, 1
  %67 = icmp sgt i32 %66, 1
  %68 = add i32 %65, -1
  %69 = select i1 %67, i32 %68, i32 %66
  store i32 %69, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %70 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %71 = add i32 %70, 1
  %72 = icmp sgt i32 %71, 1
  %73 = add i32 %70, -1
  %74 = select i1 %72, i32 %73, i32 %71
  store i32 %74, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %75 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond2.1 = icmp eq i32 %75, 1
  %spec.select9.1 = select i1 %cond2.1, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %76 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond3.1 = icmp eq i32 %76, 1
  %77 = select i1 %cond3.1, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select9.1, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %77)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %78 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %79 = add i32 %78, 1
  %80 = icmp sgt i32 %79, 1
  %81 = add i32 %78, -1
  %82 = select i1 %80, i32 %81, i32 %79
  store i32 %82, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %83 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %84 = add i32 %83, 1
  %85 = icmp sgt i32 %84, 1
  %86 = add i32 %83, -1
  %87 = select i1 %85, i32 %86, i32 %84
  store i32 %87, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %88 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond2.2 = icmp eq i32 %88, 1
  %spec.select9.2 = select i1 %cond2.2, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %89 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond3.2 = icmp eq i32 %89, 1
  %90 = select i1 %cond3.2, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select9.2, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %90)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %91 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %92 = add i32 %91, 1
  %93 = icmp sgt i32 %92, 1
  %94 = add i32 %91, -1
  %95 = select i1 %93, i32 %94, i32 %92
  store i32 %95, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %96 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %97 = add i32 %96, 1
  %98 = icmp sgt i32 %97, 1
  %99 = add i32 %96, -1
  %100 = select i1 %98, i32 %99, i32 %97
  store i32 %100, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %101 = or disjoint i64 %61, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %102 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond2.3 = icmp eq i32 %102, 1
  %spec.select9.3 = select i1 %cond2.3, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %103 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond3.3 = icmp eq i32 %103, 1
  %104 = select i1 %cond3.3, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select9.3, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %104)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %105 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %106 = add i32 %105, 1
  %107 = icmp sgt i32 %106, 1
  %108 = add i32 %105, -1
  %109 = select i1 %107, i32 %108, i32 %106
  store i32 %109, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %110 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %111 = add i32 %110, 1
  %112 = icmp sgt i32 %111, 1
  %113 = add i32 %110, -1
  %114 = select i1 %112, i32 %113, i32 %111
  store i32 %114, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %115 = add nuw nsw i64 %61, 4
  %116 = icmp samesign ult i64 %101, 31
  br i1 %116, label %.preheader11, label %.preheader

.preheader:                                       ; preds = %.preheader11, %.preheader
  %117 = phi i64 [ %171, %.preheader ], [ 0, %.preheader11 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %118 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond4 = icmp eq i32 %118, 1
  %spec.select10 = select i1 %cond4, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %119 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond5 = icmp eq i32 %119, 1
  %120 = select i1 %cond5, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select10, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %120)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %121 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %122 = add i32 %121, 1
  %123 = icmp sgt i32 %122, 1
  %124 = add i32 %121, -1
  %125 = select i1 %123, i32 %124, i32 %122
  store i32 %125, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %126 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %127 = add i32 %126, 1
  %128 = icmp sgt i32 %127, 1
  %129 = add i32 %126, -1
  %130 = select i1 %128, i32 %129, i32 %127
  store i32 %130, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %131 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond4.1 = icmp eq i32 %131, 1
  %spec.select10.1 = select i1 %cond4.1, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %132 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond5.1 = icmp eq i32 %132, 1
  %133 = select i1 %cond5.1, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select10.1, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %133)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %134 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %135 = add i32 %134, 1
  %136 = icmp sgt i32 %135, 1
  %137 = add i32 %134, -1
  %138 = select i1 %136, i32 %137, i32 %135
  store i32 %138, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %139 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %140 = add i32 %139, 1
  %141 = icmp sgt i32 %140, 1
  %142 = add i32 %139, -1
  %143 = select i1 %141, i32 %142, i32 %140
  store i32 %143, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %144 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond4.2 = icmp eq i32 %144, 1
  %spec.select10.2 = select i1 %cond4.2, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %145 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond5.2 = icmp eq i32 %145, 1
  %146 = select i1 %cond5.2, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select10.2, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %146)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %147 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %148 = add i32 %147, 1
  %149 = icmp sgt i32 %148, 1
  %150 = add i32 %147, -1
  %151 = select i1 %149, i32 %150, i32 %148
  store i32 %151, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %152 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %153 = add i32 %152, 1
  %154 = icmp sgt i32 %153, 1
  %155 = add i32 %152, -1
  %156 = select i1 %154, i32 %155, i32 %153
  store i32 %156, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %157 = or disjoint i64 %117, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %158 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %cond4.3 = icmp eq i32 %158, 1
  %spec.select10.3 = select i1 %cond4.3, ptr @Aqkv_6_cons_buff_1, ptr @Aqkv_6_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %159 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %cond5.3 = icmp eq i32 %159, 1
  %160 = select i1 %cond5.3, ptr @Cqkv_6_buff_1, ptr @Cqkv_6_buff_0
  tail call void @op0_layer_fused_qkv_gemv_bf16(i32 2, i32 0, ptr nonnull %spec.select10.3, ptr nonnull @bq_mem_5_cons_buff_0, ptr nonnull %160)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %161 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  %162 = add i32 %161, 1
  %163 = icmp sgt i32 %162, 1
  %164 = add i32 %161, -1
  %165 = select i1 %163, i32 %164, i32 %162
  store i32 %165, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %166 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %167 = add i32 %166, 1
  %168 = icmp sgt i32 %167, 1
  %169 = add i32 %166, -1
  %170 = select i1 %168, i32 %169, i32 %167
  store i32 %170, ptr getelementptr inbounds nuw (i8, ptr @_anonymous6, i20 8), align 4
  %171 = add nuw nsw i64 %117, 4
  %172 = icmp samesign ult i64 %157, 31
  br i1 %172, label %.preheader, label %173

173:                                              ; preds = %.preheader
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %174 = load i32, ptr @_anonymous6, align 4
  %175 = icmp ugt i32 %174, 2147483646
  %176 = zext i1 %175 to i32
  %177 = add i32 %174, %176
  store i32 %177, ptr @_anonymous6, align 4
  %178 = add nuw nsw i64 %3, 1
  %179 = icmp samesign ult i64 %3, 4294967294
  br i1 %179, label %2, label %180

180:                                              ; preds = %173
  %181 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %181, 9223372036854775807
  br i1 %.not, label %182, label %.preheader12

182:                                              ; preds = %180
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
