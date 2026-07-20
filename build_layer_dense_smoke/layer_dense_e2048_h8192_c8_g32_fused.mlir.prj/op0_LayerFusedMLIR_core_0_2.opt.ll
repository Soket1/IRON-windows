; ModuleID = 'layer_dense_e2048_h8192_c8_g32_fused.mlir.prj\op0_LayerFusedMLIR_core_0_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous0 = external local_unnamed_addr global [4 x i32]
@Aqkv_0_cons_buff_1 = external global [2304 x i8]
@Aqkv_0_cons_buff_0 = external global [2304 x i8]
@Cqkv_0_buff_1 = external global [2 x bfloat]
@Cqkv_0_buff_0 = external global [2 x bfloat]
@bq_L3L2_buff_0 = external global [2048 x bfloat]
@rms_in_cons_buff_2 = external global [2048 x bfloat]
@rms_in_cons_buff_1 = external global [2048 x bfloat]
@rms_in_cons_buff_0 = external global [2048 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_layer_fused_pre_rms_col0_bf16(ptr, ptr, ptr, i32) local_unnamed_addr

declare void @op0_layer_fused_qkv_gemv_static_bf16(i32, i32, ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) @_anonymous0, i8 0, i64 16, i1 false)
  br label %.preheader18

.preheader18:                                     ; preds = %0, %186
  %1 = phi i64 [ 0, %0 ], [ %187, %186 ]
  br label %2

2:                                                ; preds = %.preheader18, %183
  %3 = phi i64 [ 0, %.preheader18 ], [ %184, %183 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -2)
  %4 = load i32, ptr @_anonymous0, align 4
  %switch.selectcmp = icmp eq i32 %4, 1
  %switch.select = select i1 %switch.selectcmp, ptr @rms_in_cons_buff_1, ptr @rms_in_cons_buff_0
  %switch.selectcmp9 = icmp eq i32 %4, 2
  %switch.select10 = select i1 %switch.selectcmp9, ptr @rms_in_cons_buff_2, ptr %switch.select
  %switch.select12 = select i1 %switch.selectcmp, ptr @rms_in_cons_buff_2, ptr @rms_in_cons_buff_1
  %switch.select14 = select i1 %switch.selectcmp9, ptr @rms_in_cons_buff_0, ptr %switch.select12
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @op0_layer_fused_pre_rms_col0_bf16(ptr nonnull %switch.select10, ptr nonnull %switch.select14, ptr nonnull @bq_L3L2_buff_0, i32 2048)
  tail call void @llvm.aie2p.release(i32 54, i32 2)
  %5 = load i32, ptr @_anonymous0, align 4
  %6 = add i32 %5, 2
  %7 = icmp sgt i32 %6, 2
  %8 = add i32 %5, -1
  %9 = select i1 %7, i32 %8, i32 %6
  store i32 %9, ptr @_anonymous0, align 4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %10 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %11 = icmp ugt i32 %10, 2147483646
  %12 = zext i1 %11 to i32
  %13 = add i32 %10, %12
  store i32 %13, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  br label %14

14:                                               ; preds = %14, %2
  %15 = phi i64 [ 0, %2 ], [ %69, %14 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond = icmp eq i32 %16, 1
  %spec.select = select i1 %cond, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %17 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond1 = icmp eq i32 %17, 1
  %18 = select i1 %cond1, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select, ptr nonnull %18)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %19 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %20 = add i32 %19, 1
  %21 = icmp sgt i32 %20, 1
  %22 = add i32 %19, -1
  %23 = select i1 %21, i32 %22, i32 %20
  store i32 %23, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %24 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %25 = add i32 %24, 1
  %26 = icmp sgt i32 %25, 1
  %27 = add i32 %24, -1
  %28 = select i1 %26, i32 %27, i32 %25
  store i32 %28, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %29 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond.1 = icmp eq i32 %29, 1
  %spec.select.1 = select i1 %cond.1, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond1.1 = icmp eq i32 %30, 1
  %31 = select i1 %cond1.1, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select.1, ptr nonnull %31)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sgt i32 %33, 1
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %38 = add i32 %37, 1
  %39 = icmp sgt i32 %38, 1
  %40 = add i32 %37, -1
  %41 = select i1 %39, i32 %40, i32 %38
  store i32 %41, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %42 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond.2 = icmp eq i32 %42, 1
  %spec.select.2 = select i1 %cond.2, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %43 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond1.2 = icmp eq i32 %43, 1
  %44 = select i1 %cond1.2, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select.2, ptr nonnull %44)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %45 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %46 = add i32 %45, 1
  %47 = icmp sgt i32 %46, 1
  %48 = add i32 %45, -1
  %49 = select i1 %47, i32 %48, i32 %46
  store i32 %49, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %50 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %51 = add i32 %50, 1
  %52 = icmp sgt i32 %51, 1
  %53 = add i32 %50, -1
  %54 = select i1 %52, i32 %53, i32 %51
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %55 = or disjoint i64 %15, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %56 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond.3 = icmp eq i32 %56, 1
  %spec.select.3 = select i1 %cond.3, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %57 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond1.3 = icmp eq i32 %57, 1
  %58 = select i1 %cond1.3, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select.3, ptr nonnull %58)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %59 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %60 = add i32 %59, 1
  %61 = icmp sgt i32 %60, 1
  %62 = add i32 %59, -1
  %63 = select i1 %61, i32 %62, i32 %60
  store i32 %63, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %64 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %65 = add i32 %64, 1
  %66 = icmp sgt i32 %65, 1
  %67 = add i32 %64, -1
  %68 = select i1 %66, i32 %67, i32 %65
  store i32 %68, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %69 = add nuw nsw i64 %15, 4
  %70 = icmp samesign ult i64 %55, 255
  br i1 %70, label %14, label %.preheader17

.preheader17:                                     ; preds = %14, %.preheader17
  %71 = phi i64 [ %125, %.preheader17 ], [ 0, %14 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %72 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2 = icmp eq i32 %72, 1
  %spec.select15 = select i1 %cond2, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %73 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond3 = icmp eq i32 %73, 1
  %74 = select i1 %cond3, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select15, ptr nonnull %74)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %75 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %76 = add i32 %75, 1
  %77 = icmp sgt i32 %76, 1
  %78 = add i32 %75, -1
  %79 = select i1 %77, i32 %78, i32 %76
  store i32 %79, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %80 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %81 = add i32 %80, 1
  %82 = icmp sgt i32 %81, 1
  %83 = add i32 %80, -1
  %84 = select i1 %82, i32 %83, i32 %81
  store i32 %84, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %85 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.1 = icmp eq i32 %85, 1
  %spec.select15.1 = select i1 %cond2.1, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %86 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond3.1 = icmp eq i32 %86, 1
  %87 = select i1 %cond3.1, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select15.1, ptr nonnull %87)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %88 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %89 = add i32 %88, 1
  %90 = icmp sgt i32 %89, 1
  %91 = add i32 %88, -1
  %92 = select i1 %90, i32 %91, i32 %89
  store i32 %92, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %93 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %94 = add i32 %93, 1
  %95 = icmp sgt i32 %94, 1
  %96 = add i32 %93, -1
  %97 = select i1 %95, i32 %96, i32 %94
  store i32 %97, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %98 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.2 = icmp eq i32 %98, 1
  %spec.select15.2 = select i1 %cond2.2, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %99 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond3.2 = icmp eq i32 %99, 1
  %100 = select i1 %cond3.2, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select15.2, ptr nonnull %100)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %101 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %102 = add i32 %101, 1
  %103 = icmp sgt i32 %102, 1
  %104 = add i32 %101, -1
  %105 = select i1 %103, i32 %104, i32 %102
  store i32 %105, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %106 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %107 = add i32 %106, 1
  %108 = icmp sgt i32 %107, 1
  %109 = add i32 %106, -1
  %110 = select i1 %108, i32 %109, i32 %107
  store i32 %110, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %111 = or disjoint i64 %71, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %112 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.3 = icmp eq i32 %112, 1
  %spec.select15.3 = select i1 %cond2.3, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %113 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond3.3 = icmp eq i32 %113, 1
  %114 = select i1 %cond3.3, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select15.3, ptr nonnull %114)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %115 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %116 = add i32 %115, 1
  %117 = icmp sgt i32 %116, 1
  %118 = add i32 %115, -1
  %119 = select i1 %117, i32 %118, i32 %116
  store i32 %119, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %120 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %121 = add i32 %120, 1
  %122 = icmp sgt i32 %121, 1
  %123 = add i32 %120, -1
  %124 = select i1 %122, i32 %123, i32 %121
  store i32 %124, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %125 = add nuw nsw i64 %71, 4
  %126 = icmp samesign ult i64 %111, 63
  br i1 %126, label %.preheader17, label %.preheader

.preheader:                                       ; preds = %.preheader17, %.preheader
  %127 = phi i64 [ %181, %.preheader ], [ 0, %.preheader17 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %128 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond4 = icmp eq i32 %128, 1
  %spec.select16 = select i1 %cond4, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %129 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond5 = icmp eq i32 %129, 1
  %130 = select i1 %cond5, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select16, ptr nonnull %130)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %131 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %132 = add i32 %131, 1
  %133 = icmp sgt i32 %132, 1
  %134 = add i32 %131, -1
  %135 = select i1 %133, i32 %134, i32 %132
  store i32 %135, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %136 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %137 = add i32 %136, 1
  %138 = icmp sgt i32 %137, 1
  %139 = add i32 %136, -1
  %140 = select i1 %138, i32 %139, i32 %137
  store i32 %140, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %141 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond4.1 = icmp eq i32 %141, 1
  %spec.select16.1 = select i1 %cond4.1, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %142 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond5.1 = icmp eq i32 %142, 1
  %143 = select i1 %cond5.1, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select16.1, ptr nonnull %143)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %144 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %145 = add i32 %144, 1
  %146 = icmp sgt i32 %145, 1
  %147 = add i32 %144, -1
  %148 = select i1 %146, i32 %147, i32 %145
  store i32 %148, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %149 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %150 = add i32 %149, 1
  %151 = icmp sgt i32 %150, 1
  %152 = add i32 %149, -1
  %153 = select i1 %151, i32 %152, i32 %150
  store i32 %153, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %154 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond4.2 = icmp eq i32 %154, 1
  %spec.select16.2 = select i1 %cond4.2, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %155 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond5.2 = icmp eq i32 %155, 1
  %156 = select i1 %cond5.2, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select16.2, ptr nonnull %156)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %157 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %158 = add i32 %157, 1
  %159 = icmp sgt i32 %158, 1
  %160 = add i32 %157, -1
  %161 = select i1 %159, i32 %160, i32 %158
  store i32 %161, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %162 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %163 = add i32 %162, 1
  %164 = icmp sgt i32 %163, 1
  %165 = add i32 %162, -1
  %166 = select i1 %164, i32 %165, i32 %163
  store i32 %166, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %167 = or disjoint i64 %127, 3
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %168 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond4.3 = icmp eq i32 %168, 1
  %spec.select16.3 = select i1 %cond4.3, ptr @Aqkv_0_cons_buff_1, ptr @Aqkv_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %169 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %cond5.3 = icmp eq i32 %169, 1
  %170 = select i1 %cond5.3, ptr @Cqkv_0_buff_1, ptr @Cqkv_0_buff_0
  tail call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr nonnull %spec.select16.3, ptr nonnull %170)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %171 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %172 = add i32 %171, 1
  %173 = icmp sgt i32 %172, 1
  %174 = add i32 %171, -1
  %175 = select i1 %173, i32 %174, i32 %172
  store i32 %175, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %176 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %177 = add i32 %176, 1
  %178 = icmp sgt i32 %177, 1
  %179 = add i32 %176, -1
  %180 = select i1 %178, i32 %179, i32 %177
  store i32 %180, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 12), align 4
  %181 = add nuw nsw i64 %127, 4
  %182 = icmp samesign ult i64 %167, 63
  br i1 %182, label %.preheader, label %183

183:                                              ; preds = %.preheader
  %184 = add nuw nsw i64 %3, 1
  %185 = icmp samesign ult i64 %3, 4294967294
  br i1 %185, label %2, label %186

186:                                              ; preds = %183
  %187 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %187, 9223372036854775807
  br i1 %.not, label %188, label %.preheader18

188:                                              ; preds = %186
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
