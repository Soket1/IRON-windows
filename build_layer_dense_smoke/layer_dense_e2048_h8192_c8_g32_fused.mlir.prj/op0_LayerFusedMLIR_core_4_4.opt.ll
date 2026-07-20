; ModuleID = 'layer_dense_e2048_h8192_c8_g32_fused.mlir.prj\op0_LayerFusedMLIR_core_4_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous16 = external local_unnamed_addr global [3 x i32]
@anm_inpff_buf = external global [2048 x bfloat]
@anm_oout_buf = external global [2048 x bfloat]
@o_out_joined_0_cons_buff_1 = external global [8 x bfloat]
@o_out_joined_0_cons_buff_0 = external global [8 x bfloat]
@anm_mem_cons_buff_2 = external global [2048 x bfloat]
@anm_mem_cons_buff_1 = external global [2048 x bfloat]
@anm_mem_cons_buff_0 = external global [2048 x bfloat]
@ffi_L3L2_buff_0 = external global [2048 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_layer_fused_o_out_assemble_bf16(ptr, ptr, i32, i32, i32, i32, i32) local_unnamed_addr

declare void @op0_layer_fused_add_bf16(ptr, ptr, ptr, i32) local_unnamed_addr

declare void @op0_layer_fused_rms_norm2_bf16(ptr, ptr, ptr, i32) local_unnamed_addr

define void @core_4_4() local_unnamed_addr {
  store i32 0, ptr @_anonymous16, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous16, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous16, i20 8), align 4
  br label %.preheader7

.preheader7:                                      ; preds = %0, %51
  %1 = phi i64 [ 0, %0 ], [ %52, %51 ]
  br label %.preheader

.preheader:                                       ; preds = %.preheader7, %38
  %2 = phi i64 [ 0, %.preheader7 ], [ %49, %38 ]
  br label %3

3:                                                ; preds = %3, %.preheader
  %4 = phi i64 [ 0, %.preheader ], [ %36, %3 ]
  %5 = trunc nuw i64 %4 to i32
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = load i32, ptr @_anonymous16, align 4
  %cond = icmp eq i32 %6, 1
  %spec.select = select i1 %cond, ptr @o_out_joined_0_cons_buff_1, ptr @o_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_o_out_assemble_bf16(ptr nonnull %spec.select, ptr nonnull @anm_oout_buf, i32 %5, i32 0, i32 4, i32 2, i32 512)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %7 = load i32, ptr @_anonymous16, align 4
  %8 = add i32 %7, 1
  %9 = icmp sgt i32 %8, 1
  %10 = add i32 %7, -1
  %11 = select i1 %9, i32 %10, i32 %8
  store i32 %11, ptr @_anonymous16, align 4
  %12 = trunc i64 %4 to i32
  %13 = or disjoint i32 %12, 1
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = load i32, ptr @_anonymous16, align 4
  %cond.1 = icmp eq i32 %14, 1
  %spec.select.1 = select i1 %cond.1, ptr @o_out_joined_0_cons_buff_1, ptr @o_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_o_out_assemble_bf16(ptr nonnull %spec.select.1, ptr nonnull @anm_oout_buf, i32 %13, i32 0, i32 4, i32 2, i32 512)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = load i32, ptr @_anonymous16, align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr @_anonymous16, align 4
  %20 = trunc i64 %4 to i32
  %21 = or disjoint i32 %20, 2
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %22 = load i32, ptr @_anonymous16, align 4
  %cond.2 = icmp eq i32 %22, 1
  %spec.select.2 = select i1 %cond.2, ptr @o_out_joined_0_cons_buff_1, ptr @o_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_o_out_assemble_bf16(ptr nonnull %spec.select.2, ptr nonnull @anm_oout_buf, i32 %21, i32 0, i32 4, i32 2, i32 512)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %23 = load i32, ptr @_anonymous16, align 4
  %24 = add i32 %23, 1
  %25 = icmp sgt i32 %24, 1
  %26 = add i32 %23, -1
  %27 = select i1 %25, i32 %26, i32 %24
  store i32 %27, ptr @_anonymous16, align 4
  %28 = or disjoint i64 %4, 3
  %29 = trunc nuw i64 %28 to i32
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %30 = load i32, ptr @_anonymous16, align 4
  %cond.3 = icmp eq i32 %30, 1
  %spec.select.3 = select i1 %cond.3, ptr @o_out_joined_0_cons_buff_1, ptr @o_out_joined_0_cons_buff_0
  tail call void @op0_layer_fused_o_out_assemble_bf16(ptr nonnull %spec.select.3, ptr nonnull @anm_oout_buf, i32 %29, i32 0, i32 4, i32 2, i32 512)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %31 = load i32, ptr @_anonymous16, align 4
  %32 = add i32 %31, 1
  %33 = icmp sgt i32 %32, 1
  %34 = add i32 %31, -1
  %35 = select i1 %33, i32 %34, i32 %32
  store i32 %35, ptr @_anonymous16, align 4
  %36 = add nuw nsw i64 %4, 4
  %37 = icmp samesign ult i64 %28, 255
  br i1 %37, label %3, label %38

38:                                               ; preds = %3
  tail call void @llvm.aie2p.acquire(i32 51, i32 -2)
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous16, i20 4), align 4
  %switch.selectcmp = icmp eq i32 %39, 1
  %switch.select = select i1 %switch.selectcmp, ptr @anm_mem_cons_buff_1, ptr @anm_mem_cons_buff_0
  %switch.selectcmp1 = icmp eq i32 %39, 2
  %switch.select2 = select i1 %switch.selectcmp1, ptr @anm_mem_cons_buff_2, ptr %switch.select
  %switch.select4 = select i1 %switch.selectcmp, ptr @anm_mem_cons_buff_2, ptr @anm_mem_cons_buff_1
  %switch.select6 = select i1 %switch.selectcmp1, ptr @anm_mem_cons_buff_0, ptr %switch.select4
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @op0_layer_fused_add_bf16(ptr nonnull @anm_oout_buf, ptr nonnull %switch.select2, ptr nonnull @anm_inpff_buf, i32 2048)
  tail call void @op0_layer_fused_rms_norm2_bf16(ptr nonnull @anm_inpff_buf, ptr nonnull %switch.select6, ptr nonnull @ffi_L3L2_buff_0, i32 2048)
  tail call void @llvm.aie2p.release(i32 50, i32 2)
  %40 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous16, i20 4), align 4
  %41 = add i32 %40, 2
  %42 = icmp sgt i32 %41, 2
  %43 = add i32 %40, -1
  %44 = select i1 %42, i32 %43, i32 %41
  store i32 %44, ptr getelementptr inbounds nuw (i8, ptr @_anonymous16, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %45 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous16, i20 8), align 4
  %46 = icmp ugt i32 %45, 2147483646
  %47 = zext i1 %46 to i32
  %48 = add i32 %45, %47
  store i32 %48, ptr getelementptr inbounds nuw (i8, ptr @_anonymous16, i20 8), align 4
  %49 = add nuw nsw i64 %2, 1
  %50 = icmp samesign ult i64 %2, 4294967294
  br i1 %50, label %.preheader, label %51

51:                                               ; preds = %38
  %52 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %52, 9223372036854775807
  br i1 %.not, label %53, label %.preheader7

53:                                               ; preds = %51
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
