; ModuleID = 'smoke_fused_fused.mlir.prj\op0_RMSNorm_core_3_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@in2_weights_0_4_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_4_cons_buff_0 = external global [256 x bfloat]
@out1_4_0_cons_buff_1 = external global [256 x bfloat]
@out1_4_0_cons_buff_0 = external global [256 x bfloat]
@out2_4_0_buff_1 = external global [256 x bfloat]
@out2_4_0_buff_0 = external global [256 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op0_eltwise_mul_bf16_vector(ptr, ptr, ptr, i32) local_unnamed_addr

define void @core_3_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @op0_eltwise_mul_bf16_vector(ptr nonnull @out1_4_0_cons_buff_0, ptr nonnull @in2_weights_0_4_cons_buff_0, ptr nonnull @out2_4_0_buff_0, i32 256)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @op0_eltwise_mul_bf16_vector(ptr nonnull @out1_4_0_cons_buff_1, ptr nonnull @in2_weights_0_4_cons_buff_1, ptr nonnull @out2_4_0_buff_1, i32 256)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @op0_eltwise_mul_bf16_vector(ptr nonnull @out1_4_0_cons_buff_0, ptr nonnull @in2_weights_0_4_cons_buff_0, ptr nonnull @out2_4_0_buff_0, i32 256)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
