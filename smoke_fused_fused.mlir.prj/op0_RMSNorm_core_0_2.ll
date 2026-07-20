; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@in1_0_0_cons_buff_1 = external global [256 x bfloat]
@in1_0_0_cons_buff_0 = external global [256 x bfloat]
@in1_1_0_cons_buff_1 = external global [256 x bfloat]
@in1_1_0_cons_buff_0 = external global [256 x bfloat]
@in1_2_0_cons_buff_1 = external global [256 x bfloat]
@in1_2_0_cons_buff_0 = external global [256 x bfloat]
@in1_3_0_cons_buff_1 = external global [256 x bfloat]
@in1_3_0_cons_buff_0 = external global [256 x bfloat]
@in1_4_0_cons_buff_1 = external global [256 x bfloat]
@in1_4_0_cons_buff_0 = external global [256 x bfloat]
@in1_5_0_cons_buff_1 = external global [256 x bfloat]
@in1_5_0_cons_buff_0 = external global [256 x bfloat]
@in1_6_0_cons_buff_1 = external global [256 x bfloat]
@in1_6_0_cons_buff_0 = external global [256 x bfloat]
@in1_7_0_cons_buff_1 = external global [256 x bfloat]
@in1_7_0_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_7_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_7_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_6_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_6_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_5_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_5_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_4_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_4_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_3_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_3_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_2_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_2_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_1_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_1_cons_buff_0 = external global [256 x bfloat]
@in2_weights_0_0_cons_buff_1 = external global [256 x bfloat]
@in2_weights_0_0_cons_buff_0 = external global [256 x bfloat]
@out1_0_0_buff_1 = external global [256 x bfloat]
@out1_0_0_buff_0 = external global [256 x bfloat]
@out1_0_0_cons_buff_1 = external global [256 x bfloat]
@out1_0_0_cons_buff_0 = external global [256 x bfloat]
@out1_1_0_buff_1 = external global [256 x bfloat]
@out1_1_0_buff_0 = external global [256 x bfloat]
@out1_1_0_cons_buff_1 = external global [256 x bfloat]
@out1_1_0_cons_buff_0 = external global [256 x bfloat]
@out1_2_0_buff_1 = external global [256 x bfloat]
@out1_2_0_buff_0 = external global [256 x bfloat]
@out1_2_0_cons_buff_1 = external global [256 x bfloat]
@out1_2_0_cons_buff_0 = external global [256 x bfloat]
@out1_3_0_buff_1 = external global [256 x bfloat]
@out1_3_0_buff_0 = external global [256 x bfloat]
@out1_3_0_cons_buff_1 = external global [256 x bfloat]
@out1_3_0_cons_buff_0 = external global [256 x bfloat]
@out1_4_0_buff_1 = external global [256 x bfloat]
@out1_4_0_buff_0 = external global [256 x bfloat]
@out1_4_0_cons_buff_1 = external global [256 x bfloat]
@out1_4_0_cons_buff_0 = external global [256 x bfloat]
@out1_5_0_buff_1 = external global [256 x bfloat]
@out1_5_0_buff_0 = external global [256 x bfloat]
@out1_5_0_cons_buff_1 = external global [256 x bfloat]
@out1_5_0_cons_buff_0 = external global [256 x bfloat]
@out1_6_0_buff_1 = external global [256 x bfloat]
@out1_6_0_buff_0 = external global [256 x bfloat]
@out1_6_0_cons_buff_1 = external global [256 x bfloat]
@out1_6_0_cons_buff_0 = external global [256 x bfloat]
@out1_7_0_buff_1 = external global [256 x bfloat]
@out1_7_0_buff_0 = external global [256 x bfloat]
@out1_7_0_cons_buff_1 = external global [256 x bfloat]
@out1_7_0_cons_buff_0 = external global [256 x bfloat]
@out2_0_0_buff_1 = external global [256 x bfloat]
@out2_0_0_buff_0 = external global [256 x bfloat]
@out2_1_0_buff_1 = external global [256 x bfloat]
@out2_1_0_buff_0 = external global [256 x bfloat]
@out2_2_0_buff_1 = external global [256 x bfloat]
@out2_2_0_buff_0 = external global [256 x bfloat]
@out2_3_0_buff_1 = external global [256 x bfloat]
@out2_3_0_buff_0 = external global [256 x bfloat]
@out2_4_0_buff_1 = external global [256 x bfloat]
@out2_4_0_buff_0 = external global [256 x bfloat]
@out2_5_0_buff_1 = external global [256 x bfloat]
@out2_5_0_buff_0 = external global [256 x bfloat]
@out2_6_0_buff_1 = external global [256 x bfloat]
@out2_6_0_buff_0 = external global [256 x bfloat]
@out2_7_0_buff_1 = external global [256 x bfloat]
@out2_7_0_buff_0 = external global [256 x bfloat]

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

declare void @op0_rms_norm_bf16_vector(ptr, ptr, i32)

declare void @op0_eltwise_mul_bf16_vector(ptr, ptr, ptr, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @op0_rms_norm_bf16_vector(ptr @in1_0_0_cons_buff_0, ptr @out1_0_0_buff_0, i32 256)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @op0_rms_norm_bf16_vector(ptr @in1_0_0_cons_buff_1, ptr @out1_0_0_buff_1, i32 256)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @op0_rms_norm_bf16_vector(ptr @in1_0_0_cons_buff_0, ptr @out1_0_0_buff_0, i32 256)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
