; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous16 = external global [3 x i32]
@_anonymous15 = external global [3 x i32]
@_anonymous14 = external global [3 x i32]
@_anonymous13 = external global [3 x i32]
@_anonymous12 = external global [3 x i32]
@_anonymous11 = external global [3 x i32]
@_anonymous10 = external global [3 x i32]
@_anonymous9 = external global [3 x i32]
@_anonymous8 = external global [3 x i32]
@_anonymous7 = external global [3 x i32]
@_anonymous6 = external global [3 x i32]
@_anonymous5 = external global [3 x i32]
@_anonymous4 = external global [3 x i32]
@_anonymous3 = external global [3 x i32]
@_anonymous2 = external global [3 x i32]
@_anonymous1 = external global [3 x i32]
@_anonymous0 = external global [4 x i32]
@anm_inpff_buf = external global [2048 x bfloat]
@anm_oout_buf = external global [2048 x bfloat]
@Adp_0_cons_buff_1 = external global [2304 x i8]
@Adp_0_cons_buff_0 = external global [2304 x i8]
@Ffn_src_0_cons_buff_1 = external global [13824 x i8]
@Ffn_src_0_cons_buff_0 = external global [13824 x i8]
@Agu_0_cons_buff_1 = external global [4608 x i8]
@Agu_0_cons_buff_0 = external global [4608 x i8]
@Agu_1_cons_buff_1 = external global [4608 x i8]
@Agu_1_cons_buff_0 = external global [4608 x i8]
@Adp_1_cons_buff_1 = external global [2304 x i8]
@Adp_1_cons_buff_0 = external global [2304 x i8]
@Adp_2_cons_buff_1 = external global [2304 x i8]
@Adp_2_cons_buff_0 = external global [2304 x i8]
@Ffn_src_1_cons_buff_1 = external global [13824 x i8]
@Ffn_src_1_cons_buff_0 = external global [13824 x i8]
@Agu_2_cons_buff_1 = external global [4608 x i8]
@Agu_2_cons_buff_0 = external global [4608 x i8]
@Agu_3_cons_buff_1 = external global [4608 x i8]
@Agu_3_cons_buff_0 = external global [4608 x i8]
@Adp_3_cons_buff_1 = external global [2304 x i8]
@Adp_3_cons_buff_0 = external global [2304 x i8]
@Ao_0_cons_buff_1 = external global [2304 x i8]
@Ao_0_cons_buff_0 = external global [2304 x i8]
@Ao_src_0_cons_buff_1 = external global [4608 x i8]
@Ao_src_0_cons_buff_0 = external global [4608 x i8]
@Ao_1_cons_buff_1 = external global [2304 x i8]
@Ao_1_cons_buff_0 = external global [2304 x i8]
@Ao_2_cons_buff_1 = external global [2304 x i8]
@Ao_2_cons_buff_0 = external global [2304 x i8]
@Ao_src_1_cons_buff_1 = external global [4608 x i8]
@Ao_src_1_cons_buff_0 = external global [4608 x i8]
@Ao_3_cons_buff_1 = external global [2304 x i8]
@Ao_3_cons_buff_0 = external global [2304 x i8]
@Aqkv_0_cons_buff_1 = external global [2304 x i8]
@Aqkv_0_cons_buff_0 = external global [2304 x i8]
@Aqkv_src_0_cons_buff_1 = external global [4608 x i8]
@Aqkv_src_0_cons_buff_0 = external global [4608 x i8]
@Aqkv_1_cons_buff_1 = external global [2304 x i8]
@Aqkv_1_cons_buff_0 = external global [2304 x i8]
@Aqkv_2_cons_buff_1 = external global [2304 x i8]
@Aqkv_2_cons_buff_0 = external global [2304 x i8]
@Aqkv_src_1_cons_buff_1 = external global [4608 x i8]
@Aqkv_src_1_cons_buff_0 = external global [4608 x i8]
@Aqkv_3_cons_buff_1 = external global [2304 x i8]
@Aqkv_3_cons_buff_0 = external global [2304 x i8]
@Cdp_0_buff_1 = external global [2 x bfloat]
@Cdp_0_buff_0 = external global [2 x bfloat]
@Cdp_1_buff_1 = external global [2 x bfloat]
@Cdp_1_buff_0 = external global [2 x bfloat]
@ffn_out_joined_0_buff_1 = external global [4 x bfloat]
@ffn_out_joined_0_buff_0 = external global [4 x bfloat]
@Cdp_2_buff_1 = external global [2 x bfloat]
@Cdp_2_buff_0 = external global [2 x bfloat]
@Cdp_3_buff_1 = external global [2 x bfloat]
@Cdp_3_buff_0 = external global [2 x bfloat]
@ffn_out_joined_1_buff_1 = external global [4 x bfloat]
@ffn_out_joined_1_buff_0 = external global [4 x bfloat]
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
@Cqkv_0_buff_1 = external global [2 x bfloat]
@Cqkv_0_buff_0 = external global [2 x bfloat]
@Cqkv_1_buff_1 = external global [2 x bfloat]
@Cqkv_1_buff_0 = external global [2 x bfloat]
@Cqkv_2_buff_1 = external global [2 x bfloat]
@Cqkv_2_buff_0 = external global [2 x bfloat]
@Cqkv_3_buff_1 = external global [2 x bfloat]
@Cqkv_3_buff_0 = external global [2 x bfloat]
@anm_in_cons_buff_1 = external global [2048 x bfloat]
@anm_in_cons_buff_0 = external global [2048 x bfloat]
@anm_mem_cons_buff_2 = external global [2048 x bfloat]
@anm_mem_cons_buff_1 = external global [2048 x bfloat]
@anm_mem_cons_buff_0 = external global [2048 x bfloat]
@bo_L3L2_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_3_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_2_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_1_cons_buff_0 = external global [2048 x bfloat]
@bo_mem_0_cons_buff_0 = external global [2048 x bfloat]
@bq_L3L2_buff_0 = external global [2048 x bfloat]
@bq_L3L2_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_2_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_1_cons_buff_0 = external global [2048 x bfloat]
@bq_mem_0_cons_buff_0 = external global [2048 x bfloat]
@ffi_L3L2_buff_0 = external global [2048 x bfloat]
@ffi_L3L2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_3_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_2_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_1_cons_buff_0 = external global [2048 x bfloat]
@ffi_mem_0_cons_buff_0 = external global [2048 x bfloat]
@inter_0_buff_1 = external global [2048 x bfloat]
@inter_0_buff_0 = external global [2048 x bfloat]
@inter_1_buff_1 = external global [2048 x bfloat]
@inter_1_buff_0 = external global [2048 x bfloat]
@inter_2_buff_1 = external global [2048 x bfloat]
@inter_2_buff_0 = external global [2048 x bfloat]
@inter_3_buff_1 = external global [2048 x bfloat]
@inter_3_buff_0 = external global [2048 x bfloat]
@rms_in_cons_buff_2 = external global [2048 x bfloat]
@rms_in_cons_buff_1 = external global [2048 x bfloat]
@rms_in_cons_buff_0 = external global [2048 x bfloat]

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

declare void @op0_layer_fused_pre_rms_col0_bf16(ptr, ptr, ptr, i32)

declare void @op0_layer_fused_qkv_gemv_static_bf16(i32, i32, ptr, ptr)

declare void @op0_layer_fused_qkv_gemv_bf16(i32, i32, ptr, ptr, ptr)

declare void @op0_layer_fused_o_proj_bf16(i32, i32, ptr, ptr, ptr)

declare void @op0_layer_fused_gate_up_bf16(i32, i32, ptr, ptr, i32)

declare void @op0_layer_fused_silu_mul_bf16(ptr, i32)

declare void @op0_layer_fused_down_partial_bf16(i32, i32, ptr, ptr, ptr)

declare void @op0_layer_fused_o_out_assemble_bf16(ptr, ptr, i32, i32, i32, i32, i32)

declare void @op0_layer_fused_add_bf16(ptr, ptr, ptr, i32)

declare void @op0_layer_fused_rms_norm2_bf16(ptr, ptr, ptr, i32)

define void @core_0_2() {
  store i32 0, ptr @_anonymous0
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  store i32 0, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  br label %1

1:                                                ; preds = %104, %0
  %2 = phi i64 [ %105, %104 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %106

4:                                                ; preds = %102, %1
  %5 = phi i64 [ %103, %102 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %104

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 55, i32 -2)
  %8 = load i32, ptr @_anonymous0
  switch i32 %8, label %9 [
    i32 0, label %107
    i32 1, label %109
    i32 2, label %111
  ]

9:                                                ; preds = %107, %109, %111, %7
  %10 = phi ptr [ %112, %111 ], [ %110, %109 ], [ %108, %107 ], [ @rms_in_cons_buff_0, %7 ]
  %11 = getelementptr [2048 x bfloat], ptr %10, i32 0, i32 0
  br label %12

12:                                               ; preds = %9
  %13 = load i32, ptr @_anonymous0
  switch i32 %13, label %14 [
    i32 0, label %113
    i32 1, label %115
    i32 2, label %117
  ]

14:                                               ; preds = %113, %115, %117, %12
  %15 = phi ptr [ %118, %117 ], [ %116, %115 ], [ %114, %113 ], [ @rms_in_cons_buff_1, %12 ]
  %16 = getelementptr [2048 x bfloat], ptr %15, i32 0, i32 0
  br label %17

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @op0_layer_fused_pre_rms_col0_bf16(ptr %11, ptr %16, ptr @bq_L3L2_buff_0, i32 2048)
  call void @llvm.aie2p.release(i32 54, i32 2)
  %18 = load i32, ptr @_anonymous0
  %19 = add i32 %18, 2
  %20 = icmp sge i32 %19, 3
  %21 = add i32 %18, -1
  %22 = select i1 %20, i32 %21, i32 %19
  store i32 %22, ptr @_anonymous0
  call void @llvm.aie2p.release(i32 53, i32 1)
  %23 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4)
  %24 = add i32 %23, 1
  %25 = icmp sge i32 %24, 1
  %26 = select i1 %25, i32 %23, i32 %24
  store i32 %26, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 4)
  br label %27

27:                                               ; preds = %40, %17
  %28 = phi i64 [ %51, %40 ], [ 0, %17 ]
  %29 = icmp slt i64 %28, 256
  br i1 %29, label %30, label %52

30:                                               ; preds = %27
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %31 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  switch i32 %31, label %32 [
    i32 0, label %119
    i32 1, label %121
  ]

32:                                               ; preds = %119, %121, %30
  %33 = phi ptr [ %122, %121 ], [ %120, %119 ], [ @Aqkv_0_cons_buff_0, %30 ]
  %34 = getelementptr [2304 x i8], ptr %33, i32 0, i32 0
  br label %35

35:                                               ; preds = %32
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %36 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  switch i32 %36, label %37 [
    i32 0, label %123
    i32 1, label %125
  ]

37:                                               ; preds = %123, %125, %35
  %38 = phi ptr [ %126, %125 ], [ %124, %123 ], [ @Cqkv_0_buff_0, %35 ]
  %39 = getelementptr [2 x bfloat], ptr %38, i32 0, i32 0
  br label %40

40:                                               ; preds = %37
  call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr %34, ptr %39)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %41 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  %42 = add i32 %41, 1
  %43 = icmp sge i32 %42, 2
  %44 = add i32 %41, -1
  %45 = select i1 %43, i32 %44, i32 %42
  store i32 %45, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %46 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  %47 = add i32 %46, 1
  %48 = icmp sge i32 %47, 2
  %49 = add i32 %46, -1
  %50 = select i1 %48, i32 %49, i32 %47
  store i32 %50, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  %51 = add i64 %28, 1
  br label %27

52:                                               ; preds = %65, %27
  %53 = phi i64 [ %76, %65 ], [ 0, %27 ]
  %54 = icmp slt i64 %53, 64
  br i1 %54, label %55, label %77

55:                                               ; preds = %52
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %56 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  switch i32 %56, label %57 [
    i32 0, label %127
    i32 1, label %129
  ]

57:                                               ; preds = %127, %129, %55
  %58 = phi ptr [ %130, %129 ], [ %128, %127 ], [ @Aqkv_0_cons_buff_0, %55 ]
  %59 = getelementptr [2304 x i8], ptr %58, i32 0, i32 0
  br label %60

60:                                               ; preds = %57
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %61 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  switch i32 %61, label %62 [
    i32 0, label %131
    i32 1, label %133
  ]

62:                                               ; preds = %131, %133, %60
  %63 = phi ptr [ %134, %133 ], [ %132, %131 ], [ @Cqkv_0_buff_0, %60 ]
  %64 = getelementptr [2 x bfloat], ptr %63, i32 0, i32 0
  br label %65

65:                                               ; preds = %62
  call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr %59, ptr %64)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %66 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  %67 = add i32 %66, 1
  %68 = icmp sge i32 %67, 2
  %69 = add i32 %66, -1
  %70 = select i1 %68, i32 %69, i32 %67
  store i32 %70, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %71 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  %72 = add i32 %71, 1
  %73 = icmp sge i32 %72, 2
  %74 = add i32 %71, -1
  %75 = select i1 %73, i32 %74, i32 %72
  store i32 %75, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  %76 = add i64 %53, 1
  br label %52

77:                                               ; preds = %90, %52
  %78 = phi i64 [ %101, %90 ], [ 0, %52 ]
  %79 = icmp slt i64 %78, 64
  br i1 %79, label %80, label %102

80:                                               ; preds = %77
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %81 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  switch i32 %81, label %82 [
    i32 0, label %135
    i32 1, label %137
  ]

82:                                               ; preds = %135, %137, %80
  %83 = phi ptr [ %138, %137 ], [ %136, %135 ], [ @Aqkv_0_cons_buff_0, %80 ]
  %84 = getelementptr [2304 x i8], ptr %83, i32 0, i32 0
  br label %85

85:                                               ; preds = %82
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %86 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  switch i32 %86, label %87 [
    i32 0, label %139
    i32 1, label %141
  ]

87:                                               ; preds = %139, %141, %85
  %88 = phi ptr [ %142, %141 ], [ %140, %139 ], [ @Cqkv_0_buff_0, %85 ]
  %89 = getelementptr [2 x bfloat], ptr %88, i32 0, i32 0
  br label %90

90:                                               ; preds = %87
  call void @op0_layer_fused_qkv_gemv_static_bf16(i32 2, i32 0, ptr %84, ptr %89)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %91 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  %92 = add i32 %91, 1
  %93 = icmp sge i32 %92, 2
  %94 = add i32 %91, -1
  %95 = select i1 %93, i32 %94, i32 %92
  store i32 %95, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 8)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %96 = load i32, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  %97 = add i32 %96, 1
  %98 = icmp sge i32 %97, 2
  %99 = add i32 %96, -1
  %100 = select i1 %98, i32 %99, i32 %97
  store i32 %100, ptr getelementptr inbounds (i8, ptr @_anonymous0, i64 12)
  %101 = add i64 %78, 1
  br label %77

102:                                              ; preds = %77
  %103 = add i64 %5, 1
  br label %4

104:                                              ; preds = %4
  %105 = add i64 %2, 1
  br label %1

106:                                              ; preds = %1
  ret void

107:                                              ; preds = %7
  %108 = phi ptr [ @rms_in_cons_buff_0, %7 ]
  br label %9

109:                                              ; preds = %7
  %110 = phi ptr [ @rms_in_cons_buff_1, %7 ]
  br label %9

111:                                              ; preds = %7
  %112 = phi ptr [ @rms_in_cons_buff_2, %7 ]
  br label %9

113:                                              ; preds = %12
  %114 = phi ptr [ @rms_in_cons_buff_1, %12 ]
  br label %14

115:                                              ; preds = %12
  %116 = phi ptr [ @rms_in_cons_buff_2, %12 ]
  br label %14

117:                                              ; preds = %12
  %118 = phi ptr [ @rms_in_cons_buff_0, %12 ]
  br label %14

119:                                              ; preds = %30
  %120 = phi ptr [ @Aqkv_0_cons_buff_0, %30 ]
  br label %32

121:                                              ; preds = %30
  %122 = phi ptr [ @Aqkv_0_cons_buff_1, %30 ]
  br label %32

123:                                              ; preds = %35
  %124 = phi ptr [ @Cqkv_0_buff_0, %35 ]
  br label %37

125:                                              ; preds = %35
  %126 = phi ptr [ @Cqkv_0_buff_1, %35 ]
  br label %37

127:                                              ; preds = %55
  %128 = phi ptr [ @Aqkv_0_cons_buff_0, %55 ]
  br label %57

129:                                              ; preds = %55
  %130 = phi ptr [ @Aqkv_0_cons_buff_1, %55 ]
  br label %57

131:                                              ; preds = %60
  %132 = phi ptr [ @Cqkv_0_buff_0, %60 ]
  br label %62

133:                                              ; preds = %60
  %134 = phi ptr [ @Cqkv_0_buff_1, %60 ]
  br label %62

135:                                              ; preds = %80
  %136 = phi ptr [ @Aqkv_0_cons_buff_0, %80 ]
  br label %82

137:                                              ; preds = %80
  %138 = phi ptr [ @Aqkv_0_cons_buff_1, %80 ]
  br label %82

139:                                              ; preds = %85
  %140 = phi ptr [ @Cqkv_0_buff_0, %85 ]
  br label %87

141:                                              ; preds = %85
  %142 = phi ptr [ @Cqkv_0_buff_1, %85 ]
  br label %87
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
