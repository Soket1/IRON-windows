	.file	"bcast_gemv_rr.cpp"
	.section	.text.layer_fused_gemv_bcast_kc256_bf16,"ax",@progbits
	.globl	layer_fused_gemv_bcast_kc256_bf16
	.p2align	4
	.type	layer_fused_gemv_bcast_kc256_bf16,@function
layer_fused_gemv_bcast_kc256_bf16:
	// #135: ppbase-dense inner loop (1.5 b/vmac) adapted for KC=256 geometry
	// Bank map: dm0=bias, dm1=g0_acc, dm2=g1_acc, dm3=dequant_raw, dm4=dequant_norm
	// Pipeline: vups→dm3, vadd→dm4 (cross-bank), vsub→dm4
	//           vconv cml4/cmh4→x3/x5, vmac.f→dm1/dm2
	movxm	r0, #19201
	movxm	r1, #4096                      // KC=256: total weight bytes per K-chunk
	movxm	dj0, #4096
	nopx	;		mov	crrnd, #12
	mov	crunpacksize, #0
	mov	crupsmode, #0
	paddxm	 [sp], #128
	mova	r0, #0;		vbcst.16	 x0, r0
	vbcst.32	 x2, r0
	mov	s0, r0
	vconv.fp32.bf16	cml0, x0
	mova	r2, #16;		vmov	bmll1, x2
	mova	r4, #828;		movx	r3, #60;		vmov	bmlh1, x2
	mova	r6, #1;		vst	 bmll1, [sp, #-128];		movx	r5, #512;		vmov	cmh0, cml0
	mova	r16, #0;		vst	 bmlh1, [sp, #-64];		movx	r7, #0;		vmov	cml2, cml1
.LBB0_1:
	// --- Outer loop: 8 K-chunks ---
	nopa	;		nopb	;		lshl	 r17, r7, r6
	vlda	 bmll1, [sp, #-128];		mov	dj1, r17
	vlda	 bmlh1, [sp, #-64];		vldb	 x0, [p1, dj1];		mov	r18, r16
	// --- Prologue: dequant groups 0-1, prime x3/x5 pipeline ---
	movs	dj1, r18;		add	r18, r18, #32
	vldb.unpack	 x1, unpacksign1, [p0, dj1]
	nop
	mova	r19, #0
	add	r20, r19, #1
	nop
	vmov	cml3, cml1
	movs	dj1, r18;		add	r18, r18, #32;		vmov	cml4, cml1
	// Group 0: dequant (vups→dm3, vadd→dm4 cross-bank)
	vldb.unpack	 x1, unpacksign1, [p0, dj1];		vups.4x	dm3, x1, s0, upssign1;		vadd	dm4, dm3, dm0, r0
	add.nc	lc, r2, #-5                     // lc = 11 iterations (groups 2-13, last 2 in epilogue)
	movs	dj1, r18;		add	r18, r18, #32
	vldb.unpack	 x1, unpacksign1, [p0, dj1]    // pre-load group 1 weights for inner loop kick
	// Group 0 vsub (norm) + activation broadcast
	add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm4, dm0, r3
	add	r20, r19, #1;		vextbcst.16	 x9, x0, r20
	// Group 0 vconv (both halves from dm4)
	nopa	;		nopb	;		movs	dj1, r18;		add	r18, r18, #32
	nopa	;		nopb	;		vconv.bf16.fp32	 x3, cml4;		nopxm	;		nopv
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh4;		nopxm	;		nopv
	// Group 0 MAC (lane0→dm1) + dequant group 1 (vups→dm3, vadd→dm4)
	movxm	ls, #.LBB0_2
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm1, dm1, x3, x7, r4
	// Kick: group 1 dequant + vconv prime (no vmac for x5 yet — x5 is group 0, delayed)
	nopa	;		nopb	;		movs	dj1, r18;		add	r18, r18, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm4, dm3, dm0, r0
	nopa	;		vldb.unpack	 x1, unpacksign1, [p0, dj1];		vconv.bf16.fp32	 x3, cml4;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		nopv
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd0;		nopv
.LBB0_2:
	// --- ppbase-dense inner loop: 3+1 bundles, 2 vmac, 1.5 b/vmac ---
	// Bundle 0: address update + dequant next group
	nopa	;		nopb	;		movs	dj1, r18;		add	r18, r18, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm4, dm3, dm0, r0
	// Bundle 1: load + vconv + bcst + MAC g1 (5-slot co-issue!)
	nopa	;		vldb.unpack	 x1, unpacksign1, [p0, dj1];		vconv.bf16.fp32	 x3, cml4;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		vmac.f	dm2, dm2, x5, x9, r4
	// Bundle 2: vconv + bcst + vsub (4-slot)
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
.L_LEnd0:
	// Bundle 3: MAC g0
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm1, dm1, x3, x7, r4
	// --- Epilogue: drain last 2 groups (groups 14-15), merge, scale, store ---
	movs	p3, p0;		mov	m0, r17
	padda	 [p3], m0;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm4, dm3, dm0, r0
	vldb	 x0, [p3, dj0];		vconv.bf16.fp32	 x3, cml4
	vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
	add	 r16, r16, r5;		vextbcst.16	 x9, x0, r20;		vmac.f	dm2, dm2, x5, x9, r4
	eq	 r17, r16, r1;		vmac.f	dm1, dm1, x3, x7, r4
	add	r7, r7, #32
	nop
	nop
	vconv.bf16.fp32	 x3, cml4
	vconv.bf16.fp32	 x5, cmh4
	vmac.f	dm2, dm2, x5, x9, r4
	vmac.f	dm1, dm1, x3, x7, r4
	nop
	nop
	// Merge g0+g1 accumulators → dm1, then zero dm2 for scale fold
	vadd.f	dm1, dm1, dm2, r0
	nop
	nop
	vmov	cml2, cml1                      // zero dm2 (cml1 was initialized to zero at prologue)
	nop
	nop
	// Scale folding: merged_dm1 * scale → dm2
	vconv.bf16.fp32	 x2, cml1
	nop
	vmac.f	dm2, dm2, x2, x0, r4
	jz	 r17, #.LBB0_1
	nop
	nop
	nop
	nop
	nop
	ret	lr
	nop
	nop
	nop
	vst.conv.bf16.fp32	 cml2, [p2, #0];		paddxm	 [sp], #-128
	nop
.Lfunc_end0:
	.size	layer_fused_gemv_bcast_kc256_bf16, .Lfunc_end0-layer_fused_gemv_bcast_kc256_bf16

	.section	.text._ha_noop,"ax",@progbits
	.globl	_ha_noop
	.p2align	4
	.type	_ha_noop,@function
_ha_noop:
	ret	lr
	nop
	nop
	nop
	nop
	nop
	nop
	.size	_ha_noop, .-_ha_noop

	.section	".linker-options","e",@llvm_linker_options
	.section	".note.GNU-stack","",@progbits
	.addrsig
