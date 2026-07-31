	.file	"bcast_gemv_rr.cpp"
	.section	.text.layer_fused_gemv_bcast_kc256_bf16,"ax",@progbits
	.globl	layer_fused_gemv_bcast_kc256_bf16
	.p2align	4
	.type	layer_fused_gemv_bcast_kc256_bf16,@function
layer_fused_gemv_bcast_kc256_bf16:
	// #135: ppbase-density inner loop (1.5 bundles/vmac) + KC=256 scale folding
	movxm	r0, #19201
	movxm	r1, #4096                      // KC=256 weight bytes per outer iteration
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
	nopa	;		nopb	;		lshl	 r17, r7, r6
	vlda	 bmll1, [sp, #-128];		mov	dj1, r17
	vlda	 bmlh1, [sp, #-64];		vldb	 x0, [p1, dj1];		mov	r18, r16
	movs	dj1, r18;		add	r18, r18, #32
	vldb.unpack	 x1, unpacksign1, [p0, dj1]
	nop
	nop
	movs	dj1, r18;		add	r18, r18, #32
	vldb.unpack	 x1, unpacksign1, [p0, dj1]
	nop
	nop
	movs	dj1, r18;		add	r18, r18, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r0
	vldb.unpack	 x1, unpacksign1, [p0, dj1]
	vsub.f	dm4, dm3, dm0, r3
	movxm	ls, #.LBB0_2
	mova	r19, #0;		nopb	;		movs	dj1, r18;		add	r18, r18, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r0
	vldb.unpack	 x1, unpacksign1, [p0, dj1];		add	r20, r19, #1;		add.nc	lc, r2, #-5
	add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd0;		nopv
	// --- PROLOGUE KICK: first iteration of the ppbase-dense pattern ---
	nopa	;		nopb	;		movs	dj1, r18;		add	r18, r18, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r0
	nopa	;		vldb.unpack	 x1, unpacksign1, [p0, dj1];		vconv.bf16.fp32	 x3, cml4;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		nopv
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm1, dm1, x3, x7, r4
.LBB0_2:
	// --- ppbase-dense inner loop: 3 bundles, 2 vmac, 1.5 bundles/vmac ---
	nopa	;		nopb	;		movs	dj1, r18;		add	r18, r18, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r0
	nopa	;		vldb.unpack	 x1, unpacksign1, [p0, dj1];		vconv.bf16.fp32	 x3, cml4;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		vmac.f	dm2, dm2, x5, x9, r4
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
.L_LEnd0:
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm1, dm1, x3, x7, r4
	// --- epilogue: drain remaining 3 group-pairs (6 groups, 6 vmac) ---
	nopa	;		nopx	;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r0
	vconv.bf16.fp32	 x3, cml4;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		vmac.f	dm2, dm2, x5, x9, r4
	vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
	vmac.f	dm1, dm1, x3, x7, r4
	vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r0
	vconv.bf16.fp32	 x3, cml4;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		vmac.f	dm2, dm2, x5, x9, r4
	vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm4, dm3, dm0, r3
	vmac.f	dm1, dm1, x3, x7, r4
	nop
	vconv.bf16.fp32	 x3, cml4;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		vmac.f	dm2, dm2, x5, x9, r4
	vconv.bf16.fp32	 x5, cmh4;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19
	add	 r16, r16, r5;		vmac.f	dm1, dm1, x3, x7, r4
	eq	 r17, r16, r1
	vconv.bf16.fp32	 x3, cml4;		add	r7, r7, #32;		vextbcst.16	 x9, x0, r20;		vmac.f	dm2, dm2, x5, x9, r4
	vconv.bf16.fp32	 x5, cmh4
	vmac.f	dm1, dm1, x3, x7, r4
	nop
	vmac.f	dm2, dm2, x5, x9, r4
	jz	 r17, #.LBB0_1
	nop
	nop
	nop
	nop
	nop
	// final merge: dm1 + dm2 -> dm0, store via p2
	vadd.f	dm0, dm1, dm2, r0
	nop
	ret	lr
	nop
	nop
	nop
	vst.conv.bf16.fp32	 cml0, [p2, #0]
	nop
.Lfunc_end0:
	.size	layer_fused_gemv_bcast_kc256_bf16, .Lfunc_end0-layer_fused_gemv_bcast_kc256_bf16

	// _ha_noop for aiecc linking
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
