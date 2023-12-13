	.file	"simd-dot.c"
	.text
	.p2align 4
	.globl	hpc_gettime
	.type	hpc_gettime, @function
hpc_gettime:
.LFB0:
	.cfi_startproc
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movl	$1, %edi
	movq	%rsp, %rsi
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	call	clock_gettime@PLT
	vxorps	%xmm1, %xmm1, %xmm1
	vcvtsi2sdq	8(%rsp), %xmm1, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vcvtsi2sdq	(%rsp), %xmm1, %xmm1
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	vaddsd	%xmm1, %xmm0, %xmm0
	jne	.L5
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L5:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE0:
	.size	hpc_gettime, .-hpc_gettime
	.p2align 4
	.globl	serial_dot
	.type	serial_dot, @function
serial_dot:
.LFB13:
	.cfi_startproc
	testl	%edx, %edx
	jle	.L9
	movslq	%edx, %rdx
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	salq	$2, %rdx
	.p2align 4
	.p2align 3
.L8:
	vmovss	(%rdi,%rax), %xmm1
	vmulss	(%rsi,%rax), %xmm1, %xmm1
	addq	$4, %rax
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	cmpq	%rdx, %rax
	jne	.L8
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	ret
	.p2align 4
	.p2align 3
.L9:
	vxorps	%xmm0, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE13:
	.size	serial_dot, .-serial_dot
	.p2align 4
	.globl	simd_dot
	.type	simd_dot, @function
simd_dot:
.LFB14:
	.cfi_startproc
	movslq	%edx, %r8
	subq	$3, %r8
	cmpl	$3, %edx
	je	.L14
	xorl	%eax, %eax
	vxorps	%xmm1, %xmm1, %xmm1
	xorl	%edx, %edx
	.p2align 4
	.p2align 3
.L13:
	vmovaps	(%rdi,%rax), %xmm3
	addl	$4, %edx
	vaddps	(%rsi,%rax), %xmm3, %xmm0
	movslq	%edx, %rcx
	addq	$16, %rax
	vaddps	%xmm0, %xmm1, %xmm1
	cmpq	%r8, %rcx
	jb	.L13
	vcvtss2sd	%xmm1, %xmm1, %xmm0
	vxorpd	%xmm2, %xmm2, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vcvtss2sd	%xmm2, %xmm2, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vcvtss2sd	%xmm2, %xmm2, %xmm2
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	ret
	.p2align 4
	.p2align 3
.L14:
	vxorps	%xmm0, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE14:
	.size	simd_dot, .-simd_dot
	.p2align 4
	.globl	fill
	.type	fill, @function
fill:
.LFB15:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %r8
	movl	%edx, %edi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$64, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	vmovaps	.LC4(%rip), %xmm0
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	vmovaps	%xmm0, 16(%rsp)
	vmovaps	.LC5(%rip), %xmm0
	vmovaps	%xmm0, 32(%rsp)
	testl	%edi, %edi
	jle	.L16
	movl	$0xc0000000, (%r8)
	movq	%rsi, %r9
	movl	$0x3f000000, (%rsi)
	cmpl	$1, %edi
	je	.L16
	leal	-2(%rdi), %edx
	cmpl	$2, %edx
	jbe	.L26
	movq	%rsi, %rax
	subq	%r8, %rax
	subq	$4, %rax
	cmpq	$24, %rax
	ja	.L47
.L26:
	movl	$1, %eax
	.p2align 4
	.p2align 3
.L24:
	movq	%rax, %rdx
	andl	$3, %edx
	vmovss	32(%rsp,%rdx,4), %xmm0
	vmovss	16(%rsp,%rdx,4), %xmm1
	vmovss	%xmm1, (%r8,%rax,4)
	vmovss	%xmm0, (%r9,%rax,4)
	incq	%rax
	cmpq	%rdi, %rax
	jne	.L24
.L16:
	movq	56(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L48
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4
	.p2align 3
.L47:
	.cfi_restore_state
	leal	-1(%rdi), %eax
	cmpl	$6, %edx
	jbe	.L27
	vmovdqa	.LC3(%rip), %ymm3
	movl	%eax, %edx
	movl	%edi, 8(%rsp)
	leaq	16(%rsp), %rsi
	shrl	$3, %edx
	leaq	32(%rsp), %rcx
	movl	$4, %r13d
	movl	%eax, 12(%rsp)
	salq	$5, %rdx
	leaq	4(%rdx), %rbx
	movl	$8, %edx
	vmovd	%edx, %xmm5
	movl	$3, %edx
	movq	%rbx, %rdi
	vmovd	%edx, %xmm4
	vpbroadcastd	%xmm5, %ymm5
	vpbroadcastd	%xmm4, %ymm4
	.p2align 4
	.p2align 3
.L28:
	vmovdqa	%ymm3, %ymm0
	vpaddd	%ymm5, %ymm3, %ymm3
	vpand	%ymm4, %ymm0, %ymm0
	vmovd	%xmm0, %r10d
	vpextrd	$1, %xmm0, %r14d
	vpextrd	$2, %xmm0, %r11d
	vpextrd	$3, %xmm0, %eax
	vextracti128	$0x1, %ymm0, %xmm0
	movslq	%r11d, %r11
	movslq	%r10d, %r10
	vmovd	%xmm0, %ebx
	vpextrd	$2, %xmm0, %r12d
	vpextrd	$1, %xmm0, %r15d
	salq	$2, %r11
	movslq	%ebx, %rbx
	movslq	%r12d, %r12
	vpextrd	$3, %xmm0, %edx
	movslq	%r15d, %r15
	salq	$2, %rbx
	salq	$2, %r12
	movslq	%edx, %rdx
	salq	$2, %r10
	vmovss	(%r12,%rcx), %xmm0
	vmovss	(%rbx,%rcx), %xmm1
	cltq
	movslq	%r14d, %r14
	vinsertps	$0x10, (%rcx,%rdx,4), %xmm0, %xmm0
	vinsertps	$0x10, (%rcx,%r15,4), %xmm1, %xmm1
	vmovss	(%rsi,%r12), %xmm6
	vinsertps	$0x10, (%rsi,%rdx,4), %xmm6, %xmm6
	vmovlhps	%xmm0, %xmm1, %xmm1
	vmovss	(%r11,%rcx), %xmm0
	vinsertps	$0x10, (%rcx,%rax,4), %xmm0, %xmm2
	vmovss	(%r10,%rcx), %xmm0
	vinsertps	$0x10, (%rcx,%r14,4), %xmm0, %xmm0
	vmovlhps	%xmm2, %xmm0, %xmm0
	vmovss	(%rsi,%rbx), %xmm2
	vinsertps	$0x10, (%rsi,%r15,4), %xmm2, %xmm2
	vinsertf128	$0x1, %xmm1, %ymm0, %ymm0
	vmovss	(%rsi,%r10), %xmm1
	vinsertps	$0x10, (%rsi,%r14,4), %xmm1, %xmm1
	vmovlhps	%xmm6, %xmm2, %xmm2
	vmovss	(%rsi,%r11), %xmm6
	vinsertps	$0x10, (%rsi,%rax,4), %xmm6, %xmm6
	vmovlhps	%xmm6, %xmm1, %xmm1
	vinsertf128	$0x1, %xmm2, %ymm1, %ymm1
	vmovups	%ymm1, (%r8,%r13)
	vmovups	%ymm0, (%r9,%r13)
	addq	$32, %r13
	cmpq	%r13, %rdi
	jne	.L28
	movl	12(%rsp), %eax
	movl	8(%rsp), %edi
	movl	%eax, %esi
	andl	$-8, %esi
	leal	1(%rsi), %ecx
	movl	%ecx, %edx
	testb	$7, %al
	je	.L43
	movl	%edi, %eax
	subl	%esi, %eax
	leal	-1(%rax), %r11d
	subl	$2, %eax
	cmpl	$2, %eax
	jbe	.L44
	vzeroupper
.L30:
	vmovd	%edx, %xmm7
	movl	$3, %eax
	salq	$2, %rcx
	vpshufd	$0, %xmm7, %xmm0
	vpaddd	.LC8(%rip), %xmm0, %xmm0
	vmovd	%eax, %xmm1
	vpshufd	$0, %xmm1, %xmm1
	vpand	%xmm1, %xmm0, %xmm0
	vpextrd	$2, %xmm0, %r10d
	vmovd	%xmm0, %esi
	vpextrd	$3, %xmm0, %eax
	movslq	%r10d, %r10
	movslq	%esi, %rsi
	cltq
	vpextrd	$1, %xmm0, %ebx
	salq	$2, %r10
	salq	$2, %rsi
	movslq	%ebx, %rbx
	vmovss	32(%rsp,%r10), %xmm0
	vmovss	16(%rsp,%r10), %xmm2
	vinsertps	$0x10, 32(%rsp,%rax,4), %xmm0, %xmm1
	vmovss	32(%rsp,%rsi), %xmm0
	vinsertps	$0x10, 32(%rsp,%rbx,4), %xmm0, %xmm0
	vinsertps	$0x10, 16(%rsp,%rax,4), %xmm2, %xmm2
	movl	%r11d, %eax
	andl	$-4, %eax
	addl	%eax, %edx
	andl	$3, %r11d
	vmovlhps	%xmm1, %xmm0, %xmm0
	vmovss	16(%rsp,%rsi), %xmm1
	vinsertps	$0x10, 16(%rsp,%rbx,4), %xmm1, %xmm1
	vmovlhps	%xmm2, %xmm1, %xmm1
	vmovups	%xmm1, (%r8,%rcx)
	vmovups	%xmm0, (%r9,%rcx)
	je	.L16
.L21:
	movslq	%edx, %rax
	.p2align 4
	.p2align 3
.L23:
	movq	%rax, %rdx
	andl	$3, %edx
	vmovss	32(%rsp,%rdx,4), %xmm0
	vmovss	16(%rsp,%rdx,4), %xmm1
	vmovss	%xmm1, (%r8,%rax,4)
	vmovss	%xmm0, (%r9,%rax,4)
	incq	%rax
	cmpl	%eax, %edi
	jg	.L23
	jmp	.L16
.L43:
	vzeroupper
	jmp	.L16
.L27:
	movl	%eax, %r11d
	movl	$1, %edx
	movl	$1, %ecx
	jmp	.L30
.L44:
	vzeroupper
	jmp	.L21
.L48:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE15:
	.size	fill, .-fill
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC12:
	.string	"Usage: %s [n]\n"
.LC13:
	.string	"simd-dot.c"
.LC14:
	.string	"n > 0"
.LC15:
	.string	"size < 1024*1024*200UL"
.LC16:
	.string	"0 == ret"
.LC17:
	.string	"Array length = %d\n"
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC19:
	.string	"Serial: result=%f, avg. time=%f (%d runs)\n"
	.align 8
.LC20:
	.string	"SIMD  : result=%f, avg. time=%f (%d runs)\n"
	.section	.rodata.str1.1
.LC23:
	.string	"Check FAILED\n"
.LC24:
	.string	"Speedup (serial/SIMD) %f\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB16:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$104, %rsp
	.cfi_def_cfa_offset 160
	movq	%fs:40, %rax
	movq	%rax, 88(%rsp)
	xorl	%eax, %eax
	cmpl	$2, %edi
	jg	.L76
	je	.L77
	movl	$41943040, %ebx
	movl	$10485760, 40(%rsp)
	movl	$10485760, %r13d
.L52:
	leaq	48(%rsp), %rdi
	movq	%rbx, %rdx
	movl	$32, %esi
	call	posix_memalign@PLT
	testl	%eax, %eax
	jne	.L54
	leaq	56(%rsp), %rdi
	movq	%rbx, %rdx
	movl	$32, %esi
	movq	48(%rsp), %r15
	call	posix_memalign@PLT
	movl	%eax, 44(%rsp)
	testl	%eax, %eax
	jne	.L78
	movl	40(%rsp), %ebx
	leaq	.LC17(%rip), %rdi
	xorl	%eax, %eax
	movl	$10, %ebp
	movq	56(%rsp), %r14
	movl	%ebx, %esi
	call	printf@PLT
	movl	%ebx, %edx
	movq	%r14, %rsi
	movq	%r15, %rdi
	call	fill
	movslq	%ebx, %rdx
	movq	$0x000000000, 8(%rsp)
	leaq	64(%rsp), %rbx
	leaq	0(,%rdx,4), %r12
	.p2align 4
	.p2align 3
.L57:
	movq	%rbx, %rsi
	movl	$1, %edi
	call	clock_gettime@PLT
	vxorpd	%xmm5, %xmm5, %xmm5
	xorl	%eax, %eax
	vxorpd	%xmm2, %xmm2, %xmm2
	vcvtsi2sdq	72(%rsp), %xmm5, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vcvtsi2sdq	64(%rsp), %xmm5, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm6
	vmovsd	%xmm6, 16(%rsp)
	.p2align 4
	.p2align 3
.L58:
	vmovss	(%r15,%rax), %xmm0
	vmulss	(%r14,%rax), %xmm0, %xmm0
	addq	$4, %rax
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm2, %xmm2
	cmpq	%r12, %rax
	jne	.L58
	movq	%rbx, %rsi
	movl	$1, %edi
	vmovsd	%xmm2, 24(%rsp)
	call	clock_gettime@PLT
	vxorpd	%xmm6, %xmm6, %xmm6
	decl	%ebp
	vcvtsi2sdq	72(%rsp), %xmm6, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vcvtsi2sdq	64(%rsp), %xmm6, %xmm1
	vmovsd	24(%rsp), %xmm2
	vaddsd	%xmm1, %xmm0, %xmm0
	vsubsd	16(%rsp), %xmm0, %xmm0
	vaddsd	8(%rsp), %xmm0, %xmm3
	vmovsd	%xmm3, 8(%rsp)
	jne	.L57
	vdivsd	.LC18(%rip), %xmm3, %xmm6
	movl	40(%rsp), %edx
	movq	%r14, %rsi
	movq	%r15, %rdi
	vmovsd	%xmm2, 16(%rsp)
	leaq	-3(%r13), %rbp
	movl	$10, %r12d
	vmovsd	%xmm6, 32(%rsp)
	call	fill
	vmovsd	16(%rsp), %xmm2
	movq	$0x000000000, 8(%rsp)
	.p2align 4
	.p2align 3
.L61:
	movq	%rbx, %rsi
	movl	$1, %edi
	vmovsd	%xmm2, 24(%rsp)
	call	clock_gettime@PLT
	vxorpd	%xmm7, %xmm7, %xmm7
	cmpq	$3, %r13
	vcvtsi2sdq	72(%rsp), %xmm7, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vcvtsi2sdq	64(%rsp), %xmm7, %xmm1
	vmovsd	24(%rsp), %xmm2
	vaddsd	%xmm1, %xmm0, %xmm7
	vmovsd	%xmm7, 16(%rsp)
	je	.L67
	xorl	%edx, %edx
	xorl	%eax, %eax
	vxorps	%xmm1, %xmm1, %xmm1
	.p2align 4
	.p2align 3
.L60:
	vmovaps	(%r15,%rax), %xmm4
	addl	$4, %edx
	vaddps	(%r14,%rax), %xmm4, %xmm0
	movslq	%edx, %rsi
	addq	$16, %rax
	vaddps	%xmm0, %xmm1, %xmm1
	cmpq	%rbp, %rsi
	jb	.L60
	vcvtss2sd	%xmm1, %xmm1, %xmm0
	vxorpd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vcvtss2sd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vcvtss2sd	%xmm3, %xmm3, %xmm3
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vaddsd	%xmm3, %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm3
	vmovss	%xmm3, 40(%rsp)
.L59:
	movq	%rbx, %rsi
	movl	$1, %edi
	vmovsd	%xmm2, 24(%rsp)
	call	clock_gettime@PLT
	vxorpd	%xmm5, %xmm5, %xmm5
	decl	%r12d
	vcvtsi2sdq	72(%rsp), %xmm5, %xmm0
	vdivsd	.LC0(%rip), %xmm0, %xmm0
	vcvtsi2sdq	64(%rsp), %xmm5, %xmm1
	vmovsd	24(%rsp), %xmm2
	vaddsd	%xmm1, %xmm0, %xmm0
	vsubsd	16(%rsp), %xmm0, %xmm0
	vaddsd	8(%rsp), %xmm0, %xmm6
	vmovsd	%xmm6, 8(%rsp)
	jne	.L61
	vdivsd	.LC18(%rip), %xmm6, %xmm6
	vmovsd	32(%rsp), %xmm1
	vcvtsd2ss	%xmm2, %xmm2, %xmm2
	movl	$10, %esi
	leaq	.LC19(%rip), %rdi
	movl	$2, %eax
	vcvtss2sd	%xmm2, %xmm2, %xmm0
	vmovss	%xmm2, 8(%rsp)
	vmovq	%xmm6, %rbx
	call	printf@PLT
	vcvtss2sd	40(%rsp), %xmm0, %xmm0
	movl	$10, %esi
	vmovq	%rbx, %xmm1
	leaq	.LC20(%rip), %rdi
	movl	$2, %eax
	call	printf@PLT
	vmovss	8(%rsp), %xmm2
	vsubss	40(%rsp), %xmm2, %xmm2
	vandps	.LC21(%rip), %xmm2, %xmm2
	vcomiss	.LC22(%rip), %xmm2
	ja	.L79
	vmovsd	32(%rsp), %xmm7
	vmovq	%rbx, %xmm6
	leaq	.LC24(%rip), %rdi
	movl	$1, %eax
	vdivsd	%xmm6, %xmm7, %xmm0
	call	printf@PLT
	movq	%r15, %rdi
	call	free@PLT
	movq	%r14, %rdi
	call	free@PLT
.L49:
	movq	88(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L80
	movl	44(%rsp), %eax
	addq	$104, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4
	.p2align 3
.L67:
	.cfi_restore_state
	movl	$0x00000000, 40(%rsp)
	jmp	.L59
.L79:
	movq	stderr(%rip), %rcx
	movl	$13, %edx
	movl	$1, %esi
	leaq	.LC23(%rip), %rdi
	call	fwrite@PLT
.L51:
	movl	$1, 44(%rsp)
	jmp	.L49
.L76:
	movq	(%rsi), %rdx
	movq	stderr(%rip), %rdi
	leaq	.LC12(%rip), %rsi
	call	fprintf@PLT
	jmp	.L51
.L77:
	movq	8(%rsi), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movl	%eax, 40(%rsp)
	testl	%eax, %eax
	jle	.L81
	movslq	%eax, %r13
	leaq	0(,%r13,4), %rbx
	cmpq	$209715199, %rbx
	jbe	.L52
	leaq	__PRETTY_FUNCTION__.0(%rip), %rcx
	movl	$227, %edx
	leaq	.LC13(%rip), %rsi
	leaq	.LC15(%rip), %rdi
	call	__assert_fail@PLT
.L81:
	leaq	__PRETTY_FUNCTION__.0(%rip), %rcx
	movl	$223, %edx
	leaq	.LC13(%rip), %rsi
	leaq	.LC14(%rip), %rdi
	call	__assert_fail@PLT
.L54:
	leaq	__PRETTY_FUNCTION__.0(%rip), %rcx
	movl	$229, %edx
	leaq	.LC13(%rip), %rsi
	leaq	.LC16(%rip), %rdi
	call	__assert_fail@PLT
.L78:
	leaq	__PRETTY_FUNCTION__.0(%rip), %rcx
	movl	$231, %edx
	leaq	.LC13(%rip), %rsi
	leaq	.LC16(%rip), %rdi
	call	__assert_fail@PLT
.L80:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE16:
	.size	main, .-main
	.section	.rodata
	.type	__PRETTY_FUNCTION__.0, @object
	.size	__PRETTY_FUNCTION__.0, 5
__PRETTY_FUNCTION__.0:
	.string	"main"
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1104006501
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC3:
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.long	8
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC4:
	.long	-1073741824
	.long	0
	.long	1082130432
	.long	1073741824
	.align 16
.LC5:
	.long	1056964608
	.long	0
	.long	1031798784
	.long	1056964608
	.align 16
.LC8:
	.long	0
	.long	1
	.long	2
	.long	3
	.section	.rodata.cst8
	.align 8
.LC18:
	.long	0
	.long	1076101120
	.section	.rodata.cst16
	.align 16
.LC21:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC22:
	.long	925353388
	.ident	"GCC: (GNU) 13.2.1 20230801"
	.section	.note.GNU-stack,"",@progbits
