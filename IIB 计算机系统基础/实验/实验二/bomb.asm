
bomb:     file format elf32-i386


Disassembly of section .init:

080486f4 <_init>:
 80486f4:	53                   	push   %ebx
 80486f5:	83 ec 08             	sub    $0x8,%esp
 80486f8:	e8 13 02 00 00       	call   8048910 <__x86.get_pc_thunk.bx>
 80486fd:	81 c3 03 39 00 00    	add    $0x3903,%ebx
 8048703:	8b 83 fc ff ff ff    	mov    -0x4(%ebx),%eax
 8048709:	85 c0                	test   %eax,%eax
 804870b:	74 05                	je     8048712 <_init+0x1e>
 804870d:	e8 be 01 00 00       	call   80488d0 <__gmon_start__@plt>
 8048712:	83 c4 08             	add    $0x8,%esp
 8048715:	5b                   	pop    %ebx
 8048716:	c3                   	ret    

Disassembly of section .plt:

08048720 <.plt>:
 8048720:	ff 35 04 c0 04 08    	pushl  0x804c004
 8048726:	ff 25 08 c0 04 08    	jmp    *0x804c008
 804872c:	00 00                	add    %al,(%eax)
	...

08048730 <read@plt>:
 8048730:	ff 25 0c c0 04 08    	jmp    *0x804c00c
 8048736:	68 00 00 00 00       	push   $0x0
 804873b:	e9 e0 ff ff ff       	jmp    8048720 <.plt>

08048740 <fflush@plt>:
 8048740:	ff 25 10 c0 04 08    	jmp    *0x804c010
 8048746:	68 08 00 00 00       	push   $0x8
 804874b:	e9 d0 ff ff ff       	jmp    8048720 <.plt>

08048750 <fgets@plt>:
 8048750:	ff 25 14 c0 04 08    	jmp    *0x804c014
 8048756:	68 10 00 00 00       	push   $0x10
 804875b:	e9 c0 ff ff ff       	jmp    8048720 <.plt>

08048760 <signal@plt>:
 8048760:	ff 25 18 c0 04 08    	jmp    *0x804c018
 8048766:	68 18 00 00 00       	push   $0x18
 804876b:	e9 b0 ff ff ff       	jmp    8048720 <.plt>

08048770 <sleep@plt>:
 8048770:	ff 25 1c c0 04 08    	jmp    *0x804c01c
 8048776:	68 20 00 00 00       	push   $0x20
 804877b:	e9 a0 ff ff ff       	jmp    8048720 <.plt>

08048780 <alarm@plt>:
 8048780:	ff 25 20 c0 04 08    	jmp    *0x804c020
 8048786:	68 28 00 00 00       	push   $0x28
 804878b:	e9 90 ff ff ff       	jmp    8048720 <.plt>

08048790 <__stack_chk_fail@plt>:
 8048790:	ff 25 24 c0 04 08    	jmp    *0x804c024
 8048796:	68 30 00 00 00       	push   $0x30
 804879b:	e9 80 ff ff ff       	jmp    8048720 <.plt>

080487a0 <strcpy@plt>:
 80487a0:	ff 25 28 c0 04 08    	jmp    *0x804c028
 80487a6:	68 38 00 00 00       	push   $0x38
 80487ab:	e9 70 ff ff ff       	jmp    8048720 <.plt>

080487b0 <getenv@plt>:
 80487b0:	ff 25 2c c0 04 08    	jmp    *0x804c02c
 80487b6:	68 40 00 00 00       	push   $0x40
 80487bb:	e9 60 ff ff ff       	jmp    8048720 <.plt>

080487c0 <puts@plt>:
 80487c0:	ff 25 30 c0 04 08    	jmp    *0x804c030
 80487c6:	68 48 00 00 00       	push   $0x48
 80487cb:	e9 50 ff ff ff       	jmp    8048720 <.plt>

080487d0 <__memmove_chk@plt>:
 80487d0:	ff 25 34 c0 04 08    	jmp    *0x804c034
 80487d6:	68 50 00 00 00       	push   $0x50
 80487db:	e9 40 ff ff ff       	jmp    8048720 <.plt>

080487e0 <exit@plt>:
 80487e0:	ff 25 38 c0 04 08    	jmp    *0x804c038
 80487e6:	68 58 00 00 00       	push   $0x58
 80487eb:	e9 30 ff ff ff       	jmp    8048720 <.plt>

080487f0 <__libc_start_main@plt>:
 80487f0:	ff 25 3c c0 04 08    	jmp    *0x804c03c
 80487f6:	68 60 00 00 00       	push   $0x60
 80487fb:	e9 20 ff ff ff       	jmp    8048720 <.plt>

08048800 <write@plt>:
 8048800:	ff 25 40 c0 04 08    	jmp    *0x804c040
 8048806:	68 68 00 00 00       	push   $0x68
 804880b:	e9 10 ff ff ff       	jmp    8048720 <.plt>

08048810 <__isoc99_sscanf@plt>:
 8048810:	ff 25 44 c0 04 08    	jmp    *0x804c044
 8048816:	68 70 00 00 00       	push   $0x70
 804881b:	e9 00 ff ff ff       	jmp    8048720 <.plt>

08048820 <fopen@plt>:
 8048820:	ff 25 48 c0 04 08    	jmp    *0x804c048
 8048826:	68 78 00 00 00       	push   $0x78
 804882b:	e9 f0 fe ff ff       	jmp    8048720 <.plt>

08048830 <__errno_location@plt>:
 8048830:	ff 25 4c c0 04 08    	jmp    *0x804c04c
 8048836:	68 80 00 00 00       	push   $0x80
 804883b:	e9 e0 fe ff ff       	jmp    8048720 <.plt>

08048840 <__printf_chk@plt>:
 8048840:	ff 25 50 c0 04 08    	jmp    *0x804c050
 8048846:	68 88 00 00 00       	push   $0x88
 804884b:	e9 d0 fe ff ff       	jmp    8048720 <.plt>

08048850 <socket@plt>:
 8048850:	ff 25 54 c0 04 08    	jmp    *0x804c054
 8048856:	68 90 00 00 00       	push   $0x90
 804885b:	e9 c0 fe ff ff       	jmp    8048720 <.plt>

08048860 <__fprintf_chk@plt>:
 8048860:	ff 25 58 c0 04 08    	jmp    *0x804c058
 8048866:	68 98 00 00 00       	push   $0x98
 804886b:	e9 b0 fe ff ff       	jmp    8048720 <.plt>

08048870 <gethostbyname@plt>:
 8048870:	ff 25 5c c0 04 08    	jmp    *0x804c05c
 8048876:	68 a0 00 00 00       	push   $0xa0
 804887b:	e9 a0 fe ff ff       	jmp    8048720 <.plt>

08048880 <strtol@plt>:
 8048880:	ff 25 60 c0 04 08    	jmp    *0x804c060
 8048886:	68 a8 00 00 00       	push   $0xa8
 804888b:	e9 90 fe ff ff       	jmp    8048720 <.plt>

08048890 <connect@plt>:
 8048890:	ff 25 64 c0 04 08    	jmp    *0x804c064
 8048896:	68 b0 00 00 00       	push   $0xb0
 804889b:	e9 80 fe ff ff       	jmp    8048720 <.plt>

080488a0 <close@plt>:
 80488a0:	ff 25 68 c0 04 08    	jmp    *0x804c068
 80488a6:	68 b8 00 00 00       	push   $0xb8
 80488ab:	e9 70 fe ff ff       	jmp    8048720 <.plt>

080488b0 <__ctype_b_loc@plt>:
 80488b0:	ff 25 6c c0 04 08    	jmp    *0x804c06c
 80488b6:	68 c0 00 00 00       	push   $0xc0
 80488bb:	e9 60 fe ff ff       	jmp    8048720 <.plt>

080488c0 <__sprintf_chk@plt>:
 80488c0:	ff 25 70 c0 04 08    	jmp    *0x804c070
 80488c6:	68 c8 00 00 00       	push   $0xc8
 80488cb:	e9 50 fe ff ff       	jmp    8048720 <.plt>

Disassembly of section .plt.got:

080488d0 <__gmon_start__@plt>:
 80488d0:	ff 25 fc bf 04 08    	jmp    *0x804bffc
 80488d6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

080488e0 <_start>:
 80488e0:	31 ed                	xor    %ebp,%ebp
 80488e2:	5e                   	pop    %esi
 80488e3:	89 e1                	mov    %esp,%ecx
 80488e5:	83 e4 f0             	and    $0xfffffff0,%esp
 80488e8:	50                   	push   %eax
 80488e9:	54                   	push   %esp
 80488ea:	52                   	push   %edx
 80488eb:	68 60 9f 04 08       	push   $0x8049f60
 80488f0:	68 00 9f 04 08       	push   $0x8049f00
 80488f5:	51                   	push   %ecx
 80488f6:	56                   	push   %esi
 80488f7:	68 db 89 04 08       	push   $0x80489db
 80488fc:	e8 ef fe ff ff       	call   80487f0 <__libc_start_main@plt>
 8048901:	f4                   	hlt    
 8048902:	66 90                	xchg   %ax,%ax
 8048904:	66 90                	xchg   %ax,%ax
 8048906:	66 90                	xchg   %ax,%ax
 8048908:	66 90                	xchg   %ax,%ax
 804890a:	66 90                	xchg   %ax,%ax
 804890c:	66 90                	xchg   %ax,%ax
 804890e:	66 90                	xchg   %ax,%ax

08048910 <__x86.get_pc_thunk.bx>:
 8048910:	8b 1c 24             	mov    (%esp),%ebx
 8048913:	c3                   	ret    
 8048914:	66 90                	xchg   %ax,%ax
 8048916:	66 90                	xchg   %ax,%ax
 8048918:	66 90                	xchg   %ax,%ax
 804891a:	66 90                	xchg   %ax,%ax
 804891c:	66 90                	xchg   %ax,%ax
 804891e:	66 90                	xchg   %ax,%ax

08048920 <deregister_tm_clones>:
 8048920:	b8 a3 c3 04 08       	mov    $0x804c3a3,%eax
 8048925:	2d a0 c3 04 08       	sub    $0x804c3a0,%eax
 804892a:	83 f8 06             	cmp    $0x6,%eax
 804892d:	76 1a                	jbe    8048949 <deregister_tm_clones+0x29>
 804892f:	b8 00 00 00 00       	mov    $0x0,%eax
 8048934:	85 c0                	test   %eax,%eax
 8048936:	74 11                	je     8048949 <deregister_tm_clones+0x29>
 8048938:	55                   	push   %ebp
 8048939:	89 e5                	mov    %esp,%ebp
 804893b:	83 ec 14             	sub    $0x14,%esp
 804893e:	68 a0 c3 04 08       	push   $0x804c3a0
 8048943:	ff d0                	call   *%eax
 8048945:	83 c4 10             	add    $0x10,%esp
 8048948:	c9                   	leave  
 8048949:	f3 c3                	repz ret 
 804894b:	90                   	nop
 804894c:	8d 74 26 00          	lea    0x0(%esi,%eiz,1),%esi

08048950 <register_tm_clones>:
 8048950:	b8 a0 c3 04 08       	mov    $0x804c3a0,%eax
 8048955:	2d a0 c3 04 08       	sub    $0x804c3a0,%eax
 804895a:	c1 f8 02             	sar    $0x2,%eax
 804895d:	89 c2                	mov    %eax,%edx
 804895f:	c1 ea 1f             	shr    $0x1f,%edx
 8048962:	01 d0                	add    %edx,%eax
 8048964:	d1 f8                	sar    %eax
 8048966:	74 1b                	je     8048983 <register_tm_clones+0x33>
 8048968:	ba 00 00 00 00       	mov    $0x0,%edx
 804896d:	85 d2                	test   %edx,%edx
 804896f:	74 12                	je     8048983 <register_tm_clones+0x33>
 8048971:	55                   	push   %ebp
 8048972:	89 e5                	mov    %esp,%ebp
 8048974:	83 ec 10             	sub    $0x10,%esp
 8048977:	50                   	push   %eax
 8048978:	68 a0 c3 04 08       	push   $0x804c3a0
 804897d:	ff d2                	call   *%edx
 804897f:	83 c4 10             	add    $0x10,%esp
 8048982:	c9                   	leave  
 8048983:	f3 c3                	repz ret 
 8048985:	8d 74 26 00          	lea    0x0(%esi,%eiz,1),%esi
 8048989:	8d bc 27 00 00 00 00 	lea    0x0(%edi,%eiz,1),%edi

08048990 <__do_global_dtors_aux>:
 8048990:	80 3d c8 c3 04 08 00 	cmpb   $0x0,0x804c3c8
 8048997:	75 13                	jne    80489ac <__do_global_dtors_aux+0x1c>
 8048999:	55                   	push   %ebp
 804899a:	89 e5                	mov    %esp,%ebp
 804899c:	83 ec 08             	sub    $0x8,%esp
 804899f:	e8 7c ff ff ff       	call   8048920 <deregister_tm_clones>
 80489a4:	c6 05 c8 c3 04 08 01 	movb   $0x1,0x804c3c8
 80489ab:	c9                   	leave  
 80489ac:	f3 c3                	repz ret 
 80489ae:	66 90                	xchg   %ax,%ax

080489b0 <frame_dummy>:
 80489b0:	b8 10 bf 04 08       	mov    $0x804bf10,%eax
 80489b5:	8b 10                	mov    (%eax),%edx
 80489b7:	85 d2                	test   %edx,%edx
 80489b9:	75 05                	jne    80489c0 <frame_dummy+0x10>
 80489bb:	eb 93                	jmp    8048950 <register_tm_clones>
 80489bd:	8d 76 00             	lea    0x0(%esi),%esi
 80489c0:	ba 00 00 00 00       	mov    $0x0,%edx
 80489c5:	85 d2                	test   %edx,%edx
 80489c7:	74 f2                	je     80489bb <frame_dummy+0xb>
 80489c9:	55                   	push   %ebp
 80489ca:	89 e5                	mov    %esp,%ebp
 80489cc:	83 ec 14             	sub    $0x14,%esp
 80489cf:	50                   	push   %eax
 80489d0:	ff d2                	call   *%edx
 80489d2:	83 c4 10             	add    $0x10,%esp
 80489d5:	c9                   	leave  
 80489d6:	e9 75 ff ff ff       	jmp    8048950 <register_tm_clones>

080489db <main>:
 80489db:	8d 4c 24 04          	lea    0x4(%esp),%ecx
 80489df:	83 e4 f0             	and    $0xfffffff0,%esp
 80489e2:	ff 71 fc             	pushl  -0x4(%ecx)
 80489e5:	55                   	push   %ebp
 80489e6:	89 e5                	mov    %esp,%ebp
 80489e8:	53                   	push   %ebx
 80489e9:	51                   	push   %ecx
 80489ea:	8b 01                	mov    (%ecx),%eax
 80489ec:	8b 59 04             	mov    0x4(%ecx),%ebx
 80489ef:	83 f8 01             	cmp    $0x1,%eax
 80489f2:	0f 84 01 01 00 00    	je     8048af9 <main+0x11e>
 80489f8:	83 f8 02             	cmp    $0x2,%eax
 80489fb:	0f 85 24 01 00 00    	jne    8048b25 <main+0x14a>
 8048a01:	83 ec 08             	sub    $0x8,%esp
 8048a04:	68 88 9f 04 08       	push   $0x8049f88
 8048a09:	ff 73 04             	pushl  0x4(%ebx)
 8048a0c:	e8 0f fe ff ff       	call   8048820 <fopen@plt>
 8048a11:	a3 d0 c3 04 08       	mov    %eax,0x804c3d0
 8048a16:	83 c4 10             	add    $0x10,%esp
 8048a19:	85 c0                	test   %eax,%eax
 8048a1b:	0f 84 e7 00 00 00    	je     8048b08 <main+0x12d>
 8048a21:	e8 08 07 00 00       	call   804912e <initialize_bomb>
 8048a26:	83 ec 0c             	sub    $0xc,%esp
 8048a29:	68 0c a0 04 08       	push   $0x804a00c
 8048a2e:	e8 8d fd ff ff       	call   80487c0 <puts@plt>
 8048a33:	c7 04 24 48 a0 04 08 	movl   $0x804a048,(%esp)
 8048a3a:	e8 81 fd ff ff       	call   80487c0 <puts@plt>
 8048a3f:	e8 da 07 00 00       	call   804921e <read_line>
 8048a44:	89 04 24             	mov    %eax,(%esp)
 8048a47:	e8 f6 00 00 00       	call   8048b42 <phase_1>
 8048a4c:	e8 cc 08 00 00       	call   804931d <phase_defused>
 8048a51:	c7 04 24 74 a0 04 08 	movl   $0x804a074,(%esp)
 8048a58:	e8 63 fd ff ff       	call   80487c0 <puts@plt>
 8048a5d:	e8 bc 07 00 00       	call   804921e <read_line>
 8048a62:	89 04 24             	mov    %eax,(%esp)
 8048a65:	e8 fb 00 00 00       	call   8048b65 <phase_2>
 8048a6a:	e8 ae 08 00 00       	call   804931d <phase_defused>
 8048a6f:	c7 04 24 c1 9f 04 08 	movl   $0x8049fc1,(%esp)
 8048a76:	e8 45 fd ff ff       	call   80487c0 <puts@plt>
 8048a7b:	e8 9e 07 00 00       	call   804921e <read_line>
 8048a80:	89 04 24             	mov    %eax,(%esp)
 8048a83:	e8 44 01 00 00       	call   8048bcc <phase_3>
 8048a88:	e8 90 08 00 00       	call   804931d <phase_defused>
 8048a8d:	c7 04 24 df 9f 04 08 	movl   $0x8049fdf,(%esp)
 8048a94:	e8 27 fd ff ff       	call   80487c0 <puts@plt>
 8048a99:	e8 80 07 00 00       	call   804921e <read_line>
 8048a9e:	89 04 24             	mov    %eax,(%esp)
 8048aa1:	e8 d6 02 00 00       	call   8048d7c <phase_4>
 8048aa6:	e8 72 08 00 00       	call   804931d <phase_defused>
 8048aab:	c7 04 24 a0 a0 04 08 	movl   $0x804a0a0,(%esp)
 8048ab2:	e8 09 fd ff ff       	call   80487c0 <puts@plt>
 8048ab7:	e8 62 07 00 00       	call   804921e <read_line>
 8048abc:	89 04 24             	mov    %eax,(%esp)
 8048abf:	e8 2c 03 00 00       	call   8048df0 <phase_5>
 8048ac4:	e8 54 08 00 00       	call   804931d <phase_defused>
 8048ac9:	c7 04 24 ee 9f 04 08 	movl   $0x8049fee,(%esp)
 8048ad0:	e8 eb fc ff ff       	call   80487c0 <puts@plt>
 8048ad5:	e8 44 07 00 00       	call   804921e <read_line>
 8048ada:	89 04 24             	mov    %eax,(%esp)
 8048add:	e8 9d 03 00 00       	call   8048e7f <phase_6>
 8048ae2:	e8 36 08 00 00       	call   804931d <phase_defused>
 8048ae7:	83 c4 10             	add    $0x10,%esp
 8048aea:	b8 00 00 00 00       	mov    $0x0,%eax
 8048aef:	8d 65 f8             	lea    -0x8(%ebp),%esp
 8048af2:	59                   	pop    %ecx
 8048af3:	5b                   	pop    %ebx
 8048af4:	5d                   	pop    %ebp
 8048af5:	8d 61 fc             	lea    -0x4(%ecx),%esp
 8048af8:	c3                   	ret    
 8048af9:	a1 c0 c3 04 08       	mov    0x804c3c0,%eax
 8048afe:	a3 d0 c3 04 08       	mov    %eax,0x804c3d0
 8048b03:	e9 19 ff ff ff       	jmp    8048a21 <main+0x46>
 8048b08:	ff 73 04             	pushl  0x4(%ebx)
 8048b0b:	ff 33                	pushl  (%ebx)
 8048b0d:	68 8a 9f 04 08       	push   $0x8049f8a
 8048b12:	6a 01                	push   $0x1
 8048b14:	e8 27 fd ff ff       	call   8048840 <__printf_chk@plt>
 8048b19:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
 8048b20:	e8 bb fc ff ff       	call   80487e0 <exit@plt>
 8048b25:	83 ec 04             	sub    $0x4,%esp
 8048b28:	ff 33                	pushl  (%ebx)
 8048b2a:	68 a7 9f 04 08       	push   $0x8049fa7
 8048b2f:	6a 01                	push   $0x1
 8048b31:	e8 0a fd ff ff       	call   8048840 <__printf_chk@plt>
 8048b36:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
 8048b3d:	e8 9e fc ff ff       	call   80487e0 <exit@plt>

08048b42 <phase_1>:
 8048b42:	83 ec 14             	sub    $0x14,%esp
 8048b45:	68 c4 a0 04 08       	push   $0x804a0c4
 8048b4a:	ff 74 24 1c          	pushl  0x1c(%esp)
 8048b4e:	e8 76 05 00 00       	call   80490c9 <strings_not_equal>
 8048b53:	83 c4 10             	add    $0x10,%esp
 8048b56:	85 c0                	test   %eax,%eax
 8048b58:	75 04                	jne    8048b5e <phase_1+0x1c>
 8048b5a:	83 c4 0c             	add    $0xc,%esp
 8048b5d:	c3                   	ret    
 8048b5e:	e8 5b 06 00 00       	call   80491be <explode_bomb>
 8048b63:	eb f5                	jmp    8048b5a <phase_1+0x18>

08048b65 <phase_2>:
 8048b65:	56                   	push   %esi
 8048b66:	53                   	push   %ebx
 8048b67:	83 ec 2c             	sub    $0x2c,%esp
 8048b6a:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 8048b70:	89 44 24 24          	mov    %eax,0x24(%esp)
 8048b74:	31 c0                	xor    %eax,%eax
 8048b76:	8d 44 24 0c          	lea    0xc(%esp),%eax
 8048b7a:	50                   	push   %eax
 8048b7b:	ff 74 24 3c          	pushl  0x3c(%esp)
 8048b7f:	e8 5f 06 00 00       	call   80491e3 <read_six_numbers>
 8048b84:	83 c4 10             	add    $0x10,%esp
 8048b87:	83 7c 24 04 01       	cmpl   $0x1,0x4(%esp)
 8048b8c:	74 05                	je     8048b93 <phase_2+0x2e>
 8048b8e:	e8 2b 06 00 00       	call   80491be <explode_bomb>
 8048b93:	8d 5c 24 04          	lea    0x4(%esp),%ebx
 8048b97:	8d 74 24 18          	lea    0x18(%esp),%esi
 8048b9b:	eb 07                	jmp    8048ba4 <phase_2+0x3f>
 8048b9d:	83 c3 04             	add    $0x4,%ebx
 8048ba0:	39 f3                	cmp    %esi,%ebx
 8048ba2:	74 10                	je     8048bb4 <phase_2+0x4f>
 8048ba4:	8b 03                	mov    (%ebx),%eax
 8048ba6:	01 c0                	add    %eax,%eax
 8048ba8:	39 43 04             	cmp    %eax,0x4(%ebx)
 8048bab:	74 f0                	je     8048b9d <phase_2+0x38>
 8048bad:	e8 0c 06 00 00       	call   80491be <explode_bomb>
 8048bb2:	eb e9                	jmp    8048b9d <phase_2+0x38>
 8048bb4:	8b 44 24 1c          	mov    0x1c(%esp),%eax
 8048bb8:	65 33 05 14 00 00 00 	xor    %gs:0x14,%eax
 8048bbf:	75 06                	jne    8048bc7 <phase_2+0x62>
 8048bc1:	83 c4 24             	add    $0x24,%esp
 8048bc4:	5b                   	pop    %ebx
 8048bc5:	5e                   	pop    %esi
 8048bc6:	c3                   	ret    
 8048bc7:	e8 c4 fb ff ff       	call   8048790 <__stack_chk_fail@plt>

08048bcc <phase_3>:
 8048bcc:	83 ec 28             	sub    $0x28,%esp
 8048bcf:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 8048bd5:	89 44 24 18          	mov    %eax,0x18(%esp)
 8048bd9:	31 c0                	xor    %eax,%eax
 8048bdb:	8d 44 24 14          	lea    0x14(%esp),%eax
 8048bdf:	50                   	push   %eax
 8048be0:	8d 44 24 13          	lea    0x13(%esp),%eax
 8048be4:	50                   	push   %eax
 8048be5:	8d 44 24 18          	lea    0x18(%esp),%eax
 8048be9:	50                   	push   %eax
 8048bea:	68 1a a1 04 08       	push   $0x804a11a
 8048bef:	ff 74 24 3c          	pushl  0x3c(%esp)
 8048bf3:	e8 18 fc ff ff       	call   8048810 <__isoc99_sscanf@plt>
 8048bf8:	83 c4 20             	add    $0x20,%esp
 8048bfb:	83 f8 02             	cmp    $0x2,%eax
 8048bfe:	7e 16                	jle    8048c16 <phase_3+0x4a>
 8048c00:	83 7c 24 04 07       	cmpl   $0x7,0x4(%esp)
 8048c05:	0f 87 03 01 00 00    	ja     8048d0e <phase_3+0x142>
 8048c0b:	8b 44 24 04          	mov    0x4(%esp),%eax
 8048c0f:	ff 24 85 40 a1 04 08 	jmp    *0x804a140(,%eax,4)
 8048c16:	e8 a3 05 00 00       	call   80491be <explode_bomb>
 8048c1b:	eb e3                	jmp    8048c00 <phase_3+0x34>
 8048c1d:	b8 78 00 00 00       	mov    $0x78,%eax
 8048c22:	81 7c 24 08 66 01 00 	cmpl   $0x166,0x8(%esp)
 8048c29:	00 
 8048c2a:	0f 84 e8 00 00 00    	je     8048d18 <phase_3+0x14c>
 8048c30:	e8 89 05 00 00       	call   80491be <explode_bomb>
 8048c35:	b8 78 00 00 00       	mov    $0x78,%eax
 8048c3a:	e9 d9 00 00 00       	jmp    8048d18 <phase_3+0x14c>
 8048c3f:	b8 76 00 00 00       	mov    $0x76,%eax
 8048c44:	81 7c 24 08 b1 02 00 	cmpl   $0x2b1,0x8(%esp)
 8048c4b:	00 
 8048c4c:	0f 84 c6 00 00 00    	je     8048d18 <phase_3+0x14c>
 8048c52:	e8 67 05 00 00       	call   80491be <explode_bomb>
 8048c57:	b8 76 00 00 00       	mov    $0x76,%eax
 8048c5c:	e9 b7 00 00 00       	jmp    8048d18 <phase_3+0x14c>
 8048c61:	b8 6d 00 00 00       	mov    $0x6d,%eax
 8048c66:	81 7c 24 08 bc 00 00 	cmpl   $0xbc,0x8(%esp)
 8048c6d:	00 
 8048c6e:	0f 84 a4 00 00 00    	je     8048d18 <phase_3+0x14c>
 8048c74:	e8 45 05 00 00       	call   80491be <explode_bomb>
 8048c79:	b8 6d 00 00 00       	mov    $0x6d,%eax
 8048c7e:	e9 95 00 00 00       	jmp    8048d18 <phase_3+0x14c>
 8048c83:	b8 73 00 00 00       	mov    $0x73,%eax
 8048c88:	81 7c 24 08 a0 00 00 	cmpl   $0xa0,0x8(%esp)
 8048c8f:	00 
 8048c90:	0f 84 82 00 00 00    	je     8048d18 <phase_3+0x14c>
 8048c96:	e8 23 05 00 00       	call   80491be <explode_bomb>
 8048c9b:	b8 73 00 00 00       	mov    $0x73,%eax
 8048ca0:	eb 76                	jmp    8048d18 <phase_3+0x14c>
 8048ca2:	b8 64 00 00 00       	mov    $0x64,%eax
 8048ca7:	81 7c 24 08 ff 01 00 	cmpl   $0x1ff,0x8(%esp)
 8048cae:	00 
 8048caf:	74 67                	je     8048d18 <phase_3+0x14c>
 8048cb1:	e8 08 05 00 00       	call   80491be <explode_bomb>
 8048cb6:	b8 64 00 00 00       	mov    $0x64,%eax
 8048cbb:	eb 5b                	jmp    8048d18 <phase_3+0x14c>
 8048cbd:	b8 6f 00 00 00       	mov    $0x6f,%eax
 8048cc2:	81 7c 24 08 c7 02 00 	cmpl   $0x2c7,0x8(%esp)
 8048cc9:	00 
 8048cca:	74 4c                	je     8048d18 <phase_3+0x14c>
 8048ccc:	e8 ed 04 00 00       	call   80491be <explode_bomb>
 8048cd1:	b8 6f 00 00 00       	mov    $0x6f,%eax
 8048cd6:	eb 40                	jmp    8048d18 <phase_3+0x14c>
 8048cd8:	b8 72 00 00 00       	mov    $0x72,%eax
 8048cdd:	81 7c 24 08 9d 03 00 	cmpl   $0x39d,0x8(%esp)
 8048ce4:	00 
 8048ce5:	74 31                	je     8048d18 <phase_3+0x14c>
 8048ce7:	e8 d2 04 00 00       	call   80491be <explode_bomb>
 8048cec:	b8 72 00 00 00       	mov    $0x72,%eax
 8048cf1:	eb 25                	jmp    8048d18 <phase_3+0x14c>
 8048cf3:	b8 79 00 00 00       	mov    $0x79,%eax
 8048cf8:	81 7c 24 08 e9 01 00 	cmpl   $0x1e9,0x8(%esp)
 8048cff:	00 
 8048d00:	74 16                	je     8048d18 <phase_3+0x14c>
 8048d02:	e8 b7 04 00 00       	call   80491be <explode_bomb>
 8048d07:	b8 79 00 00 00       	mov    $0x79,%eax
 8048d0c:	eb 0a                	jmp    8048d18 <phase_3+0x14c>
 8048d0e:	e8 ab 04 00 00       	call   80491be <explode_bomb>
 8048d13:	b8 6b 00 00 00       	mov    $0x6b,%eax
 8048d18:	3a 44 24 03          	cmp    0x3(%esp),%al
 8048d1c:	74 05                	je     8048d23 <phase_3+0x157>
 8048d1e:	e8 9b 04 00 00       	call   80491be <explode_bomb>
 8048d23:	8b 44 24 0c          	mov    0xc(%esp),%eax
 8048d27:	65 33 05 14 00 00 00 	xor    %gs:0x14,%eax
 8048d2e:	75 04                	jne    8048d34 <phase_3+0x168>
 8048d30:	83 c4 1c             	add    $0x1c,%esp
 8048d33:	c3                   	ret    
 8048d34:	e8 57 fa ff ff       	call   8048790 <__stack_chk_fail@plt>

08048d39 <func4>:
 8048d39:	57                   	push   %edi
 8048d3a:	56                   	push   %esi
 8048d3b:	53                   	push   %ebx
 8048d3c:	8b 5c 24 10          	mov    0x10(%esp),%ebx
 8048d40:	8b 7c 24 14          	mov    0x14(%esp),%edi
 8048d44:	85 db                	test   %ebx,%ebx
 8048d46:	7e 2d                	jle    8048d75 <func4+0x3c>
 8048d48:	89 f8                	mov    %edi,%eax
 8048d4a:	83 fb 01             	cmp    $0x1,%ebx
 8048d4d:	74 22                	je     8048d71 <func4+0x38>
 8048d4f:	83 ec 08             	sub    $0x8,%esp
 8048d52:	57                   	push   %edi
 8048d53:	8d 43 ff             	lea    -0x1(%ebx),%eax
 8048d56:	50                   	push   %eax
 8048d57:	e8 dd ff ff ff       	call   8048d39 <func4>
 8048d5c:	83 c4 08             	add    $0x8,%esp
 8048d5f:	8d 34 07             	lea    (%edi,%eax,1),%esi
 8048d62:	57                   	push   %edi
 8048d63:	83 eb 02             	sub    $0x2,%ebx
 8048d66:	53                   	push   %ebx
 8048d67:	e8 cd ff ff ff       	call   8048d39 <func4>
 8048d6c:	83 c4 10             	add    $0x10,%esp
 8048d6f:	01 f0                	add    %esi,%eax
 8048d71:	5b                   	pop    %ebx
 8048d72:	5e                   	pop    %esi
 8048d73:	5f                   	pop    %edi
 8048d74:	c3                   	ret    
 8048d75:	b8 00 00 00 00       	mov    $0x0,%eax
 8048d7a:	eb f5                	jmp    8048d71 <func4+0x38>

08048d7c <phase_4>:
 8048d7c:	83 ec 1c             	sub    $0x1c,%esp
 8048d7f:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 8048d85:	89 44 24 0c          	mov    %eax,0xc(%esp)
 8048d89:	31 c0                	xor    %eax,%eax
 8048d8b:	8d 44 24 04          	lea    0x4(%esp),%eax
 8048d8f:	50                   	push   %eax
 8048d90:	8d 44 24 0c          	lea    0xc(%esp),%eax
 8048d94:	50                   	push   %eax
 8048d95:	68 af a2 04 08       	push   $0x804a2af
 8048d9a:	ff 74 24 2c          	pushl  0x2c(%esp)
 8048d9e:	e8 6d fa ff ff       	call   8048810 <__isoc99_sscanf@plt>
 8048da3:	83 c4 10             	add    $0x10,%esp
 8048da6:	83 f8 02             	cmp    $0x2,%eax
 8048da9:	74 32                	je     8048ddd <phase_4+0x61>
 8048dab:	e8 0e 04 00 00       	call   80491be <explode_bomb>
 8048db0:	83 ec 08             	sub    $0x8,%esp
 8048db3:	ff 74 24 0c          	pushl  0xc(%esp)
 8048db7:	6a 08                	push   $0x8
 8048db9:	e8 7b ff ff ff       	call   8048d39 <func4>
 8048dbe:	83 c4 10             	add    $0x10,%esp
 8048dc1:	3b 44 24 08          	cmp    0x8(%esp),%eax
 8048dc5:	74 05                	je     8048dcc <phase_4+0x50>
 8048dc7:	e8 f2 03 00 00       	call   80491be <explode_bomb>
 8048dcc:	8b 44 24 0c          	mov    0xc(%esp),%eax
 8048dd0:	65 33 05 14 00 00 00 	xor    %gs:0x14,%eax
 8048dd7:	75 12                	jne    8048deb <phase_4+0x6f>
 8048dd9:	83 c4 1c             	add    $0x1c,%esp
 8048ddc:	c3                   	ret    
 8048ddd:	8b 44 24 04          	mov    0x4(%esp),%eax
 8048de1:	83 e8 02             	sub    $0x2,%eax
 8048de4:	83 f8 02             	cmp    $0x2,%eax
 8048de7:	76 c7                	jbe    8048db0 <phase_4+0x34>
 8048de9:	eb c0                	jmp    8048dab <phase_4+0x2f>
 8048deb:	e8 a0 f9 ff ff       	call   8048790 <__stack_chk_fail@plt>

08048df0 <phase_5>:
 8048df0:	83 ec 1c             	sub    $0x1c,%esp
 8048df3:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 8048df9:	89 44 24 0c          	mov    %eax,0xc(%esp)
 8048dfd:	31 c0                	xor    %eax,%eax
 8048dff:	8d 44 24 08          	lea    0x8(%esp),%eax
 8048e03:	50                   	push   %eax
 8048e04:	8d 44 24 08          	lea    0x8(%esp),%eax
 8048e08:	50                   	push   %eax
 8048e09:	68 af a2 04 08       	push   $0x804a2af
 8048e0e:	ff 74 24 2c          	pushl  0x2c(%esp)
 8048e12:	e8 f9 f9 ff ff       	call   8048810 <__isoc99_sscanf@plt>
 8048e17:	83 c4 10             	add    $0x10,%esp
 8048e1a:	83 f8 01             	cmp    $0x1,%eax
 8048e1d:	7e 54                	jle    8048e73 <phase_5+0x83>
 8048e1f:	8b 44 24 04          	mov    0x4(%esp),%eax
 8048e23:	83 e0 0f             	and    $0xf,%eax
 8048e26:	89 44 24 04          	mov    %eax,0x4(%esp)
 8048e2a:	83 f8 0f             	cmp    $0xf,%eax
 8048e2d:	74 2e                	je     8048e5d <phase_5+0x6d>
 8048e2f:	b9 00 00 00 00       	mov    $0x0,%ecx
 8048e34:	ba 00 00 00 00       	mov    $0x0,%edx
 8048e39:	83 c2 01             	add    $0x1,%edx
 8048e3c:	8b 04 85 60 a1 04 08 	mov    0x804a160(,%eax,4),%eax
 8048e43:	01 c1                	add    %eax,%ecx
 8048e45:	83 f8 0f             	cmp    $0xf,%eax
 8048e48:	75 ef                	jne    8048e39 <phase_5+0x49>
 8048e4a:	c7 44 24 04 0f 00 00 	movl   $0xf,0x4(%esp)
 8048e51:	00 
 8048e52:	83 fa 0f             	cmp    $0xf,%edx
 8048e55:	75 06                	jne    8048e5d <phase_5+0x6d>
 8048e57:	3b 4c 24 08          	cmp    0x8(%esp),%ecx
 8048e5b:	74 05                	je     8048e62 <phase_5+0x72>
 8048e5d:	e8 5c 03 00 00       	call   80491be <explode_bomb>
 8048e62:	8b 44 24 0c          	mov    0xc(%esp),%eax
 8048e66:	65 33 05 14 00 00 00 	xor    %gs:0x14,%eax
 8048e6d:	75 0b                	jne    8048e7a <phase_5+0x8a>
 8048e6f:	83 c4 1c             	add    $0x1c,%esp
 8048e72:	c3                   	ret    
 8048e73:	e8 46 03 00 00       	call   80491be <explode_bomb>
 8048e78:	eb a5                	jmp    8048e1f <phase_5+0x2f>
 8048e7a:	e8 11 f9 ff ff       	call   8048790 <__stack_chk_fail@plt>

08048e7f <phase_6>:
 8048e7f:	56                   	push   %esi
 8048e80:	53                   	push   %ebx
 8048e81:	83 ec 4c             	sub    $0x4c,%esp
 8048e84:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 8048e8a:	89 44 24 44          	mov    %eax,0x44(%esp)
 8048e8e:	31 c0                	xor    %eax,%eax
 8048e90:	8d 44 24 14          	lea    0x14(%esp),%eax
 8048e94:	50                   	push   %eax
 8048e95:	ff 74 24 5c          	pushl  0x5c(%esp)
 8048e99:	e8 45 03 00 00       	call   80491e3 <read_six_numbers>
 8048e9e:	83 c4 10             	add    $0x10,%esp
 8048ea1:	be 00 00 00 00       	mov    $0x0,%esi
 8048ea6:	eb 1c                	jmp    8048ec4 <phase_6+0x45>
 8048ea8:	83 c6 01             	add    $0x1,%esi
 8048eab:	83 fe 06             	cmp    $0x6,%esi
 8048eae:	74 2e                	je     8048ede <phase_6+0x5f>
 8048eb0:	89 f3                	mov    %esi,%ebx
 8048eb2:	8b 44 9c 0c          	mov    0xc(%esp,%ebx,4),%eax
 8048eb6:	39 44 b4 08          	cmp    %eax,0x8(%esp,%esi,4)
 8048eba:	74 1b                	je     8048ed7 <phase_6+0x58>
 8048ebc:	83 c3 01             	add    $0x1,%ebx
 8048ebf:	83 fb 05             	cmp    $0x5,%ebx
 8048ec2:	7e ee                	jle    8048eb2 <phase_6+0x33>
 8048ec4:	8b 44 b4 0c          	mov    0xc(%esp,%esi,4),%eax
 8048ec8:	83 e8 01             	sub    $0x1,%eax
 8048ecb:	83 f8 05             	cmp    $0x5,%eax
 8048ece:	76 d8                	jbe    8048ea8 <phase_6+0x29>
 8048ed0:	e8 e9 02 00 00       	call   80491be <explode_bomb>
 8048ed5:	eb d1                	jmp    8048ea8 <phase_6+0x29>
 8048ed7:	e8 e2 02 00 00       	call   80491be <explode_bomb>
 8048edc:	eb de                	jmp    8048ebc <phase_6+0x3d>
 8048ede:	8d 44 24 0c          	lea    0xc(%esp),%eax
 8048ee2:	8d 5c 24 24          	lea    0x24(%esp),%ebx
 8048ee6:	b9 07 00 00 00       	mov    $0x7,%ecx
 8048eeb:	89 ca                	mov    %ecx,%edx
 8048eed:	2b 10                	sub    (%eax),%edx
 8048eef:	89 10                	mov    %edx,(%eax)
 8048ef1:	83 c0 04             	add    $0x4,%eax
 8048ef4:	39 c3                	cmp    %eax,%ebx
 8048ef6:	75 f3                	jne    8048eeb <phase_6+0x6c>
 8048ef8:	bb 00 00 00 00       	mov    $0x0,%ebx
 8048efd:	89 de                	mov    %ebx,%esi
 8048eff:	8b 4c 9c 0c          	mov    0xc(%esp,%ebx,4),%ecx
 8048f03:	b8 01 00 00 00       	mov    $0x1,%eax
 8048f08:	ba 3c c1 04 08       	mov    $0x804c13c,%edx
 8048f0d:	83 f9 01             	cmp    $0x1,%ecx
 8048f10:	7e 0a                	jle    8048f1c <phase_6+0x9d>
 8048f12:	8b 52 08             	mov    0x8(%edx),%edx
 8048f15:	83 c0 01             	add    $0x1,%eax
 8048f18:	39 c8                	cmp    %ecx,%eax
 8048f1a:	75 f6                	jne    8048f12 <phase_6+0x93>
 8048f1c:	89 54 b4 24          	mov    %edx,0x24(%esp,%esi,4)
 8048f20:	83 c3 01             	add    $0x1,%ebx
 8048f23:	83 fb 06             	cmp    $0x6,%ebx
 8048f26:	75 d5                	jne    8048efd <phase_6+0x7e>
 8048f28:	8b 5c 24 24          	mov    0x24(%esp),%ebx
 8048f2c:	89 d9                	mov    %ebx,%ecx
 8048f2e:	b8 01 00 00 00       	mov    $0x1,%eax
 8048f33:	8b 54 84 24          	mov    0x24(%esp,%eax,4),%edx
 8048f37:	89 51 08             	mov    %edx,0x8(%ecx)
 8048f3a:	83 c0 01             	add    $0x1,%eax
 8048f3d:	89 d1                	mov    %edx,%ecx
 8048f3f:	83 f8 06             	cmp    $0x6,%eax
 8048f42:	75 ef                	jne    8048f33 <phase_6+0xb4>
 8048f44:	c7 42 08 00 00 00 00 	movl   $0x0,0x8(%edx)
 8048f4b:	be 05 00 00 00       	mov    $0x5,%esi
 8048f50:	eb 08                	jmp    8048f5a <phase_6+0xdb>
 8048f52:	8b 5b 08             	mov    0x8(%ebx),%ebx
 8048f55:	83 ee 01             	sub    $0x1,%esi
 8048f58:	74 10                	je     8048f6a <phase_6+0xeb>
 8048f5a:	8b 43 08             	mov    0x8(%ebx),%eax
 8048f5d:	8b 00                	mov    (%eax),%eax
 8048f5f:	39 03                	cmp    %eax,(%ebx)
 8048f61:	7d ef                	jge    8048f52 <phase_6+0xd3>
 8048f63:	e8 56 02 00 00       	call   80491be <explode_bomb>
 8048f68:	eb e8                	jmp    8048f52 <phase_6+0xd3>
 8048f6a:	8b 44 24 3c          	mov    0x3c(%esp),%eax
 8048f6e:	65 33 05 14 00 00 00 	xor    %gs:0x14,%eax
 8048f75:	75 06                	jne    8048f7d <phase_6+0xfe>
 8048f77:	83 c4 44             	add    $0x44,%esp
 8048f7a:	5b                   	pop    %ebx
 8048f7b:	5e                   	pop    %esi
 8048f7c:	c3                   	ret    
 8048f7d:	e8 0e f8 ff ff       	call   8048790 <__stack_chk_fail@plt>

08048f82 <fun7>:
 8048f82:	53                   	push   %ebx
 8048f83:	83 ec 08             	sub    $0x8,%esp
 8048f86:	8b 54 24 10          	mov    0x10(%esp),%edx
 8048f8a:	8b 4c 24 14          	mov    0x14(%esp),%ecx
 8048f8e:	85 d2                	test   %edx,%edx
 8048f90:	74 3a                	je     8048fcc <fun7+0x4a>
 8048f92:	8b 1a                	mov    (%edx),%ebx
 8048f94:	39 cb                	cmp    %ecx,%ebx
 8048f96:	7f 21                	jg     8048fb9 <fun7+0x37>
 8048f98:	b8 00 00 00 00       	mov    $0x0,%eax
 8048f9d:	39 cb                	cmp    %ecx,%ebx
 8048f9f:	74 13                	je     8048fb4 <fun7+0x32>
 8048fa1:	83 ec 08             	sub    $0x8,%esp
 8048fa4:	51                   	push   %ecx
 8048fa5:	ff 72 08             	pushl  0x8(%edx)
 8048fa8:	e8 d5 ff ff ff       	call   8048f82 <fun7>
 8048fad:	83 c4 10             	add    $0x10,%esp
 8048fb0:	8d 44 00 01          	lea    0x1(%eax,%eax,1),%eax
 8048fb4:	83 c4 08             	add    $0x8,%esp
 8048fb7:	5b                   	pop    %ebx
 8048fb8:	c3                   	ret    
 8048fb9:	83 ec 08             	sub    $0x8,%esp
 8048fbc:	51                   	push   %ecx
 8048fbd:	ff 72 04             	pushl  0x4(%edx)
 8048fc0:	e8 bd ff ff ff       	call   8048f82 <fun7>
 8048fc5:	83 c4 10             	add    $0x10,%esp
 8048fc8:	01 c0                	add    %eax,%eax
 8048fca:	eb e8                	jmp    8048fb4 <fun7+0x32>
 8048fcc:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8048fd1:	eb e1                	jmp    8048fb4 <fun7+0x32>

08048fd3 <secret_phase>:
 8048fd3:	53                   	push   %ebx
 8048fd4:	83 ec 08             	sub    $0x8,%esp
 8048fd7:	e8 42 02 00 00       	call   804921e <read_line>
 8048fdc:	83 ec 04             	sub    $0x4,%esp
 8048fdf:	6a 0a                	push   $0xa
 8048fe1:	6a 00                	push   $0x0
 8048fe3:	50                   	push   %eax
 8048fe4:	e8 97 f8 ff ff       	call   8048880 <strtol@plt>
 8048fe9:	89 c3                	mov    %eax,%ebx
 8048feb:	8d 40 ff             	lea    -0x1(%eax),%eax
 8048fee:	83 c4 10             	add    $0x10,%esp
 8048ff1:	3d e8 03 00 00       	cmp    $0x3e8,%eax
 8048ff6:	77 32                	ja     804902a <secret_phase+0x57>
 8048ff8:	83 ec 08             	sub    $0x8,%esp
 8048ffb:	53                   	push   %ebx
 8048ffc:	68 88 c0 04 08       	push   $0x804c088
 8049001:	e8 7c ff ff ff       	call   8048f82 <fun7>
 8049006:	83 c4 10             	add    $0x10,%esp
 8049009:	83 f8 06             	cmp    $0x6,%eax
 804900c:	74 05                	je     8049013 <secret_phase+0x40>
 804900e:	e8 ab 01 00 00       	call   80491be <explode_bomb>
 8049013:	83 ec 0c             	sub    $0xc,%esp
 8049016:	68 f4 a0 04 08       	push   $0x804a0f4
 804901b:	e8 a0 f7 ff ff       	call   80487c0 <puts@plt>
 8049020:	e8 f8 02 00 00       	call   804931d <phase_defused>
 8049025:	83 c4 18             	add    $0x18,%esp
 8049028:	5b                   	pop    %ebx
 8049029:	c3                   	ret    
 804902a:	e8 8f 01 00 00       	call   80491be <explode_bomb>
 804902f:	eb c7                	jmp    8048ff8 <secret_phase+0x25>

08049031 <sig_handler>:
 8049031:	83 ec 18             	sub    $0x18,%esp
 8049034:	68 a0 a1 04 08       	push   $0x804a1a0
 8049039:	e8 82 f7 ff ff       	call   80487c0 <puts@plt>
 804903e:	c7 04 24 03 00 00 00 	movl   $0x3,(%esp)
 8049045:	e8 26 f7 ff ff       	call   8048770 <sleep@plt>
 804904a:	83 c4 08             	add    $0x8,%esp
 804904d:	68 62 a2 04 08       	push   $0x804a262
 8049052:	6a 01                	push   $0x1
 8049054:	e8 e7 f7 ff ff       	call   8048840 <__printf_chk@plt>
 8049059:	83 c4 04             	add    $0x4,%esp
 804905c:	ff 35 c4 c3 04 08    	pushl  0x804c3c4
 8049062:	e8 d9 f6 ff ff       	call   8048740 <fflush@plt>
 8049067:	c7 04 24 01 00 00 00 	movl   $0x1,(%esp)
 804906e:	e8 fd f6 ff ff       	call   8048770 <sleep@plt>
 8049073:	c7 04 24 6a a2 04 08 	movl   $0x804a26a,(%esp)
 804907a:	e8 41 f7 ff ff       	call   80487c0 <puts@plt>
 804907f:	c7 04 24 10 00 00 00 	movl   $0x10,(%esp)
 8049086:	e8 55 f7 ff ff       	call   80487e0 <exit@plt>

0804908b <invalid_phase>:
 804908b:	83 ec 10             	sub    $0x10,%esp
 804908e:	ff 74 24 14          	pushl  0x14(%esp)
 8049092:	68 72 a2 04 08       	push   $0x804a272
 8049097:	6a 01                	push   $0x1
 8049099:	e8 a2 f7 ff ff       	call   8048840 <__printf_chk@plt>
 804909e:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
 80490a5:	e8 36 f7 ff ff       	call   80487e0 <exit@plt>

080490aa <string_length>:
 80490aa:	8b 54 24 04          	mov    0x4(%esp),%edx
 80490ae:	80 3a 00             	cmpb   $0x0,(%edx)
 80490b1:	74 10                	je     80490c3 <string_length+0x19>
 80490b3:	b8 00 00 00 00       	mov    $0x0,%eax
 80490b8:	83 c0 01             	add    $0x1,%eax
 80490bb:	80 3c 02 00          	cmpb   $0x0,(%edx,%eax,1)
 80490bf:	75 f7                	jne    80490b8 <string_length+0xe>
 80490c1:	f3 c3                	repz ret 
 80490c3:	b8 00 00 00 00       	mov    $0x0,%eax
 80490c8:	c3                   	ret    

080490c9 <strings_not_equal>:
 80490c9:	57                   	push   %edi
 80490ca:	56                   	push   %esi
 80490cb:	53                   	push   %ebx
 80490cc:	8b 5c 24 10          	mov    0x10(%esp),%ebx
 80490d0:	8b 74 24 14          	mov    0x14(%esp),%esi
 80490d4:	53                   	push   %ebx
 80490d5:	e8 d0 ff ff ff       	call   80490aa <string_length>
 80490da:	89 c7                	mov    %eax,%edi
 80490dc:	89 34 24             	mov    %esi,(%esp)
 80490df:	e8 c6 ff ff ff       	call   80490aa <string_length>
 80490e4:	83 c4 04             	add    $0x4,%esp
 80490e7:	ba 01 00 00 00       	mov    $0x1,%edx
 80490ec:	39 c7                	cmp    %eax,%edi
 80490ee:	74 06                	je     80490f6 <strings_not_equal+0x2d>
 80490f0:	89 d0                	mov    %edx,%eax
 80490f2:	5b                   	pop    %ebx
 80490f3:	5e                   	pop    %esi
 80490f4:	5f                   	pop    %edi
 80490f5:	c3                   	ret    
 80490f6:	0f b6 03             	movzbl (%ebx),%eax
 80490f9:	84 c0                	test   %al,%al
 80490fb:	74 23                	je     8049120 <strings_not_equal+0x57>
 80490fd:	3a 06                	cmp    (%esi),%al
 80490ff:	75 26                	jne    8049127 <strings_not_equal+0x5e>
 8049101:	83 c3 01             	add    $0x1,%ebx
 8049104:	83 c6 01             	add    $0x1,%esi
 8049107:	0f b6 03             	movzbl (%ebx),%eax
 804910a:	84 c0                	test   %al,%al
 804910c:	74 0b                	je     8049119 <strings_not_equal+0x50>
 804910e:	3a 06                	cmp    (%esi),%al
 8049110:	74 ef                	je     8049101 <strings_not_equal+0x38>
 8049112:	ba 01 00 00 00       	mov    $0x1,%edx
 8049117:	eb d7                	jmp    80490f0 <strings_not_equal+0x27>
 8049119:	ba 00 00 00 00       	mov    $0x0,%edx
 804911e:	eb d0                	jmp    80490f0 <strings_not_equal+0x27>
 8049120:	ba 00 00 00 00       	mov    $0x0,%edx
 8049125:	eb c9                	jmp    80490f0 <strings_not_equal+0x27>
 8049127:	ba 01 00 00 00       	mov    $0x1,%edx
 804912c:	eb c2                	jmp    80490f0 <strings_not_equal+0x27>

0804912e <initialize_bomb>:
 804912e:	83 ec 14             	sub    $0x14,%esp
 8049131:	68 31 90 04 08       	push   $0x8049031
 8049136:	6a 02                	push   $0x2
 8049138:	e8 23 f6 ff ff       	call   8048760 <signal@plt>
 804913d:	83 c4 1c             	add    $0x1c,%esp
 8049140:	c3                   	ret    

08049141 <initialize_bomb_solve>:
 8049141:	f3 c3                	repz ret 

08049143 <blank_line>:
 8049143:	56                   	push   %esi
 8049144:	53                   	push   %ebx
 8049145:	83 ec 04             	sub    $0x4,%esp
 8049148:	8b 74 24 10          	mov    0x10(%esp),%esi
 804914c:	0f b6 1e             	movzbl (%esi),%ebx
 804914f:	84 db                	test   %bl,%bl
 8049151:	74 1b                	je     804916e <blank_line+0x2b>
 8049153:	e8 58 f7 ff ff       	call   80488b0 <__ctype_b_loc@plt>
 8049158:	83 c6 01             	add    $0x1,%esi
 804915b:	0f be db             	movsbl %bl,%ebx
 804915e:	8b 00                	mov    (%eax),%eax
 8049160:	f6 44 58 01 20       	testb  $0x20,0x1(%eax,%ebx,2)
 8049165:	75 e5                	jne    804914c <blank_line+0x9>
 8049167:	b8 00 00 00 00       	mov    $0x0,%eax
 804916c:	eb 05                	jmp    8049173 <blank_line+0x30>
 804916e:	b8 01 00 00 00       	mov    $0x1,%eax
 8049173:	83 c4 04             	add    $0x4,%esp
 8049176:	5b                   	pop    %ebx
 8049177:	5e                   	pop    %esi
 8049178:	c3                   	ret    

08049179 <skip>:
 8049179:	53                   	push   %ebx
 804917a:	83 ec 08             	sub    $0x8,%esp
 804917d:	83 ec 04             	sub    $0x4,%esp
 8049180:	ff 35 d0 c3 04 08    	pushl  0x804c3d0
 8049186:	6a 50                	push   $0x50
 8049188:	a1 cc c3 04 08       	mov    0x804c3cc,%eax
 804918d:	8d 04 80             	lea    (%eax,%eax,4),%eax
 8049190:	c1 e0 04             	shl    $0x4,%eax
 8049193:	05 e0 c3 04 08       	add    $0x804c3e0,%eax
 8049198:	50                   	push   %eax
 8049199:	e8 b2 f5 ff ff       	call   8048750 <fgets@plt>
 804919e:	89 c3                	mov    %eax,%ebx
 80491a0:	83 c4 10             	add    $0x10,%esp
 80491a3:	85 c0                	test   %eax,%eax
 80491a5:	74 10                	je     80491b7 <skip+0x3e>
 80491a7:	83 ec 0c             	sub    $0xc,%esp
 80491aa:	50                   	push   %eax
 80491ab:	e8 93 ff ff ff       	call   8049143 <blank_line>
 80491b0:	83 c4 10             	add    $0x10,%esp
 80491b3:	85 c0                	test   %eax,%eax
 80491b5:	75 c6                	jne    804917d <skip+0x4>
 80491b7:	89 d8                	mov    %ebx,%eax
 80491b9:	83 c4 08             	add    $0x8,%esp
 80491bc:	5b                   	pop    %ebx
 80491bd:	c3                   	ret    

080491be <explode_bomb>:
 80491be:	83 ec 18             	sub    $0x18,%esp
 80491c1:	68 83 a2 04 08       	push   $0x804a283
 80491c6:	e8 f5 f5 ff ff       	call   80487c0 <puts@plt>
 80491cb:	c7 04 24 8c a2 04 08 	movl   $0x804a28c,(%esp)
 80491d2:	e8 e9 f5 ff ff       	call   80487c0 <puts@plt>
 80491d7:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
 80491de:	e8 fd f5 ff ff       	call   80487e0 <exit@plt>

080491e3 <read_six_numbers>:
 80491e3:	83 ec 0c             	sub    $0xc,%esp
 80491e6:	8b 44 24 14          	mov    0x14(%esp),%eax
 80491ea:	8d 50 14             	lea    0x14(%eax),%edx
 80491ed:	52                   	push   %edx
 80491ee:	8d 50 10             	lea    0x10(%eax),%edx
 80491f1:	52                   	push   %edx
 80491f2:	8d 50 0c             	lea    0xc(%eax),%edx
 80491f5:	52                   	push   %edx
 80491f6:	8d 50 08             	lea    0x8(%eax),%edx
 80491f9:	52                   	push   %edx
 80491fa:	8d 50 04             	lea    0x4(%eax),%edx
 80491fd:	52                   	push   %edx
 80491fe:	50                   	push   %eax
 80491ff:	68 a3 a2 04 08       	push   $0x804a2a3
 8049204:	ff 74 24 2c          	pushl  0x2c(%esp)
 8049208:	e8 03 f6 ff ff       	call   8048810 <__isoc99_sscanf@plt>
 804920d:	83 c4 20             	add    $0x20,%esp
 8049210:	83 f8 05             	cmp    $0x5,%eax
 8049213:	7e 04                	jle    8049219 <read_six_numbers+0x36>
 8049215:	83 c4 0c             	add    $0xc,%esp
 8049218:	c3                   	ret    
 8049219:	e8 a0 ff ff ff       	call   80491be <explode_bomb>

0804921e <read_line>:
 804921e:	57                   	push   %edi
 804921f:	56                   	push   %esi
 8049220:	53                   	push   %ebx
 8049221:	e8 53 ff ff ff       	call   8049179 <skip>
 8049226:	85 c0                	test   %eax,%eax
 8049228:	74 4b                	je     8049275 <read_line+0x57>
 804922a:	8b 15 cc c3 04 08    	mov    0x804c3cc,%edx
 8049230:	8d 1c 92             	lea    (%edx,%edx,4),%ebx
 8049233:	c1 e3 04             	shl    $0x4,%ebx
 8049236:	81 c3 e0 c3 04 08    	add    $0x804c3e0,%ebx
 804923c:	b8 00 00 00 00       	mov    $0x0,%eax
 8049241:	b9 ff ff ff ff       	mov    $0xffffffff,%ecx
 8049246:	89 df                	mov    %ebx,%edi
 8049248:	f2 ae                	repnz scas %es:(%edi),%al
 804924a:	f7 d1                	not    %ecx
 804924c:	83 e9 01             	sub    $0x1,%ecx
 804924f:	83 f9 4e             	cmp    $0x4e,%ecx
 8049252:	0f 8f 8d 00 00 00    	jg     80492e5 <read_line+0xc7>
 8049258:	8d 04 92             	lea    (%edx,%edx,4),%eax
 804925b:	c1 e0 04             	shl    $0x4,%eax
 804925e:	c6 84 01 df c3 04 08 	movb   $0x0,0x804c3df(%ecx,%eax,1)
 8049265:	00 
 8049266:	83 c2 01             	add    $0x1,%edx
 8049269:	89 15 cc c3 04 08    	mov    %edx,0x804c3cc
 804926f:	89 d8                	mov    %ebx,%eax
 8049271:	5b                   	pop    %ebx
 8049272:	5e                   	pop    %esi
 8049273:	5f                   	pop    %edi
 8049274:	c3                   	ret    
 8049275:	a1 c0 c3 04 08       	mov    0x804c3c0,%eax
 804927a:	39 05 d0 c3 04 08    	cmp    %eax,0x804c3d0
 8049280:	74 40                	je     80492c2 <read_line+0xa4>
 8049282:	83 ec 0c             	sub    $0xc,%esp
 8049285:	68 d3 a2 04 08       	push   $0x804a2d3
 804928a:	e8 21 f5 ff ff       	call   80487b0 <getenv@plt>
 804928f:	83 c4 10             	add    $0x10,%esp
 8049292:	85 c0                	test   %eax,%eax
 8049294:	75 45                	jne    80492db <read_line+0xbd>
 8049296:	a1 c0 c3 04 08       	mov    0x804c3c0,%eax
 804929b:	a3 d0 c3 04 08       	mov    %eax,0x804c3d0
 80492a0:	e8 d4 fe ff ff       	call   8049179 <skip>
 80492a5:	85 c0                	test   %eax,%eax
 80492a7:	75 81                	jne    804922a <read_line+0xc>
 80492a9:	83 ec 0c             	sub    $0xc,%esp
 80492ac:	68 b5 a2 04 08       	push   $0x804a2b5
 80492b1:	e8 0a f5 ff ff       	call   80487c0 <puts@plt>
 80492b6:	c7 04 24 00 00 00 00 	movl   $0x0,(%esp)
 80492bd:	e8 1e f5 ff ff       	call   80487e0 <exit@plt>
 80492c2:	83 ec 0c             	sub    $0xc,%esp
 80492c5:	68 b5 a2 04 08       	push   $0x804a2b5
 80492ca:	e8 f1 f4 ff ff       	call   80487c0 <puts@plt>
 80492cf:	c7 04 24 08 00 00 00 	movl   $0x8,(%esp)
 80492d6:	e8 05 f5 ff ff       	call   80487e0 <exit@plt>
 80492db:	83 ec 0c             	sub    $0xc,%esp
 80492de:	6a 00                	push   $0x0
 80492e0:	e8 fb f4 ff ff       	call   80487e0 <exit@plt>
 80492e5:	83 ec 0c             	sub    $0xc,%esp
 80492e8:	68 de a2 04 08       	push   $0x804a2de
 80492ed:	e8 ce f4 ff ff       	call   80487c0 <puts@plt>
 80492f2:	a1 cc c3 04 08       	mov    0x804c3cc,%eax
 80492f7:	8d 50 01             	lea    0x1(%eax),%edx
 80492fa:	89 15 cc c3 04 08    	mov    %edx,0x804c3cc
 8049300:	6b c0 50             	imul   $0x50,%eax,%eax
 8049303:	05 e0 c3 04 08       	add    $0x804c3e0,%eax
 8049308:	ba f9 a2 04 08       	mov    $0x804a2f9,%edx
 804930d:	b9 04 00 00 00       	mov    $0x4,%ecx
 8049312:	89 c7                	mov    %eax,%edi
 8049314:	89 d6                	mov    %edx,%esi
 8049316:	f3 a5                	rep movsl %ds:(%esi),%es:(%edi)
 8049318:	e8 a1 fe ff ff       	call   80491be <explode_bomb>

0804931d <phase_defused>:
 804931d:	83 ec 6c             	sub    $0x6c,%esp
 8049320:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 8049326:	89 44 24 5c          	mov    %eax,0x5c(%esp)
 804932a:	31 c0                	xor    %eax,%eax
 804932c:	83 3d cc c3 04 08 06 	cmpl   $0x6,0x804c3cc
 8049333:	74 11                	je     8049346 <phase_defused+0x29>
 8049335:	8b 44 24 5c          	mov    0x5c(%esp),%eax
 8049339:	65 33 05 14 00 00 00 	xor    %gs:0x14,%eax
 8049340:	75 7b                	jne    80493bd <phase_defused+0xa0>
 8049342:	83 c4 6c             	add    $0x6c,%esp
 8049345:	c3                   	ret    
 8049346:	83 ec 0c             	sub    $0xc,%esp
 8049349:	8d 44 24 18          	lea    0x18(%esp),%eax
 804934d:	50                   	push   %eax
 804934e:	8d 44 24 18          	lea    0x18(%esp),%eax
 8049352:	50                   	push   %eax
 8049353:	8d 44 24 18          	lea    0x18(%esp),%eax
 8049357:	50                   	push   %eax
 8049358:	68 09 a3 04 08       	push   $0x804a309
 804935d:	68 d0 c4 04 08       	push   $0x804c4d0
 8049362:	e8 a9 f4 ff ff       	call   8048810 <__isoc99_sscanf@plt>
 8049367:	83 c4 20             	add    $0x20,%esp
 804936a:	83 f8 03             	cmp    $0x3,%eax
 804936d:	74 12                	je     8049381 <phase_defused+0x64>
 804936f:	83 ec 0c             	sub    $0xc,%esp
 8049372:	68 38 a2 04 08       	push   $0x804a238
 8049377:	e8 44 f4 ff ff       	call   80487c0 <puts@plt>
 804937c:	83 c4 10             	add    $0x10,%esp
 804937f:	eb b4                	jmp    8049335 <phase_defused+0x18>
 8049381:	83 ec 08             	sub    $0x8,%esp
 8049384:	68 12 a3 04 08       	push   $0x804a312
 8049389:	8d 44 24 18          	lea    0x18(%esp),%eax
 804938d:	50                   	push   %eax
 804938e:	e8 36 fd ff ff       	call   80490c9 <strings_not_equal>
 8049393:	83 c4 10             	add    $0x10,%esp
 8049396:	85 c0                	test   %eax,%eax
 8049398:	75 d5                	jne    804936f <phase_defused+0x52>
 804939a:	83 ec 0c             	sub    $0xc,%esp
 804939d:	68 d8 a1 04 08       	push   $0x804a1d8
 80493a2:	e8 19 f4 ff ff       	call   80487c0 <puts@plt>
 80493a7:	c7 04 24 00 a2 04 08 	movl   $0x804a200,(%esp)
 80493ae:	e8 0d f4 ff ff       	call   80487c0 <puts@plt>
 80493b3:	e8 1b fc ff ff       	call   8048fd3 <secret_phase>
 80493b8:	83 c4 10             	add    $0x10,%esp
 80493bb:	eb b2                	jmp    804936f <phase_defused+0x52>
 80493bd:	e8 ce f3 ff ff       	call   8048790 <__stack_chk_fail@plt>

080493c2 <sigalrm_handler>:
 80493c2:	83 ec 0c             	sub    $0xc,%esp
 80493c5:	6a 00                	push   $0x0
 80493c7:	68 68 a3 04 08       	push   $0x804a368
 80493cc:	6a 01                	push   $0x1
 80493ce:	ff 35 a0 c3 04 08    	pushl  0x804c3a0
 80493d4:	e8 87 f4 ff ff       	call   8048860 <__fprintf_chk@plt>
 80493d9:	c7 04 24 01 00 00 00 	movl   $0x1,(%esp)
 80493e0:	e8 fb f3 ff ff       	call   80487e0 <exit@plt>

080493e5 <rio_readlineb>:
 80493e5:	55                   	push   %ebp
 80493e6:	57                   	push   %edi
 80493e7:	56                   	push   %esi
 80493e8:	53                   	push   %ebx
 80493e9:	83 ec 1c             	sub    $0x1c,%esp
 80493ec:	83 f9 01             	cmp    $0x1,%ecx
 80493ef:	76 79                	jbe    804946a <rio_readlineb+0x85>
 80493f1:	89 d7                	mov    %edx,%edi
 80493f3:	89 c3                	mov    %eax,%ebx
 80493f5:	89 4c 24 0c          	mov    %ecx,0xc(%esp)
 80493f9:	bd 01 00 00 00       	mov    $0x1,%ebp
 80493fe:	8d 70 0c             	lea    0xc(%eax),%esi
 8049401:	eb 0a                	jmp    804940d <rio_readlineb+0x28>
 8049403:	e8 28 f4 ff ff       	call   8048830 <__errno_location@plt>
 8049408:	83 38 04             	cmpl   $0x4,(%eax)
 804940b:	75 66                	jne    8049473 <rio_readlineb+0x8e>
 804940d:	8b 43 04             	mov    0x4(%ebx),%eax
 8049410:	85 c0                	test   %eax,%eax
 8049412:	7f 23                	jg     8049437 <rio_readlineb+0x52>
 8049414:	83 ec 04             	sub    $0x4,%esp
 8049417:	68 00 20 00 00       	push   $0x2000
 804941c:	56                   	push   %esi
 804941d:	ff 33                	pushl  (%ebx)
 804941f:	e8 0c f3 ff ff       	call   8048730 <read@plt>
 8049424:	89 43 04             	mov    %eax,0x4(%ebx)
 8049427:	83 c4 10             	add    $0x10,%esp
 804942a:	85 c0                	test   %eax,%eax
 804942c:	78 d5                	js     8049403 <rio_readlineb+0x1e>
 804942e:	85 c0                	test   %eax,%eax
 8049430:	74 48                	je     804947a <rio_readlineb+0x95>
 8049432:	89 73 08             	mov    %esi,0x8(%ebx)
 8049435:	eb d6                	jmp    804940d <rio_readlineb+0x28>
 8049437:	8b 4b 08             	mov    0x8(%ebx),%ecx
 804943a:	0f b6 11             	movzbl (%ecx),%edx
 804943d:	83 c1 01             	add    $0x1,%ecx
 8049440:	89 4b 08             	mov    %ecx,0x8(%ebx)
 8049443:	83 e8 01             	sub    $0x1,%eax
 8049446:	89 43 04             	mov    %eax,0x4(%ebx)
 8049449:	83 c7 01             	add    $0x1,%edi
 804944c:	88 57 ff             	mov    %dl,-0x1(%edi)
 804944f:	80 fa 0a             	cmp    $0xa,%dl
 8049452:	74 09                	je     804945d <rio_readlineb+0x78>
 8049454:	83 c5 01             	add    $0x1,%ebp
 8049457:	3b 6c 24 0c          	cmp    0xc(%esp),%ebp
 804945b:	75 b0                	jne    804940d <rio_readlineb+0x28>
 804945d:	c6 07 00             	movb   $0x0,(%edi)
 8049460:	89 e8                	mov    %ebp,%eax
 8049462:	83 c4 1c             	add    $0x1c,%esp
 8049465:	5b                   	pop    %ebx
 8049466:	5e                   	pop    %esi
 8049467:	5f                   	pop    %edi
 8049468:	5d                   	pop    %ebp
 8049469:	c3                   	ret    
 804946a:	89 d7                	mov    %edx,%edi
 804946c:	bd 01 00 00 00       	mov    $0x1,%ebp
 8049471:	eb ea                	jmp    804945d <rio_readlineb+0x78>
 8049473:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049478:	eb 05                	jmp    804947f <rio_readlineb+0x9a>
 804947a:	b8 00 00 00 00       	mov    $0x0,%eax
 804947f:	85 c0                	test   %eax,%eax
 8049481:	75 0c                	jne    804948f <rio_readlineb+0xaa>
 8049483:	83 fd 01             	cmp    $0x1,%ebp
 8049486:	75 d5                	jne    804945d <rio_readlineb+0x78>
 8049488:	bd 00 00 00 00       	mov    $0x0,%ebp
 804948d:	eb d1                	jmp    8049460 <rio_readlineb+0x7b>
 804948f:	bd ff ff ff ff       	mov    $0xffffffff,%ebp
 8049494:	eb ca                	jmp    8049460 <rio_readlineb+0x7b>

08049496 <submitr>:
 8049496:	55                   	push   %ebp
 8049497:	57                   	push   %edi
 8049498:	56                   	push   %esi
 8049499:	53                   	push   %ebx
 804949a:	81 ec 60 a0 00 00    	sub    $0xa060,%esp
 80494a0:	8b 9c 24 74 a0 00 00 	mov    0xa074(%esp),%ebx
 80494a7:	8b 84 24 7c a0 00 00 	mov    0xa07c(%esp),%eax
 80494ae:	89 44 24 0c          	mov    %eax,0xc(%esp)
 80494b2:	8b 84 24 80 a0 00 00 	mov    0xa080(%esp),%eax
 80494b9:	89 44 24 10          	mov    %eax,0x10(%esp)
 80494bd:	8b 84 24 84 a0 00 00 	mov    0xa084(%esp),%eax
 80494c4:	89 44 24 14          	mov    %eax,0x14(%esp)
 80494c8:	8b b4 24 88 a0 00 00 	mov    0xa088(%esp),%esi
 80494cf:	8b 84 24 8c a0 00 00 	mov    0xa08c(%esp),%eax
 80494d6:	89 44 24 18          	mov    %eax,0x18(%esp)
 80494da:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 80494e0:	89 84 24 50 a0 00 00 	mov    %eax,0xa050(%esp)
 80494e7:	31 c0                	xor    %eax,%eax
 80494e9:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%esp)
 80494f0:	00 
 80494f1:	6a 00                	push   $0x0
 80494f3:	6a 01                	push   $0x1
 80494f5:	6a 02                	push   $0x2
 80494f7:	e8 54 f3 ff ff       	call   8048850 <socket@plt>
 80494fc:	83 c4 10             	add    $0x10,%esp
 80494ff:	85 c0                	test   %eax,%eax
 8049501:	0f 88 04 01 00 00    	js     804960b <submitr+0x175>
 8049507:	89 c5                	mov    %eax,%ebp
 8049509:	83 ec 0c             	sub    $0xc,%esp
 804950c:	53                   	push   %ebx
 804950d:	e8 5e f3 ff ff       	call   8048870 <gethostbyname@plt>
 8049512:	83 c4 10             	add    $0x10,%esp
 8049515:	85 c0                	test   %eax,%eax
 8049517:	0f 84 40 01 00 00    	je     804965d <submitr+0x1c7>
 804951d:	8d 5c 24 30          	lea    0x30(%esp),%ebx
 8049521:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%esp)
 8049528:	00 
 8049529:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%esp)
 8049530:	00 
 8049531:	c7 44 24 38 00 00 00 	movl   $0x0,0x38(%esp)
 8049538:	00 
 8049539:	c7 44 24 3c 00 00 00 	movl   $0x0,0x3c(%esp)
 8049540:	00 
 8049541:	66 c7 44 24 30 02 00 	movw   $0x2,0x30(%esp)
 8049548:	6a 0c                	push   $0xc
 804954a:	ff 70 0c             	pushl  0xc(%eax)
 804954d:	8b 40 10             	mov    0x10(%eax),%eax
 8049550:	ff 30                	pushl  (%eax)
 8049552:	8d 44 24 40          	lea    0x40(%esp),%eax
 8049556:	50                   	push   %eax
 8049557:	e8 74 f2 ff ff       	call   80487d0 <__memmove_chk@plt>
 804955c:	0f b7 84 24 84 a0 00 	movzwl 0xa084(%esp),%eax
 8049563:	00 
 8049564:	66 c1 c8 08          	ror    $0x8,%ax
 8049568:	66 89 44 24 42       	mov    %ax,0x42(%esp)
 804956d:	83 c4 0c             	add    $0xc,%esp
 8049570:	6a 10                	push   $0x10
 8049572:	53                   	push   %ebx
 8049573:	55                   	push   %ebp
 8049574:	e8 17 f3 ff ff       	call   8048890 <connect@plt>
 8049579:	83 c4 10             	add    $0x10,%esp
 804957c:	85 c0                	test   %eax,%eax
 804957e:	0f 88 49 01 00 00    	js     80496cd <submitr+0x237>
 8049584:	ba ff ff ff ff       	mov    $0xffffffff,%edx
 8049589:	b8 00 00 00 00       	mov    $0x0,%eax
 804958e:	89 d1                	mov    %edx,%ecx
 8049590:	89 f7                	mov    %esi,%edi
 8049592:	f2 ae                	repnz scas %es:(%edi),%al
 8049594:	89 cb                	mov    %ecx,%ebx
 8049596:	f7 d3                	not    %ebx
 8049598:	89 d1                	mov    %edx,%ecx
 804959a:	8b 7c 24 08          	mov    0x8(%esp),%edi
 804959e:	f2 ae                	repnz scas %es:(%edi),%al
 80495a0:	89 4c 24 18          	mov    %ecx,0x18(%esp)
 80495a4:	89 d1                	mov    %edx,%ecx
 80495a6:	8b 7c 24 0c          	mov    0xc(%esp),%edi
 80495aa:	f2 ae                	repnz scas %es:(%edi),%al
 80495ac:	89 cf                	mov    %ecx,%edi
 80495ae:	f7 d7                	not    %edi
 80495b0:	89 7c 24 1c          	mov    %edi,0x1c(%esp)
 80495b4:	89 d1                	mov    %edx,%ecx
 80495b6:	8b 7c 24 10          	mov    0x10(%esp),%edi
 80495ba:	f2 ae                	repnz scas %es:(%edi),%al
 80495bc:	8b 54 24 1c          	mov    0x1c(%esp),%edx
 80495c0:	2b 54 24 18          	sub    0x18(%esp),%edx
 80495c4:	29 ca                	sub    %ecx,%edx
 80495c6:	8d 44 5b fd          	lea    -0x3(%ebx,%ebx,2),%eax
 80495ca:	8d 44 02 7b          	lea    0x7b(%edx,%eax,1),%eax
 80495ce:	3d 00 20 00 00       	cmp    $0x2000,%eax
 80495d3:	0f 87 56 01 00 00    	ja     804972f <submitr+0x299>
 80495d9:	8d 94 24 4c 40 00 00 	lea    0x404c(%esp),%edx
 80495e0:	b9 00 08 00 00       	mov    $0x800,%ecx
 80495e5:	b8 00 00 00 00       	mov    $0x0,%eax
 80495ea:	89 d7                	mov    %edx,%edi
 80495ec:	f3 ab                	rep stos %eax,%es:(%edi)
 80495ee:	b9 ff ff ff ff       	mov    $0xffffffff,%ecx
 80495f3:	89 f7                	mov    %esi,%edi
 80495f5:	f2 ae                	repnz scas %es:(%edi),%al
 80495f7:	f7 d1                	not    %ecx
 80495f9:	89 cb                	mov    %ecx,%ebx
 80495fb:	83 eb 01             	sub    $0x1,%ebx
 80495fe:	0f 84 07 06 00 00    	je     8049c0b <submitr+0x775>
 8049604:	89 d7                	mov    %edx,%edi
 8049606:	e9 b0 01 00 00       	jmp    80497bb <submitr+0x325>
 804960b:	8b 44 24 14          	mov    0x14(%esp),%eax
 804960f:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 8049615:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
 804961c:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
 8049623:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
 804962a:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
 8049631:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
 8049638:	c7 40 18 63 72 65 61 	movl   $0x61657263,0x18(%eax)
 804963f:	c7 40 1c 74 65 20 73 	movl   $0x73206574,0x1c(%eax)
 8049646:	c7 40 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%eax)
 804964d:	66 c7 40 24 74 00    	movw   $0x74,0x24(%eax)
 8049653:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049658:	e9 d0 04 00 00       	jmp    8049b2d <submitr+0x697>
 804965d:	8b 44 24 14          	mov    0x14(%esp),%eax
 8049661:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 8049667:	c7 40 04 72 3a 20 44 	movl   $0x44203a72,0x4(%eax)
 804966e:	c7 40 08 4e 53 20 69 	movl   $0x6920534e,0x8(%eax)
 8049675:	c7 40 0c 73 20 75 6e 	movl   $0x6e752073,0xc(%eax)
 804967c:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
 8049683:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
 804968a:	c7 40 18 72 65 73 6f 	movl   $0x6f736572,0x18(%eax)
 8049691:	c7 40 1c 6c 76 65 20 	movl   $0x2065766c,0x1c(%eax)
 8049698:	c7 40 20 73 65 72 76 	movl   $0x76726573,0x20(%eax)
 804969f:	c7 40 24 65 72 20 61 	movl   $0x61207265,0x24(%eax)
 80496a6:	c7 40 28 64 64 72 65 	movl   $0x65726464,0x28(%eax)
 80496ad:	66 c7 40 2c 73 73    	movw   $0x7373,0x2c(%eax)
 80496b3:	c6 40 2e 00          	movb   $0x0,0x2e(%eax)
 80496b7:	83 ec 0c             	sub    $0xc,%esp
 80496ba:	55                   	push   %ebp
 80496bb:	e8 e0 f1 ff ff       	call   80488a0 <close@plt>
 80496c0:	83 c4 10             	add    $0x10,%esp
 80496c3:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 80496c8:	e9 60 04 00 00       	jmp    8049b2d <submitr+0x697>
 80496cd:	8b 44 24 14          	mov    0x14(%esp),%eax
 80496d1:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 80496d7:	c7 40 04 72 3a 20 55 	movl   $0x55203a72,0x4(%eax)
 80496de:	c7 40 08 6e 61 62 6c 	movl   $0x6c62616e,0x8(%eax)
 80496e5:	c7 40 0c 65 20 74 6f 	movl   $0x6f742065,0xc(%eax)
 80496ec:	c7 40 10 20 63 6f 6e 	movl   $0x6e6f6320,0x10(%eax)
 80496f3:	c7 40 14 6e 65 63 74 	movl   $0x7463656e,0x14(%eax)
 80496fa:	c7 40 18 20 74 6f 20 	movl   $0x206f7420,0x18(%eax)
 8049701:	c7 40 1c 74 68 65 20 	movl   $0x20656874,0x1c(%eax)
 8049708:	c7 40 20 73 65 72 76 	movl   $0x76726573,0x20(%eax)
 804970f:	66 c7 40 24 65 72    	movw   $0x7265,0x24(%eax)
 8049715:	c6 40 26 00          	movb   $0x0,0x26(%eax)
 8049719:	83 ec 0c             	sub    $0xc,%esp
 804971c:	55                   	push   %ebp
 804971d:	e8 7e f1 ff ff       	call   80488a0 <close@plt>
 8049722:	83 c4 10             	add    $0x10,%esp
 8049725:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 804972a:	e9 fe 03 00 00       	jmp    8049b2d <submitr+0x697>
 804972f:	8b 44 24 14          	mov    0x14(%esp),%eax
 8049733:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 8049739:	c7 40 04 72 3a 20 52 	movl   $0x52203a72,0x4(%eax)
 8049740:	c7 40 08 65 73 75 6c 	movl   $0x6c757365,0x8(%eax)
 8049747:	c7 40 0c 74 20 73 74 	movl   $0x74732074,0xc(%eax)
 804974e:	c7 40 10 72 69 6e 67 	movl   $0x676e6972,0x10(%eax)
 8049755:	c7 40 14 20 74 6f 6f 	movl   $0x6f6f7420,0x14(%eax)
 804975c:	c7 40 18 20 6c 61 72 	movl   $0x72616c20,0x18(%eax)
 8049763:	c7 40 1c 67 65 2e 20 	movl   $0x202e6567,0x1c(%eax)
 804976a:	c7 40 20 49 6e 63 72 	movl   $0x72636e49,0x20(%eax)
 8049771:	c7 40 24 65 61 73 65 	movl   $0x65736165,0x24(%eax)
 8049778:	c7 40 28 20 53 55 42 	movl   $0x42555320,0x28(%eax)
 804977f:	c7 40 2c 4d 49 54 52 	movl   $0x5254494d,0x2c(%eax)
 8049786:	c7 40 30 5f 4d 41 58 	movl   $0x58414d5f,0x30(%eax)
 804978d:	c7 40 34 42 55 46 00 	movl   $0x465542,0x34(%eax)
 8049794:	83 ec 0c             	sub    $0xc,%esp
 8049797:	55                   	push   %ebp
 8049798:	e8 03 f1 ff ff       	call   80488a0 <close@plt>
 804979d:	83 c4 10             	add    $0x10,%esp
 80497a0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 80497a5:	e9 83 03 00 00       	jmp    8049b2d <submitr+0x697>
 80497aa:	88 17                	mov    %dl,(%edi)
 80497ac:	8d 7f 01             	lea    0x1(%edi),%edi
 80497af:	83 c6 01             	add    $0x1,%esi
 80497b2:	83 eb 01             	sub    $0x1,%ebx
 80497b5:	0f 84 50 04 00 00    	je     8049c0b <submitr+0x775>
 80497bb:	0f b6 16             	movzbl (%esi),%edx
 80497be:	8d 4a d6             	lea    -0x2a(%edx),%ecx
 80497c1:	b8 01 00 00 00       	mov    $0x1,%eax
 80497c6:	80 f9 0f             	cmp    $0xf,%cl
 80497c9:	77 0d                	ja     80497d8 <submitr+0x342>
 80497cb:	b8 d9 ff 00 00       	mov    $0xffd9,%eax
 80497d0:	d3 e8                	shr    %cl,%eax
 80497d2:	83 f0 01             	xor    $0x1,%eax
 80497d5:	83 e0 01             	and    $0x1,%eax
 80497d8:	80 fa 5f             	cmp    $0x5f,%dl
 80497db:	74 cd                	je     80497aa <submitr+0x314>
 80497dd:	84 c0                	test   %al,%al
 80497df:	74 c9                	je     80497aa <submitr+0x314>
 80497e1:	89 d0                	mov    %edx,%eax
 80497e3:	83 e0 df             	and    $0xffffffdf,%eax
 80497e6:	83 e8 41             	sub    $0x41,%eax
 80497e9:	3c 19                	cmp    $0x19,%al
 80497eb:	76 bd                	jbe    80497aa <submitr+0x314>
 80497ed:	80 fa 20             	cmp    $0x20,%dl
 80497f0:	74 58                	je     804984a <submitr+0x3b4>
 80497f2:	8d 42 e0             	lea    -0x20(%edx),%eax
 80497f5:	3c 5f                	cmp    $0x5f,%al
 80497f7:	76 09                	jbe    8049802 <submitr+0x36c>
 80497f9:	80 fa 09             	cmp    $0x9,%dl
 80497fc:	0f 85 c5 03 00 00    	jne    8049bc7 <submitr+0x731>
 8049802:	83 ec 0c             	sub    $0xc,%esp
 8049805:	0f b6 d2             	movzbl %dl,%edx
 8049808:	52                   	push   %edx
 8049809:	68 74 a4 04 08       	push   $0x804a474
 804980e:	6a 08                	push   $0x8
 8049810:	6a 01                	push   $0x1
 8049812:	8d 84 24 68 80 00 00 	lea    0x8068(%esp),%eax
 8049819:	50                   	push   %eax
 804981a:	e8 a1 f0 ff ff       	call   80488c0 <__sprintf_chk@plt>
 804981f:	0f b6 84 24 6c 80 00 	movzbl 0x806c(%esp),%eax
 8049826:	00 
 8049827:	88 07                	mov    %al,(%edi)
 8049829:	0f b6 84 24 6d 80 00 	movzbl 0x806d(%esp),%eax
 8049830:	00 
 8049831:	88 47 01             	mov    %al,0x1(%edi)
 8049834:	0f b6 84 24 6e 80 00 	movzbl 0x806e(%esp),%eax
 804983b:	00 
 804983c:	88 47 02             	mov    %al,0x2(%edi)
 804983f:	83 c4 20             	add    $0x20,%esp
 8049842:	8d 7f 03             	lea    0x3(%edi),%edi
 8049845:	e9 65 ff ff ff       	jmp    80497af <submitr+0x319>
 804984a:	c6 07 2b             	movb   $0x2b,(%edi)
 804984d:	8d 7f 01             	lea    0x1(%edi),%edi
 8049850:	e9 5a ff ff ff       	jmp    80497af <submitr+0x319>
 8049855:	01 c6                	add    %eax,%esi
 8049857:	29 c3                	sub    %eax,%ebx
 8049859:	74 24                	je     804987f <submitr+0x3e9>
 804985b:	83 ec 04             	sub    $0x4,%esp
 804985e:	53                   	push   %ebx
 804985f:	56                   	push   %esi
 8049860:	55                   	push   %ebp
 8049861:	e8 9a ef ff ff       	call   8048800 <write@plt>
 8049866:	83 c4 10             	add    $0x10,%esp
 8049869:	85 c0                	test   %eax,%eax
 804986b:	7f e8                	jg     8049855 <submitr+0x3bf>
 804986d:	e8 be ef ff ff       	call   8048830 <__errno_location@plt>
 8049872:	83 38 04             	cmpl   $0x4,(%eax)
 8049875:	0f 85 b0 00 00 00    	jne    804992b <submitr+0x495>
 804987b:	89 f8                	mov    %edi,%eax
 804987d:	eb d6                	jmp    8049855 <submitr+0x3bf>
 804987f:	83 7c 24 08 00       	cmpl   $0x0,0x8(%esp)
 8049884:	0f 88 a1 00 00 00    	js     804992b <submitr+0x495>
 804988a:	89 6c 24 40          	mov    %ebp,0x40(%esp)
 804988e:	c7 44 24 44 00 00 00 	movl   $0x0,0x44(%esp)
 8049895:	00 
 8049896:	8d 44 24 4c          	lea    0x4c(%esp),%eax
 804989a:	89 44 24 48          	mov    %eax,0x48(%esp)
 804989e:	b9 00 20 00 00       	mov    $0x2000,%ecx
 80498a3:	8d 94 24 4c 20 00 00 	lea    0x204c(%esp),%edx
 80498aa:	8d 44 24 40          	lea    0x40(%esp),%eax
 80498ae:	e8 32 fb ff ff       	call   80493e5 <rio_readlineb>
 80498b3:	85 c0                	test   %eax,%eax
 80498b5:	0f 8e d6 00 00 00    	jle    8049991 <submitr+0x4fb>
 80498bb:	83 ec 0c             	sub    $0xc,%esp
 80498be:	8d 84 24 58 80 00 00 	lea    0x8058(%esp),%eax
 80498c5:	50                   	push   %eax
 80498c6:	8d 44 24 3c          	lea    0x3c(%esp),%eax
 80498ca:	50                   	push   %eax
 80498cb:	8d 84 24 60 60 00 00 	lea    0x6060(%esp),%eax
 80498d2:	50                   	push   %eax
 80498d3:	68 7b a4 04 08       	push   $0x804a47b
 80498d8:	8d 84 24 68 20 00 00 	lea    0x2068(%esp),%eax
 80498df:	50                   	push   %eax
 80498e0:	e8 2b ef ff ff       	call   8048810 <__isoc99_sscanf@plt>
 80498e5:	8b 44 24 4c          	mov    0x4c(%esp),%eax
 80498e9:	83 c4 20             	add    $0x20,%esp
 80498ec:	3d c8 00 00 00       	cmp    $0xc8,%eax
 80498f1:	0f 84 a6 01 00 00    	je     8049a9d <submitr+0x607>
 80498f7:	83 ec 08             	sub    $0x8,%esp
 80498fa:	8d 94 24 54 80 00 00 	lea    0x8054(%esp),%edx
 8049901:	52                   	push   %edx
 8049902:	50                   	push   %eax
 8049903:	68 8c a3 04 08       	push   $0x804a38c
 8049908:	6a ff                	push   $0xffffffff
 804990a:	6a 01                	push   $0x1
 804990c:	ff 74 24 30          	pushl  0x30(%esp)
 8049910:	e8 ab ef ff ff       	call   80488c0 <__sprintf_chk@plt>
 8049915:	83 c4 14             	add    $0x14,%esp
 8049918:	55                   	push   %ebp
 8049919:	e8 82 ef ff ff       	call   80488a0 <close@plt>
 804991e:	83 c4 10             	add    $0x10,%esp
 8049921:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049926:	e9 02 02 00 00       	jmp    8049b2d <submitr+0x697>
 804992b:	8b 44 24 14          	mov    0x14(%esp),%eax
 804992f:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 8049935:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
 804993c:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
 8049943:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
 804994a:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
 8049951:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
 8049958:	c7 40 18 77 72 69 74 	movl   $0x74697277,0x18(%eax)
 804995f:	c7 40 1c 65 20 74 6f 	movl   $0x6f742065,0x1c(%eax)
 8049966:	c7 40 20 20 74 68 65 	movl   $0x65687420,0x20(%eax)
 804996d:	c7 40 24 20 73 65 72 	movl   $0x72657320,0x24(%eax)
 8049974:	c7 40 28 76 65 72 00 	movl   $0x726576,0x28(%eax)
 804997b:	83 ec 0c             	sub    $0xc,%esp
 804997e:	55                   	push   %ebp
 804997f:	e8 1c ef ff ff       	call   80488a0 <close@plt>
 8049984:	83 c4 10             	add    $0x10,%esp
 8049987:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 804998c:	e9 9c 01 00 00       	jmp    8049b2d <submitr+0x697>
 8049991:	8b 44 24 14          	mov    0x14(%esp),%eax
 8049995:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 804999b:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
 80499a2:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
 80499a9:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
 80499b0:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
 80499b7:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
 80499be:	c7 40 18 72 65 61 64 	movl   $0x64616572,0x18(%eax)
 80499c5:	c7 40 1c 20 66 69 72 	movl   $0x72696620,0x1c(%eax)
 80499cc:	c7 40 20 73 74 20 68 	movl   $0x68207473,0x20(%eax)
 80499d3:	c7 40 24 65 61 64 65 	movl   $0x65646165,0x24(%eax)
 80499da:	c7 40 28 72 20 66 72 	movl   $0x72662072,0x28(%eax)
 80499e1:	c7 40 2c 6f 6d 20 73 	movl   $0x73206d6f,0x2c(%eax)
 80499e8:	c7 40 30 65 72 76 65 	movl   $0x65767265,0x30(%eax)
 80499ef:	66 c7 40 34 72 00    	movw   $0x72,0x34(%eax)
 80499f5:	83 ec 0c             	sub    $0xc,%esp
 80499f8:	55                   	push   %ebp
 80499f9:	e8 a2 ee ff ff       	call   80488a0 <close@plt>
 80499fe:	83 c4 10             	add    $0x10,%esp
 8049a01:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049a06:	e9 22 01 00 00       	jmp    8049b2d <submitr+0x697>
 8049a0b:	8b 44 24 14          	mov    0x14(%esp),%eax
 8049a0f:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 8049a15:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
 8049a1c:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
 8049a23:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
 8049a2a:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
 8049a31:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
 8049a38:	c7 40 18 72 65 61 64 	movl   $0x64616572,0x18(%eax)
 8049a3f:	c7 40 1c 20 68 65 61 	movl   $0x61656820,0x1c(%eax)
 8049a46:	c7 40 20 64 65 72 73 	movl   $0x73726564,0x20(%eax)
 8049a4d:	c7 40 24 20 66 72 6f 	movl   $0x6f726620,0x24(%eax)
 8049a54:	c7 40 28 6d 20 73 65 	movl   $0x6573206d,0x28(%eax)
 8049a5b:	c7 40 2c 72 76 65 72 	movl   $0x72657672,0x2c(%eax)
 8049a62:	c6 40 30 00          	movb   $0x0,0x30(%eax)
 8049a66:	83 ec 0c             	sub    $0xc,%esp
 8049a69:	55                   	push   %ebp
 8049a6a:	e8 31 ee ff ff       	call   80488a0 <close@plt>
 8049a6f:	83 c4 10             	add    $0x10,%esp
 8049a72:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049a77:	e9 b1 00 00 00       	jmp    8049b2d <submitr+0x697>
 8049a7c:	85 c0                	test   %eax,%eax
 8049a7e:	74 4b                	je     8049acb <submitr+0x635>
 8049a80:	b9 00 20 00 00       	mov    $0x2000,%ecx
 8049a85:	8d 94 24 4c 20 00 00 	lea    0x204c(%esp),%edx
 8049a8c:	8d 44 24 40          	lea    0x40(%esp),%eax
 8049a90:	e8 50 f9 ff ff       	call   80493e5 <rio_readlineb>
 8049a95:	85 c0                	test   %eax,%eax
 8049a97:	0f 8e 6e ff ff ff    	jle    8049a0b <submitr+0x575>
 8049a9d:	0f b6 94 24 4c 20 00 	movzbl 0x204c(%esp),%edx
 8049aa4:	00 
 8049aa5:	b8 0d 00 00 00       	mov    $0xd,%eax
 8049aaa:	29 d0                	sub    %edx,%eax
 8049aac:	75 ce                	jne    8049a7c <submitr+0x5e6>
 8049aae:	0f b6 94 24 4d 20 00 	movzbl 0x204d(%esp),%edx
 8049ab5:	00 
 8049ab6:	b8 0a 00 00 00       	mov    $0xa,%eax
 8049abb:	29 d0                	sub    %edx,%eax
 8049abd:	75 bd                	jne    8049a7c <submitr+0x5e6>
 8049abf:	0f b6 84 24 4e 20 00 	movzbl 0x204e(%esp),%eax
 8049ac6:	00 
 8049ac7:	f7 d8                	neg    %eax
 8049ac9:	eb b1                	jmp    8049a7c <submitr+0x5e6>
 8049acb:	b9 00 20 00 00       	mov    $0x2000,%ecx
 8049ad0:	8d 94 24 4c 20 00 00 	lea    0x204c(%esp),%edx
 8049ad7:	8d 44 24 40          	lea    0x40(%esp),%eax
 8049adb:	e8 05 f9 ff ff       	call   80493e5 <rio_readlineb>
 8049ae0:	85 c0                	test   %eax,%eax
 8049ae2:	7e 68                	jle    8049b4c <submitr+0x6b6>
 8049ae4:	83 ec 08             	sub    $0x8,%esp
 8049ae7:	8d 84 24 54 20 00 00 	lea    0x2054(%esp),%eax
 8049aee:	50                   	push   %eax
 8049aef:	8b 7c 24 20          	mov    0x20(%esp),%edi
 8049af3:	57                   	push   %edi
 8049af4:	e8 a7 ec ff ff       	call   80487a0 <strcpy@plt>
 8049af9:	89 2c 24             	mov    %ebp,(%esp)
 8049afc:	e8 9f ed ff ff       	call   80488a0 <close@plt>
 8049b01:	0f b6 17             	movzbl (%edi),%edx
 8049b04:	b8 4f 00 00 00       	mov    $0x4f,%eax
 8049b09:	83 c4 10             	add    $0x10,%esp
 8049b0c:	29 d0                	sub    %edx,%eax
 8049b0e:	75 13                	jne    8049b23 <submitr+0x68d>
 8049b10:	0f b6 57 01          	movzbl 0x1(%edi),%edx
 8049b14:	b8 4b 00 00 00       	mov    $0x4b,%eax
 8049b19:	29 d0                	sub    %edx,%eax
 8049b1b:	75 06                	jne    8049b23 <submitr+0x68d>
 8049b1d:	0f b6 47 02          	movzbl 0x2(%edi),%eax
 8049b21:	f7 d8                	neg    %eax
 8049b23:	85 c0                	test   %eax,%eax
 8049b25:	0f 95 c0             	setne  %al
 8049b28:	0f b6 c0             	movzbl %al,%eax
 8049b2b:	f7 d8                	neg    %eax
 8049b2d:	8b bc 24 4c a0 00 00 	mov    0xa04c(%esp),%edi
 8049b34:	65 33 3d 14 00 00 00 	xor    %gs:0x14,%edi
 8049b3b:	0f 85 2a 01 00 00    	jne    8049c6b <submitr+0x7d5>
 8049b41:	81 c4 5c a0 00 00    	add    $0xa05c,%esp
 8049b47:	5b                   	pop    %ebx
 8049b48:	5e                   	pop    %esi
 8049b49:	5f                   	pop    %edi
 8049b4a:	5d                   	pop    %ebp
 8049b4b:	c3                   	ret    
 8049b4c:	8b 44 24 14          	mov    0x14(%esp),%eax
 8049b50:	c7 00 45 72 72 6f    	movl   $0x6f727245,(%eax)
 8049b56:	c7 40 04 72 3a 20 43 	movl   $0x43203a72,0x4(%eax)
 8049b5d:	c7 40 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%eax)
 8049b64:	c7 40 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%eax)
 8049b6b:	c7 40 10 61 62 6c 65 	movl   $0x656c6261,0x10(%eax)
 8049b72:	c7 40 14 20 74 6f 20 	movl   $0x206f7420,0x14(%eax)
 8049b79:	c7 40 18 72 65 61 64 	movl   $0x64616572,0x18(%eax)
 8049b80:	c7 40 1c 20 73 74 61 	movl   $0x61747320,0x1c(%eax)
 8049b87:	c7 40 20 74 75 73 20 	movl   $0x20737574,0x20(%eax)
 8049b8e:	c7 40 24 6d 65 73 73 	movl   $0x7373656d,0x24(%eax)
 8049b95:	c7 40 28 61 67 65 20 	movl   $0x20656761,0x28(%eax)
 8049b9c:	c7 40 2c 66 72 6f 6d 	movl   $0x6d6f7266,0x2c(%eax)
 8049ba3:	c7 40 30 20 73 65 72 	movl   $0x72657320,0x30(%eax)
 8049baa:	c7 40 34 76 65 72 00 	movl   $0x726576,0x34(%eax)
 8049bb1:	83 ec 0c             	sub    $0xc,%esp
 8049bb4:	55                   	push   %ebp
 8049bb5:	e8 e6 ec ff ff       	call   80488a0 <close@plt>
 8049bba:	83 c4 10             	add    $0x10,%esp
 8049bbd:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049bc2:	e9 66 ff ff ff       	jmp    8049b2d <submitr+0x697>
 8049bc7:	a1 bc a3 04 08       	mov    0x804a3bc,%eax
 8049bcc:	8b 5c 24 14          	mov    0x14(%esp),%ebx
 8049bd0:	89 03                	mov    %eax,(%ebx)
 8049bd2:	a1 fb a3 04 08       	mov    0x804a3fb,%eax
 8049bd7:	89 43 3f             	mov    %eax,0x3f(%ebx)
 8049bda:	8d 7b 04             	lea    0x4(%ebx),%edi
 8049bdd:	83 e7 fc             	and    $0xfffffffc,%edi
 8049be0:	29 fb                	sub    %edi,%ebx
 8049be2:	89 d8                	mov    %ebx,%eax
 8049be4:	be bc a3 04 08       	mov    $0x804a3bc,%esi
 8049be9:	29 de                	sub    %ebx,%esi
 8049beb:	83 c0 43             	add    $0x43,%eax
 8049bee:	c1 e8 02             	shr    $0x2,%eax
 8049bf1:	89 c1                	mov    %eax,%ecx
 8049bf3:	f3 a5                	rep movsl %ds:(%esi),%es:(%edi)
 8049bf5:	83 ec 0c             	sub    $0xc,%esp
 8049bf8:	55                   	push   %ebp
 8049bf9:	e8 a2 ec ff ff       	call   80488a0 <close@plt>
 8049bfe:	83 c4 10             	add    $0x10,%esp
 8049c01:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049c06:	e9 22 ff ff ff       	jmp    8049b2d <submitr+0x697>
 8049c0b:	8d 84 24 4c 40 00 00 	lea    0x404c(%esp),%eax
 8049c12:	50                   	push   %eax
 8049c13:	ff 74 24 14          	pushl  0x14(%esp)
 8049c17:	ff 74 24 14          	pushl  0x14(%esp)
 8049c1b:	ff 74 24 14          	pushl  0x14(%esp)
 8049c1f:	68 00 a4 04 08       	push   $0x804a400
 8049c24:	68 00 20 00 00       	push   $0x2000
 8049c29:	6a 01                	push   $0x1
 8049c2b:	8d bc 24 68 20 00 00 	lea    0x2068(%esp),%edi
 8049c32:	57                   	push   %edi
 8049c33:	e8 88 ec ff ff       	call   80488c0 <__sprintf_chk@plt>
 8049c38:	b8 00 00 00 00       	mov    $0x0,%eax
 8049c3d:	b9 ff ff ff ff       	mov    $0xffffffff,%ecx
 8049c42:	f2 ae                	repnz scas %es:(%edi),%al
 8049c44:	f7 d1                	not    %ecx
 8049c46:	8d 41 ff             	lea    -0x1(%ecx),%eax
 8049c49:	89 44 24 28          	mov    %eax,0x28(%esp)
 8049c4d:	83 c4 20             	add    $0x20,%esp
 8049c50:	89 c3                	mov    %eax,%ebx
 8049c52:	8d b4 24 4c 20 00 00 	lea    0x204c(%esp),%esi
 8049c59:	bf 00 00 00 00       	mov    $0x0,%edi
 8049c5e:	85 c0                	test   %eax,%eax
 8049c60:	0f 85 f5 fb ff ff    	jne    804985b <submitr+0x3c5>
 8049c66:	e9 1f fc ff ff       	jmp    804988a <submitr+0x3f4>
 8049c6b:	e8 20 eb ff ff       	call   8048790 <__stack_chk_fail@plt>

08049c70 <init_timeout>:
 8049c70:	53                   	push   %ebx
 8049c71:	83 ec 08             	sub    $0x8,%esp
 8049c74:	8b 5c 24 10          	mov    0x10(%esp),%ebx
 8049c78:	85 db                	test   %ebx,%ebx
 8049c7a:	74 24                	je     8049ca0 <init_timeout+0x30>
 8049c7c:	83 ec 08             	sub    $0x8,%esp
 8049c7f:	68 c2 93 04 08       	push   $0x80493c2
 8049c84:	6a 0e                	push   $0xe
 8049c86:	e8 d5 ea ff ff       	call   8048760 <signal@plt>
 8049c8b:	85 db                	test   %ebx,%ebx
 8049c8d:	b8 00 00 00 00       	mov    $0x0,%eax
 8049c92:	0f 48 d8             	cmovs  %eax,%ebx
 8049c95:	89 1c 24             	mov    %ebx,(%esp)
 8049c98:	e8 e3 ea ff ff       	call   8048780 <alarm@plt>
 8049c9d:	83 c4 10             	add    $0x10,%esp
 8049ca0:	83 c4 08             	add    $0x8,%esp
 8049ca3:	5b                   	pop    %ebx
 8049ca4:	c3                   	ret    

08049ca5 <init_driver>:
 8049ca5:	57                   	push   %edi
 8049ca6:	56                   	push   %esi
 8049ca7:	53                   	push   %ebx
 8049ca8:	83 ec 28             	sub    $0x28,%esp
 8049cab:	8b 74 24 38          	mov    0x38(%esp),%esi
 8049caf:	65 a1 14 00 00 00    	mov    %gs:0x14,%eax
 8049cb5:	89 44 24 24          	mov    %eax,0x24(%esp)
 8049cb9:	31 c0                	xor    %eax,%eax
 8049cbb:	6a 01                	push   $0x1
 8049cbd:	6a 0d                	push   $0xd
 8049cbf:	e8 9c ea ff ff       	call   8048760 <signal@plt>
 8049cc4:	83 c4 08             	add    $0x8,%esp
 8049cc7:	6a 01                	push   $0x1
 8049cc9:	6a 1d                	push   $0x1d
 8049ccb:	e8 90 ea ff ff       	call   8048760 <signal@plt>
 8049cd0:	83 c4 08             	add    $0x8,%esp
 8049cd3:	6a 01                	push   $0x1
 8049cd5:	6a 1d                	push   $0x1d
 8049cd7:	e8 84 ea ff ff       	call   8048760 <signal@plt>
 8049cdc:	83 c4 0c             	add    $0xc,%esp
 8049cdf:	6a 00                	push   $0x0
 8049ce1:	6a 01                	push   $0x1
 8049ce3:	6a 02                	push   $0x2
 8049ce5:	e8 66 eb ff ff       	call   8048850 <socket@plt>
 8049cea:	83 c4 10             	add    $0x10,%esp
 8049ced:	85 c0                	test   %eax,%eax
 8049cef:	0f 88 a9 00 00 00    	js     8049d9e <init_driver+0xf9>
 8049cf5:	89 c3                	mov    %eax,%ebx
 8049cf7:	83 ec 0c             	sub    $0xc,%esp
 8049cfa:	68 8c a4 04 08       	push   $0x804a48c
 8049cff:	e8 6c eb ff ff       	call   8048870 <gethostbyname@plt>
 8049d04:	83 c4 10             	add    $0x10,%esp
 8049d07:	85 c0                	test   %eax,%eax
 8049d09:	0f 84 da 00 00 00    	je     8049de9 <init_driver+0x144>
 8049d0f:	8d 7c 24 0c          	lea    0xc(%esp),%edi
 8049d13:	c7 44 24 0c 00 00 00 	movl   $0x0,0xc(%esp)
 8049d1a:	00 
 8049d1b:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%esp)
 8049d22:	00 
 8049d23:	c7 44 24 14 00 00 00 	movl   $0x0,0x14(%esp)
 8049d2a:	00 
 8049d2b:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%esp)
 8049d32:	00 
 8049d33:	66 c7 44 24 0c 02 00 	movw   $0x2,0xc(%esp)
 8049d3a:	6a 0c                	push   $0xc
 8049d3c:	ff 70 0c             	pushl  0xc(%eax)
 8049d3f:	8b 40 10             	mov    0x10(%eax),%eax
 8049d42:	ff 30                	pushl  (%eax)
 8049d44:	8d 44 24 1c          	lea    0x1c(%esp),%eax
 8049d48:	50                   	push   %eax
 8049d49:	e8 82 ea ff ff       	call   80487d0 <__memmove_chk@plt>
 8049d4e:	66 c7 44 24 1e 3b 6e 	movw   $0x6e3b,0x1e(%esp)
 8049d55:	83 c4 0c             	add    $0xc,%esp
 8049d58:	6a 10                	push   $0x10
 8049d5a:	57                   	push   %edi
 8049d5b:	53                   	push   %ebx
 8049d5c:	e8 2f eb ff ff       	call   8048890 <connect@plt>
 8049d61:	83 c4 10             	add    $0x10,%esp
 8049d64:	85 c0                	test   %eax,%eax
 8049d66:	0f 88 e9 00 00 00    	js     8049e55 <init_driver+0x1b0>
 8049d6c:	83 ec 0c             	sub    $0xc,%esp
 8049d6f:	53                   	push   %ebx
 8049d70:	e8 2b eb ff ff       	call   80488a0 <close@plt>
 8049d75:	66 c7 06 4f 4b       	movw   $0x4b4f,(%esi)
 8049d7a:	c6 46 02 00          	movb   $0x0,0x2(%esi)
 8049d7e:	83 c4 10             	add    $0x10,%esp
 8049d81:	b8 00 00 00 00       	mov    $0x0,%eax
 8049d86:	8b 54 24 1c          	mov    0x1c(%esp),%edx
 8049d8a:	65 33 15 14 00 00 00 	xor    %gs:0x14,%edx
 8049d91:	0f 85 eb 00 00 00    	jne    8049e82 <init_driver+0x1dd>
 8049d97:	83 c4 20             	add    $0x20,%esp
 8049d9a:	5b                   	pop    %ebx
 8049d9b:	5e                   	pop    %esi
 8049d9c:	5f                   	pop    %edi
 8049d9d:	c3                   	ret    
 8049d9e:	c7 06 45 72 72 6f    	movl   $0x6f727245,(%esi)
 8049da4:	c7 46 04 72 3a 20 43 	movl   $0x43203a72,0x4(%esi)
 8049dab:	c7 46 08 6c 69 65 6e 	movl   $0x6e65696c,0x8(%esi)
 8049db2:	c7 46 0c 74 20 75 6e 	movl   $0x6e752074,0xc(%esi)
 8049db9:	c7 46 10 61 62 6c 65 	movl   $0x656c6261,0x10(%esi)
 8049dc0:	c7 46 14 20 74 6f 20 	movl   $0x206f7420,0x14(%esi)
 8049dc7:	c7 46 18 63 72 65 61 	movl   $0x61657263,0x18(%esi)
 8049dce:	c7 46 1c 74 65 20 73 	movl   $0x73206574,0x1c(%esi)
 8049dd5:	c7 46 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%esi)
 8049ddc:	66 c7 46 24 74 00    	movw   $0x74,0x24(%esi)
 8049de2:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049de7:	eb 9d                	jmp    8049d86 <init_driver+0xe1>
 8049de9:	c7 06 45 72 72 6f    	movl   $0x6f727245,(%esi)
 8049def:	c7 46 04 72 3a 20 44 	movl   $0x44203a72,0x4(%esi)
 8049df6:	c7 46 08 4e 53 20 69 	movl   $0x6920534e,0x8(%esi)
 8049dfd:	c7 46 0c 73 20 75 6e 	movl   $0x6e752073,0xc(%esi)
 8049e04:	c7 46 10 61 62 6c 65 	movl   $0x656c6261,0x10(%esi)
 8049e0b:	c7 46 14 20 74 6f 20 	movl   $0x206f7420,0x14(%esi)
 8049e12:	c7 46 18 72 65 73 6f 	movl   $0x6f736572,0x18(%esi)
 8049e19:	c7 46 1c 6c 76 65 20 	movl   $0x2065766c,0x1c(%esi)
 8049e20:	c7 46 20 73 65 72 76 	movl   $0x76726573,0x20(%esi)
 8049e27:	c7 46 24 65 72 20 61 	movl   $0x61207265,0x24(%esi)
 8049e2e:	c7 46 28 64 64 72 65 	movl   $0x65726464,0x28(%esi)
 8049e35:	66 c7 46 2c 73 73    	movw   $0x7373,0x2c(%esi)
 8049e3b:	c6 46 2e 00          	movb   $0x0,0x2e(%esi)
 8049e3f:	83 ec 0c             	sub    $0xc,%esp
 8049e42:	53                   	push   %ebx
 8049e43:	e8 58 ea ff ff       	call   80488a0 <close@plt>
 8049e48:	83 c4 10             	add    $0x10,%esp
 8049e4b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049e50:	e9 31 ff ff ff       	jmp    8049d86 <init_driver+0xe1>
 8049e55:	83 ec 0c             	sub    $0xc,%esp
 8049e58:	68 8c a4 04 08       	push   $0x804a48c
 8049e5d:	68 4c a4 04 08       	push   $0x804a44c
 8049e62:	6a ff                	push   $0xffffffff
 8049e64:	6a 01                	push   $0x1
 8049e66:	56                   	push   %esi
 8049e67:	e8 54 ea ff ff       	call   80488c0 <__sprintf_chk@plt>
 8049e6c:	83 c4 14             	add    $0x14,%esp
 8049e6f:	53                   	push   %ebx
 8049e70:	e8 2b ea ff ff       	call   80488a0 <close@plt>
 8049e75:	83 c4 10             	add    $0x10,%esp
 8049e78:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
 8049e7d:	e9 04 ff ff ff       	jmp    8049d86 <init_driver+0xe1>
 8049e82:	e8 09 e9 ff ff       	call   8048790 <__stack_chk_fail@plt>

08049e87 <driver_post>:
 8049e87:	53                   	push   %ebx
 8049e88:	83 ec 08             	sub    $0x8,%esp
 8049e8b:	8b 54 24 10          	mov    0x10(%esp),%edx
 8049e8f:	8b 44 24 18          	mov    0x18(%esp),%eax
 8049e93:	8b 5c 24 1c          	mov    0x1c(%esp),%ebx
 8049e97:	85 c0                	test   %eax,%eax
 8049e99:	75 17                	jne    8049eb2 <driver_post+0x2b>
 8049e9b:	85 d2                	test   %edx,%edx
 8049e9d:	74 05                	je     8049ea4 <driver_post+0x1d>
 8049e9f:	80 3a 00             	cmpb   $0x0,(%edx)
 8049ea2:	75 34                	jne    8049ed8 <driver_post+0x51>
 8049ea4:	66 c7 03 4f 4b       	movw   $0x4b4f,(%ebx)
 8049ea9:	c6 43 02 00          	movb   $0x0,0x2(%ebx)
 8049ead:	83 c4 08             	add    $0x8,%esp
 8049eb0:	5b                   	pop    %ebx
 8049eb1:	c3                   	ret    
 8049eb2:	83 ec 04             	sub    $0x4,%esp
 8049eb5:	ff 74 24 18          	pushl  0x18(%esp)
 8049eb9:	68 9a a4 04 08       	push   $0x804a49a
 8049ebe:	6a 01                	push   $0x1
 8049ec0:	e8 7b e9 ff ff       	call   8048840 <__printf_chk@plt>
 8049ec5:	66 c7 03 4f 4b       	movw   $0x4b4f,(%ebx)
 8049eca:	c6 43 02 00          	movb   $0x0,0x2(%ebx)
 8049ece:	83 c4 10             	add    $0x10,%esp
 8049ed1:	b8 00 00 00 00       	mov    $0x0,%eax
 8049ed6:	eb d5                	jmp    8049ead <driver_post+0x26>
 8049ed8:	83 ec 04             	sub    $0x4,%esp
 8049edb:	53                   	push   %ebx
 8049edc:	ff 74 24 1c          	pushl  0x1c(%esp)
 8049ee0:	68 b1 a4 04 08       	push   $0x804a4b1
 8049ee5:	52                   	push   %edx
 8049ee6:	68 b9 a4 04 08       	push   $0x804a4b9
 8049eeb:	68 6e 3b 00 00       	push   $0x3b6e
 8049ef0:	68 8c a4 04 08       	push   $0x804a48c
 8049ef5:	e8 9c f5 ff ff       	call   8049496 <submitr>
 8049efa:	83 c4 20             	add    $0x20,%esp
 8049efd:	eb ae                	jmp    8049ead <driver_post+0x26>
 8049eff:	90                   	nop

08049f00 <__libc_csu_init>:
 8049f00:	55                   	push   %ebp
 8049f01:	57                   	push   %edi
 8049f02:	56                   	push   %esi
 8049f03:	53                   	push   %ebx
 8049f04:	e8 07 ea ff ff       	call   8048910 <__x86.get_pc_thunk.bx>
 8049f09:	81 c3 f7 20 00 00    	add    $0x20f7,%ebx
 8049f0f:	83 ec 0c             	sub    $0xc,%esp
 8049f12:	8b 6c 24 20          	mov    0x20(%esp),%ebp
 8049f16:	8d b3 0c ff ff ff    	lea    -0xf4(%ebx),%esi
 8049f1c:	e8 d3 e7 ff ff       	call   80486f4 <_init>
 8049f21:	8d 83 08 ff ff ff    	lea    -0xf8(%ebx),%eax
 8049f27:	29 c6                	sub    %eax,%esi
 8049f29:	c1 fe 02             	sar    $0x2,%esi
 8049f2c:	85 f6                	test   %esi,%esi
 8049f2e:	74 25                	je     8049f55 <__libc_csu_init+0x55>
 8049f30:	31 ff                	xor    %edi,%edi
 8049f32:	8d b6 00 00 00 00    	lea    0x0(%esi),%esi
 8049f38:	83 ec 04             	sub    $0x4,%esp
 8049f3b:	ff 74 24 2c          	pushl  0x2c(%esp)
 8049f3f:	ff 74 24 2c          	pushl  0x2c(%esp)
 8049f43:	55                   	push   %ebp
 8049f44:	ff 94 bb 08 ff ff ff 	call   *-0xf8(%ebx,%edi,4)
 8049f4b:	83 c7 01             	add    $0x1,%edi
 8049f4e:	83 c4 10             	add    $0x10,%esp
 8049f51:	39 fe                	cmp    %edi,%esi
 8049f53:	75 e3                	jne    8049f38 <__libc_csu_init+0x38>
 8049f55:	83 c4 0c             	add    $0xc,%esp
 8049f58:	5b                   	pop    %ebx
 8049f59:	5e                   	pop    %esi
 8049f5a:	5f                   	pop    %edi
 8049f5b:	5d                   	pop    %ebp
 8049f5c:	c3                   	ret    
 8049f5d:	8d 76 00             	lea    0x0(%esi),%esi

08049f60 <__libc_csu_fini>:
 8049f60:	f3 c3                	repz ret 

Disassembly of section .fini:

08049f64 <_fini>:
 8049f64:	53                   	push   %ebx
 8049f65:	83 ec 08             	sub    $0x8,%esp
 8049f68:	e8 a3 e9 ff ff       	call   8048910 <__x86.get_pc_thunk.bx>
 8049f6d:	81 c3 93 20 00 00    	add    $0x2093,%ebx
 8049f73:	83 c4 08             	add    $0x8,%esp
 8049f76:	5b                   	pop    %ebx
 8049f77:	c3                   	ret    
