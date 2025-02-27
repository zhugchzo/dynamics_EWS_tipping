!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   model: Fortran script to specify model for AUTO
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,b,&
                       a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,&
                       a13,a14,a15,a16,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,&
                       b11,b12,b13,b14,b15,b16,c1,c2,c3,c4,c5,c6,c7,c8,&  
                       c9,c10,c11,c12,c13,c14,c15,c16,d1,d2,d3,d4,d5,d6,&  
                       d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,e1,e2,e3,e4,&  
                       e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,f1,f2,&  
                       f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,&  
                       g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,&  
                       g15,g16,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,&  
                       h13,h14,h15,h16,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,&  
                       i11,i12,i13,i14,i15,i16,j1,j2,j3,j4,j5,j6,j7,j8,&  
                       j9,j10,j11,j12,j13,j14,j15,j16

       x1=U(1)
       x2=U(2)
       x3=U(3)
       x4=U(4)
       x5=U(5)
       x6=U(6)
       x7=U(7)
       x8=U(8)
       x9=U(9)
       x10=U(10)
       b=PAR(1)
       a1=PAR(2)
       a2=PAR(3)
       a3=PAR(4)
       a4=PAR(5)
       a5=PAR(6)
       a6=PAR(7)
       a7=PAR(8)
       a8=PAR(9)
       a9=PAR(10)
!------omit PAR(11) since saved for AUTO computing period
       a10=PAR(12)
       a11=PAR(13)
       a12=PAR(14)
       a13=PAR(15)
       a14=PAR(16)
       a15=PAR(17)
       a16=PAR(18)
       b1=PAR(19)
       b2=PAR(20)
       b3=PAR(21)
       b4=PAR(22)
       b5=PAR(23)
       b6=PAR(24)
       b7=PAR(25)
       b8=PAR(26)
       b9 = PAR(27)
       b10 = PAR(28)
       b11 = PAR(29)
       b12 = PAR(30)
       b13 = PAR(31)
       b14 = PAR(32)
       b15 = PAR(33)
       b16 = PAR(34)
       c1 = PAR(35)
       c2 = PAR(36)
       c3 = PAR(37)
       c4 = PAR(38)
       c5 = PAR(39)
       c6 = PAR(40)
       c7 = PAR(41)
       c8 = PAR(42)
       c9 = PAR(43)
       c10 = PAR(44)
       c11 = PAR(45)
       c12 = PAR(46)
       c13 = PAR(47)
       c14 = PAR(48)
       c15 = PAR(49)
       c16 = PAR(50)
       d1 = PAR(51)
       d2 = PAR(52)
       d3 = PAR(53)
       d4 = PAR(54)
       d5 = PAR(55)
       d6 = PAR(56)
       d7 = PAR(57)
       d8 = PAR(58)
       d9 = PAR(59)
       d10 = PAR(60)
       d11 = PAR(61)
       d12 = PAR(62)
       d13 = PAR(63)
       d14 = PAR(64)
       d15 = PAR(65)
       d16 = PAR(66)
       e1 = PAR(67)
       e2 = PAR(68)
       e3 = PAR(69)
       e4 = PAR(70)
       e5 = PAR(71)
       e6 = PAR(72)
       e7 = PAR(73)
       e8 = PAR(74)
       e9 = PAR(75)
       e10 = PAR(76)
       e11 = PAR(77)
       e12 = PAR(78)
       e13 = PAR(79)
       e14 = PAR(80)
       e15 = PAR(81)
       e16 = PAR(82)
       f1 = PAR(83)
       f2 = PAR(84)
       f3 = PAR(85)
       f4 = PAR(86)
       f5 = PAR(87)
       f6 = PAR(88)
       f7 = PAR(89)
       f8 = PAR(90)
       f9 = PAR(91)
       f10 = PAR(92)
       f11 = PAR(93)
       f12 = PAR(94)
       f13 = PAR(95)
       f14 = PAR(96)
       f15 = PAR(97)
       f16 = PAR(98)
       g1 = PAR(99)
       g2 = PAR(100)
       g3 = PAR(101)
       g4 = PAR(102)
       g5 = PAR(103)
       g6 = PAR(104)
       g7 = PAR(105)
       g8 = PAR(106)
       g9 = PAR(107)
       g10 = PAR(108)
       g11 = PAR(109)
       g12 = PAR(110)
       g13 = PAR(111)
       g14 = PAR(112)
       g15 = PAR(113)
       g16 = PAR(114)
       h1 = PAR(115)
       h2 = PAR(116)
       h3 = PAR(117)
       h4 = PAR(118)
       h5 = PAR(119)
       h6 = PAR(120)
       h7 = PAR(121)
       h8 = PAR(122)
       h9 = PAR(123)
       h10 = PAR(124)
       h11 = PAR(125)
       h12 = PAR(126)
       h13 = PAR(127)
       h14 = PAR(128)
       h15 = PAR(129)
       h16 = PAR(130)
       i1 = PAR(131)
       i2 = PAR(132)
       i3 = PAR(133)
       i4 = PAR(134)
       i5 = PAR(135)
       i6 = PAR(136)
       i7 = PAR(137)
       i8 = PAR(138)
       i9 = PAR(139)
       i10 = PAR(140)
       i11 = PAR(141)
       i12 = PAR(142)
       i13 = PAR(143)
       i14 = PAR(144)
       i15 = PAR(145)
       i16 = PAR(146)
       j1 = PAR(147)
       j2 = PAR(148)
       j3 = PAR(149)
       j4 = PAR(150)
       j5 = PAR(151)
       j6 = PAR(152)
       j7 = PAR(153)
       j8 = PAR(154)
       j9 = PAR(155)
       j10 = PAR(156)
       j11 = PAR(157)
       j12 = PAR(158)
       j13 = PAR(159)
       j14 = PAR(160)
       j15 = PAR(161)
       j16 = PAR(162)


       F(1) = a1 + a2*x1 + a3*b + a4*x1**2 + a5*b*x1 + a6*x1**3 + &
       a7*b*x1**2 + a8*x2 + a9*x3 + a10*x4 + a11*x5 + &
       a12*x6 + a13*x7 + a14*x8 + a15*x9 + a16*x10

       F(2) = b1 + b2*x2 + b3*b + b4*x2**2 + b5*b*x2 + b6*x2**3 + &
       b7*b*x2**2 + b8*x1 + b9*x3 + b10*x4 + b11*x5 + &
       b12*x6 + b13*x7 + b14*x8 + b15*x9 + b16*x10

       F(3) = c1 + c2*x3 + c3*b + c4*x3**2 + c5*b*x3 + c6*x3**3 + &
       c7*b*x3**2 + c8*x1 + c9*x2 + c10*x4 + c11*x5 + &
       c12*x6 + c13*x7 + c14*x8 + c15*x9 + c16*x10

       F(4) = d1 + d2*x4 + d3*b + d4*x4**2 + d5*b*x4 + d6*x4**3 + &
       d7*b*x4**2 + d8*x1 + d9*x2 + d10*x3 + d11*x5 + &
       d12*x6 + d13*x7 + d14*x8 + d15*x9 + d16*x10

       F(5) = e1 + e2*x5 + e3*b + e4*x5**2 + e5*b*x5 + e6*x5**3 + &
       e7*b*x5**2 + e8*x1 + e9*x2 + e10*x3 + e11*x4 + &
       e12*x6 + e13*x7 + e14*x8 + e15*x9 + e16*x10

       F(6) = f1 + f2*x6 + f3*b + f4*x6**2 + f5*b*x6 + f6*x6**3 + &
       f7*b*x6**2 + f8*x1 + f9*x2 + f10*x3 + f11*x4 + &
       f12*x5 + f13*x7 + f14*x8 + f15*x9 + f16*x10

       F(7) = g1 + g2*x7 + g3*b + g4*x7**2 + g5*b*x7 + g6*x7**3 + &
       g7*b*x7**2 + g8*x1 + g9*x2 + g10*x3 + g11*x4 + &
       g12*x5 + g13*x6 + g14*x8 + g15*x9 + g16*x10

       F(8) = h1 + h2*x8 + h3*b + h4*x8**2 + h5*b*x8 + h6*x8**3 + &
       h7*b*x8**2 + h8*x1 + h9*x2 + h10*x3 + h11*x4 + &
       h12*x5 + h13*x6 + h14*x7 + h15*x9 + h16*x10

       F(9) = i1 + i2*x9 + i3*b + i4*x9**2 + i5*b*x9 + i6*x9**3 + &
       i7*b*x9**2 + i8*x1 + i9*x2 + i10*x3 + i11*x4 + &
       i12*x5 + i13*x6 + i14*x7 + i15*x8 + i16*x10

       F(10) = j1 + j2*x10 + j3*b + j4*x10**2 + j5*b*x10 + j6*x10**3 + &
       j7*b*x10**2 + j8*x1 + j9*x2 + j10*x3 + j11*x4 + &
       j12*x5 + j13*x6 + j14*x7 + j15*x8 + j16*x9

      END SUBROUTINE FUNC


      SUBROUTINE STPNT(NDIM,U,PAR,T)

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
