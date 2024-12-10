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

      DOUBLE PRECISION x,y,b,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,&
                                        a12,a13,a14,a15,a16,a17,a18,a19,a20,&
                                        b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,&
                                        b12,b13,b14,b15,b16,b17,b18,b19,b20

       x=U(1)
       y=U(2)
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
       a17=PAR(19)
       a18=PAR(20)
       a19=PAR(21)
       a20=PAR(22)
       b1=PAR(23)
       b2=PAR(24)
       b3=PAR(25)
       b4=PAR(26)
       b5=PAR(27)
       b6=PAR(28)
       b7=PAR(29)
       b8=PAR(30)
       b9=PAR(31)
       b10=PAR(32)
       b11=PAR(33)
       b12=PAR(34)
       b13=PAR(35)
       b14=PAR(36)
       b15=PAR(37)
       b16=PAR(38)
       b17=PAR(39)
       b18=PAR(40)
       b19=PAR(41)
       b20=PAR(42)

       F(1) = a1 + a2*x + a3*y + a4*b + a5*x**2 + a6*x*y + a7*b*x + &
       a8*y**2 + a9*b*y + a10*b**2 + a11*x**3 + a12*x**2*y + &
       a13*b*x**2 + a14*x*y**2 + a15*b*x*y + a16*b**2*x + &
       a17*y**3 + a18*b*y**2 + a19*b**2*y + a20*b**3
       F(2) = b1 + b2*x + b3*y + b4*b + b5*x**2 + b6*x*y + b7*b*x + &
       b8*y**2 + b9*b*y + b10*b**2 + b11*x**3 + b12*x**2*y + &
       b13*b*x**2 + b14*x*y**2 + b15*b*x*y + b16*b**2*x + &
       b17*y**3 + b18*b*y**2 + b19*b**2*y + b20*b**3
       
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
