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

      DOUBLE PRECISION x,h,g,k,s

       x=U(1)
       h=PAR(1)
       g=PAR(2)
       k=PAR(3)
       s=PAR(4)

       F(1) = g*x*(1-x/k) - h*x**2/(s**2 + x**2)
       
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
