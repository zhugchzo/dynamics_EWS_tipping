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

      DOUBLE PRECISION x,y,k,r,a,b,e,d

       x=U(1)
       y=U(2)
       k=PAR(1)
       r=PAR(2)
       a=PAR(3)
       b=PAR(4)
       e=PAR(5)       
       d=PAR(6)

       F(1) = r*x*(1-x/k) - (a*x*y)/(b+x)
       F(2) = e*a*x*y/(b+x) - d*y
       
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
