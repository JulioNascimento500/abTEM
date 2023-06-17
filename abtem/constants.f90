!=============================================================================!
!                              Constants                                      !
!=============================================================================!
! This module stores all of the globally accessible constants used by the     !
! code.                                                                       !
!=============================================================================!

module constants

  ! Impose strong typing

  implicit none

  ! Everything is private ...

  private 

  !... unless exposed here.

  ! Define the kind parameter for double precison

  integer, parameter, public :: dp=selected_real_kind(15,300)

  ! Define the square root of -1.

  complex(kind=dp), parameter, public :: cmplx_i=(0.0_dp,1.0_dp)

  ! Define complex 1.0.

  complex(kind=dp), parameter, public :: cmplx_1=(1.0_dp,0.0_dp)

  ! Define complex 0.0.

  complex(kind=dp), parameter, public :: cmplx_0=(0.0_dp,0.0_dp)


  ! Define pi

  real(kind=dp), parameter, public :: pi=3.141592653589793238462643383279502884197_dp

  ! Define 2*pi

  real(kind=dp), parameter, public :: two_pi=2.0_dp*pi

  ! Define sqrt(2*pi) (Can't calculate at compile time)

  real(kind=dp), parameter, public :: sqrt_two_pi=2.506628274631000241612355239340104162693_dp

end module constants
