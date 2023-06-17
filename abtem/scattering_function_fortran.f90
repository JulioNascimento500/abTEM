subroutine magnons_function(Qgrid,Egrid,plotValues,N,NMax,T,EigVecR,EigVals,exp_sum_j,omegaX,delta,Vplus,Vminus,alpha,beta)

  use constants                      !Minimal useage where practical 
  use omp_lib 

  implicit none

  !### Integer ###
  integer             :: i,j,k,stat
  integer,intent(in)  :: N,NMax,alpha,beta
  integer,intent(in)  :: Qgrid,Egrid

  !### Arrays ##
  complex(kind=8), dimension(Egrid,Qgrid)      , intent(inout)  :: plotValues(Egrid,Qgrid)
  complex(kind=8), dimension(NMax,NMax,Qgrid)  , intent(in)     :: EigVecR(NMax,NMax,Qgrid) 
  complex(kind=8), dimension(Qgrid,NMax)       , intent(in)     :: EigVals(Qgrid,NMax)
  complex(kind=8), dimension(Egrid)            , intent(in)     :: omegaX(Egrid)

  complex(kind=8), dimension(3,N)              , intent(in)     :: Vplus(3,N)
  complex(kind=8), dimension(3,N)              , intent(in)     :: Vminus(3,N) 
  
  complex(kind=8), dimension(Qgrid,N)          , intent(in)     :: exp_sum_j(Qgrid,N)

  complex(kind=8), dimension(N)                                 :: XR(N)

  complex(kind=8), dimension(N)                                 :: omegaN(N)
  

  complex(kind=8)                                           ::  BD, BDning
  !### Reals ###
  complex(kind=8)                                             ::  total

  real(kind=dp), intent(in)                                 :: T
  real(kind=dp), intent(in)                                 :: delta

  

  
  !#### f2py intent block ####
  !f2py intent(in)       :: N
  !f2py intent(in)       :: Egrid
  !f2py intent(in)       :: Qgrid
  !f2py intent(in)       :: T
  !f2py intent(in)       :: N
  !f2py intent(in)       :: NMax
  !f2py intent(in)       :: delta


  !f2py intent(inout)             :: plotValues
  !f2py depend(Egrid,Qgrid)       :: plotValues

  !f2py intent(in)                 :: EigVecR
  !f2py depend(NMax,NMax,Qgrid)    :: EigVecR

  !f2py intent(in)            :: EigVals
  !f2py depend(NMax,Qgrid)    :: EigVals

  !f2py intent(in)            :: exp_sum_j
  !f2py depend(Qgrid,N)       :: exp_sum_j

  !f2py intent(in)            :: omegaX
  !f2py depend(Egrid)         :: omegaX

  !f2py intent(in)            :: Vplus
  !f2py depend(N)             :: Vplus

  !f2py intent(in)            :: Vminus
  !f2py depend(N)             :: Vminus  


  print *, 'threads ', omp_get_num_threads()

  ! parallel do default(none) shared(EigVecR,N,NMax,exp_sum_j,egrid,eigvals,omegax,T,delta,plotValues,Vplus,Vminus) &
  ! private(j,total,k,omegaN,XR,BD,BDning)

  do i=1,Qgrid
    do j=1,Egrid
      total=0.0_dp
      do k=1,N

        XR = Vminus(alpha,k)*EigVecR(1:N,k+N,i) + Vplus(beta,k)*EigVecR(N+1:NMax,k+N,i)
        
        BD=boson_dist(omegaX(j),EigVals(i,k+N),T) + 1.0_dp

        BDning=broadening(omegaX(j),EigVals(i,k+N),delta)
        
        !print *, 'BD',BD
        !print *, 'BDning',BDning

        if (BDning .ne. complex(0.0,0.0)) then

          total = total + (sum(outer_product(conjg(exp_sum_j(i,:)*XR),exp_sum_j(i,:)*XR)) * BD * BDning)

        end if


        !total = total + ((conjg(dot_product(exp_sum_j(i,:),XR))*dot_product(exp_sum_j(i,:),XR))) * BD * BDning
        ! if (i==1 .and. j==1) then
        !   ! print *, 'EigVals', EigVals(i,k+N)
        !   ! print *, 'omegaX', omegaX(j)
        !   ! print *, 'BD', BD
        !   ! print *, 'BDning', BDning
        ! end if      
      end do
      plotValues(j,i)=total
    end do
  end do
 
  ! end parallel do

contains

  function outer_product(A,B) result(AB)
    complex(kind=8), intent(in) :: A(:),B(:)
    complex(kind=8), allocatable :: AB(:,:)
    integer :: nA,nB
    nA=size(A)
    nB=size(B)
    allocate(AB(nA,nB))
    AB = spread(source = A, dim = 2, ncopies = nB) * &
        spread(source = B, dim = 1, ncopies = nA)
  end function outer_product

  real(kind=dp) function boson_dist(omega,mu,Temp)
  !
  !################
  ! This function calculates 
  ! The boson distribution
  !
  ! In : Given omega (omega)
  ! In : Particular Eigenvalue of the system (omegaN)
  ! In : Temperature (T)
  ! In : Size of array X (N)
  !
  ! Out: Value for the distribution in omega
  !
  !################    
  complex(kind=8)               :: omega,mu
  real(kind=dp)  , intent(in)   :: Temp
  real(kind=dp)                 :: Kb
 
    kb=8.617333e-2_dp !meV/K

    boson_dist = 1/(exp((omega)/(Kb*Temp)) - 1.0_dp)
    if (isnan(boson_dist)) boson_dist=complex(0.0_dp,0.0_dp)
    !if (real(omega) < real(mu)) boson_dist=complex(0.0_dp,0.0_dp)
  end function boson_dist

  real(kind=dp) function broadening(omega,omegaN,deltaV)
  !
  !################
  ! This function calculates 
  ! The boson distribution
  !
  ! In : Given omega (omega)
  ! In : Particular Eigenvalue of the system (omegaN)
  ! In : Temperature (T)
  ! 
  !
  ! Out: Value for the broadening in omega
  !
  !################    
  use constants
  
  complex(kind=8), intent(in)      :: omega,omegaN
  real(kind=dp),   intent(in)      :: deltaV

    !print *, (omega-omegaN)
    !print *, (2.0_dp*(delta**2.0_dp))
    !delta=0.2_dp*abs(Jij)*0.5_dp
    broadening = (1.0_dp/sqrt(2.0_dp*pi*(deltaV**2.0_dp)))*exp(-(((omega-omegaN)*conjg(omega-omegaN))/(2.0_dp*(deltaV**2.0_dp))))

  end function broadening
end