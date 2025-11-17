module extFGH

  interface ! interface for C++ subroutines
  
    subroutine myfgh ( need, nx, x, nf, nh, iusr, rusr, f, g, h ) bind ( c )
      use iso_c_binding
      integer (c_int) :: need(*),nx,nf,nh,iusr(*)
      real (c_double) :: x(*),rusr(*),f(*),g(*),h(*)       
    end subroutine myfgh
	
	subroutine mymix ( nx, ptcl1, ptcl2, alpha, iusr, rusr ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx,iusr(*)
      real (c_double) :: ptcl1(*),ptcl2(*),rusr(*),alpha(*)      
    end subroutine mymix
	
	subroutine toxhat ( ptcl, x, nx, rusr ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx
      real (c_double) :: x(*),rusr(*),ptcl(*)      
    end subroutine toxhat
	
	subroutine myfnn ( nx, x, fnn ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx
      real (c_double) :: x(*),fnn(*)      
    end subroutine myfnn
	
	subroutine fromxhat ( x, ptcl, nx, rusr ) bind ( c )
      use iso_c_binding
      integer (c_int) :: nx
      real (c_double) :: x(*),rusr(*),ptcl(*)      
    end subroutine fromxhat
	
  end interface      

end module extFGH

program main

  use extFGH

  implicit none  
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     SIMULATION PARAMETERS     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  integer, parameter :: nTest = 250 ! when storage retrieval accuracy is tested, errors are computed every nTest time steps
  integer, parameter :: mode = 3
  ! mode determines the behavior of the code
  ! mode == 1 does no storage retrieval, and outputs a low number (1,000,000) of PaSR samples
  ! mode == 2 performs standard preconditioned ISAT
  ! mode == 3 performs a standalone ISAT simulation
  ! mode == 4 performs a standalone MLP simulation  
  double precision, parameter :: errtol = 64e-5 					! ISAT error tolerance  
  integer, parameter :: nx = 164 									! number of variables (species + 1 for temperature)  
  integer, parameter :: nPtcl = 100, nSteps = 10 					! number of particles and steps in the PaSR reactor
  integer, parameter :: nC12H26 = 99, nO2 = 5, nN2 = 8
  integer, parameter ::nCO2 = 12, nH2O = 7 							! locations of the H2, O2 and N2 mass fractions in the ptcl(:,:) array. ptcl(1,:) is temperature
  ! and the species are arranged in the same order as they are in the chemical mechanism
  double precision, parameter :: mFrac = 0.05, pFrac = 0.1      	! mFrac is the fuel to oxidizer mass fraction, pFrac is the pilot to reactant mass fraction
  double precision, parameter :: flowThroughTime = 2.5e-4       	! expected time for a particle to be replaced in the PaSR reactor
  double precision, parameter :: mixTime = 2.5e-5               	! characteristic time scale of particle mixing
  double precision, parameter :: dt = 5e-7                      	! length of one PaSR time step
  double precision, parameter :: Tcold = 1000.0, Trange = 1000.0	! base temperature and temperature range for normalization
  double precision, parameter :: atolODE = 1e-8, rtolODE = 1e-8		! absolute and relative error control parameters for ODE integration
  double precision, parameter :: pressure = 101325.0                ! PaSR pressure (in Pascals)
  double precision, parameter :: stomby = 3000.0                    ! maximum size of ISAT table, in MB
  integer, parameter		  :: nPtclOut = 10                     	! write the post-reaction particles every "nPtclOut" steps
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!       END SIMULATION PARAMETERS     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!       VARIABLE ALLOCATIONS          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  integer :: nf=nx,nh=1 											! dimensions of x and f (h is for an ISAT functionality which is not used here)
  integer :: ns=nx-1, info(100), idd, counter, i1, i2
  ! ns is the number of species, nPtcl is the number of particles in the reactor, nSteps is the number of time steps
  double precision, pointer    :: x(:), ptcl1(:), ptcl2(:), rusr(:), f(:), fisat(:), fisatsum(:), fnn(:), g(:,:), h(:), ptcls(:,:), alpha(:)
  double precision, pointer    :: pilot(:), fuel(:), oxidizer(:)
  double precision :: rinfo(70), stats(100)
  integer :: start, finish, rate
  double precision :: error(164)
  ! mFrac controls the mass fraction of the H2 stream in the PaSR, pFrac controls the mass fraction of 
  ! equilibrium combustion products in the PaSR, Tfuel and Tox are the inflow temperatures of the fuel and
  ! oxidizer streams  
  double precision :: rr, rr2, dummy ! work variables used for random outcomes in particle replacement and mixing
  integer, pointer :: need(:),need2(:),iusr(:) ! ISAT inputs
  integer :: ii,jj,kk 
  allocate(need(3),need2(3),iusr(3),x(nx),ptcl1(nx),ptcl2(nx),rusr(2*nx+5),f(nx),fisat(nx),fisatsum(nx),fnn(nx),g(nx,nx),h(1),ptcls(nx+1,nPtcl),alpha(1))
  allocate(pilot(nx),fuel(nx),oxidizer(nx))  
  iusr(1) = mode ! used to pass the code's behavior to the C++ routines (determines whether f^{MLP} is subtracted from f(x))
  need = (/ 1, 0, 0 /) ! if we only need f(x) from ISAT
  need2 = (/ 1, 1, 0 /) ! if we need both f(x) and its Jacobian
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       END VARIABLE ALLOCATIONS        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  !!!!!!!!!!!!!!!!!!!!!       DEFINITION OF NORMALIZATION, FUEL AND PILOT STREAMS      !!!!!!!!!!!!!!!!!!!!!   
  
  rusr(1) = Tcold
  rusr(nx+1) = Trange ! temperature normalization
  
  open(unit=7, file="normalize.dat",status="old",action="read")
  open(unit=8, file="pilot.dat",status="old",action="read")
  open(unit=9, file="fuel.dat",status="old",action="read")
  open(unit=10, file="oxidizer.dat",status="old",action="read")
  open(unit=11, file="output.dat",status="replace",action="write")
  open(unit=12, file="verbose.dat",status="replace",action="write")
  
  do ii = 2,nx
	read(7,*) rusr(ii)
	rusr(ii) = max(rusr(ii),1e-3)
  end do
  
  do ii = 1,nx
	read(8,*) pilot(ii)
	read(9,*) fuel(ii)
	read(10,*) oxidizer(ii)
  end do
  
  do ii = 1,nx-1
	rusr(nx+1+ii) = 1.0 ! these are not used at present
  end do
  
  !!!!!!!!!!!!!!!!!!!!!     END DEFINITION OF NORMALIZATION, FUEL AND PILOT STREAMS      !!!!!!!!!!!!!!!!!!!!!  


  !!!!!!!!!!!!!!!!!!!!!                INITIALIZATION OF ISAT PARAMETERS                !!!!!!!!!!!!!!!!!!!!!!  
  
  rusr(2*nx+1) = atolODE 					! absolute error tolerance for the Cantera ODE integration
  rusr(2*nx+2) = rtolODE 					! relative error tolerance for the Cantera ODE integration
  rusr(2*nx+3) = dt 						! dt
  rusr(2*nx+4) = 1e-6 						! dx ( not used )
  rusr(2*nx+5) = pressure 					! reactor pressure, in Pa  
  info  = 0
  rinfo = 0
  info(2)  = 0								! if_g (no piecewise constant approximation of the Jacobian, see ISAT documentation)
  info(12) = 2								! isat op (output ISAT performance data)
  info(28) = 0   							! idites (accuracy test every idites-th ISAT retrieval, not used here)
  info(25) = 1   							! no growing
  !info(40) = 10   							! degenerate leaf after 10 grows
  rinfo(1) = errtol 						! absolute error tolerance which ISAT aims for
  rinfo(2) = errtol 						! relative error tolerance which ISAT aims for
  rinfo(3) = 1e2 							! tolerance for EOA growth (not essential here), see ISAT documentation
  rinfo(7) = 0.1  							! limit number of grows
  rinfo(8) = stomby  						!  stomby - size of ISAT table in MB  
  error = 0.0 								! initialize the error   
  call random_seed(put=(/10,11/)) 			! set a random seed for the PaSR reactor
  ! the random seed for ISAT is separate and not set here
  
  !!!!!!!!!!!!!!!!!!!!!              END INITIALIZATION OF ISAT PARAMETERS              !!!!!!!!!!!!!!!!!!!!!!
  
  
  !!!!!!!!!!!!!!!!!!!!!!                 INITIAL STATE OF THE PASR                     !!!!!!!!!!!!!!!!!!!!!!!
  
  counter = 0 								! initialize the particle counter
  
  do ii = 1,nPtcl 							! begin initialization of particles in the reactor
    ! the particles can come from either the fuel stream, oxidizer stream or equilibrium products stream  
  
	call random_number(rr)
	call random_number(rr2) 
	
	if ( rr2 < pFrac ) then 				! this occurs with probability pFrac	
		ptcls(1:nx,ii) = pilot 				! initialize a pilor particle	
	else	
		if ( rr < mFrac ) then 				! this occurs with probability (1-pFrac)*mFrac
			ptcls(1:nx,ii) = fuel			! initialize a fuel particle
		else
			ptcls(1:nx,ii) = oxidizer 		! initialize an oxidizer particle
		end if  	
	end if	
	counter = counter + 1					! increment the particle counter
	ptcls(nx+1,ii) = counter				! add the particle number 
  end do 									! end initialization
  
  call toxhat( ptcls(1:nx,1), x, nx, rusr )
  call myfgh( need, nx, x, nf, nh, iusr, rusr, f, g, h ) ! normalize one particle and call f(x) on it, so that 
  ! the chemistry is initialized
  
  !!!!!!!!!!!!!!!!!!!!!!                 END INITIAL STATE OF THE PASR                     !!!!!!!!!!!!!!!!!!!!!!!
  
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!           PASR TIME LOOP               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  
  call system_clock(count = start, count_rate = rate) 		! start the timer
  
  do jj = 1,nSteps 											! begin time step
  
	print *, jj											! output time step number
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        REACTION       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
	do ii = 1,nPtcl 										! perform chemical kinetics on all particles
		call toxhat( ptcls(1:nx,ii), x, nx, rusr ) 			! normalize the particle
		
		if ( mode.eq.1 ) then
			write(11,*) x 
				call myfgh( need, nx, x, nf, nh, iusr, rusr, f, g, h )
			write(11,*) f
		end if 												! output x and f(x) for the particle,
		! if the code is collecting data forf^{MLP} training
		
		if ( (mode.eq.2).or.(mode.eq.3) ) then
			call isatab( idd,0,nx,x,nf,nh,nh, myfgh, iusr,rusr, info, rinfo, fisat ,g,h,stats)
		end if 												! call ISAT, if needed
			
		if ( (mode.eq.2).or.(mode.eq.4) ) then
			call myfnn( nx, x, fnn )
		end if 												! call f^{MLP}, if needed
		
		if ( mode.gt.1 ) then 								! if the code does storage retrieval
			if (modulo(jj,nTest).eq.0) then                 ! perform test of the storage retrieval accuracy
			
				call myfgh( need, nx, x, nf, nh, iusr, rusr, f, g, h )
				! every nTest-th time step, perform a DE on all particles and calculate the storage retrieval error
			
				do kk = 1,nx
					if ( mode.lt.4 ) then
						error(kk) = error(kk) + (f(kk)-fisat(kk))**2 
						! ISAT error, when ISAT is used for storage retrieval
					elseif ( mode.eq.4 ) then
						error(kk) = error(kk) + (f(kk)-fnn(kk))**2 
						! f^{MLP} error, when f^{MLP} is used for storage retrieval
					end if
						
				end do
			
			end if
		end if
		
		if ( mode.eq.1 ) then 								! add the appropriate time increment to the particle's composition
			x = x + f 										! DE increment
		elseif ( mode.eq.2 ) then
			x = x + fisat + fnn 							! preconditioned ISAT increment
		elseif ( mode.eq.3 ) then 
			x = x + fisat 									! standalone ISAT increment
		elseif ( mode.eq.4 ) then
			x = x + fnn 									! MLP increment
		end if
		
		call fromxhat( x, ptcls(1:nx,ii), nx, rusr )		
		! convert the particles back to dimensional form
		
	end do ! end of reaction loop
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        END  REACTION       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	if ( modulo(jj,nPtclOut).eq.0 ) then
		write(12,*) ptcls
	end if                        							! output particles after the reaction time step, every nPtclOut-th step
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     INFLOW AND OUTFLOW    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	do ii = 1,nPtcl ! start the inflow/outflow loop
	
		call random_number(rr)
		if ( rr < dt/flowThroughTime ) then ! a particle is replaced with probability dt/flowThroughTime
		
			call random_number(rr)		
			call random_number(rr2)
		
				if ( rr2 < pFrac ) then ! same probabilities and compositions of the three streams
					! as when the particles are initialized
	
					ptcls(1:nx,ii) = pilot
	
				else
	
					if ( rr < mFrac ) then
						ptcls(1:nx,ii) = fuel
					else
						ptcls(1:nx,ii) = oxidizer
					end if  
				
				end if
			
			counter = counter + 1
			ptcls(nx+1,ii) = counter	
			
		end if
	end do
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     END INFLOW AND OUTFLOW    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        PARTICLE MIXING       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	do ii = 1,nPtcl ! begin particle mixing
		call random_number(rr)
		i1 = int(ceiling(rr*real(nPtcl)))
		if (i1 .eq. 0) i1 = 1
		call random_number(rr)
		i2 = int(ceiling(rr*real(nPtcl)))
		if (i2 .eq. 0) i2 = 1 ! pick two particles at random
		
		call random_number(rr)
		
		ptcl1 = ptcls(1:nx,i1)
		ptcl2 = ptcls(1:nx,i2)
		
		call mymix( nx, ptcl1, ptcl2, rr*dt/mixTime, iusr, rusr )
		
		ptcls(1:nx,i1) = ptcl1
		ptcls(1:nx,i2) = ptcl2
		
	end do
	
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       END PARTICLE MIXING       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  end do ! end time step
  
  call system_clock(count = finish)
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!           END PASR TIME LOOP               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!             OUTPUT RESULTS               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  print *, '("Time = ",e10.3," seconds.")',real(finish-start)/real(rate)
  
  print *, 'Mean temp: ', sum(ptcls(1,:)/nPtcl)
  
  print *, 'Mean H: ', sum(ptcls(2,:)/nPtcl)
  
  if ( mode.gt.1 ) then
  
	  print *, 'RMS error: ', sqrt(sum(error)/(nPtcl*nSteps/nTest))
	  
	  if ( (mode.eq.2).or.(mode.eq.3) ) then
	  
		  call isatab( idd,6,nx,x,nf,nh,nh, myfgh, iusr,rusr, info, rinfo, fisat ,g,h,stats)
		  
		  print *, 'Number of queries: ', stats(1)
		  
		  print *, 'Number of leaves: ', stats(12)
		  
		  print *, 'Number of grows: ', stats(4)
		  
		  print *, 'Number of adds: ', stats(5)
		  
		  print *, 'Fraction of retrieves: ', real(stats(2) + stats(3))/real(stats(1))
		  
		  print *, 'Number of unresolved: ', stats(8)
		  
		  print *, 'Number of DEs: ', stats(7)
		  
	  end if
	  
  end if 
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!          END OUTPUT RESULTS       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
end program main
