PROGRAM lingk
!-------------------------------------------------------------------------------
!
!    Simulation of linear gyrokinetic equation in a local flux-tube geometry 
!
!      Gyrocenter distribution function  fk(zz,vl,mu,is)
!               Electrostatic potential  pk(zz)
!             Parallel vector potential  ak(zz)
!
!      Modified gyrocenter distribution function hk(zz,vl,mu,is) 
!      for time advance, dhk/dt = L(hk).
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : set_geometry
  use fileio, only : fileio_open, fileio_close
  use clock, only : clock_init, clock_sta, clock_end, elt
  implicit none

  real(kind=DP) :: time = 0._DP
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: fk, hk
  complex(kind=DP), dimension(-nz:nz-1)                    :: pk, ak

  real(kind=DP) :: time_out
  logical :: stop_signal = .false.
  integer :: itime

    call set_param

    call fileio_open

    call clock_init
                                                            call clock_sta(1)

                                                            call clock_sta(2)
!- initialize -!
    itime = 0
    call set_geometry
    call initial_condition(time, fk, hk, pk, ak)
!--------------!
                                                            call clock_end(2)
                                                            call clock_sta(4)
    write(*,*) 'time = ', time
    call out_fkinzv(time, fk, nm/4)
    !call out_fkinzv(time, fk, nm/2)
    !call out_fkinzv(time, fk, 3*nm/4)
    call out_mominzt(time, fk, pk, ak)
    call out_linfreq(time, pk, stop_signal)
    !call out_fkinzvm_paraview(time, fk)
    time_out = time + dt_out - eps
                                                            call clock_end(4)

!= time-evolution =!
    do itime = 0, litime

                                                            call clock_sta(3)
    call rkg4(time, fk, hk, pk, ak)
    time = time + dt
                                                            call clock_end(3)

                                                            call clock_sta(4)
    !- output variables -!
    if (time > time_out) then
      write(*,*) 'time = ', time
      call out_fkinzv(time, fk, nm/4)
      !call out_fkinzv(time, fk, nm/2)
      !call out_fkinzv(time, fk, 3*nm/4)
      call out_mominzt(time, fk, pk, ak)
      call out_linfreq(time, pk, stop_signal)
      !call out_fkinzvm_paraview(time, fk)
      time_out = time_out + dt_out
    end if
    !--------------------!
                                                            call clock_end(4)

    if (elt(2) + elt(3) + elt(4) > elt_limit) then
      write( *, * ) 'Elapsed time limit is close.'
      exit
    end if
    if (time > time_limit) then
      write( *, * ) 'Simulation time limit is close.'
      exit
    end if
    if (stop_signal) then
      write( *, * ) 'Stop signal is detected.'
      exit
    end if

    end do
!==================!

                                                            call clock_end(1)

    write( *, * ) '### Elapsed time ###'
    write( *, * ) '# Time steps = ', itime
    write( *, * ) '#'
    write( *, * ) '#      Total = ', elt(1)
    write( *, * ) '#       Init = ', elt(2)
    write( *, * ) '#        RKG = ', elt(3)
    write( *, * ) '#     Output = ', elt(4)
    write( *, * ) '#      Other = ', elt(1) - (elt(2) + elt(3) + elt(4))
    call fileio_close
    write( *, * ) 'End program.'


 CONTAINS


!SUBROUTINE restart(time, fk)
!!-------------------------------------------------------------------------------
!!
!!    Read restart binary data
!!
!!-------------------------------------------------------------------------------
!  use parameters
!  use fileio, only : ifkk
!  implicit none
!
!  real(kind=DP), intent(out) :: time
!  complex(kind=DP), dimension(-nkx:nkx,0:nky,0:nm-1), intent(out) :: fk
!  integer :: iost
!
!
!    do
!      read( ifkk, iostat=iost ) time, fk
!      if (iost == -1) then
!        exit
!      else if (iost /= 0) then
!        write(*,*) "Input file error! iostat = ", iost
!        stop
!      end if
!    end do
!
!
!END SUBROUTINE restart


SUBROUTINE out_fkinzv(time, fk, im)
!-------------------------------------------------------------------------------
!
!    Output variables
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : zz, vl
  use fileio, only : ofzv
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(in) :: fk
  integer, intent(in) :: im

  character(len=8) :: ctime
  character(len=4) :: cim
  integer :: iz, iv, is

    write(ctime,'(i8.8)') int(time / dt_out)
    write(cim,'(i4.4)') im

!--- ascii data for gnuplot ---
!   open( ofzv, file="./data/fkinzv_im"//cim//"_t"//ctime//".dat" )
!     write( ofzv, * ) "# Time = ", time
!     write( ofzv, "(99a17)" ) "#              zz", "vl", "Re[fk]", "Im[fk]"
!     do iv = 1, 2*nv
!       do iz = -nz, nz-1
!         write( ofzv, "(99e17.7e3)" ) zz(iz), vl(iv),  &
!           (real(fk(iz,iv,im,is),kind=DP), aimag(fk(iz,iv,im,is)), is=0,ns-1)
!       end do
!       write( ofzv, * )
!     end do
!   close( ofzv )
!--- binary data for gnuplot ---
    open( unit=ofzv, file="./data/fkinzv_im"//cim//"_t"//ctime//".dat",  &
          status="replace", action="write", form="unformatted", access="stream" )
      do iv = 1, 2*nv
        do iz = -nz, nz-1
          write( ofzv ) zz(iz), vl(iv),  &
            (real(fk(iz,iv,im,is),kind=DP), aimag(fk(iz,iv,im,is)), is=0,ns-1)
        end do
      end do
    close( ofzv )
!-------------------------------


END SUBROUTINE out_fkinzv


SUBROUTINE out_mominzt(time, fk, pk, ak)
!-------------------------------------------------------------------------------
!
!    Output variables
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : zz
  use fileio, only : omzt
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(in) :: fk
  complex(kind=DP), dimension(-nz:nz-1)                   , intent(in) :: pk, ak

  complex(kind=DP), dimension(-nz:nz-1,0:ns-1) :: nk
  integer :: iz, is

    do is = 0, ns-1
      call vintegral_z(fk(:,:,:,is), nk(:,is))
    end do
    do iz = -nz, nz-1
      write( omzt, "(99e17.7e3)" ) zz(iz), time,  &
           real(pk(iz), kind=DP), aimag(pk(iz)),  &
           real(ak(iz), kind=DP), aimag(ak(iz)),  &
          (real(nk(iz,is), kind=DP), aimag(nk(iz,is)), is=0,ns-1)
    end do
    write( omzt, * )


END SUBROUTINE out_mominzt


SUBROUTINE out_linfreq(time, pk, stop_signal)
!-------------------------------------------------------------------------------
!
!    Output variables
!
!-------------------------------------------------------------------------------
  use parameters
  use fileio, only : ofrq
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nz:nz-1), intent(in) :: pk
  logical, intent(out) :: stop_signal

  real(kind=DP), save :: time0, pk0_norm2
  complex(kind=DP), save :: omega0, pk0(-nz:nz-1)
  complex(kind=DP) :: pk0pk, omega, diff
  real(kind=DP) :: pk_norm2, ineq
  integer :: iz

    if (time < dt_out) then

      time0 = time
      pk0(:) = pk(:)
      omega0 = (0._DP, 0._DP)
      pk0_norm2 = sum(abs(pk0(-nz+1:nz-1))**2)

    else

    !- calculate interior products -
      pk0pk = (0._DP, 0._DP)
      do iz = -nz+1, nz-1
        pk0pk = pk0pk + conjg(pk0(iz)) * pk(iz)
      end do
      pk_norm2 = sum(abs(pk(-nz+1:nz-1)**2))

    !- calculate frequency - 
      omega = log(pk0pk / pk0_norm2) / (ci * (time0 - time))
      diff = abs(real(omega - omega0, kind=DP) / real(omega, kind=DP))  &
           + ci * abs(aimag(omega - omega0) / aimag(omega))
      ineq = abs(pk0pk)**2 / (pk0_norm2 * pk_norm2)

      write( ofrq, "(99e17.7e3)" ) time, aimag(omega), real(omega, kind=DP),  &
           aimag(diff), real(diff, kind=DP), 1._DP - ineq
      flush( ofrq )

      if ( real(diff, kind=DP) < 0.0001_DP .and.  &
           aimag(diff) < 0.0001_DP .and.  &
           (1._DP - ineq) < 0.0001_DP ) then
        write( ofrq, * ) "# Well converged."
        write( ofrq, "(99a17)" ) "#              kx","ky","Growthrate",  &
                   "Frequency", "Diff(grow)", "Diff(freq)", "1 - Ineq"
        write( ofrq, "(a,99e17.7e3)" ) "# ", kx, ky,  &
               aimag(omega), real(omega, kind=DP),  &
               aimag(diff), real(diff, kind=DP), 1._DP - ineq
        stop_signal = .true.
      end if

    !- remember the values -
      time0 = time
      pk0(:) = pk(:)
      omega0 = omega
      pk0_norm2 = pk_norm2

    end if

END SUBROUTINE out_linfreq


SUBROUTINE out_fkinzvm_paraview(time, fk)
!-------------------------------------------------------------------------------
!
!    Output variables
!
!-------------------------------------------------------------------------------
  use parameters
!  use geometry, only : zz, vl, mu
  use fileio, only : ozvm
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(in) :: fk

  character(len=8) :: ctime
  character(len=1) :: cis
!  real(kind=DP), dimension(:,:,:,:), allocatable :: coord
  integer :: iz, iv, im, is, irec

  do is = 0, ns-1
    write(cis,'(i1.1)') is
    write(ctime,'(i8.8)') int(time / dt_out)

!--- ascii data output for vtk ---
!    open( ozvm, file="./data/pv_fkinzvm_is"//cis//"_t"//ctime//".vtk", &
!                status="replace", action="write",                      &
!                form="formatted", access="sequential" )
!      write( ozvm, "(A)" ) "# vtk DataFile Version 2.0"
!      write( ozvm, "(A,F15.7,A,F15.7,A,F15.7)" ) "lz= ",lz," lv= ",lv," lm= ",lm
!      write( ozvm, "(A)" ) "ASCII"
!      write( ozvm, "(A)" ) "DATASET STRUCTURED_POINTS"
!      write( ozvm, "(A,3I15)" ) "DIMENSIONS ", 2*nz, 2*nv, nm+1 
!      write( ozvm, "(A)" ) "ORIGIN 0.0 0.0 0.0"
!      write( ozvm, "(A,3F15.7)" ) "SPACING ", 3.d0/dble(2*nz), 2.d0/dble(2*nv), 1.d0/dble(nm+1)
!      write( ozvm, * )
!      write( ozvm, "(A,I15)" ) "POINT_DATA ", (2*nz)*(2*nv)*(nm+1)
!      write( ozvm, "(3A)" ) "SCALARS ", "|fk|", " float 1"
!      write( ozvm, "(A)" ) "LOOKUP_TABLE deault"
!      do im = 0, nm
!        do iv = 1, 2*nv
!          do iz = -nz, nz-1
!            write( ozvm, "(E15.7e3)" ) abs(fk(iz,iv,im,is))
!          end do
!        end do
!      end do
!    close( ozvm )
!--- binary data output for vtk ---
    open( ozvm, file="./data/pv_fkinzvm_is"//cis//"_t"//ctime//".vtk_head", &
                status="replace", action="write",                           &
                form="formatted", access="sequential" )
      write( ozvm, "(A)" ) "# vtk DataFile Version 2.0"
      write( ozvm, "(A,F15.7,A,F15.7,A,F15.7)" ) "lz= ",lz," lv= ",lv," lm= ",lm
      write( ozvm, "(A)" ) "BINARY"
      write( ozvm, "(A)" ) "DATASET STRUCTURED_POINTS"
      write( ozvm, "(A,3I15)" ) "DIMENSIONS ", 2*nz, 2*nv, nm+1
      write( ozvm, "(A)" ) "ORIGIN 0.0 0.0 0.0"
      write( ozvm, "(A,3F15.7)" ) "SPACING ", 3.d0/dble(2*nz), 2.d0/dble(2*nv), 1.d0/dble(nm+1)
      write( ozvm, * )
      write( ozvm, "(A,I15)" ) "POINT_DATA ", (2*nz)*(2*nv)*(nm+1)
      write( ozvm, "(3A)" ) "SCALARS ", "|fk|", " float 1"
      write( ozvm, "(A)" ) "LOOKUP_TABLE deault"
    close( ozvm ) 
    open( ozvm, file="./data/pv_fkinzvm_is"//cis//"_t"//ctime//".vtk_bin", &
                status="replace", action="write",                          &
                !form="binary", convert="BIG_ENDIAN" )
                !form="unformatted", access="stream", convert="BIG_ENDIAN" )
                form="unformatted", access="direct", recl=4, convert="BIG_ENDIAN" )
      irec = 1
      do im = 0, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            write( ozvm, rec=irec ) real(abs(fk(iz,iv,im,is)), kind=4)
            irec = irec + 1
          end do
        end do
      end do
    close( ozvm )
!--- binary data output for plot3d2xdmf ---
!    if ( int(time/dt_out) == 0 ) then
!      allocate( coord(-nz:nz-1,1:2*nv,0:nm,1:3) )
!      do im = 0, nm
!        do iv = 1, 2*nv
!          do iz = -nz, nz-1
!            coord(iz,iv,im,1) = zz(iz)
!            coord(iz,iv,im,2) = vl(iv)
!            coord(iz,iv,im,3) = mu(im)
!          end do
!        end do
!      end do
!      open( ozvm, file="./data/caseg_d0_is"//cis//".g", form="unformatted" )
!        write( ozvm ) 2*nz, 2*nv, nm+1
!        write( ozvm ) real(coord, kind=8)
!      close( ozvm )
!      deallocate( coord )
!    end if
!    open( ozvm, file="./data/casef_d0_is"//cis//"_t"//ctime//".f", form="unformatted" )
!      write( ozvm ) 2*nz, 2*nv, nm+1, 1
!      write( ozvm ) real(abs(fk(:,:,:,is)), kind=8)
!    close( ozvm )
!----------------------------------------------

  end do


END SUBROUTINE out_fkinzvm_paraview


SUBROUTINE vintegral_z(wf, wn)
!-------------------------------------------------------------------------------
!
!    Integrate wf in vl, vp
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : vp, dv, dvp
  implicit none
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm), intent(in) :: wf
  complex(kind=DP), dimension(-nz:nz-1), intent(out) :: wn

  integer :: iz, iv, im

    wn(:) = (0._DP, 0._DP)

    do im = 1, nm-1
      do iv = 1, 2*nv
        do iz = -nz, nz-1
          wn(iz) = wn(iz) + wf(iz,iv,im) * (2._DP*pi*vp(iz,im)*dv*dvp(iz))
        end do
      end do
    end do

  !- edge compensation
    im = 1
      do iv = 1, 2*nv
        do iz = -nz, nz-1
          wn(iz) = wn(iz)                                     &
                 - ( - wf(iz,iv,im  ) * vp(iz,im  ) / 12._DP  &
                   + ( wf(iz,iv,im+1) * vp(iz,im+1)           &
                     - wf(iz,iv,im  ) * vp(iz,im  ) * 2._DP   &
                     ) * 11._DP / 720._DP                     &
                   ) * (2._DP*pi*dv*dvp(iz))
        end do
      end do

END SUBROUTINE vintegral_z



SUBROUTINE initial_condition(time, fk, hk, pk, ak)
!-------------------------------------------------------------------------------
!
!    Set initial condition
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : fmx, j0, zz, vl
  implicit none

  real(kind=DP), intent(out) :: time
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(out) :: fk,hk
  complex(kind=DP), dimension(-nz:nz-1)                   , intent(out) :: pk,ak


  real(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: rr,ri
  real(kind=DP), parameter :: initval = 0.001_DP
  integer :: iz, iv, im, is

    time = 0._DP

    fk(:,:,:,:) = (0._DP, 0._DP)
    hk(:,:,:,:) = (0._DP, 0._DP)
    pk(:) = (0._DP, 0._DP)
    ak(:) = (0._DP, 0._DP)

    if (flag_runs > 1) then
      !call restart(time, fk)
    else
      do is = 0, ns-1
        do im = 0, nm
          do iv = 1, 2*nv
            do iz = -nz, nz-1
              fk(iz,iv,im,is) = initval * (1._DP + zz(iz) + vl(iv))**2  &
                              * exp( -zz(iz)**2 / (0.2_DP*pi)**2 ) &
                              * fmx(iz,iv,im)
            end do
          end do
        end do
      end do
    end if
    call random_number(rr)
    call random_number(ri)
    fk(:,:,:,:) = rr + ci * ri

    call fld_esfield(fk, pk)
    call fld_emfield_ff(fk, ak)

    do is = 0, ns-1
      do im = 0, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            hk(iz,iv,im,is) = fk(iz,iv,im,is)  &
               + sgn(is) * Znum(is)  / sqrt( Anum(is) * tau(is) )  &
                     * fmx(iz,iv,im) * vl(iv) * j0(iz,im,is) * ak(iz)
          end do
        end do
      end do
    end do


END SUBROUTINE initial_condition


SUBROUTINE rkg4(time, fk, hk, pk, ak)
!-------------------------------------------------------------------------------
!
!    Time evolution by Runge_Kutta_Gill
!
!-------------------------------------------------------------------------------
  use parameters
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), intent(inout),  &
    dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: fk, hk
  complex(kind=DP), dimension(-nz:nz-1), intent(inout) :: pk, ak

  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), save :: q = (0._DP, 0._DP)
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: dh
  complex(kind=DP) :: k, r
  integer :: iz, iv, im, is


    call time_diff(time, fk, pk, ak, dh)
!$OMP parallel default(none) shared(dt,dh,q,hk) private(iz,iv,im,is,k,r)
    do is = 0, ns-1
!$OMP do
      do im = 0, nm-1
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            k = dt * dh(iz,iv,im,is)
            r = 0.5_DP * (k - 2._DP * q(iz,iv,im,is))
            hk(iz,iv,im,is) = hk(iz,iv,im,is) + r
            q(iz,iv,im,is) = q(iz,iv,im,is) + 3._DP * r - 0.5_DP * k
          end do
        end do
      end do
!$OMP end do nowait
    end do
!$OMP end parallel
    call fld_emfield_hh(hk, ak)
    call fld_hh2ff(hk, ak, fk)
    call fld_esfield(fk, pk)

    call time_diff(time + 0.5_DP * dt, fk, pk, ak, dh)
!$OMP parallel default(none) shared(dt,dh,q,hk) private(iz,iv,im,is,k,r)
    do is = 0, ns-1
!$OMP do
      do im = 0, nm-1
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            k = dt * dh(iz,iv,im,is)
            r = (1._DP - sqrt(0.5_DP)) * (k - q(iz,iv,im,is))
            hk(iz,iv,im,is) = hk(iz,iv,im,is) + r
            q(iz,iv,im,is) = q(iz,iv,im,is)+3._DP*r-(1._DP-sqrt(0.5_DP))*k
          end do
        end do
      end do
!$OMP end do nowait
    end do
!$OMP end parallel
    call fld_emfield_hh(hk, ak)
    call fld_hh2ff(hk, ak, fk)
    call fld_esfield(fk, pk)

    call time_diff(time + 0.5_DP * dt, fk, pk, ak, dh)
!$OMP parallel default(none) shared(dt,dh,q,hk) private(iz,iv,im,is,k,r)
    do is = 0, ns-1
!$OMP do
      do im = 0, nm-1
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            k = dt * dh(iz,iv,im,is)
            r = (1._DP + sqrt(0.5_DP)) * (k - q(iz,iv,im,is))
            hk(iz,iv,im,is) = hk(iz,iv,im,is) + r
            q(iz,iv,im,is) = q(iz,iv,im,is) + 3._DP * r - (1._DP + sqrt(0.5_DP)) * k
          end do
        end do
      end do
!$OMP end do nowait
    end do
!$OMP end parallel
    call fld_emfield_hh(hk, ak)
    call fld_hh2ff(hk, ak, fk)
    call fld_esfield(fk, pk)

    call time_diff(time + dt, fk, pk, ak, dh)
!$OMP parallel default(none) shared(dt,dh,q,hk) private(iz,iv,im,is,k,r)
    do is = 0, ns-1
!$OMP do
      do im = 0, nm-1
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            k = dt * dh(iz,iv,im,is)
            r = 1._DP/6._DP * (k - 2._DP * q(iz,iv,im,is))
            hk(iz,iv,im,is) = hk(iz,iv,im,is) + r
            q(iz,iv,im,is) = q(iz,iv,im,is) + 3._DP * r - 0.5_DP * k
          end do
        end do
      end do
!$OMP end do nowait
    end do
!$OMP end parallel
    call fld_emfield_hh(hk, ak)
    call fld_hh2ff(hk, ak, fk)
    call fld_esfield(fk, pk)


END SUBROUTINE rkg4


SUBROUTINE time_diff(time, fk, pk, ak, dh)
!-------------------------------------------------------------------------------
!
!    Calculate time-differential term
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : j0, kvd, kvs, vl, fmx, dpara, dv, mir
  implicit none

  real(kind=DP), intent(in) :: time
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(in) :: fk
  complex(kind=DP), dimension(-nz:nz-1)                   , intent(in) :: pk, ak
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(out) :: dh

  complex(kind=DP),  &
    dimension(-nz-nzb:nz-1+nzb,1-nvb:2*nv+nvb,0-nvb:nm+nvb,0:ns-1) :: ff
  complex(kind=DP), dimension(-nz-nzb:nz-1+nzb,0:nm,0:ns-1) :: psi, chi
  real(kind=DP) :: cs1, cs2, cefz(-nz:nz-1), cefv
  integer :: iz, iv, im, is

!$OMP parallel default(none)  &
!$OMP shared(dh,fk,pk,ak,ff,psi,chi,j0,kvd,kvs,vl,fmx,dpara,dv,mir,sgn,Znum,tau,Anum)  &
!$OMP private(iz,iv,im,is,cs1,cs2,cefz,cefv)
!$OMP workshare
    ff(:,:,:,:) = (0._DP, 0._DP)
    psi(:,:,:) = (0._DP, 0._DP)
    chi(:,:,:) = (0._DP, 0._DP)
!$OMP end workshare
    do is = 0, ns-1
!$OMP do
      do im = 0, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            ff(iz,iv,im,is) = fk(iz,iv,im,is)
          end do
          if (vl(iv) > 0._DP) then
            do iz = -nzb, -1
              ff(-nz+iz,iv,im,is) = (0._DP, 0._DP)
            end do
            ff(nz  ,iv,im,is) =   ff(nz-1,iv,im,is)
            ff(nz+1,iv,im,is) = - ff(nz-2,iv,im,is) + 2._DP * ff(nz-1,iv,im,is)
          else
            ff(-nz-1,iv,im,is) =   ff(-nz  ,iv,im,is)
            ff(-nz-2,iv,im,is) = - ff(-nz+1,iv,im,is) + 2._DP * ff(-nz,iv,im,is)
            do iz = 1, nzb
              ff(nz-1+iz,iv,im,is) = (0._DP, 0._DP)
            end do
          end if
        end do
      end do
!$OMP end do
    end do
    do is = 0, ns-1
!$OMP do
      do im = 0, nm
        do iz = -nz, nz-1
          psi(iz,im,is) = j0(iz,im,is) * pk(iz)
          chi(iz,im,is) = j0(iz,im,is) * ak(iz)
        end do
        !psi(-nz-1,im,is) =   psi(-nz  ,im,is)
        !psi(-nz-2,im,is) = - psi(-nz+1,im,is) + 2._DP * psi(-nz  ,im,is)
        !chi(-nz-1,im,is) =   chi(-nz  ,im,is)
        !chi(-nz-2,im,is) = - chi(-nz+1,im,is) + 2._DP * chi(-nz  ,im,is)
        !psi( nz  ,im,is) =   psi( nz-1,im,is)
        !psi( nz+1,im,is) = - psi( nz-2,im,is) + 2._DP * psi( nz-1,im,is)
        !chi( nz  ,im,is) =   chi( nz-1,im,is)
        !chi( nz+1,im,is) = - chi( nz-2,im,is) + 2._DP * chi( nz-1,im,is)
      end do
!$OMP end do
    end do



    do is = 0, ns-1
      cs1 = sgn(is) * Znum(is) / tau(is)
      cs2 = sqrt( tau(is) / Anum(is) )
!$OMP do
      do im = 0, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            dh(iz,iv,im,is) = - ci * kvd(iz,iv,im,is) * ff(iz,iv,im,is)        &
                              - cs1 * fmx(iz,iv,im) * (                        &
                                    + ci * kvd(iz,iv,im,is) * psi(iz,im,is)    &
                                    - ci * kvs(iz,iv,im,is)                    &
                            * ( psi(iz,im,is) - cs2 * vl(iv) * chi(iz,im,is) ) )
          end do
        end do
      end do
!$OMP end do
    end do

    do is = 0, ns-1
      cs1 = sgn(is) * Znum(is) / tau(is)
      do iz = -nz, nz-1
        cefz(iz) = 1._DP / ( 12._DP * dpara(iz) ) * sqrt( tau(is) / Anum(is) )
      end do
      cefv = 1._DP / ( 12._DP * dv ) * sqrt( tau(is) / Anum(is) )
!$OMP do
      do im = 0, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            dh(iz,iv,im,is) = dh(iz,iv,im,is)         &
                 - vl(iv) * cefz(iz) * (              &
                     -         ff(iz+2,iv,im,is)      &
                     + 8._DP * ff(iz+1,iv,im,is)      &
                     - 8._DP * ff(iz-1,iv,im,is)      &
                     +         ff(iz-2,iv,im,is) )    &
                 + mir(iz,im) * cefv * (              &
                     -         ff(iz,iv+2,im,is)      &
                     + 8._DP * ff(iz,iv+1,im,is)      &
                     - 8._DP * ff(iz,iv-1,im,is)      &
                     +         ff(iz,iv-2,im,is) )    &
                 - cs1 * fmx(iz,iv,im) * (            &
                       vl(iv) * cefz(iz) * (          &
                         -         psi(iz+2,im,is)    &
                         + 8._DP * psi(iz+1,im,is)    &
                         - 8._DP * psi(iz-1,im,is)    &
                         +         psi(iz-2,im,is) ) )
            end do
          end do
        end do
!$OMP end do
    end do
!$OMP end parallel

    call collision(ff, dh)


END SUBROUTINE time_diff

      
SUBROUTINE collision(ff, dh)
!-------------------------------------------------------------------------------
!
!    Collision term - Lenard-Bernstein model
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : dv, dvp, vp, ksq, omg, vl
  implicit none

  complex(kind=DP), intent(in),  &
    dimension(-nz-nzb:nz-1+nzb,1-nvb:2*nv+nvb,0-nvb:nm+nvb,0:ns-1) :: ff
  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(inout) :: dh

  real(kind=DP) :: nu_s, cef1, cef2, cef3(-nz:nz-1), cef4(-nz:nz-1)
  integer :: iz, iv, im, is


!$OMP parallel default(none)  &
!$OMP shared(ff,dh,dv,dvp,vp,ksq,omg,vl,tau,Anum,fcs,Znum,nu)  &
!$OMP private(iz,iv,im,is,nu_s,cef1,cef2,cef3,cef4)
    do is = 0, ns-1
      nu_s = nu(is) * sqrt(tau(is) / Anum(is))*fcs(is)*Znum(is)**3/tau(is)**2
      cef1   = nu_s / (12._DP * dv * dv)
      cef2   = nu_s / (12._DP * dv)
      do iz = -nz, nz-1
        cef3(iz)   = nu_s / (12._DP * dvp(iz) * dvp(iz))
        cef4(iz)   = nu_s / (12._DP * dvp(iz))
      end do

      im = 0
!$OMP do schedule(dynamic)
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            dh(iz,iv,im,is) = dh(iz,iv,im,is)                 &
                            + ( -          ff(iz,iv+2,im,is)  &
                                + 16._DP * ff(iz,iv+1,im,is)  &
                                - 30._DP * ff(iz,iv  ,im,is)  &
                                + 16._DP * ff(iz,iv-1,im,is)  &
                                -          ff(iz,iv-2,im,is)  &
                              ) * cef1                        &
                            + ( -          ff(iz,iv+2,im,is)  &
                                +  8._DP * ff(iz,iv+1,im,is)  &
                                -  8._DP * ff(iz,iv-1,im,is)  &
                                +          ff(iz,iv-2,im,is)  &
                              ) * cef2 * vl(iv)               &
                            + ( -          ff(iz,iv,im+2,is)  &
                                + 16._DP * ff(iz,iv,im+1,is)  &
                                - 30._DP * ff(iz,iv,im  ,is)  &
                                + 16._DP * ff(iz,iv,im+1,is)  &
                                -          ff(iz,iv,im+2,is)  &
                              ) * cef3(iz) * 2._DP            &
                            + nu_s * 3._DP * ff(iz,iv,im,is)  &
                          ! FLR on collision
                            - nu_s * ksq(iz) * Anum(is) * tau(is)  &
                              / (Znum(is) * omg(iz))**2 * ff(iz,iv,im,is)
          end do
        end do
!$OMP end do nowait

      im = 1
!$OMP do schedule(dynamic)
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            dh(iz,iv,im,is) = dh(iz,iv,im,is)                 &
                            + ( -          ff(iz,iv+2,im,is)  &
                                + 16._DP * ff(iz,iv+1,im,is)  &
                                - 30._DP * ff(iz,iv  ,im,is)  &
                                + 16._DP * ff(iz,iv-1,im,is)  &
                                -          ff(iz,iv-2,im,is)  &
                              ) * cef1                        &
                            + ( -          ff(iz,iv+2,im,is)  &
                                +  8._DP * ff(iz,iv+1,im,is)  &
                                -  8._DP * ff(iz,iv-1,im,is)  &
                                +          ff(iz,iv-2,im,is)  &
                              ) * cef2 * vl(iv)               &
                            + ( -          ff(iz,iv,im+2,is)  &
                                + 16._DP * ff(iz,iv,im+1,is)  &
                                - 30._DP * ff(iz,iv,im  ,is)  &
                                + 16._DP * ff(iz,iv,im-1,is)  &
                                -          ff(iz,iv,im  ,is)  &
                              ) * cef3(iz)                    &
                            + ( -          ff(iz,iv,im+2,is)  &
                                +  8._DP * ff(iz,iv,im+1,is)  &
                                -  8._DP * ff(iz,iv,im-1,is)  &
                                +          ff(iz,iv,im  ,is)  &
                            ) * cef4(iz) * (vp(iz,im) + 1._DP / vp(iz,im))  &
                            + nu_s * 3._DP * ff(iz,iv,im,is)  &
                          ! FLR on collision
                            - nu_s * ksq(iz) * Anum(is) * tau(is)  &
                              / (Znum(is) * omg(iz))**2 * ff(iz,iv,im,is)
          end do
        end do
!$OMP end do nowait

!$OMP do schedule(dynamic)
      do im = 2, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            dh(iz,iv,im,is) = dh(iz,iv,im,is)                 &
                            + ( -          ff(iz,iv+2,im,is)  &
                                + 16._DP * ff(iz,iv+1,im,is)  &
                                - 30._DP * ff(iz,iv  ,im,is)  &
                                + 16._DP * ff(iz,iv-1,im,is)  &
                                -          ff(iz,iv-2,im,is)  &
                              ) * cef1                        &
                            + ( -          ff(iz,iv+2,im,is)  &
                                +  8._DP * ff(iz,iv+1,im,is)  &
                                -  8._DP * ff(iz,iv-1,im,is)  &
                                +          ff(iz,iv-2,im,is)  &
                              ) * cef2 * vl(iv)               &
                            + ( -          ff(iz,iv,im+2,is)  &
                                + 16._DP * ff(iz,iv,im+1,is)  &
                                - 30._DP * ff(iz,iv,im  ,is)  &
                                + 16._DP * ff(iz,iv,im-1,is)  &
                                -          ff(iz,iv,im-2,is)  &
                              ) * cef3(iz)                    &
                            + ( -          ff(iz,iv,im+2,is)  &
                                +  8._DP * ff(iz,iv,im+1,is)  &
                                -  8._DP * ff(iz,iv,im-1,is)  &
                                +          ff(iz,iv,im-2,is)  &
                              ) * cef4(iz) * (vp(iz,im) + 1._DP / vp(iz,im)) &
                            + nu_s * 3._DP * ff(iz,iv,im,is)  &
                          ! FLR on collision
                            - nu_s * ksq(iz) * Anum(is) * tau(is)  &
                              / (Znum(is) * omg(iz))**2 * ff(iz,iv,im,is)
          end do
        end do
      end do
!$OMP end do nowait

    end do
!$OMP end parallel


END SUBROUTINE collision


SUBROUTINE fld_esfield(fk, pk)
!-------------------------------------------------------------------------------
!
!    Solve Poisson eq.
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : fct_poisson, j0, g0, vp, dv, dvp
  implicit none

  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(in) :: fk
  complex(kind=DP), dimension(-nz:nz-1), intent(out) :: pk

  complex(kind=DP), dimension(-nz:nz-1) :: nk
  complex(kind=DP) :: wf, wfvp, wfvp1
  integer :: iz, iv, im, is

    nk(:) = (0._DP, 0._DP)
!$OMP parallel default(none)  &
!$OMP shared(fk,nk,j0,vp,dv,dvp,sgn,fcs)  &
!$OMP private(iz,iv,im,is,wf,wfvp,wfvp1)
    do is = 0, ns-1
!$OMP do reduction(+:nk)
      do im = 1, nm-1
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            wf = fk(iz,iv,im,is) * j0(iz,im,is) * sgn(is) * fcs(is)
            nk(iz) = nk(iz) + wf * (2._DP*pi*vp(iz,im)*dv*dvp(iz))
          end do
        end do
      end do
!$OMP end do
     !- edge compensation -
      im = 1
!$OMP do reduction(+:nk)
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            wfvp  = fk(iz,iv,im  ,is) * j0(iz,im  ,is) * sgn(is) * fcs(is) * vp(iz,im  ) 
            wfvp1 = fk(iz,iv,im+1,is) * j0(iz,im+1,is) * sgn(is) * fcs(is) * vp(iz,im+1) 
            nk(iz) = nk(iz) - ( - wfvp  / 12._DP        &
                              + ( wfvp1                 &
                                - wfvp  * 2._DP         &
                                ) * 11._DP / 720._DP    &
                              ) * (2._DP*pi*dv*dvp(iz))
          end do
        end do
!$OMP end do
    end do
!$OMP end parallel

    if (ns == 1) then
      do iz = -nz, nz-1
        pk(iz) = nk(iz) / ((1._DP - g0(iz,0))/tau(0) + 1._DP)
      end do
    else
      do iz = -nz, nz-1
        pk(iz) = nk(iz) * fct_poisson(iz)
      end do
    end if

END SUBROUTINE fld_esfield


SUBROUTINE fld_emfield_ff(fk, ak)
!-------------------------------------------------------------------------------
!
!    Solve Ampere eq. using fk
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : j0, vl, ksq, vp, dv, dvp
  implicit none

  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(in) :: fk
  complex(kind=DP), dimension(-nz:nz-1), intent(out) :: ak

  complex(kind=DP), dimension(-nz:nz-1) :: nk
  complex(kind=DP) :: wf, wfvp, wfvp1
  integer :: iz, iv, im, is

    nk(:) = (0._DP, 0._DP)
!$OMP parallel default(none)  &
!$OMP shared(fk,nk,j0,vl,vp,dv,dvp,sgn,fcs,tau,Anum)  &
!$OMP private(iz,iv,im,is,wf,wfvp,wfvp1)
    do is = 0, ns-1
!$OMP do reduction(+:nk)
      do im = 1, nm-1
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            wf = fk(iz,iv,im,is) * j0(iz,im,is) * sgn(is) * fcs(is) &
                                     * sqrt( tau(is) / Anum(is) ) * vl(iv)
            nk(iz) = nk(iz) + wf * (2._DP*pi*vp(iz,im)*dv*dvp(iz))
          end do
        end do
      end do
!$OMP end do
     !- edge compensation -
      im = 1
!$OMP do reduction(+:nk)
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            wfvp  = fk(iz,iv,im  ,is) * j0(iz,im  ,is) * sgn(is) * fcs(is) &
                                     * sqrt( tau(is) / Anum(is) ) * vl(iv) * vp(iz,im  )
            wfvp1 = fk(iz,iv,im+1,is) * j0(iz,im+1,is) * sgn(is) * fcs(is) &
                                     * sqrt( tau(is) / Anum(is) ) * vl(iv) * vp(iz,im+1)
            nk(iz) = nk(iz) - ( - wfvp  / 12._DP        &
                              + ( wfvp1                 &
                                - wfvp  * 2._DP         &
                                ) * 11._DP / 720._DP    &
                              ) * (2._DP*pi*dv*dvp(iz))
          end do
        end do
!$OMP end do
    end do
!$OMP end parallel

    do iz = -nz, nz-1
      ak(iz) = nk(iz) * beta / ksq(iz)
    end do

END SUBROUTINE fld_emfield_ff


SUBROUTINE fld_emfield_hh(hk, ak)
!-------------------------------------------------------------------------------
!
!    Solve Ampere eq. using hk
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : fct_ampere, vl, j0, vp, dv, dvp
  implicit none

  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1), intent(in) :: hk
  complex(kind=DP), dimension(-nz:nz-1), intent(out) :: ak

  complex(kind=DP), dimension(-nz:nz-1) :: nk
  complex(kind=DP) :: wf, wfvp, wfvp1
  integer :: iz, iv, im, is

    nk(:) = (0._DP, 0._DP)
!$OMP parallel default(none)  &
!$OMP shared(hk,nk,j0,vl,vp,dv,dvp,sgn,fcs,tau,Anum)  &
!$OMP private(iz,iv,im,is,wf,wfvp,wfvp1)
    do is = 0, ns-1
!$OMP do reduction(+:nk)
      do im = 1, nm-1
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            wf = hk(iz,iv,im,is) * j0(iz,im,is) * sgn(is) * fcs(is) &
                                     * sqrt( tau(is) / Anum(is) ) * vl(iv)
            nk(iz) = nk(iz) + wf * (2._DP*pi*vp(iz,im)*dv*dvp(iz))
          end do
        end do
      end do
!$OMP end do
     !- edge compensation -
      im = 1
!$OMP do reduction(+:nk)
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            wfvp  = hk(iz,iv,im  ,is) * j0(iz,im  ,is) * sgn(is) * fcs(is) &
                                     * sqrt( tau(is) / Anum(is) ) * vl(iv) * vp(iz,im  )
            wfvp1 = hk(iz,iv,im+1,is) * j0(iz,im+1,is) * sgn(is) * fcs(is) &
                                     * sqrt( tau(is) / Anum(is) ) * vl(iv) * vp(iz,im+1)
            nk(iz) = nk(iz) - ( - wfvp  / 12._DP        &
                              + ( wfvp1                 &
                                - wfvp  * 2._DP         &
                                ) * 11._DP / 720._DP    &
                              ) * (2._DP*pi*dv*dvp(iz))
          end do
        end do
!$OMP end do
    end do
!$OMP end parallel

    do iz = -nz, nz-1
      ak(iz) = nk(iz) * beta * fct_ampere(iz)
    end do

END SUBROUTINE fld_emfield_hh


SUBROUTINE fld_hh2ff(hk, ak, fk)
!-------------------------------------------------------------------------------
!
!    Compute hk, ak -> fk
!
!-------------------------------------------------------------------------------
  use parameters
  use geometry, only : fmx, vl, j0
  implicit none

  complex(kind=DP), intent(in),  &
    dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: hk
  complex(kind=DP), dimension(-nz:nz-1), intent(in) :: ak
  complex(kind=DP), intent(out),  &
    dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: fk

  integer :: iz, iv, im, is

!$OMP parallel default(none) shared(fk,hk,ak,fmx,vl,j0,sgn,Znum,Anum,tau) private(iz,iv,im,is)
    do is = 0, ns-1
!$OMP do
      do im = 0, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            fk(iz,iv,im,is) =  hk(iz,iv,im,is)  &
                     - sgn(is) * Znum(is)  / sqrt(Anum(is) * tau(is))  &
                     * fmx(iz,iv,im) * vl(iv) * j0(iz,im,is) * ak(iz)
          end do
        end do
      end do
!$OMP end do nowait
    end do
!$OMP end parallel


END SUBROUTINE fld_hh2ff


END PROGRAM lingk


!SUBROUTINE matvec(hk, Lhk)
!!-------------------------------------------------------------------------------
!!
!!    Calculate L(hk)
!!
!!-------------------------------------------------------------------------------
!  use parameters
!  use geometry, only : fmx, vl, j0
!  implicit none
!
!  complex(kind=DP), intent(in),  &
!    dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: hk
!  complex(kind=DP), intent(out),  &
!    dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: Lhk
!
!  complex(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: fk
!  complex(kind=DP), dimension(-nz:nz-1) :: pk, ak
!  real(kind=DP) :: time
!  integer :: iz, iv, im, is
!
!    time = 99999.9999_DP
!    call fld_emfield_hh(hk, ak)
!    call fld_hh2ff(hk, ak, fk)
!    call fld_esfield(fk, pk)
!    call time_diff(time, fk, pk, ak, Lhk)
!
!END SUBROUTINE matvec
