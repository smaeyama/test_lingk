MODULE geometry
!-------------------------------------------------------------------------------
!
!  Module for geometry
!
!-------------------------------------------------------------------------------
  use parameters
  implicit none
  private

  public :: set_geometry,  &
            dz, dv, dm, cfsrf, zz, vl, mu, omg, rootg, ksq, dpara, dvp,  &
            fct_poisson, fct_ampere, vp, mir, fmx, kvd, kvs, j0, g0

  real(kind=DP) :: dz, dv, dm, cfsrf
  real(kind=DP), dimension(-nz:nz-1) :: zz
  real(kind=DP), dimension(1:2*nv)   :: vl
  real(kind=DP), dimension(0:nm)     :: mu
  real(kind=DP), dimension(-nz:nz-1) :: omg, rootg, ksq, dpara, dvp,  &
                                        fct_poisson, fct_ampere
  real(kind=DP), dimension(-nz:nz-1,0:nm) :: vp, mir
  real(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm) :: fmx
  real(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm,0:ns-1) :: kvd, kvs
  real(kind=DP), dimension(-nz:nz-1,0:nm,0:ns-1) :: j0
  real(kind=DP), dimension(-nz:nz-1,0:ns-1) :: g0

 CONTAINS

SUBROUTINE set_geometry

  integer :: iz, iv, im, is
  real(kind=DP), external :: dbesj0, dbesi0
  real(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm) :: wr3
  real(kind=DP), dimension(-nz:nz-1) :: wr, wn
  real(kind=DP) :: x
  real(kind=DP) :: kvd_max, vl_max, mir_max, nu_max, dt_perp, dt_zz, dt_vl, dt_col, cs, nu_temp
  real(kind=DP), parameter :: courant_num = 0.5_DP

    dz = lz / real(nz, kind=DP)
    dv = 2._DP * lv / real(2*nv-1, kind=DP)
    dm = sqrt(2._DP * lm) / real(nm, kind=DP)

    do iz = -nz, nz-1
      zz(iz) = dz * real(iz, kind=DP)
      omg(iz) = 1._DP - eps_r * cos(zz(iz))
      rootg(iz) = q_0 / omg(iz)
      ksq(iz) = (kx + s_hat * zz(iz) * ky)**2 + ky**2
      dpara(iz) = dz * q_0
    end do
    cfsrf = 0._DP
    do iz = -nz, nz-1
      cfsrf = cfsrf + rootg(iz)
    end do

    do iv = 1, 2*nv
      vl(iv) = - lv + dv * real(iv-1, kind=DP)
    end do

    do im = 0, nm
      mu(im) = 0.5_DP * (dm * real(im, kind=DP))**2
    end do
    do im = 0, nm
      do iz = -nz, nz-1
        vp(iz,im) = sqrt(2._DP * mu(im) * omg(iz))
        mir(iz,im) = mu(im) * eps_r * sin(zz(iz)) / q_0
      end do
    end do
    do iz = -nz, nz-1
      dvp(iz) = vp(iz,1)
    end do

    do im = 0, nm
      do iv = 1, 2*nv
        do iz = -nz, nz-1
          fmx(iz,iv,im) = exp(- 0.5_DP * vl(iv)**2 - mu(im) * omg(iz)) &
                         / sqrt(2._DP * pi)**3
        end do
      end do
    end do

    do is = 0, ns-1
      do im = 0, nm
        do iv = 1, 2*nv
          do iz = -nz, nz-1
            kvd(iz,iv,im,is) = - (vl(iv)**2 + omg(iz)*mu(im))                  &
               * (ky * cos(zz(iz)) + (kx + s_hat * zz(iz) * ky) *sin(zz(iz)))  &
               * (sgn(is) * tau(is) / Znum(is))
            kvs(iz,iv,im,is) = - sgn(is) * tau(is) / Znum(is) * ky        &
               * (R0_Ln(is)   &
                + R0_Lt(is) * (0.5_DP*vl(iv)**2 + omg(iz)*mu(im) - 1.5_DP))
          end do
        end do
      end do
    end do

    do is = 0, ns-1
      do iz = -nz, nz-1
        x = ksq(iz)*tau(is)*Anum(is)/(Znum(is)**2*omg(iz)**2)
        if ( x < 150._DP ) then
          g0(iz,is) = dbesi0(x) * exp(- x)
        else
          g0(iz,is) = ( 1._DP                                    &
                      + 0.25_DP              / (2._DP * x)      &
                      + 9._DP / 32._DP       / (2._DP * x)**2   &
                      + 75._DP / 128._DP     / (2._DP * x)**3   &
                      + 3675._DP / 2048._DP  / (2._DP * x)**4   &
                      + 59535._DP / 8192._DP / (2._DP * x)**5 ) &
                      / sqrt(2._DP * pi * x)
        end if
      end do
    end do

    do is = 0, ns-1
      do im = 0, nm
        do iz = -nz, nz-1
          x = sqrt(2._DP * ksq(iz) * mu(im) / omg(iz))  &
                 * sqrt(tau(is) * Anum(is)) / Znum(is)
          j0(iz,im,is) = dbesj0(x)
        end do
      end do
    end do

    do iz = -nz, nz-1
      wr(iz) = lambda * ksq(iz)
    end do
    do is = 0, ns-1
      do iz = -nz, nz-1
        wr(iz) = wr(iz) + Znum(is) * fcs(is) / tau(is) * (1._DP - g0(iz,is))
      end do
    end do
    do iz = -nz, nz-1
      fct_poisson(iz) = 1._DP / wr(iz)
    end do

    if ( beta > 0._DP ) then
      do iz = -nz, nz-1
        wr(iz) = ksq(iz)
      end do
      do is = 0, ns-1
        do im = 0, nm
          do iv = 1, 2*nv
            do iz = -nz, nz-1
              wr3(iz,iv,im) = Znum(is) * fcs(is) / Anum(is)  &
                           * vl(iv)**2 * j0(iz,im,is)**2 * fmx(iz,iv,im)
            end do
          end do
        end do
        call vintegral_r(wr3, wn)
        do iz = -nz, nz-1
          wr(iz) = wr(iz) + beta * wn(iz)
        end do
      end do
      do iz = -nz, nz-1
        fct_ampere(iz) = 1._DP / wr(iz)
      end do
    else
      fct_ampere(:) = 0._DP
    end if
  
    if (flag_dtc) then
      kvd_max = 0._DP
      do is = 0, ns-1
        do im = 0, nm
          do iv = 1, 2*nv
            do iz = -nz, nz-1
              if ( kvd_max < kvd(iz,iv,im,is) ) kvd_max = kvd(iz,iv,im,is)
            end do
          end do
        end do
      end do
      dt_perp = courant_num * pi / kvd_max

      vl_max = 0._DP
      do is = 0, ns-1
        cs = sqrt( tau(is) / Anum(is) )
        do iz = -nz, nz-1
          if ( vl_max < cs * lv / dpara(iz) ) vl_max = cs * lv / dpara(iz)
        end do
      end do
      dt_zz = courant_num / vl_max

      mir_max = 0._DP
      do is = 0, ns-1
        cs = sqrt( tau(is) / Anum(is) )
        do im = 0, nm
          do iz = -nz, nz-1
            if ( mir_max < cs * mir(iz,im) ) mir_max = cs * mir(iz,im)
          end do
        end do
      end do
      dt_vl = courant_num * dv / mir_max

      nu_max = 0._DP
      do is = 0, ns-1
        nu_temp = nu(is)*sqrt(tau(is)/Anum(is))*fcs(is)*Znum(is)**3/tau(is)**2 &
                * (2._DP/dv**2)
        if ( nu_max < nu_temp ) nu_max = nu_temp
        do iz = -nz, nz-1
          nu_temp = nu(is)*sqrt(tau(is)/Anum(is))*fcs(is)*Znum(is)**3/tau(is)**2 &
                  * (2._DP/dvp(iz)**2)
          if ( nu_max < nu_temp ) nu_max = nu_temp
        end do
      end do
      if (nu_max == 0._DP) then
        dt_col = 99999.9999_DP
      else
        dt_col = courant_num / nu_max
      end if

      dt = min(dt_perp, dt_zz, dt_vl, dt_col)

      write(*,*) " # Time step size control"
      write(*,*) ""
      write(*,*) " # courant num. = ", courant_num
      write(*,*) " # dt_perp      = ", dt_perp
      write(*,*) " # dt_zz        = ", dt_zz
      write(*,*) " # dt_vl        = ", dt_vl
      write(*,*) " # dt_col       = ", dt_col
      write(*,*) " # dt           = ", dt
      write(*,*) ""
    end if

    !--- check ---
    !do iz = -nz, nz-1
    !  write(100000,"(99G15.7)") zz(iz), omg(iz), rootg(iz), ksq(iz), dpara(iz), dvp(iz), fct_poisson(iz)
    !end do

    !  do iz = -nz, nz-1
    !do im = 0, nm
    !    write(100010,"(99G15.7)") zz(iz), mu(im), vp(iz,im), mir(iz,im)
    !  end do
    !end do

    !    do iz = -nz, nz-1
    !  do iv = 1, 2*nv
    !do im = 0, nm
    !      write(100020,"(99G15.7)") zz(iz), vl(iv), mu(im), fmx(iz,iv,im)
    !    end do
    !  end do
    !end do

    !do is = 0, ns-1
    !      do iz = -nz, nz-1
    !    do iv = 1, 2*nv
    !  do im = 0, nm
    !        write(100030,"(99G15.7)") zz(iz), vl(iv), mu(im), kvd(iz,iv,im,is), kvs(iz,iv,im,is)
    !      end do
    !    end do
    !  end do
    !end do

    !do is = 0, ns-1
    !    do iz = -nz, nz-1
    !  do im = 0, nm
    !        write(100040,"(99G15.7)") zz(iz), mu(im), j0(iz,im,is)
    !    end do
    !  end do
    !end do

    !do is = 0, ns-1
    !  do iz = -nz, nz-1
    !    write(100050,"(99G15.7)") zz(iz), g0(iz,is)
    !  end do
    !end do


END SUBROUTINE set_geometry


SUBROUTINE vintegral_r(wf, wn)
!-------------------------------------------------------------------------------
!
!    Integrate j0*fk in vp
!
!-------------------------------------------------------------------------------

  real(kind=DP), dimension(-nz:nz-1,1:2*nv,0:nm), intent(in) :: wf
  real(kind=DP), dimension(-nz:nz-1), intent(out) :: wn

  integer :: iz, iv, im

    wn(:) = 0._DP

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

END SUBROUTINE vintegral_r


END MODULE geometry
