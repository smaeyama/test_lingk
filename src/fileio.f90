MODULE fileio
!------------------------------------------------------------------------------!
!
!  Set input/output files
!
!------------------------------------------------------------------------------!
  use parameters
  implicit none

  integer, parameter :: omzt = 100, &
                        ofrq = 200, &
                        ofzv = 300, &
                        ozvm = 400

 CONTAINS

SUBROUTINE fileio_open

  character(len=3) :: cruns, pruns

    write( cruns, '(i3.3)' ) flag_runs
    write( pruns, '(i3.3)' ) flag_runs-1

    !if (flag_runs > 1) then
    !  open( ifkk, file = "./data/fk."//pruns, form = "unformatted" )
    !end if

    open( omzt, file = "./data/mominzt."//cruns )
    write( omzt, "(99a17)" ) "#              zz", "time", "phi", "Al", "dens"

    open( ofrq, file = "./data/frq."//cruns )
    write( ofrq, "(99a17)" ) "#            time", "growth", "frequency",  &
                             "diff(grow)", "diff(freq)", "1-Ineq."


END SUBROUTINE fileio_open

SUBROUTINE fileio_close

    close( omzt )
    close( ofrq )

END SUBROUTINE fileio_close


END MODULE fileio
