package gonum

import "gonum.org/v1/gonum/blas"

func (impl Implementation) Dtgsyl(trans string, ijob int, m int, n int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, d []float64, ldd int, e []float64, lde int, f []float64, ldf int, dif []float64, scale []float64, work []float64, lwork int, iwork []float64, info int) int {

	//parameter( zero == 0, one == 1 )
	var lquery, notran bool
	var i, ie, ifunc, iround, is, isolve, j, je, js, k int
	var linfo, lwmin, mb, nb, p, ppqq, pq, q int
	var dscale, dsum, scale2, scaloc []float64
	info = 0
	notran = (trans == "N")
	lquery = (lwork == -1)

	if !(notran) && !(trans == "T") {
		info = -1
	} else if notran {
		if (ijob < 0) || (ijob > 4) {
			info = -2
		}
	}

	if info == 0 {
		if m <= 0 {
			info = -3
		} else if n <= 0 {
			info = -4
		} else if lda < max(1, m) {
			info = -6
		} else if ldb < max(1, n) {
			info = -8
		} else if ldc < max(1, m) {
			info = -10
		} else if ldd < max(1, m) {
			info = -12
		} else if lde < max(1, n) {
			info = -14
		} else if ldf < max(1, m) {
			info = -16
		}
	}
	if info == 0 {
		if notran {
			if (ijob == 1) || (ijob == 2) {
				lwmin = max(1, 2*m*n)
			} else {
				lwmin = 1
			}
		} else {
			lwmin = 1
		}

		//work[1] = lwmin
		work[1] = float64(lwmin)
		if (lwork < lwmin) && !(lquery) {
			info = -20
		}
	}

	if info != 0 {
		//xerbla( "DTGSYL", -info )
		//RETURN ;
		panic(-info)
	} else if lquery {
		panic(lquery)
	}
	//     Quick return if possible

	if (m == 0) || (n == 0) {
		scale[0] = float64(1)
		if notran {
			if ijob <= 0 {
				dif[0] = float64(0)
			}
		}
		//RETURN
	}

	//    Determine optimal block sizes MB and NB
	mb = impl.Ilaenv(2, "DTGSYL", trans, m, n, -1, -1)
	nb = impl.Ilaenv(5, "DTGSYL", trans, m, n, -1, -1)

	isolve = 1
	ifunc = 0
	if notran {
		if ijob >= 3 {
			ifunc = ijob - 2
			impl.Dlaset(blas.All, m, n, 0, 0, c, ldc)
			impl.Dlaset(blas.All, m, n, 0, 0, f, ldf)
		} else if ijob >= 1 {
			isolve = 2
		}
	}

}
