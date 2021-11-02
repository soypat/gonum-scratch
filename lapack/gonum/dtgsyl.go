package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

// Dpstrf computes the Cholesky factorization with complete pivoting of an n×n
// symmetric positive semidefinite matrix A.
//

// The factorization has the form
//  Pᵀ * A * P = Uᵀ * U ,  if uplo = blas.Upper,
//  Pᵀ * A * P = L  * Lᵀ,  if uplo = blas.Lower,
// where U is an upper triangular matrix and L is lower triangular, and P is
// stored as vector piv.
//
// Dpstrf does not attempt to check that A is positive semidefinite.
//
// The length of piv must be n and the length of work must be at least 2*n,
// otherwise Dpstrf will panic.
//
// Dpstrf is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgsyl(trans string, ijob int, m int, n int, a [][]float64, lda int, b [][]float64, ldb int, c [][]float64, ldc int, d []float64, ldd int, e []float64, lde int, f []float64, ldf int, dif float64, scale float64, work []float64, lwork int, iwork []float64, info int) int {

	//parameter( zero == 0, one == 1 )
	var lquery, notran bool
	var i, ie, ifunc, iround, is, isolve, j, je, js, k int
	var linfo, lwmin, mb, nb, p, ppqq, pq, q int
	var scale2, scaloc float64
	var dsum, dscale float64
	info = 0
	notran = (trans == "N")
	lquery = (lwork == -1)
	bi := blas64.Implementation()
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
		//scale[0] = float64(1)
		scale = float64(1)
		if notran {
			if ijob <= 0 {
				//dif[0] = float64(0)
				dif = float64(0)
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
			//impl.Dlaset('F', m, n, 0, 0, c, ldc)
			impl.Dlaset('F', m, n, 0, 0, f, ldf)
		} else if ijob >= 1 {
			isolve = 2
		}
	}

	if ((mb <= 1) && (nb <= 1)) || ((mb >= m) && (nb >= n)) {
		//DO 30 iround = 1, isolve
		for iround := 1; iround <= isolve; iround++ {
			//           Use unblocked Level 2 solver
			dscale = 0
			dsum = 1
			pq = 0
			//dtgsy2(trans, ifunc, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, scale, dsum, dscale, iwork, pq, info)
			if dscale != 0 {
				if (ijob == 1) || (ijob == 3) {
					//dif = Sqrt(dble(2*m*n)) / (dscale * Sqrt(dsum))
					dif = math.Sqrt((2 * float64(m) * float64(n))) / (dscale * math.Sqrt(dsum))
				} else {
					//dif = math.Sqrt(dble(pq)) / (dscale * math.Sqrt(dsum))
					dif = math.Sqrt((float64(pq))) / (dscale * math.Sqrt(dsum))
				}
			}

			if (isolve == 2) && (iround == 1) {
				if notran {
					ifunc = ijob
				}
				scale2 = scale
				//impl.Dlacpy('F', m, n, c, ldc, work, m)
				//impl.Dlacpy('F', m, n, c[1:], ldc, work, m) no hay forma que lo tome, no sera que esta funcion habra q reformarla a dlacopy?

				//impl.Dlacpy('F', m, n, f, ldf, work[m*n+1], m)  work[m*n+1]?????? que onda con esto porque dlacpy en el penultimo argumento pide []float64
				impl.Dlacpy('F', m, n, f, ldf, work, m)
				//impl.Dlaset('F', m, n, 0, 0, c, ldc) c[1:] tampoco lo toma
				impl.Dlaset('F', m, n, 0, 0, f, ldf)
			} else if (isolve == 2) && (iround == 2) {
				//impl.Dlacpy('F', m, n, work, m, c, ldc) c[1:] tampoco lo toma
				//impl.Dlacpy('F', m, n, work(m*n+1), m, f, ldf) work[m*n+1]?????
				impl.Dlacpy('F', m, n, work, m, f, ldf)
				scale = scale2
			}
			//cierro for
		} //cierro for
		//*retrun ????
	}

	//     Determine block structure of A

	p = 0
	i = 1
G40: //CONTINUE
	if i > m {
		goto G50
	}
	p = p + 1
	iwork[p] = float64(i)
	i = i + mb
	if i >= m {
		goto G50
	}
	if a[i][i-1] != 0 {
		i = i + 1
	}
	goto G40
G50: //CONTINUE
	iwork[p+1] = float64(m + 1)
	if iwork[p] == iwork[p+1] {
		p = p - 1
	}

	//     Determine block structure of B

	q = p + 1
	j = 1
G60: //CONTINUE
	if j > n {
		goto G70
	}
	q = q + 1
	iwork[q] = float64(j)
	j = j + nb
	if j >= n {
		goto G70
	}
	if b[j][j-1] != 0 {
		j = j + 1
	}
	goto G60
G70: //CONTINUE
	iwork[q+1] = float64(n + 1)
	if iwork[q] == iwork[q+1] {
		q = q - 1
	}
	if notran {

		//DO 150 iround = 1, isolve
		for iround := 1; iround <= isolve; iround++ {
			//           Solve (I, J)-subsystem
			//              A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
			//              D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
			//          for I = P, P - 1,..., 1; J = 1, 2,..., Q
			dscale = 0
			dsum = 1
			pq = 0
			scale = 1
			//DO 130 j = p + 2, q
			for j := (p + 2); j <= q; j++ {
				js = int(iwork[j])
				je = int(iwork[j+1] - 1)
				nb = je - js + 1
				//DO 120 i = p, 1, -1
				for i := p; i <= 1; i-- {
					is = int(iwork[i])
					ie = int(iwork[i+1] - 1)
					mb = ie - is + 1
					ppqq = 0
					//dtgsy2( trans, ifunc, mb, nb, a( is, is ), lda,b( js, js ), ldb, c( is, js ), ldc,d( is, is ), ldd, e( js, js ), lde,f( is, js ), ldf, scaloc, dsum, dscale,iwork( q+2 ), ppqq, linfo )
					if linfo > 0 {
						info = linfo
					}
					pq = pq + ppqq
					if scaloc <= float64(1) {
						//DO 80 k = 1, js - 1
						for k := 1; k <= js; js-- {
							//bi.Dscal(m, scaloc, c(1, k), 1)
							//bi.Dscal(m, scaloc, c[k:], ldc) no lo toma tampoco
							bi.Dscal(m, scaloc, f[k:], 1)
							//80                CONTINUE
						}
						//DO 90 k = js, je
						for k := js; k <= je; je++ {
							//bi.Dscal(is-1, scaloc, c(1, k), 1)
							bi.Dscal(is-1, scaloc, f[k:], 1)
							//90                CONTINUE
						}
						//DO 100 k = js, je
						for k := js; k <= js; je++ {
							//bi.Dscal(m-ie, scaloc, c(ie+1, k), 1)
							//bi.Dscal(m-ie, scaloc, f(ie+1, k), 1)
							bi.Dscal(m-ie, scaloc, f[(ie+1)*ldc+k:], 1)
							//100                CONTINUE
						}
						//DO 110 k = je + 1, n
						for k := je + 1; k <= (je + 1); k++ {
							//bi.Dscal(m, scaloc, c(1, k), 1)
							//bi.Dscal(m, scaloc, f(1, k), 1)
							bi.Dscal(m, scaloc, f[(ie+1)*ldc+k:], 1)
							//110                CONTINUE
						}
						scale = scale * scaloc
					}
					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						//bi.Dgemm('N', 'N', is-1, nb, mb, -1, a[1][is] , lda, c[is][js], ldc, 1, c[1][js], ldc)
						//bi.Dgemm('N', 'N', is-1, nb, mb, -1, d[1][is], ldd, c[is][js], ldc, 1, f(1, js), ldf)
					}
					if j < q {
						//bi.Dgemm('N', 'N', mb, n-je, nb, one, f(is, js), ldf, b(js, je+1), ldb, one, c(is, je+1), ldc)
						//bi.Dgemm('N', 'N', mb, n-je, nb, one, f(is, js), ldf, e(js, je+1), lde, one, f(is, je+1), ldf)
					}
					//120          CONTINUE
				}
				//130       CONTINUE
			}
			if dscale != 0 {
				if (ijob == 1) || (ijob == 3) {
					dif = math.Sqrt(float64(2*m*n)) / (dscale * math.Sqrt(dsum))
				} else {
					dif = math.Sqrt(float64(pq)) / (dscale * math.Sqrt(dsum))
				}

				if (isolve == 2) && (iround == 1) {
					if notran {
						ifunc = ijob
					}
					scale2 = scale
					//impl.Dlacpy('F', m, n, c, ldc, work, m)
					//impl.Dlacpy('F', m, n, f, ldf, work(m*n+1), m)
					//impl.Dlaset('F', m, n, 0, 0, c, ldc)
					//impl.Dlaset('F', m, n, 0, 0, f, ldf)
				} else if (isolve == 2) && (iround == 2) {
					//impl.Dlacpy('F', m, n, work, m, c, ldc)
					//impl.Dlacpy('F', m, n, work[m*n+1], m, f, ldf)
					scale = scale2
				}
				//150    CONTINUE
			}
		}
	}

}
