
//#include "mkl.h"
//#define BLAS_DECLS
//#define LAPACK_DECLS


//#define USE_BLAS 1

#include "matrices.h"
#include <time.h>

b32 LU(f64 *a, i32 *ipiv, f64 *tmp, u32 n, f64 tol)
{

    u32 i, j, k, imax;
    f64 maxA, absA;

    for (i = 0; i <= n; i++)
        ipiv[i] = i;  /* Unit permutation matrix, ipiv[n] initialized with n */

    for (i = 0; i < n; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < n; k++)
            if ((absA = fabs(a[k*n+i])) > maxA) {
                maxA = absA;
                imax = k;
            }

        if (maxA < tol) {
            return 0;  /* failure, matrix is degenerate */
        }

        if (imax != i) {
            /* pivoting ipiv */
            j = ipiv[i];
            ipiv[i] = ipiv[imax];
            ipiv[imax] = j;

            /* TODO: this is very expensive! */
            memcpy( tmp, a + i*n, n * sizeof(*a) );
            memcpy( a + i*n, a + imax*n, n * sizeof(*a) );
            memcpy( a + imax*n, tmp, n * sizeof(*a) );

            /* counting pivots starting from n (for determinant) */
            ipiv[n]++;
        }

        for (j = i + 1; j < n; j++) {
            a[j*n+i] /= a[i*n+i];

            for (k = i + 1; k < n; k++)
                a[j*n+k] -= a[j*n+i] * a[j*n+k];
        }
    }
    return 1;  /* decomposition done */
}


b32 MatInvN(f64Mat m, f64Mat dst)
{
    ASSERT( m.dim0 == m.dim1 && m.dim0 == dst.dim0 && m.dim1 == dst.dim1 );
    u32 n = m.dim0;

    u32 scratchSize = (n * n + n)* sizeof(f64);
    scratchSize    += n * sizeof(i32);
    ArenaAllocatorCheck( &ScratchArena, &ScratchBuffer, scratchSize );

    f64Mat tmp = f64MatMake( ScratchBuffer, n, n );
    f64MatCopy( m, tmp );

    i32 *ipiv = Alloc( ScratchBuffer, n * sizeof(i32) );
    f64 *tmpV = Alloc( ScratchBuffer, n * sizeof(f64) );

    b32 ret = LU(tmp.data, ipiv, tmpV, n, 1.0E-40);

    if ( ! ret ) {
        FreeAll( ScratchBuffer );
        return 0;
    }

    f64 *im = dst.data;
    f64 *a = tmp.data;

    for ( u32 j = 0; j < n; ++j ) {
        for ( u32 i = 0; i < n; ++i ) {
            if ( ipiv[i] == j )
                im[i*n+j] = 1.0;
            else
                im[i*n+j] = 0.0;

            for ( u32 k = 0; k < i; ++k )
                im[i*n+j] -= a[i*n+k] * im[k*n+j];
        }

        for ( i32 l = n - 1; l >= 0; --l ) {
            for ( u32 k = l + 1; k < n; ++k )
                im[l*n+j] -= a[l*n+k] * im[k*n+j];

            im[l*n+j] = im[l*n+j] / a[l*n+l];
        }
    }
    return 1;
}




#ifndef TEST
int main( int argn, const char **args )
{

#ifdef _MKL_H_
    i32 numThreads = 1;

//    MKL_Set_Num_Threads( numThreads );
    MKL_Set_Num_Threads_Local( numThreads );
#endif

#if USE_BLAS
    printf("Using BLAS...\n\n");
#endif
    
 
//   clock_t t0, t1;
//   f64 t;
//
//   u32 N = 1000;
//
//   f64Mat a = f64MatMake( DefaultAllocator, N, N );
//   f64Mat b = f64MatMake( DefaultAllocator, N, N );
//   f64Mat c = f64MatMake( DefaultAllocator, N, N );
//
////    for ( u32 i=0; i<(N*N); ++i ) {
////        a.data[i] = (f64) i;
////        b.data[i] = (f64) 2*i;
////    }
//
//
//   for ( u32 i=0; i<(N*N); ++i ) {
//       a.data[i] = (f64) i;
//       b.data[i] = (f64) 2*i;
//   }
//
//
//
//   t0 = clock();
//
////    f64MatMul( a, b, c );
//
//   f64MatScale( a, 2, c );
//
//   t1 = clock();
//   t = (f64) (t1 - t0) / CLOCKS_PER_SEC;
//   printf("MatMul in: %.4f sec.\n", t);
//
//   printf( "%.4f\n", c.data[0] );



    f64Mat c = f64MatIdentMake( DefaultAllocator, 3 );
    f64Mat e = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat f = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat g = f64MatMake( DefaultAllocator, 3, 3 );


    f.data[0] = 1.0;
    f.data[1] = 1.0;
    f.data[2] = 0.0;
    f.data[3] = 0.0;
    f.data[4] = 1.0;
    f.data[5] = 0.0;
    f.data[6] = 2.0;
    f.data[7] = 1.0;
    f.data[8] = 1.0;


    f64MatPrint( f, "f" );
    printf("det = %.6f\n", f64MatDet( f ));

#if USE_BLAS
    f64MatInv( f, e );
#else
    MatInvN(f, e);
#endif

    f64MatMul(f, e, g);

    f64MatPrint( g, "g" );

#if USE_BLAS
    f64MatInv( f, e );
#else
    MatInvN(f, e);
#endif

    f64MatPrint( e, "e" );


    f.data[0] = 0.0;
    f.data[1] = 3.0;
    f.data[2] = 5.0;
    f.data[3] = 5.0;
    f.data[4] = 5.0;
    f.data[5] = 2.0;
    f.data[6] = 3.0;
    f.data[7] = 4.0;
    f.data[8] = 3.0;


    f64MatPrint( f, "f" );
    printf("det = %.6f\n", f64MatDet( f ));

#if USE_BLAS
    f64MatInv( f, e );
#else
    MatInvN(f, e);
#endif

    f64MatPrint( e, "e" );

    f64MatMul(f, e, g);

    f64MatPrint( g, "g" );

//    ASSERT( f64MatEqual( c, g, 1E-6 ) );

    f64MatFree( DefaultAllocator, &c );
    f64MatFree( DefaultAllocator, &e );
    f64MatFree( DefaultAllocator, &f );
    f64MatFree( DefaultAllocator, &g );

    ScratchBufferDestroy();

    return 0;
}
#endif

