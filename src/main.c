
//#include "mkl.h"
//#define BLAS_DECLS
//#define LAPACK_DECLS


//#define USE_BLAS

#include "matrices.h"
#include <time.h>


#ifndef TEST
int main( int argn, const char **args )
{

#ifdef _MKL_H_
    i32 numThreads = 1;

//    MKL_Set_Num_Threads( numThreads );
    MKL_Set_Num_Threads_Local( numThreads );
#endif

//    clock_t t0, t1;
//    f64 t;
//
//    u32 N = 1000;
//
//    f64Mat a = f64MatMake( DefaultAllocator, N, N );
//    f64Mat b = f64MatMake( DefaultAllocator, N, N );
//    f64Mat c = f64MatMake( DefaultAllocator, N, N );
//
//    for ( u32 i=0; i<(N*N); ++i ) {
//        a.data[i] = (f64) i;
//        b.data[i] = (f64) 2*i;
//    }
//
//    t0 = clock();
//
//    f64MatMul( a, b, c );
//
//    t1 = clock();
//    t = (f64) (t1 - t0) / CLOCKS_PER_SEC;
//    printf("MatMul in: %.4f sec.\n", t);
//
//    printf( "%.4f\n", c.data[0] );


    u32 matSize = 3;

    f64Mat a = f64MatMake( DefaultAllocator, matSize, matSize );
    f64Mat b = f64MatMake( DefaultAllocator, matSize, matSize );
    f64Mat c = f64MatMake( DefaultAllocator, matSize, matSize );

    for ( u32 i=0; i<matSize*matSize; ++i ) {
        a.data[i] = (f64) i;
        b.data[i] = (f64) 2*i;
    }

    f64MatMul( a, b, c );

    f64MatPrint( c, "c" );
    
    return 0;
}
#endif

