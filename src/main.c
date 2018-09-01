
#include "utilities.h"
#include "matrices.h"
#include <math.h>



void MatCovScale( f64Mat cov, f64Mat x, f64Mat dst )
{
    ASSERT( cov.dim0 == cov.dim1 );
    ASSERT( x.dim1 == 1 );
    ASSERT( cov.dim0 == x.dim0 );
    ASSERT( x.dim0 == dst.dim0 && dst.dim1 == 1 );

    f64Mat tmpC = f64MatMake( DefaultAllocator, cov.dim0, cov.dim0 );
    f64Mat tmpX = f64MatMake( DefaultAllocator, x.dim0, x.dim1 );
    f64MatCopy( cov, tmpC );
    f64MatCopy( x, tmpX );

    LAPACKE_dpotrf ( DEFAULT_MAJOR, 'U', tmpC.dim0, tmpC.data, tmpC.dim0 );

    cblas_dtrmv( DEFAULT_MAJOR, CblasUpper, NO_TRANS, CblasNonUnit, tmpC.dim0, tmpC.data, tmpC.dim0, tmpX.data, 1);

    f64MatCopy( tmpX, dst );

    f64MatFree( DefaultAllocator, &tmpC );
    f64MatFree( DefaultAllocator, &tmpX );
}


#ifndef TEST
int main( int argn, const char **args )
{
    
    f64Mat f = f64MatMake( DefaultAllocator, 3, 3 );
    
    f.data[0] = 1; f.data[1] = 0;   f.data[2] = 0.5;
    f.data[3] = 0; f.data[4] = 0.5; f.data[5] = 0;
    f.data[6] = 0; f.data[7] = 0;   f.data[8] = 2;
    
    f64Mat x       = f64MatMake( DefaultAllocator, 3, 1 );
    f64Mat xScaled = f64MatMake( DefaultAllocator, 3, 1 );
    
    x.data[0] = 1; x.data[1] = 2; x.data[2] = 3;
    
    MatCovScale( f, x, xScaled );
    
    f64MatPrint( xScaled, "xScaled" );
    
//    f64 x = log(-1);
//    printf("log(-1) = %.4f\n", x);
//    PrintBits(sizeof(f64), &x);
//
//    u32 a = 3;
//    PrintBits(sizeof(u32), &a);
    
    return 0;
}
#endif

