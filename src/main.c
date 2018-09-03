
#include "utilities.h"
#include <math.h>
#include <time.h>


#ifndef TEST
int main( int argn, const char **args )
{
    
    InitializeMatrices( 4, 'l' );
    
    clock_t t0;
    clock_t t1;
    f64 t;
    
    u32 N = 4000;
    
    f64Mat a = f64MatMake( DefaultAllocator, N, N );
    f64Mat b = f64MatMake( DefaultAllocator, N, N );
    f64Mat c = f64MatMake( DefaultAllocator, N, N );
    
    for ( u32 i=0; i<(N*N); ++i ) {
        a.data[i] = (f64) i;
        b.data[i] = (f64) 2*i;
    }
    
    t0 = clock();
    
    f64MatMul( a, b, c );
    
    t1 = clock();
    t = (f64) (t1 - t0) / CLOCKS_PER_SEC;
    printf("MatMul in: %.4f sec.\n", t);
    
    printf( "%.4f\n", c.data[0] );
    
    return 0;
}
#endif

