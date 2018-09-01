
#include "utilities.h"
#include "matrices.h"
#include <math.h>


#ifndef TEST
int main( int argn, const char **args )
{
    
    f64Mat a = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat b = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat c = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat e = f64MatMake( DefaultAllocator, 3, 3 );
    
    a.data[0] = 1; a.data[1] = 0;   a.data[2] = 0.5;
    a.data[3] = 0; a.data[4] = 0.5; a.data[5] = 0;
    a.data[6] = 0; a.data[7] = 0;   a.data[8] = 2;
    
    f64MatSet( b, 2.0 );
    
    // add
    f64MatElMul( a, b, c );
    
    f64MatPrint( c, "c" );
    
    e.data[0] = 2;  e.data[1] = 0; e.data[2] = 1;
    e.data[3] = 0;  e.data[4] = 1; e.data[5] = 0;
    e.data[6] = 0;  e.data[7] = 0; e.data[8] = 4;
    
    f64MatPrint( e, "e" );
    
    b32 yes = f64MatEqual( c, e, 1 );
    
    printf("out = %d\n", yes );
    
    
    return 0;
}
#endif

