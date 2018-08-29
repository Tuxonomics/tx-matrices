
#include "utilities.h"
#include "matrices.h"
#include <math.h>

#ifndef TEST
int main( int argn, const char **args )
{
    f64 x = log(-1);
    printf("log(-1) = %.4f\n", x);
    PrintBits(sizeof(f64), &x);
    
    u32 a = 3;
    PrintBits(sizeof(u32), &a);
    
    return 0;
}
#endif

