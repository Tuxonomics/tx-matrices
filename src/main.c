
#include "utilities.h"
#include "matrices.h"
#include <math.h>


#ifndef TEST
int main( int argn, const char **args )
{
    
    u32 aSize = 2;
    
    Arena arena;
    ArenaDefaultInit( &arena, aSize * sizeof( u32 ) );
    
    Allocator arenaAllocator = ArenaAllocatorMake( &arena );
    
    u32 *a = Alloc(  arenaAllocator, sizeof( u32 ) );
    u32 *b = Calloc( arenaAllocator, 1, sizeof( u32 ) );
    
    ASSERT( ((u64) b - (u64) a) == 4 );
    
    u32 bSize = 3;
    
    ArenaDestroyResize( &arena, bSize * sizeof( u32 ) );
    
    a = Alloc(  arenaAllocator, sizeof( u32 ) );
    b = Calloc( arenaAllocator, 1, sizeof( u32 ) );
    
    u32 *c = Alloc( arenaAllocator, sizeof( u32 ) );
    
    ASSERT( ((u64) c - (u64) b) == 4 );
    
    
    return 0;
}
#endif

