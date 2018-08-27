
#include "mkl.h"


/* row-major matrix definition */

#ifndef MAT_DECL

#define DEFAULT_MAJOR CblasRowMajor
#define NO_TRANS CblasNoTrans
#define TRANS CblasTrans
#define UPPER CblasUpper
#define LOWER CblasLower

// TODO(jonas): intel-based default allocator and arena for alignment


#define MAT_DECL(type) typedef struct type##Mat type##Mat; \
    struct type##Mat { \
        u32 dim0; \
        u32 dim1; \
        type *data; \
    };

#define MAT_MAKE(type) type##Mat \
    type##MatMake(Allocator al, u32 dim0, u32 dim1) \
    { \
        type##Mat m; \
        m.dim0  = dim0; \
        m.dim1  = dim1; \
        m.data  = (type *) Alloc(al, dim0 * dim1 * sizeof(type)); \
        return m; \
    }

#define MAT_FREE(type) void \
    type##MatFree(Allocator al, type##Mat *m) \
    { \
        ASSERT(m->data); \
        Free(al, m->data); \
    }

#define MAT_PRINT(type, baseFun) void \
    type##MatPrint(type##Mat m, const char* name) \
    { \
        printf("Mat (%s): {\n", name); \
        printf("\t.dim0 = %u\n", m.dim0); \
        printf("\t.dim1 = %u\n", m.dim1); \
        printf("\t.data = {\n"); \
        u32 dim; \
        for (u32 i=0; i<m.dim0; ++i) { \
        for (u32 j=0; j<m.dim1; ++j) { \
        dim = i*m.dim1 + j; \
        printf("[%d]\n", dim); \
        baseFun(m.data[dim]); \
        printf("\n"); \
        } \
        printf("\n"); \
        } \
        printf("\n\t}\n\n"); \
        printf("}\n"); \
    }

#define MATPRINT(type, x) type##MatPrint(x, #x)

#define MAT_EQUAL(type, equalFun) b32 \
    type##MatEqual( type##Mat a, type##Mat b, f64 eps ) \
    { \
        if ( a.dim0 != b.dim0 ) \
            return 0; \
        if ( a.dim1 != b.dim1 ) \
            return 0; \
        \
        for ( u32 i=0; i<(a.dim0 * a.dim1); ++i ) { \
            if ( ! equalFun( a.data[i], b.data[i], eps ) ) \
                return 0; \
        } \
        return 1; \
    }

#define MAT_ZERO(type) type##Mat \
    type##MatZeroMake(Allocator al, u32 dim0, u32 dim1) \
    { \
        type##Mat m = type##MatMake(al, dim0, dim1); \
        memset(m.data, 0, dim0 * dim1 * sizeof(type)); \
        return m; \
    }

#define MAT_SET(type) void \
    type##MatSet( type##Mat m, type val ) \
    { \
        for ( u32 i=0; i<(m.dim0 * m.dim1); ++i ) { \
            m.data[i] = val; \
        } \
    }

#define MAT_SETELEMENT(type) Inline void \
    type##MatSetElement(type##Mat m, u32 dim0, u32 dim1, type val) \
    { \
        ASSERT( m.data ); \
        ASSERT( dim0 <= m.dim0 ); \
        ASSERT( dim1 <= m.dim1 ); \
        m.data[dim0 * m.dim1 + dim1] = val; \
    }

#define MAT_GETELEMENT(type) Inline type \
    type##MatGetElement(type##Mat m, u32 dim0, u32 dim1) \
    { \
        ASSERT( dim0 <= m.dim0 ); \
        ASSERT( dim1 <= m.dim1 ); \
        return m.data[dim0 * m.dim1 + dim1]; \
    }

#define MAT_COPY(type, blas_prefix) void \
    type##MatCopy( type##Mat src, type##Mat dst ) \
    { \
        ASSERT( (src.dim0 * src.dim1) == (dst.dim0 * dst.dim1) ); \
        cblas_##blas_prefix##copy( src.dim0 * src.dim1, src.data, 1, dst.data, 1); \
    }

#define MAT_SETCOL(type) Inline void \
    type##MatSetCol(type##Mat m, u32 dim, type##Mat newCol) \
    { \
        ASSERT( m.data ); \
        ASSERT( newCol.dim0 * newCol.dim1 == m.dim0 ); \
        \
        u32 idx = dim*m.dim1; \
        memcpy( &m.data[idx], newCol.data, sizeof( type ) * m.dim0 ); \
    }

#define MAT_GETCOL(type) Inline void \
    type##MatGetCol(type##Mat m, u32 dim, type##Mat newCol) \
    { \
        ASSERT( m.data ); \
        ASSERT( newCol.dim0 * newCol.dim1 == m.dim0 ); \
        \
        u32 idx = dim*m.dim1; \
        memcpy( newCol.data, &m.data[idx], sizeof( type ) * m.dim0 ); \
    }

#define MAT_SCALE(type, blas_prefix) void \
    type##MatScale( type##Mat m, type val ) \
    { \
        cblas_##blas_prefix##scal( (m.dim0 * m.dim1), val, m.data, 1); \
    }

#define MAT_ADD(type, fun) Inline void \
    type##MatAdd(type##Mat a, type##Mat b, type##Mat c) /* c = a + b */ \
    { \
        ASSERT(a.dim0 == b.dim0 && a.dim1 == b.dim1 && a.dim0 == c.dim0 && a.dim1 == c.dim1); \
        \
        for ( u32 i=0; i<(a.dim0*a.dim1); ++i ) { \
            c.data[i] = fun( a.data[i], b.data[i] ); \
        } \
    }

#define MAT_SUB(type, fun) Inline void \
    type##MatSub(type##Mat a, type##Mat b, type##Mat c) /* c = a - b */ \
    { \
        ASSERT(a.dim0 == b.dim0 && a.dim1 == b.dim1 && a.dim0 == c.dim0 && a.dim1 == c.dim1); \
        \
        for ( u32 i=0; i<(a.dim0*a.dim1); ++i ) { \
            c.data[i] = fun( a.data[i], b.data[i] ); \
        } \
    }




#endif


MAT_DECL(f64);
MAT_MAKE(f64);
MAT_FREE(f64);
MAT_PRINT(f64, f64Print);
MAT_EQUAL(f64, f64Equal);
MAT_ZERO(f64);
MAT_SET(f64);
MAT_SETELEMENT(f64);
MAT_GETELEMENT(f64);
MAT_COPY(f64, d);
MAT_SETCOL(f64);
MAT_GETCOL(f64);
MAT_SCALE(f64, d);
MAT_ADD(f64, f64Add);
MAT_SUB(f64, f64Sub);


#if TEST
void test_f64Matrices()
{
#define EPS 1E-10
    
    f64Mat a = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat b = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat c = f64MatMake( DefaultAllocator, 3, 3 );
    
    for ( u32 i=0; i<9; ++i ) {
        a.data[i] = (f64) i;
        b.data[i] = (f64) 2*i;
    }
    
    f64Mat d = f64MatZeroMake( DefaultAllocator, 3, 3 );
    
    
    TEST_ASSERT( f64MatEqual( a, a, EPS ) );
    TEST_ASSERT( ! f64MatEqual( a, b, EPS ) );
    
    
    /* test matrix operations */
    
    f64Mat e = f64MatMake( DefaultAllocator, 3, 3 );

    f64MatSet( e, 0 );
    
    TEST_ASSERT( f64MatEqual( d, e, EPS ) );
    
    f64MatCopy( a, e );
    
    TEST_ASSERT( f64MatEqual( a, e, EPS ) );
    
    for ( u32 i=0; i<9; ++i )
        e.data[i] = e.data[i] * 5;
    
    f64MatScale( a, 5 );
    
    TEST_ASSERT( f64MatEqual( a, e, EPS ) );
    
    f64MatScale( a, 0.2 );
    
    /* subtraction */
    f64MatSub( a, b, c );
    
    e.data[0] = 0;
    e.data[1] = -1;
    e.data[2] = -2;
    
    e.data[3] = -3;
    e.data[4] = -4;
    e.data[5] = -5;
    
    e.data[6] = -6;
    e.data[7] = -7;
    e.data[8] = -8;
    
    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
    
    
    
    
    /* addition */
    f64MatAdd( a, b, c );
    
    e.data[0] = 0;
    e.data[1] = 3;
    e.data[2] = 6;
    
    e.data[3] = 9;
    e.data[4] = 12;
    e.data[5] = 15;
    
    e.data[6] = 18;
    e.data[7] = 21;
    e.data[8] = 24;
    
    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
    
    
//    /* multiplication */
//    memset( c.data, 0, c.dim0 * c.dim1 * sizeof(f64) );
//    
//    f64MatMul( a, b, c );
//    
//    e.data[0] = 30;
//    e.data[1] = 36;
//    e.data[2] = 42;
//    
//    e.data[3] = 84;
//    e.data[4] = 108;
//    e.data[5] = 132;
//    
//    e.data[6] = 138;
//    e.data[7] = 180;
//    e.data[8] = 222;
//    
//    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
//    
//    memset( e.data, 0, e.dim0 * e.dim1 * sizeof(f64) );
//    
//    f64MatMul_Naive( a, b, e );
//    
//    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
    
    
    f64MatFree( DefaultAllocator, &a );
    f64MatFree( DefaultAllocator, &b );
    f64MatFree( DefaultAllocator, &c );
    f64MatFree( DefaultAllocator, &d );
    f64MatFree( DefaultAllocator, &e );
    
#undef EPS
}
#endif
