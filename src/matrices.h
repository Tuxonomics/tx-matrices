

// TODO(jonas): namespacing
// TODO(jonas): QR decomposition
// TODO(jonas): default routines without BLAS
// TODO(jonas): use general BLAS interface not only MKL
// TODO(jonas): aligned memory


#ifndef BLAS_DECLS
#define BLAS_DECLS
/* BLAS Declarations */

enum CBLAS_LAYOUT {
    CblasRowMajor = 101,   /* row-major arrays */
    CblasColMajor = 102,   /* column-major arrays */
};

enum CBLAS_TRANSPOSE {
    CblasNoTrans   = 111,  /* trans='N' */
    CblasTrans     = 112,  /* trans='T' */
    CblasConjTrans = 113   /* trans='C' */
};

enum CBLAS_UPLO {
    CblasUpper = 121, /* uplo ='U' */
    CblasLower = 122  /* uplo ='L' */
};

enum CBLAS_DIAG {
    CblasNonUnit = 131, /* diag ='N' */
    CblasUnit    = 132  /* diag ='U' */
};

enum CBLAS_SIDE {
    CblasLeft  = 141, /* side ='L' */
    CblasRight = 142  /* side ='R' */
};


void cblas_scopy(
    const int n, const float *x, const int incx, float *y, const int incy
);
void cblas_dcopy(
    const int n, const double *x, const int incx, double *y, const int incy
);

void cblas_sscal( const int n, const float a, float *x, const int incx );
void cblas_dscal( const int n, const double a, double *x, const int incx );

float cblas_snrm2( const int n, const float *x, const int incx );
double cblas_dnrm2 (const int n, const double *x, const int incx);

void cblas_sgemm(
    const int Layout, const int transa, const int transb, const int m,
    const int n, const int k, const float alpha, const float *a, const int lda,
    const float *b, const int ldb, const float beta, float *c, const int ldc
);
void cblas_dgemm(
    const int Layout, const int transa, const int transb, const int m,
    const int n, const int k, const double alpha, const double *a,
    const int lda, const double *b, const int ldb, const double beta,
    double *c, const int ldc
);

#endif


#ifndef LAPACK_DECLS
#define LAPACK_DECLS

/* LAPACK Declarations */
int LAPACKE_sgetrf(
    int matrix_layout, int m, int n, float *a, int lda , int *ipiv
);
int LAPACKE_dgetrf(
    int matrix_layout, int m, int n, double *a, int lda , int *ipiv
);

int LAPACKE_sgetri(
    int matrix_layout, int n, float *a, int lda, const int *ipiv
);
int LAPACKE_dgetri(
    int matrix_layout, int n, double *a, int lda, const int *ipiv
);


#endif


#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <execinfo.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <math.h>


// tuxonomics prefix
#ifndef TX_PF
    #define TX_PF(x) tx_##x
#endif


#ifndef TUXONOMICS_BASIC_TYPES
    #define TUXONOMICS_BASIC_TYPES

    typedef uint8_t  u8;
    typedef uint16_t u16;
    typedef uint32_t u32;
    typedef uint64_t u64;

    typedef int8_t  i8;
    typedef int16_t i16;
    typedef int32_t i32;
    typedef int64_t i64;

    typedef float  f32;
    typedef double f64;

    typedef i8  b8;
    typedef i32 b32;
#endif


#ifndef TUXONOMICS_UTILITIES

    #define TUXONOMICS_UTILITIES
    
    #if defined(_MSC_VER)
        #if _MSC_VER < 1300
            #define DEBUG_TRAP() __asm int 3
        #else
            #define DEBUG_TRAP() __debugbreak()
        #endif
    #else
        #define DEBUG_TRAP() __builtin_trap()
    #endif
        
    #ifndef TEST
    #if !defined(RELEASE) && !defined(ASSERTS)
        #define ASSERT_MSG_VA(cond, msg, ...) do { \
            if (!(cond)) { \
            assertHandler(__FILE__, (i32)__LINE__, msg, __VA_ARGS__); \
            DEBUG_TRAP(); \
            } \
            } while(0)
    
        #define ASSERT_MSG(cond, msg) ASSERT_MSG_VA(cond, msg, 0)
    
        #define ASSERT(cond) ASSERT_MSG_VA(cond, 0, 0)
        #define PANIC(msg) ASSERT_MSG_VA(0, msg, 0)
        #define UNIMPLEMENTED() ASSERT_MSG_VA(0, "unimplemented", 0);
    #else
        #define ASSERT_MSG_VA(cond, msg, ...)
        #define ASSERT_MSG(cond, msg)
        #define ASSERT(cond)
        #define PANIC(msg)
        #define UNIMPLEMENTED()
    #endif
    #endif


    #if !defined(Inline)
        #if defined(_MSC_VER)
            #if _MSC_VER < 1300
                #define Inline
            #else
                #define Inline __forceinline
            #endif
        #else
            #define Inline __attribute__ ((__always_inline__))
        #endif
    #endif


    #if !defined(_Threadlocal)
        #if defined(_MSC_VER)
            #define _Threadlocal __declspec( thread )
        #else
            #define _Threadlocal __thread
        #endif
    #endif


    void Backtrace() {
    #define BACKTRACE_MAX_STACK_DEPTH 50
    #if SYSTEM_POSIX
        void* callstack[BACKTRACE_MAX_STACK_DEPTH];
        int i, frames = backtrace(callstack, BACKTRACE_MAX_STACK_DEPTH);
        char** strs = backtrace_symbols(callstack, frames);
        for (i = 0; i < frames; ++i) {
            fprintf(stderr, "%s\n", strs[i]);
        }
        free(strs);
    #elif SYSTEM_WINDOWS
        UNIMPLEMENTED();
    #endif
    }

    void assertHandler(char const *file, i32 line, char const *msg, ...) {
        va_list args;
        va_start(args, msg);
        Backtrace();
        
        if (msg) {
            fprintf(stderr, "Assert failure: %s:%d: ", file, line);
            vfprintf(stderr, msg, args);
            fprintf(stderr, "\n");
        } else {
            fprintf(stderr, "Assert failure: %s:%d\n", file, line);
        }
        va_end(args);
    }

#endif


#ifndef MIN
    #define MIN(x, y) ((x) <= (y) ? (x) : (y))
    #define MAX(x, y) ((x) >= (y) ? (x) : (y))
#endif


#ifndef TUXONOMICS_ALLOCATOR
    #define TUXONOMICS_ALLOCATOR

    typedef enum AllocType {
        AT_Alloc,
        AT_Calloc,
        AT_Realloc,
        AT_Free,
        AT_FreeAll,
    } AllocType;


    #define ALLOC_FUNC(name) void *name(void *payload, enum AllocType alType, size_t count, size_t size, void *old)
    typedef void *allocFunc(void *payload, enum AllocType alType, size_t count, size_t size, void *old);


    typedef struct Allocator Allocator;
    struct Allocator {
        allocFunc *func;
        void *payload;
    };


    void *Alloc(Allocator al, size_t count) {
        return al.func(al.payload, AT_Alloc, count, 0, NULL);
    }

    void *Calloc(Allocator al, size_t count, size_t size) {
        return al.func(al.payload, AT_Calloc, count, size, NULL);
    }

    void *Free(Allocator al, void* ptr) {
        if (ptr)
            al.func(al.payload, AT_Free, 0, 0, ptr);
        return NULL;
    }

    void *FreeAll(Allocator al) {
        al.func(al.payload, AT_FreeAll, 0, 0, NULL);
        return NULL;
    }

    void *Realloc(Allocator al, void *ptr, size_t size, size_t oldsize) {
        return al.func(al.payload, AT_Realloc, size, oldsize, ptr);
    }


    void *checkedCalloc(size_t num_elems, size_t elem_size) {
        void *ptr = calloc(num_elems, elem_size);
        if (!ptr) {
            perror("calloc failed");
            exit(1);
        }
        return ptr;
    }

    void *checkedRealloc(void *ptr, size_t num_bytes) {
        ptr = realloc(ptr, num_bytes);
        if (!ptr) {
            perror("realloc failed");
            exit(1);
        }
        return ptr;
    }

    void *checkedMalloc(size_t num_bytes) {
        void *ptr = malloc(num_bytes);
        if (!ptr) {
            perror("malloc failed");
            exit(1);
        }
        return ptr;
    }

    void *heapAllocFunc(void *payload, enum AllocType alType, size_t count, size_t size, void *old) {
        switch (alType) {
            case AT_Alloc:
                return checkedMalloc(count);
            case AT_Calloc:
                return checkedCalloc(count, size);
            case AT_Free:
            case AT_FreeAll: {
                free(old);
                return NULL;
            }
            case AT_Realloc:
                return checkedRealloc(old, count);
        }
        return NULL;
    }

    Allocator DefaultAllocator = { .func = heapAllocFunc, .payload = 0 };


    typedef struct Arena {
        Allocator allocator;
        u8  *raw;
        u64 cap;
        u64 len;
    } Arena;

    void *arenaAllocFunc(void *payload, enum AllocType alType, size_t count, size_t size, void *old) {
        Arena *arena = (Arena *) payload;

        switch (alType) {
            case AT_Alloc: {
                if (arena->len + count > arena->cap) {
                    return NULL;
                }
                u8 *ptr = &arena->raw[arena->len];
                arena->len += count;
                return ptr;
            }
            case AT_Calloc: {
                u8 * ptr = arenaAllocFunc( payload, AT_Alloc, count * size, 0, old );
                memset( ptr, 0, (count * size) );
                return ptr;
            }
            case AT_Free:
            case AT_FreeAll: {
                arena->len = 0;
                break;
            }
            case AT_Realloc: {
                break;
            }
        }

        return NULL;
    }

    Allocator ArenaAllocatorMake(Arena *arena) {
        Allocator al;
        al.func    = arenaAllocFunc;
        al.payload = arena;
        return al;
    }

    void ArenaInit(Arena *arena, Allocator al, u64 size) {
        arena->allocator = al;
        arena->raw = Alloc(al, size);
        arena->cap = size;
        arena->len = 0;
    }

    void ArenaDefaultInit(Arena *arena, u64 size) {
        ArenaInit(arena, DefaultAllocator, size);
    }

    void ArenaDestroy(Arena *arena) {
        ASSERT( arena->raw );
        Free( arena->allocator, arena->raw );
    }

    void ArenaDestroyResize( Arena *arena, u64 size ) {
        ArenaDestroy( arena );

        arena->raw = Alloc( arena->allocator, size );
        arena->cap = size;
        arena->len = 0;
    }

    void ArenaInitAndAllocator( Allocator alOnArena, Arena *arena, Allocator *al, u64 size )
    {
        ArenaInit( arena, alOnArena, size );
        *al = ArenaAllocatorMake( arena );
    }

    void ArenaAllocatorCheck( Arena *arena, Allocator *al, u64 size )
    {
        if ( arena->cap < size ) {
            if ( arena->raw ) {
                ArenaDestroyResize( arena, size );
            }
            else {
                ArenaInitAndAllocator( DefaultAllocator, arena, al, size );
            }
        }
    }
#endif

#if TEST
void test_arena()
{
    u32 aSize = 2;

    Arena arena;
    ArenaDefaultInit( &arena, aSize * sizeof( u32 ) );

    Allocator arenaAllocator = ArenaAllocatorMake( &arena );

    u32 *a = Alloc(  arenaAllocator, sizeof( u32 ) );
    u32 *b = Calloc( arenaAllocator, 1, sizeof( u32 ) );

    TEST_ASSERT( ((u64) b - (u64) a) == 4 );

    u32 bSize = 3;

    ArenaDestroyResize( &arena, bSize * sizeof( u32 ) );

    a = Alloc(  arenaAllocator, sizeof( u32 ) );
    b = Calloc( arenaAllocator, 1, sizeof( u32 ) );

    u32 *c = Alloc(  arenaAllocator, sizeof( u32 ) );

    TEST_ASSERT( ((u64) c - (u64) b) == 4 );
}
#endif

#ifndef BASIC_FUNCS

    #define ADD(type) Inline \
    type type##Add( type a, type b ) { return a + b; }

    #define SUB(type) Inline \
    type type##Sub( type a, type b ) { return a - b; }

    #define MUL(type) Inline \
    type type##Mul( type a, type b ) { return a * b; }

    #define DIV(type) Inline \
    type type##Div( type a, type b ) { return a / b; }

    #define NEG(type) Inline \
    type type##Neg( type a ) { return -a; }

    #define CONST(type) Inline \
    type type##Const( type a ) { return a; }

    #define PRINT(type) Inline \
    void type##Print( type a ) { printf("%.4f", a); }

    #define EQUAL(type) Inline \
    b32 type##Equal( type a, type b, type eps ) { return fabs( a - b ) < eps; }

    #define COPY(type) Inline \
    void type##Copy( type src, type *dst ) { *dst = src; }


    #define BASIC_FUNCS(t) \
        ADD(t); \
        SUB(t); \
        MUL(t); \
        DIV(t); \
        NEG(t); \
        CONST(t); \
        PRINT(t); \
        EQUAL(t); \
        COPY(t)


    BASIC_FUNCS(f32);
    BASIC_FUNCS(f64);


    void i32Print( i32 a ) { printf("%d\n",   a); }


    #undef ADD
    #undef SUB
    #undef MUL
    #undef DIV
    #undef NEG
    #undef CONST
    #undef PRINT
    #undef EQUAL
    #undef BASIC_FUNCS

#endif


#if TEST
void test_basic_funcs()
{
    TEST_ASSERT( f32Equal( 1.0f, 0.51f, 0.5 ) );
    TEST_ASSERT( ! f32Equal( 0.0f, 0.01f, 1E-2 ) );

    TEST_ASSERT( f64Equal( 1, 0.51, 0.5 ) );
    TEST_ASSERT( ! f64Equal( 0.0, 0.01, 1E-2 ) );

    f64 a = 5;
    f64 b = 3;

    f64Copy( a, &b );
    TEST_ASSERT( f64Equal(a, b, 1E-10) );

}
#endif


Arena     ScratchArena  = {0};
Allocator ScratchBuffer = {0};


void ScratchBufferDestroy( void )
{
    ArenaDestroy( &ScratchArena );
}


#ifndef EPS
    #define EPS 1E-10
#endif


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

#define MAT_PRINTLONG(type, baseFun) void \
    type##MatPrintLong(type##Mat m, const char* name) \
    { \
        printf("Mat (%s): {\n", name); \
        printf("\t.dim0 = %u\n", m.dim0); \
        printf("\t.dim1 = %u\n", m.dim1); \
        printf("\t.data = {\n\n"); \
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
        printf("\n\t}\n"); \
        printf("}\n"); \
    }

#define MAT_PRINT(type, baseFun) void \
    type##MatPrint(type##Mat m, const char* name) \
    { \
        printf("Mat (%s): {\n", name); \
        printf("\t.dim0 = %u\n", m.dim0); \
        printf("\t.dim1 = %u\n", m.dim1); \
        printf("\t.data = {\n\n"); \
        u32 dim; \
        for (u32 i=0; i<m.dim0; ++i) { \
            for (u32 j=0; j<m.dim1; ++j) { \
                dim = i*m.dim1 + j; \
                printf("[%d] ", dim); \
                baseFun(m.data[dim]); \
                printf("  "); \
            } \
            printf("\n"); \
        } \
        printf("\n\t}\n"); \
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

#define MAT_COPY(type, blas_prefix) Inline void \
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

#define MAT_SETROW(type) Inline void \
    type##MatSetRow(type##Mat m, u32 dim, type##Mat newCol) \
    { \
        ASSERT( m.data ); \
        ASSERT( newCol.dim0 * newCol.dim1 == m.dim0 ); \
        \
        for ( u32 i=0; i<m.dim1; ++i ) { \
            m.data[dim * m.dim1 + i] = newCol.data[i]; \
        } \
    }

#define MAT_GETROW(type) Inline void \
    type##MatGetRow(type##Mat m, u32 dim, type##Mat newCol) \
    { \
        ASSERT( m.data ); \
        ASSERT( newCol.dim0 * newCol.dim1 == m.dim0 ); \
        \
        for ( u32 i=0; i<m.dim1; ++i ) { \
            newCol.data[i] = m.data[dim * m.dim1 + i]; \
        } \
    }

/* vector euclidean norm */
#define MAT_VNORM(type, blas_prefix) Inline type \
    type##MatVNorm(type##Mat m) \
    { \
        ASSERT( m.dim0 == 1 || m.dim1 == 1 ); \
         \
        return cblas_##blas_prefix##nrm2( MAX(m.dim0, m.dim1), m.data, 1 ); \
    }

/* matrix scaling */
#define MAT_SCALE(type, blas_prefix) Inline void \
    type##MatScale( type##Mat m, type val, type##Mat dst ) \
    { \
        ASSERT( m.dim0 == dst.dim0 && m.dim1 == dst.dim1 ); \
        type##MatCopy( m, dst ); \
        cblas_##blas_prefix##scal( (dst.dim0 * dst.dim1), val, dst.data, 1); \
    }

/* matrix addition */
#define MAT_ADD(type, fun) Inline void \
    type##MatAdd( type##Mat a, type##Mat b, type##Mat c ) /* c = a + b */ \
    { \
        ASSERT(a.dim0 == b.dim0 && a.dim1 == b.dim1 && a.dim0 == c.dim0 && a.dim1 == c.dim1); \
        \
        for ( u32 i=0; i<(a.dim0*a.dim1); ++i ) { \
            c.data[i] = fun( a.data[i], b.data[i] ); \
        } \
    }

/* matrix subtraction */
#define MAT_SUB(type, fun) Inline void \
    type##MatSub( type##Mat a, type##Mat b, type##Mat c ) /* c = a - b */ \
    { \
        ASSERT(a.dim0 == b.dim0 && a.dim1 == b.dim1 && a.dim0 == c.dim0 && a.dim1 == c.dim1); \
        \
        for ( u32 i=0; i<(a.dim0*a.dim1); ++i ) { \
            c.data[i] = fun( a.data[i], b.data[i] ); \
        } \
    }

/* matrix operation with element-wise multiplication / division */
#define MAT_ELOP(type, suffix, fun) void \
    type##MatEl##suffix( type##Mat a, type##Mat b, type##Mat c ) /* c = a .* b */ \
    { \
        ASSERT(a.dim0 == b.dim0 && a.dim1 == b.dim1 && a.dim0 == c.dim0 && a.dim1 == c.dim1); \
        \
        for ( u32 i=0; i<(a.dim0*a.dim1); ++i ) { \
            c.data[i] = fun( a.data[i], b.data[i] ); \
        } \
    }

#define MAT_MUL_NAIVE(type, addFun, mulFun) void \
    type##MatMul( type##Mat a, type##Mat b, type##Mat c )  /* c = a * b */ \
    { \
        ASSERT(a.dim0 == c.dim0 && a.dim1 == b.dim0 && b.dim1 == c.dim1); \
        /* A is n x m, B is m x p, C is n x p */ \
        \
        u32 n = a.dim0; \
        u32 m = a.dim1; \
        u32 p = b.dim1; \
        \
        i32 i, j, k; \
        type tmpA; \
        \
        type *ra  = a.data; \
        type *rb  = b.data; \
        type *rc  = c.data; \
        \
        for ( i = 0; i < n; ++i ) { \
            for ( k = 0; k < m; ++k ) { \
                for ( j = 0; j < p; ++j ) { \
                    rc[j] = addFun( rc[j], mulFun( ra[k], rb[j] ) ); \
                } \
                rb += p; \
            } \
            ra += k; \
            rc += p; \
            rb -= m*p; \
        } \
    }

/* matrix multiplicaton */
#define MAT_MUL(type, blas_prefix) Inline void \
    type##MatMul( type##Mat a, type##Mat b, type##Mat c ) /* c = a * b */ \
    { \
        ASSERT(a.dim0 == c.dim0 && a.dim1 == b.dim0 && b.dim1 == c.dim1); \
         \
        /* lda, ldb, ldc - leading dimensions of matrices: number of columns in matrices */ \
        cblas_##blas_prefix##gemm ( \
            CblasRowMajor, CblasNoTrans, CblasNoTrans, \
            a.dim0, b.dim1, a.dim1, \
            1.0, \
            a.data, a.dim1, \
            b.data, b.dim1, \
            0, \
            c.data, b.dim1 \
        ); \
    }

/* matrix trace */
#define MAT_TRACE(type) type \
    type##MatTrace( type##Mat m ) /* out = trace(m) */ \
    { \
        ASSERT( m.dim0 == m.dim1 ); \
        type res = 0; \
        for ( u32 i = 0; i < m.dim0; ++i ) { \
            res += m.data[ i * (m.dim0 + 1) ]; \
        } \
        return res; \
    }

/* matrix transpose */
// TODO(jonas): use cache-aware transposition
#define MAT_T(type) void \
    type##MatT(type##Mat m, type##Mat dst) \
    { \
        ASSERT( m.dim0 == dst.dim1 && m.dim1 == dst.dim0 ); \
        \
        for ( u32 i=0; i<m.dim0; ++i ) { \
            for ( u32 j=0; j<m.dim1; ++j ) { \
                dst.data[ i * m.dim0 + j ] = m.data[ j * dst.dim0 + i ]; \
            } \
        } \
    }

/* matrix inverse */
#define MAT_INV(type, lapack_prefix) i32 \
    type##MatInv(type##Mat m, type##Mat dst) \
    {\
        ASSERT( m.dim0 == m.dim1 && m.dim0 == dst.dim0 && m.dim1 == dst.dim1 ); \
        \
        u32 scratchSize = m.dim0 * m.dim1 * sizeof(type); \
        ArenaAllocatorCheck( &ScratchArena, &ScratchBuffer, scratchSize ); \
        \
        type##Mat tmp = type##MatMake( ScratchBuffer, m.dim0, m.dim1 ); \
        type##MatCopy( m, tmp ); \
        u32 n = tmp.dim0; \
        i32 ipiv[n+1]; \
        \
        i32 ret = LAPACKE_##lapack_prefix##getrf( \
            CblasRowMajor, n, n, tmp.data, n, ipiv \
        ); \
        \
        if (ret !=0) { \
            return ret; \
        } \
        \
        ret = LAPACKE_dgetri( \
            CblasRowMajor, n, tmp.data, n, ipiv \
        ); \
        type##MatCopy( tmp, dst ); \
        FreeAll( ScratchBuffer ); \
        \
        return ret; \
    }

// TODO(jonas): maybe use QR decomposition for determinant to improve accuracy
#define MAT_DET(type, lapack_prefix) type \
    type##MatDet(type##Mat m) \
    {\
        ASSERT( m.dim0 == m.dim1 ); \
        \
        u32 scratchSize = m.dim0 * m.dim1 * sizeof(type); \
        ArenaAllocatorCheck( &ScratchArena, &ScratchBuffer, scratchSize ); \
        \
        type##Mat tmp = type##MatMake( ScratchBuffer, m.dim0, m.dim0 ); \
        type##MatCopy( m, tmp ); \
        \
        u32 n = tmp.dim0; \
        i32 ipiv[n+1]; \
        \
        i32 ret1 = LAPACKE_##lapack_prefix##getrf( \
            CblasRowMajor, n, n, tmp.data, n, ipiv \
        ); \
        \
        if (ret1 !=0) { \
            return 0; \
        } \
        type det = 1; \
        for ( u32 i=0; i<n; ++i ) { \
            det *= tmp.data[ i*n + i]; \
        } \
        u32 j; \
        f64 detp= 1; \
        for ( j=0; j<n; ++j ) { \
            if ( j+1 != ipiv[j] ) { \
                detp=-detp; \
            } \
        } \
        \
        FreeAll( ScratchBuffer ); \
        \
        return det * detp; \
    }


MAT_DECL(f64);
MAT_MAKE(f64);
MAT_FREE(f64);
MAT_PRINTLONG(f64, f64Print);
MAT_PRINT(f64, f64Print);
MAT_EQUAL(f64, f64Equal);
MAT_ZERO(f64);
MAT_SET(f64);
MAT_SETELEMENT(f64);
MAT_GETELEMENT(f64);
MAT_COPY(f64, d);
MAT_SETCOL(f64);
MAT_GETCOL(f64);
MAT_SETROW(f64);
MAT_GETROW(f64);


MAT_SCALE(f64, d);             // tested
MAT_VNORM(f64, d);             // tested
MAT_ADD(f64, f64Add);          // tested
MAT_SUB(f64, f64Sub);          // tested
MAT_MUL(f64, d);               // tested
MAT_ELOP(f64, Mul, f64Mul);    // tested
MAT_ELOP(f64, Div, f64Div);    // tested
MAT_TRACE(f64);                // tested
MAT_T(f64);                    // tested
MAT_INV(f64, d);               // tested
MAT_DET(f64, d);               // tested


#if TEST
void test_scale()
{
    f64Mat a = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat b = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat c = f64MatMake( DefaultAllocator, 3, 3 );
    
    for ( u32 i=0; i<9; ++i ) {
        a.data[i] = (f64) i;
        b.data[i] = (f64) 2*i;
    }
    
    f64MatScale( a, 2, c );
    
    TEST_ASSERT( f64MatEqual( c, b, EPS ) );
}
#endif


#if TEST
void test_vnorm()
{
    f64Mat f = f64MatMake( DefaultAllocator, 9, 1 );
    
    f.data[0] = 0.3306201; f.data[1] = 0.6187407; f.data[2] = 0.6796355;
    f.data[3] = 0.4953877; f.data[4] = 0.9147741; f.data[5] = 0.3992435;
    f.data[6] = 0.5875585; f.data[7] = 0.4554847; f.data[8] = 0.8567403;
    
    TEST_ASSERT( f64Equal( f64MatVNorm(f), 1.86610966, 1E-7 ) );
    
    f64MatFree( DefaultAllocator, &f );
}
#endif


#if TEST
void test_add_sub_mul()
{
    f64Mat a = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat b = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat c = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat e = f64MatMake( DefaultAllocator, 3, 3 );
    
    for ( u32 i=0; i<9; ++i ) {
        a.data[i] = (f64) i;
        b.data[i] = (f64) 2*i;
    }
    
    // add
    f64MatAdd( a, b, c );

    e.data[0] = 0;  e.data[1] = 3;  e.data[2] = 6;
    e.data[3] = 9;  e.data[4] = 12; e.data[5] = 15;
    e.data[6] = 18; e.data[7] = 21; e.data[8] = 24;

    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
    
    // sub
    f64MatSub( a, b, c );

    e.data[0] = 0;  e.data[1] = -1; e.data[2] = -2;
    e.data[3] = -3; e.data[4] = -4; e.data[5] = -5;
    e.data[6] = -6; e.data[7] = -7; e.data[8] = -8;

    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
    
    // mul
    f64MatMul( a, b, c );

    e.data[0] = 30;  e.data[1] = 36;  e.data[2] = 42;
    e.data[3] = 84;  e.data[4] = 108; e.data[5] = 132;
    e.data[6] = 138; e.data[7] = 180; e.data[8] = 222;

    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
    
    f64MatFree( DefaultAllocator, &a );
    f64MatFree( DefaultAllocator, &b );
    f64MatFree( DefaultAllocator, &c );
    f64MatFree( DefaultAllocator, &e );
}
#endif


#if TEST
void test_element_wise_ops()
{
    f64Mat a = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat b = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat c = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat e = f64MatMake( DefaultAllocator, 3, 3 );
    
    a.data[0] = 1;   a.data[1] = 0;   a.data[2] = 0.5;
    a.data[3] = 0;   a.data[4] = 0.5; a.data[5] = 0;
    a.data[6] = 0.5; a.data[7] = 0;   a.data[8] = 2;
    
    f64MatSet( b, 2.0 );
    
    // add
    f64MatElMul( a, b, c );
    
    e.data[0] = 2;  e.data[1] = 0; e.data[2] = 1;
    e.data[3] = 0;  e.data[4] = 1; e.data[5] = 0;
    e.data[6] = 1;  e.data[7] = 0; e.data[8] = 4;
    
    TEST_ASSERT( f64MatEqual( c, e, 1E-4 ) );
    
    // sub
    f64MatElDiv( a, b, c );
    
    e.data[0] = 0.5;  e.data[1] = 0;    e.data[2] = 0.25;
    e.data[3] = 0;    e.data[4] = 0.25; e.data[5] = 0;
    e.data[6] = 0.25; e.data[7] = 0;    e.data[8] = 1;
    
    TEST_ASSERT( f64MatEqual( c, e, EPS ) );
    
    f64MatFree( DefaultAllocator, &a );
    f64MatFree( DefaultAllocator, &b );
    f64MatFree( DefaultAllocator, &c );
    f64MatFree( DefaultAllocator, &e );
}
#endif


#if TEST
void test_trace()
{
    f64Mat c = f64MatZeroMake( DefaultAllocator, 3, 3 );
    
    for ( u32 i=0; i<c.dim0; ++i )
        c.data[ i * c.dim0 + i ] = (f64) i + 0.5;
    
    TEST_ASSERT( f64Equal( f64MatTrace(c), 4.5, EPS ) );
    
    f64MatFree( DefaultAllocator, &c );
}
#endif


#if TEST
void test_inverse()
{
    f64Mat c = f64MatZeroMake( DefaultAllocator, 3, 3 );
    f64Mat e = f64MatZeroMake( DefaultAllocator, 3, 3 );
    f64Mat f = f64MatMake( DefaultAllocator, 3, 3 );
    
    for ( u32 i=0; i<c.dim0; ++i )
        c.data[ i * c.dim0 + i ] = (f64) i + 0.5;
    
    f64MatInv( c, f );
    
    for ( u32 i=0; i<e.dim0; ++i )
        e.data[ i * e.dim0 + i ] = 1 / ((f64) i + 0.5);
    
    TEST_ASSERT( f64MatEqual( f, e, EPS ) );
    
    f64MatFree( DefaultAllocator, &c );
    f64MatFree( DefaultAllocator, &e );
    f64MatFree( DefaultAllocator, &f );
    
    ScratchBufferDestroy();
}
#endif


#if TEST
void test_transpose()
{
    f64Mat a = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat b = f64MatMake( DefaultAllocator, 3, 3 );
    f64Mat c = f64MatMake( DefaultAllocator, 3, 3 );
    
    for ( u32 i=0; i<9; ++i ) {
        a.data[i] = (f64) i;
    }
    for ( u32 i=0; i<a.dim0; ++i ) {
        for ( u32 j=0; j<a.dim1; ++j ) {
            b.data[ j*b.dim0 + i ] = a.data[ i*a.dim0 + j ];
        }
    }
    
    f64MatT( a, c );
    
    TEST_ASSERT( f64MatEqual( c, b, EPS ) );
}
#endif


#if TEST
void test_determinant()
{
    f64Mat f = f64MatMake( DefaultAllocator, 3, 3 );

    f.data[0] = 0.3306201;
    f.data[1] = 0.6187407;
    f.data[2] = 0.6796355;
    f.data[3] = 0.4953877;
    f.data[4] = 0.9147741;
    f.data[5] = 0.3992435;
    f.data[6] = 0.5875585;
    f.data[7] = 0.4554847;
    f.data[8] = 0.8567403;

    TEST_ASSERT( f64Equal( f64MatDet(f), -0.130408, 1E-6 ) );

    f64MatFree( DefaultAllocator, &f );
    
    ScratchBufferDestroy();
}
#endif


MAT_DECL(i32);
MAT_MAKE(i32);
MAT_FREE(i32);
MAT_PRINT(i32, i32Print);



#undef EPS


#ifdef __cplusplus
}
#endif


