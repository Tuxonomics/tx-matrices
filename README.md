# Matrices (WIP)
[![CircleCI](https://circleci.com/gh/Tuxonomics/matrices-c.svg?style=svg)](https://circleci.com/gh/Tuxonomics/matrices-c)

This repository provides a matrix abstraction for C89 and up. So far the
matrices are not dynamically sized or buffered. For some routines it is
possible to link against some BLAS or LAPACK libraries, however, all routines
should have a default routines such that linking against BLAS and LAPACK is
completely optionally and only required to achieve a different performance.

This library will remain a single header and will have C++ compatibility soon.

The matrices are currently defined only in row-major.

An important aspect is that the matrices can be allocated using custom
allocators. The pre-defined allcator is the `DefaultAllocator` which resorts
to a system call on every allocation.


## Table of Contents
- [Namespacing](#namespacing)
- [Inverse, Determinant and Solving of Linear Equations](#inverse)
- [Scratch Buffer](#scratchbuffer)


## [Namespacing](#namespacing)
The namespacing is not complete yet. So far the matrix routines can be
declared by using the macro `DECL_MAT_OPS`. Soon the macros will recursively
extend to any base type (currently only `float` (`f32`) and `double` (`f64`)
should fully work). Once the base matrix `struct` and all other routines are
defined, the namespacing is done by the base type.

For instance, using `f64` as a base type will declare the matrix as

```C
typedef struct f64Mat f64Mat;
struct f64Mat {
    u32 dim0;
    u32 dim1;
    f64 *data;
};
```

and the corresponding functions are then namespaced by `f64Mat`:

```C
f64Mat f64MatMake(Allocator al, u32 dim0, u32 dim1);

void f64MatFree(Allocator al, f64Mat *m);

void f64MatPrintLong(f64Mat m, const char* name);

void f64MatPrint(f64Mat m, const char* name);

b32 f64MatEqual( f64Mat a, f64Mat b, f64 eps );

f64Mat f64MatZeroMake(Allocator al, u32 dim0, u32 dim1);

f64Mat f64MatIdentMake(Allocator al, u32 dim);

void f64MatSet( f64Mat m, type val );

void f64MatSetElement(f64Mat m, u32 dim0, u32 dim1, type val);

f64 f64MatGetElement(f64Mat m, u32 dim0, u32 dim1);

void f64MatCopy( f64Mat src, f64Mat dst );

void f64MatSetCol(f64Mat m, u32 dim, f64Mat srcCol);

void f64MatGetCol(f64Mat m, u32 dim, f64Mat dstCol);

void f64MatSetRow(f64Mat m, u32 dim, f64Mat srcRow);

void f64MatGetRow(f64Mat m, u32 dim, f64Mat dstRow);

f64 f64MatVNorm(f64Mat m);

void f64MatScale( f64Mat m, type val, f64Mat dst );

void f64MatAdd( f64Mat a, f64Mat b, f64Mat c );

void f64MatSub( f64Mat a, f64Mat b, f64Mat c );

void f64MatElMul( f64Mat a, f64Mat b, f64Mat c );

void f64MatElDiv( f64Mat a, f64Mat b, f64Mat c );

void f64MatMul( f64Mat a, f64Mat b, f64Mat c );

f64 f64MatTrace( f64Mat m );

void f64MatT(f64Mat m, f64Mat dst);

b32 f64MatInv(f64Mat m, f64Mat dst);

f64 f64MatDet(f64Mat m);
```


## [Inverse, Determinant and Solving of Linear Equations](#inverse)
The inverse and determinant are solved using the LU decomposition.

Soon: solving for any rhs vector.


## [Scratch Buffer](#scratchbuffer)
Some functions that need to allocate additional memory that would potentially
be too large for the stack, allocate to a scratch buffer. The scratch buffer
is automatically resized if the existing size is not enough. If the scratch
buffer needs to be completely freed by the user, invoke
```C
ScratchBufferDestroy();
```



