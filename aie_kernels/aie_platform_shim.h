/*
 * Minimal C library shim for bare-metal AIE targets on Windows.
 *
 * Peano from llvm-aie does not ship a full C standard library.
 * libc++ needs basic C primitives to compile.
 * This header is included via -include before any other header.
 *
 * Strategy:
 * - Define header guards for Peano's LLVM libc headers to skip them
 *   (prevents ::remove conflict with std::remove from <algorithm>)
 * - Provide minimal type/macro/function declarations needed by libc++
 */

#ifndef _AIE_PLATFORM_SHIM_H
#define _AIE_PLATFORM_SHIM_H

/* ---- Skip Peano's LLVM libc headers ----
   These declare ::remove etc. which conflict with std::remove (algorithm).
   By defining their guards, #include_next from libc++ wrappers is a no-op. */
#define LLVM_LIBC_STDIO_H
#define LLVM_LIBC_STDLIB_H
#define LLVM_LIBC_STRING_H
#define LLVM_LIBC_ERRNO_H
#define LLVM_LIBC_MATH_H
#define LLVM_LIBC_FLOAT_H
#define LLVM_LIBC_FENV_H
#define LLVM_LIBC_ASSERT_H

/* ---- Skip libc++ __mbstate_t.h ----
   Uses __has_include_next(<wchar.h>) which fails on bare-metal AIE. */
#ifndef _LIBCPP___MBSTATE_T_H
#define _LIBCPP___MBSTATE_T_H
#endif
#ifndef __mbstate_t_defined
#define __mbstate_t_defined
typedef struct {
    char __mbstate8[8];
} mbstate_t;
#endif

/* ---- size_t ---- */
#ifndef __SIZE_TYPE__
#define __SIZE_TYPE__ unsigned long
#endif
typedef __SIZE_TYPE__ size_t;

/* ---- NULL ---- */
#ifndef NULL
#ifdef __cplusplus
#define NULL __null
#else
#define NULL ((void *)0)
#endif
#endif

/* ---- ptrdiff_t ---- */
#ifndef __PTRDIFF_TYPE__
#define __PTRDIFF_TYPE__ long
#endif
typedef __PTRDIFF_TYPE__ ptrdiff_t;

/* ---- div_t / ldiv_t / lldiv_t ---- */
#ifndef __div_t_defined
#define __div_t_defined
typedef struct { int quot, rem; } div_t;
#endif
#ifndef __ldiv_t_defined
#define __ldiv_t_defined
typedef struct { long quot, rem; } ldiv_t;
#endif
#ifndef __lldiv_t_defined
#define __lldiv_t_defined
typedef struct { long long quot, rem; } lldiv_t;
#endif

/* ---- FILE / fpos_t (for libc++ cstdio / char_traits) ---- */
#ifndef __FILE_defined
#define __FILE_defined
typedef struct _FILE FILE;
#endif
#ifndef __fpos_t_defined
#define __fpos_t_defined
typedef long fpos_t;
#endif

/* ---- stdin / stdout / stderr ---- */
#ifndef stdin
extern FILE *stdin;
#define stdin stdin
#endif
#ifndef stdout
extern FILE *stdout;
#define stdout stdout
#endif
#ifndef stderr
extern FILE *stderr;
#define stderr stderr
#endif

/* ---- va_list (for cstdio v*printf) ---- */
#ifndef __va_list_defined
#define __va_list_defined
typedef __builtin_va_list va_list;
#endif
#ifndef va_start
#define va_start(ap, param) __builtin_va_start(ap, param)
#endif
#ifndef va_end
#define va_end(ap) __builtin_va_end(ap)
#endif
#ifndef va_arg
#define va_arg(ap, type) __builtin_va_arg(ap, type)
#endif
#ifndef va_copy
#define va_copy(dest, src) __builtin_va_copy(dest, src)
#endif

/* ---- errno ---- */
#ifndef EDOM
#define EDOM   33
#endif
#ifndef ERANGE
#define ERANGE 34
#endif
#ifndef EILSEQ
#define EILSEQ 84
#endif
#ifndef EINVAL
#define EINVAL 22
#endif
#ifndef ENOMEM
#define ENOMEM 12
#endif
#ifndef ENOSYS
#define ENOSYS 38
#endif
#ifndef errno
extern int *__errno(void);
#define errno (*__errno())
#endif

/* ---- stdlib macros ---- */
#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#ifndef RAND_MAX
#define RAND_MAX 32767
#endif
#ifndef MB_CUR_MAX
#define MB_CUR_MAX 4
#endif

/* ---- stdio macros (for libc++ cstdio / char_traits) ---- */
#ifndef EOF
#define EOF (-1)
#endif
#ifndef BUFSIZ
#define BUFSIZ 1024
#endif
#ifndef SEEK_SET
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif
#ifndef FILENAME_MAX
#define FILENAME_MAX 260
#endif
#ifndef FOPEN_MAX
#define FOPEN_MAX 20
#endif
#ifndef TMP_MAX
#define TMP_MAX 238328
#endif
#ifndef L_tmpnam
#define L_tmpnam 260
#endif
#ifndef _IOFBF
#define _IOFBF 0
#define _IOLBF 1
#define _IONBF 2
#endif

/* ---- FP classification macros (for libc++ math.h) ---- */
#ifndef FP_NAN
#define FP_NAN         0
#define FP_INFINITE    1
#define FP_NORMAL      2
#define FP_SUBNORMAL   3
#define FP_ZERO        4
#endif
#ifndef FP_ILOGB0
#define FP_ILOGB0      (-__INT_MAX__ - 1)
#define FP_ILOGBNAN    __INT_MAX__
#endif
#ifndef math_errhandling
#define math_errhandling 0
#endif

/* ---- float_t / double_t (for libc++ cmath) ---- */
#ifndef __FLT_EVAL_METHOD__
#define __FLT_EVAL_METHOD__ 0
#endif
#if __FLT_EVAL_METHOD__ == 0
typedef float float_t;
typedef double double_t;
#elif __FLT_EVAL_METHOD__ == 1
typedef double float_t;
typedef double double_t;
#else
typedef double float_t;
typedef long double double_t;
#endif

/* ---- float.h limits (for libc++ cfloat) ---- */
#ifndef FLT_RADIX
#define FLT_RADIX      2
#endif
#ifndef FLT_MANT_DIG
#define FLT_MANT_DIG   24
#endif
#ifndef DBL_MANT_DIG
#define DBL_MANT_DIG   53
#endif
#ifndef LDBL_MANT_DIG
#define LDBL_MANT_DIG  53
#endif
#ifndef FLT_DIG
#define FLT_DIG        6
#endif
#ifndef DBL_DIG
#define DBL_DIG        15
#endif
#ifndef FLT_MIN_EXP
#define FLT_MIN_EXP    (-125)
#endif
#ifndef DBL_MIN_EXP
#define DBL_MIN_EXP    (-1021)
#endif
#ifndef FLT_MAX_EXP
#define FLT_MAX_EXP    128
#endif
#ifndef DBL_MAX_EXP
#define DBL_MAX_EXP    1024
#endif
#ifndef FLT_EPSILON
#define FLT_EPSILON    1.19209290e-07f
#endif
#ifndef DBL_EPSILON
#define DBL_EPSILON    2.2204460492503131e-16
#endif
#ifndef FLT_MIN
#define FLT_MIN        1.17549435e-38f
#endif
#ifndef DBL_MIN
#define DBL_MIN        2.2250738585072014e-308
#endif
#ifndef FLT_MAX
#define FLT_MAX        3.40282347e+38f
#endif
#ifndef DBL_MAX
#define DBL_MAX        1.7976931348623157e+308
#endif

/* ---- math constants ---- */
#ifndef M_PI
#define M_PI           3.14159265358979323846
#define M_PI_2         1.57079632679489661923
#define M_LN2          0.69314718055994530942
#define M_LN10         2.30258509299404568402
#define M_LOG2E        1.44269504088896340736
#define M_E            2.71828182845904523536
#endif

/* ---- INFINITY / NAN / HUGE_VAL ---- */
#ifndef INFINITY
#define INFINITY       __builtin_inf()
#endif
#ifndef NAN
#define NAN            __builtin_nan("")
#endif
#ifndef HUGE_VAL
#define HUGE_VAL       __builtin_inf()
#endif
#ifndef HUGE_VALF
#define HUGE_VALF      __builtin_inff()
#endif
#ifndef HUGE_VALL
#define HUGE_VALL      __builtin_infl()
#endif

/* ---- C string / memory functions ---- */
#ifdef __cplusplus
extern "C" {
#endif

/* These two MUST be declared before libc++ <cstdio> tries
   "using ::remove" / "using ::rename" — otherwise it conflicts
   with std::remove from <algorithm>. */
int remove(const char *__filename);
int rename(const char *__old, const char *__new);

void *memcpy(void *__dst, const void *__src, size_t __n);
void *memmove(void *__dst, const void *__src, size_t __n);
void *memset(void *__s, int __c, size_t __n);
int memcmp(const void *__s1, const void *__s2, size_t __n);
const void *memchr(const void *__s, int __c, size_t __n);

size_t strlen(const char *__s);
char *strcpy(char *__dst, const char *__src);
char *strncpy(char *__dst, const char *__src, size_t __n);
int strcmp(const char *__s1, const char *__s2);
int strncmp(const char *__s1, const char *__s2, size_t __n);
char *strcat(char *__dst, const char *__src);
char *strncat(char *__dst, const char *__src, size_t __n);
char *strchr(const char *__s, int __c);
char *strrchr(const char *__s, int __c);
char *strstr(const char *__haystack, const char *__needle);
char *strerror(int __errnum);
size_t strspn(const char *__s, const char *__accept);
size_t strcspn(const char *__s, const char *__reject);
char *strpbrk(const char *__s, const char *__accept);

/* ---- stdlib: bsearch / qsort (may be instantiated by templates) ---- */
void *bsearch(const void *__key, const void *__base, size_t __nmemb,
              size_t __size, int (*__compar)(const void *, const void *));
void qsort(void *__base, size_t __nmemb, size_t __size,
           int (*__compar)(const void *, const void *));

/* ---- stdlib primitives ---- */
int abs(int __n);
long labs(long __n);
long long llabs(long long __n);
div_t div(int __x, int __y);
ldiv_t ldiv(long __x, long __y);
lldiv_t lldiv(long long __x, long long __y);

/* ---- abort / exit ---- */
void abort(void);
void exit(int __status);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
/* C++ overload for memchr (non-const version) */
inline void *memchr(void *__s, int __c, size_t __n) {
    return const_cast<void *>(memchr(static_cast<const void *>(__s), __c, __n));
}
#endif

#endif /* _AIE_PLATFORM_SHIM_H */
