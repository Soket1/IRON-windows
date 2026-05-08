/*
 * Minimal C library shim for bare-metal AIE targets on Windows.
 *
 * Peano from the win64.o conda package does not ship a C standard library.
 * libc++ (from llvm-aie) needs basic C primitives to compile.
 * This header is included via -include before any other header.
 */

#ifndef _AIE_PLATFORM_SHIM_H
#define _AIE_PLATFORM_SHIM_H

/* Tell libc++ there is no C library.  Must be defined BEFORE any libc++
   header is included.  libc++ uses this to define mbstate_t internally
   and to use _LIBCPP_USING_IF_EXISTS for missing C symbols. */
#ifndef _LIBCPP_HAS_NO_LIBC
#define _LIBCPP_HAS_NO_LIBC
#endif
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
#define _LIBCPP_HAS_NO_WIDE_CHARACTERS
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

/* Note: wchar_t is a built-in C++ keyword in Peano/Clang — no typedef needed. */

/* ---- mbstate_t ----
   libc++ __mbstate_t.h uses __has_include_next(<wchar.h>) which does not
   work reliably on Windows bare-metal.  Define the header guard to skip
   __mbstate_t.h entirely, and provide mbstate_t ourselves. */
#ifndef _LIBCPP___MBSTATE_T_H
#define _LIBCPP___MBSTATE_T_H
#endif
#ifndef __mbstate_t_defined
#define __mbstate_t_defined
typedef struct {
    char __mbstate8[8];
} mbstate_t;
#endif

/* ---- div_t / ldiv_t / lldiv_t ---- */
typedef struct { int quot, rem; } div_t;
typedef struct { long quot, rem; } ldiv_t;
typedef struct { long long quot, rem; } lldiv_t;

/* ---- FP classification macros (for libc++ math.h) ---- */
#ifndef FP_NAN
#define FP_NAN         0
#define FP_INFINITE    1
#define FP_NORMAL      2
#define FP_SUBNORMAL   3
#define FP_ZERO        4
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
