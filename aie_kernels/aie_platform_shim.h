/*
 * Minimal C library shim for bare-metal AIE targets on Windows.
 *
 * Peano from the win64.o conda package does not ship a C standard library.
 * libc++ (from llvm-aie) needs basic C primitives (size_t, memcpy, mbstate_t,
 * wchar.h) to compile.  This header provides just enough to satisfy libc++.
 *
 * Included via -include before any other header.
 */

#ifndef _AIE_PLATFORM_SHIM_H
#define _AIE_PLATFORM_SHIM_H

/* ---- size_t ---- */
#ifndef __SIZE_TYPE__
#define __SIZE_TYPE__ unsigned long
#endif

typedef __SIZE_TYPE__ size_t;

/* ---- NULL ---- */
#ifndef NULL
#ifdef __cplusplus
#define NULL nullptr
#else
#define NULL ((void *)0)
#endif
#endif

/* ---- mbstate_t ---- */
#ifndef _MBSTATE_T
#define _MBSTATE_T
typedef struct {
    char __opaque[8];
} mbstate_t;
#endif

/* ---- ptrdiff_t ---- */
#ifndef __PTRDIFF_TYPE__
#define __PTRDIFF_TYPE__ long
#endif
typedef __PTRDIFF_TYPE__ ptrdiff_t;

/* ---- C string / memory functions ---- */
#ifdef __cplusplus
extern "C" {
#endif

void *memcpy(void *__dst, const void *__src, size_t __n);
void *memmove(void *__dst, const void *__src, size_t __n);
void *memset(void *__s, int __c, size_t __n);
int memcmp(const void *__s1, const void *__s2, size_t __n);
size_t strlen(const char *__s);
char *strcpy(char *__dst, const char *__src);
char *strncpy(char *__dst, const char *__src, size_t __n);
int strcmp(const char *__s1, const char *__s2);
int strncmp(const char *__s1, const char *__s2, size_t __n);
char *strcat(char *__dst, const char *__src);
char *strncat(char *__dst, const char *__src, size_t __n);
const void *memchr(const void *__s, int __c, size_t __n);
void *memchr(void *__s, int __c, size_t __n);

/* ---- wchar stubs (enough for mbstate_t usage) ---- */
#ifndef WEOF
#define WEOF ((wchar_t)-1)
#endif

#ifdef __cplusplus
}
#endif

#endif /* _AIE_PLATFORM_SHIM_H */
