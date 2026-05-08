/*
 * Minimal <wchar.h> shim for bare-metal AIE targets.
 *
 * libc++ __mbstate_t.h uses __has_include_next(<wchar.h>) to find mbstate_t.
 * On bare-metal AIE there is no <wchar.h>, so we provide this stub.
 * It must be on the include path AFTER the libc++ directory so that
 * __has_include_next (which skips the current directory) can find it.
 */
#ifndef _AIE_WCHAR_SHIM_H
#define _AIE_WCHAR_SHIM_H

/* mbstate_t: opaque struct, matches the standard layout */
#ifndef __mbstate_t_defined
#define __mbstate_t_defined
typedef struct {
    char __mbstate8[8];
} mbstate_t;
#endif

/* wchar_t is a C++ builtin — no typedef needed */
/* WCHAR_MIN / WCHAR_MAX */
#ifndef WCHAR_MIN
#define WCHAR_MIN 0
#define WCHAR_MAX 0x7fffffff
#endif

/* WEOF */
#ifndef WEOF
#define WEOF ((wchar_t)-1)
#endif

#endif /* _AIE_WCHAR_SHIM_H */
