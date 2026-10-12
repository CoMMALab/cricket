#pragma once

#include <cstdio>
#include <cstdlib>

/// Aborts with the failing expression and its location when `condition` is false.
/// It stays active in every build type, so tests keep their checks when NDEBUG is set.
#define CRICKET_CHECK(condition)                                                                             \
    do                                                                                                       \
    {                                                                                                        \
        if (!(condition))                                                                                    \
        {                                                                                                    \
            std::fprintf(stderr, "%s:%d: Check `%s` failed.\n", __FILE__, __LINE__, #condition);             \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (false)
