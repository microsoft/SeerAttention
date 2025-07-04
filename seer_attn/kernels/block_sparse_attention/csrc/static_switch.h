// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define FWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = 32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = 96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = 160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 224) {           \
      constexpr static int kHeadDim = 224; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()


#define WARP_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND == 4) {                            \
      constexpr static int CONST_NAME = 4;      \
      return __VA_ARGS__();                     \
    } else if (COND == 8) {                     \
      constexpr static int CONST_NAME = 8;      \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 2;      \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define BLOCKM_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                         \
    if (COND == 64) {                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    } else if (COND == 128) {                   \
      constexpr static int CONST_NAME = 128;    \
      return __VA_ARGS__();                     \
    } else if (COND == 256) {                   \
      constexpr static int CONST_NAME = 256;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define BLOCKN_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                         \
    if (COND == 32) {                           \
      constexpr static int CONST_NAME = 32;     \
      return __VA_ARGS__();                     \
    } else if (COND == 64) {                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    } else if (COND == 128) {                   \
      constexpr static int CONST_NAME = 128;    \
      return __VA_ARGS__();                     \
    } else if (COND == 256) {                   \
      constexpr static int CONST_NAME = 256;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define STAGE_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                         \
    if (COND == 2) {                            \
      constexpr static int CONST_NAME = 2;      \
      return __VA_ARGS__();                     \
    } else if (COND == 3) {                     \
      constexpr static int CONST_NAME = 3;      \
      return __VA_ARGS__();                     \
    } else if (COND == 4) {                     \
      constexpr static int CONST_NAME = 4;      \
      return __VA_ARGS__();                     \
    } else if (COND == 5) {                     \
      constexpr static int CONST_NAME = 5;      \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 2;      \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif
