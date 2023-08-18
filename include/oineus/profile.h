#pragma once

#ifdef OINEUS_USE_CALIPER
#include <caliper/cali.h>
#else
#define CALI_CXX_MARK_FUNCTION
#define CALI_MARK_BEGIN(x) x;
#define CALI_MARK_END(x) x;
#endif
