#ifndef STDINT_TYPES_H
#define STDINT_TYPES_H

/* Standard integer types */
typedef signed char int8;
typedef unsigned char uint8;
typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned int uint32;

#ifdef _LP64
typedef long int64;
typedef unsigned long uint64;
#else
typedef long long int64;
typedef unsigned long long uint64;
#endif

#endif /* STDINT_TYPES_H */
