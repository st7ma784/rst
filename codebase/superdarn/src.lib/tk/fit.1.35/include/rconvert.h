#ifndef RCONVERT_H
#define RCONVERT_H

#include "rtypes.h"

/* Byte order conversion functions */
int16 ConvertINT2(int16 val);
int32 ConvertINT4(int32 val);
float32 ConvertREAL4(float32 val);
float64 ConvertREAL8(float64 val);

/* Array conversion functions */
void ConvertArrayINT2(int16 *arr, int n);
void ConvertArrayINT4(int32 *arr, int n);
void ConvertArrayREAL4(float32 *arr, int n);
void ConvertArrayREAL8(float64 *arr, int n);

/* Data type conversion */
int16 FloatToInt16(float32 val);
int32 FloatToInt32(float32 val);
float32 Int16ToFloat(int16 val);
float32 Int32ToFloat(int32 val);

/* String conversion */
int StringToInt(const char *str, int *val);
int StringToFloat(const char *str, float32 *val);
int StringToDouble(const char *str, float64 *val);

/* Time conversion */
int TimeStringToEpoch(const char *str, double *epoch);
int EpochToTimeString(double epoch, char *str, int maxlen);

#endif /* RCONVERT_H */
