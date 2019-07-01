#pragma once

#include <string>

typedef unsigned char byte;
using namespace std;

#define DllExport extern "C" __declspec( dllimport )

DllExport void  expert_numAvailableDevices(int * numCUDADevices);									/// Revisado											

DllExport void  expert_resetDevice(int deviceId, int * error);										/// Revisado											

DllExport void  expert_setDevice(int deviceId, int * error);										/// Revisado											

DllExport void  expert_resetAllDevices(int * error);												/// Revisado											

//DllExport string expert_descriptionError(int error);

DllExport void reservationMemory_CopyHostToDeviceOnce(byte * src, int w, int h, int * error);		/// Revisado

DllExport void freeMemory_CopyDeviceToHostOnce(byte * dst, int * error);							/// Revisado

DllExport void reservationMemoryOnce(int w, int h, int * error);									/// Revisado

DllExport void freeMemoryOnce(int * error);															/// Revisado

DllExport void copyHostToDeviceOnce(byte * src, int * error);										/// Revisado

DllExport void copyDeviceToHostOnce(byte * dst, int * error);										/// Revisado

DllExport void dev_threshold_manualOnce(byte min, byte max, int threads, int blocks, int* error);	/// Revisado

DllExport void dev_threshold_automaticoOnce(byte min, byte max, int* error);						/// Revisado

DllExport void dev_erode_manualOnce(int radio, int threads, int blocks, int * error);

DllExport void dev_erode_automaticoOnce(int radio, int * error);

DllExport void dev_erode_twoSteps_manualOnce(int radio, int threads, int blocks, int * error);

DllExport void dev_erode_twoSteps_automaticOnce(int radio, int * error);

DllExport void dev_dilate_manualOnce(int radio, int threads, int blocks, int * error);

DllExport void dev_dilate_automaticOnce(int radio, int * error);

DllExport void dev_dilate_twoSteps_manualOnce(int radio, int threads, int blocks, int * error);

DllExport void dev_dilate_twoSteps_automaticOnce(int radio, int * error);

DllExport void dev_reverseThreshold_manualOnce(byte min, byte max, int threads, int blocks, int * error);

DllExport void dev_reverseThreshold_automaticOnce(byte min, byte max, int * error);

DllExport void dev_open_fast_manualOnce(int radio, int threads, int blocks, int * error);

DllExport void dev_open_fast_automaticOnce(int radio, int * error);

DllExport void dev_close_fast_manualOnce(int radio, int threads, int blocks, int * error);

DllExport void dev_close_fast_automaticOnce(int radio, int * error);

//DllExport void automask(byte * src, byte * dst, byte * dev_src, byte * dev_dst, int size, byte min, byte max, int * error);

