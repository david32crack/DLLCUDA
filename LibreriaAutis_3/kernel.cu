
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>
#include "kernel.h"


byte* dev_src2 = NULL;
byte* dev_dst2 = NULL;
byte* dev_aux2 = NULL;

int threadsInX;
int threadsInY;
int blocksInX;
int blocksInY;

int width;
int height;
int size_img;



//**************************************************** Funciones GPU **********************************************************//

__global__ void threshold(byte* src, byte* dst, byte min, byte max, int stride, int size)
{
	for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += stride)
		dst[pos] = (src[pos] >= min && src[pos] <= max) ? 1 : 0;
}


__global__ void erode(byte* src, byte* dst, int w, int h, int radio)
{
	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	if (posx >= w || posy >= h)
		return;

	unsigned int start_y = max(posy - radio, 0);
	unsigned int end_y = min(h - 1, posy + radio);
	unsigned int start_x = max(posx - radio, 0);
	unsigned int end_x = min(w - 1, posx + radio);

	int _min = 255;

	for (int y = start_y; y <= end_y; y++)
		for (int x = start_x; x <= end_x; x++)
			_min = min(_min, src[y * w + x]);

	dst[posy * w + posx] = _min;
}

__global__ void erode_separable_step2(byte* src, byte* dst, int w, int h, int radio) {

	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	if (posx >= w || posy >= h)
		return;

	unsigned int start_y = max(posy - radio, 0);
	unsigned int end_y = min(h - 1, posy + radio);

	int _min = 255;
	for (int y = start_y; y <= end_y; y++) {
		_min = min(_min, src[y * w + posx]);
	}
	dst[posy * w + posx] = _min;
}

__global__ void erode_separable_step1(byte* src, byte* dst, int w, int h, int radio) {

	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	if (posx >= w || posy >= h)
		return;

	unsigned int start_x = max(posx - radio, 0);
	unsigned int end_x = min(w - 1, posx + radio);

	int _min = 255;
	for (int x = start_x; x <= end_x; x++) {
		_min = min(_min, src[posy * w + x]);
	}
	dst[posy * w + posx] = _min;

}

__global__ void dilate(byte * src, byte *dst, int w, int h, int radio)
{
	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	if (posx >= w || posy >= h)
		return;

	unsigned int start_y = max(posy - radio, 0);
	unsigned int end_y = min(h - 1, posy + radio);
	unsigned int start_x = max(posx - radio, 0);
	unsigned int end_x = min(w - 1, posx + radio);

	int _max = 0;

	for (int y = start_y; y <= end_y; y++)
		for (int x = start_x; x <= end_x; x++)
			_max = max(_max, src[y * w + x]);

	dst[posy * w + posx] = _max;
}

__global__ void dilate_separable_step2(byte* src, byte* dst, int w, int h, int radio) {

	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	if (posx >= w || posy >= h)
		return;

	unsigned int start_y = max(posy - radio, 0);
	unsigned int end_y = min(h - 1, posy + radio);

	int _max = 0;
	for (int y = start_y; y <= end_y; y++) {
		_max = max(_max, src[y * w + posx]);
	}
	dst[posy * w + posx] = _max;
}

__global__ void dilate_separable_step1(byte* src, byte* dst, int w, int h, int radio) {

	int posx = threadIdx.x + blockIdx.x * blockDim.x;
	int posy = threadIdx.y + blockIdx.y * blockDim.y;

	if (posx >= w || posy >= h)
		return;

	unsigned int start_x = max(posx - radio, 0);
	unsigned int end_x = min(w - 1, posx + radio);

	int _max = 0;
	for (int x = start_x; x <= end_x; x++) {
		_max = max(_max, src[posy * w + x]);
	}
	dst[posy * w + posx] = _max;

}

__global__ void reverseThreshold(byte* src, byte* dst, byte min, byte max, int stride, int size)
{
	for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += stride)
		dst[pos] = (src[pos] >= min && src[pos] <= max) ? 0 : 1;
}





//********************************************************* Llamadas a GPU *************************************************************//

void dev_threshold(byte *src, byte *dst, byte min, byte max, int threads, int blocks, int stride, int size, int* error) {

	cudaError_t cudaStatus;

	threshold << < blocks, threads >> > (src, dst, min, max, stride, size);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		*error = -7;
}

void dev_erode(byte *src, byte *dst, int width, int height, int radio, dim3 threads, dim3 blocks, int* error) {

	cudaError_t cudaStatus;

	erode << < blocks, threads >> > (src, dst, width, height, radio);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		*error = -8;

}

void dev_erode_twoSteps(byte* src, byte* dst, byte* aux, int radio, dim3 threads, dim3 blocks, int* error) {

	cudaError_t cudaStatus;

	erode_separable_step1 << <blocks, threads >> > (src, aux, width, height, radio);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		*error = -10;
		return;
	}
		
	erode_separable_step2 << <blocks, threads >> > (aux, dst, width, height, radio);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
		*error = -10;

}

void dev_dilate(byte *src, byte *dst, int width, int height, int radio, dim3 threads, dim3 blocks, int* error) {

	cudaError_t cudaStatus;

	dilate << < blocks, threads >> > (src, dst, width, height, radio);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		*error = -11;
}

void dev_dilate_twoSteps(byte* src, byte* dst, byte* aux, int radio, dim3 threads, dim3 blocks, int* error) {

	cudaError_t cudaStatus;

	dilate_separable_step1 << <blocks, threads >> > (src, aux, width, height, radio);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		*error = -14;
		return;
	}

	dilate_separable_step2 << <blocks, threads >> > (aux, dst, width, height, radio);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		*error = -14;

}

void dev_reverseThreshold(byte* src, byte* dst, byte min, byte max, int threads, int blocks, int stride, int size, int* error) {

	cudaError_t cudaStatus;

	reverseThreshold << < blocks, threads >> > (src, dst, min, max, stride, size);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		*error = -12;
}




//******************************************************* Herramientas para GPU ********************************************************//


// DETERMINA EL NUMERO DE DEVICES DISPONIBLES
void expert_numAvailableDevices(int *numCUDADevices)
{
	cudaError_t cudaStatus = cudaGetDeviceCount(numCUDADevices);
	if (cudaStatus != cudaSuccess)
		*numCUDADevices = 0;
}

// RESETEA EL DEVICE SELECCIONADO
void  expert_resetDevice(int deviceId, int* error)
{
	int numDevices;
	expert_numAvailableDevices(&numDevices);

	if (deviceId < 0 || deviceId >= numDevices)
		*error = -222;
	else
	{
		cudaError_t cudaStatus = cudaSetDevice(deviceId);

		if (cudaStatus == cudaSuccess)
			cudaStatus = cudaDeviceReset();

		if (cudaStatus == cudaSuccess)
			*error = 0;
		else
			*error = -333;
	}
}

// ESTABLECE EL DEVICE SELECCIONADO 
void  expert_setDevice(int deviceId, int *error)
{
	int numDevices;
	expert_numAvailableDevices(&numDevices);
	if (deviceId < 0 || deviceId >= numDevices)
		*error = -222;
	else
	{
		cudaError_t cudaStatus = cudaSetDevice(deviceId);
		if (cudaStatus == cudaSuccess)
			*error = 0;
		else
			*error = -444;
	}
}

// RESETEA TODOS LOS DEVICES DISPONIBLES
void  expert_resetAllDevices(int* error)
{
	int numDevices;
	expert_numAvailableDevices(&numDevices);

	*error = 0;
	if (numDevices < 1)
		*error = -111;
	else
	{
		for (int dev = 0; dev < numDevices && *error == 0; dev++)
			expert_resetDevice(dev, error);
		expert_setDevice(0, error);
	}
}



/*string expert_descriptionError(int* error) {

	string error_msg;
	switch (*error) {
	case -111:
		error_msg = "Al intentar resetear todos los devices, no había devices";
		break;
	case -222:
		error_msg = "Numero del device metido no esta en el rango";
		break;
	case -333:
		error_msg = "Error en reseteo del device metido";
		break;
	case -444:
		error_msg = "Error en el establecimiento del device";
		break;
	case -1:
		error_msg = "Al intentar reservar memoria para la imagen fuente";
		break;
	case -2:
		error_msg = "Al intentar reservar memoria para la imagen destino";
		break;
	case -3:
		error_msg = "Copia de memoria de CPU a GPU";
		break;
	case -4:
		error_msg = "Copia de memoria de GPU a CPU";
		break;
	case -5:
		error_msg = "Al intentar liberar memoria para la imagen fuente";
		break;
	case -6:
		error_msg = "Al intentar liberar memoria para la imagen destino";
		break;
	case -7:
		error_msg = "Falla en threshold";
		break;
	case -8:
		error_msg = "Falla en Erode Low";
		break;
	case -9:
		error_msg = "Falla en Erode Fast(TwoSteps) al intentar reservar memoria en GPU";
		break;
	case -10:
		error_msg = "Falla en Erode Fast(TwoSteps) al hacer el algoritmo";
		break;
	case -11:
		error_msg = "Falla en Dilate";
		break;
	case -12:
		error_msg = "Falla en ReverseThreshold";
		break;
	case -13:
		error_msg = "Falla en Dilate Fast(TwoSteps) al intentar reservar memoria en GPU";
		break;
	case -14:
		error_msg = "Falla en Dilate Fast(TwoSteps) al hacer algoritmo";
		break;
	case -15:
		error_msg = "Fallo en reescaldo de imagen";
		break;
	default: 
		error_msg = "NO ERROR";
	}
	return error_msg;
}
*/



//************************************************ Determinar Threads Y Bloques *********************************************************//

void setNumberThreads1D(int threadsX, int blocksX, bool automatic) {
	
	if (automatic) {

		threadsInX = 1024;
		blocksInX = 640;  ///////// 65535 / 1024 = 64 * 10

	}
	else {

		if (threadsX == 0 || threadsX > 1024)
			threadsX = 1024;
		if (blocksX == 0 || blocksX > 65535)
			blocksX = 65535;

		threadsInX = threadsX;
		blocksInX = blocksX;
	}

}

void setNumberThreads2D(int threadsX, int threadsY, int blocksX, int blocksY, bool automatic) {

	if (automatic) {

		if (width > 3000) {
			threadsInX = 16;
			blocksInX = 500;
		}
		else if (width > 1600) {
			threadsInX = 8;
			blocksInX = 400;
		}
		else {
			threadsInX = 8;
			blocksInX = 240;
		}

		if (height > 3000){
			threadsInY = 16;
			blocksInY = 500;
		}

		else if (width > 1600) {
			threadsInY = 8;
			blocksInY = 400;
		}

		else {
			threadsInY = 8;
			blocksInY = 240;
		}

	}
	else {

		if (threadsX == 0 || threadsX > 512)
			threadsX = 512;
		if (blocksX == 0 || blocksX > 65535)
			blocksX = 65535;

		threadsInX = threadsInY = threadsX;
		blocksInX = blocksInY = blocksX;

	}
	
	//dim3 grid(width / threadsPerBlock.x, height / threadsPerBlock.y);
}

void setDimensionNumber_Threads_Blocks(int size, int threadsX, int threadsY, int blocksX, int blocksY, bool automatic) {

	if (size > 2 || size <= 0)
		size = 1;
	if (size == 1)
		setNumberThreads1D(threadsX, blocksX, automatic);
		
	else
		setNumberThreads2D(threadsX, threadsY, blocksX, blocksY, automatic);
}




//***************************************************** INTERCAMBIAR ARRAYS *********************************************************//

void swapBuffers(byte** a, byte** b)
{
	byte* aux = *a;
	*a = *b;
	*b = aux;
}


//***********************************************************************************************************************************//
/* ERRORES de procesamiento:
/* 
 *
 * -111: Al intentar resetear todos los devices, no había devices
 * -222: Numero del device metido no esta en el rango
 * -333: Error en reseteo del device metido
 * -444: Error en el establecimiento del device
 *
 *   -1: Al intentar reservar memoria para la imagen fuente
 *   -2: Al intentar reservar memoria para la imagen destino
 *   -3: Copia de memoria de CPU a GPU
 *   -4: Copia de memoria de GPU a CPU
 *   -5: Al intentar liberar memoria para la imagen fuente 
 *   -6: Al intentar liberar memoria para la imagen destino
 *   -7: Falla en threshold
 *   -8: Falla en Erode Low 
 *   -9: Falla en Erode Fast (TwoSteps) al intentar reservar memoria en GPU 
 *  -10: Falla en Erode Fast (TwoSteps) al hacer el algoritmo
 *  -11: Falla en Dilate
 *  -12: Falla en ReverseThreshold
 *  -13: Falla en Dilate Fast (TwoSteps) al intentar reservar memoria en GPU
 *  -14: Falla en Dilate Fast (TwoSteps) al hacer algoritmo
 *  -15: Fallo en reescaldo de imagen
 *  -16: Fallo en Close
 *  -17: Fallo en Open
 *
 */


//************************************ Reserva de memoria en GPU y copia de imagen en GPU ****************************************//

void reservationMemory_CopyHostToDeviceOnce(byte* src, int w, int h, int* error) {

	cudaError_t cudaStatus;

	width = w;
	height = h;
	size_img = width * height;

	cudaDeviceSynchronize();

	//if (dev_src2 == NULL)
	//{
	cudaStatus = cudaMalloc(&dev_src2, size_img);
	if (cudaStatus != cudaSuccess) {
		//printf("Error en reserva de memoria del dev_src");
		*error = -1;
		//dev_src2 = NULL;
		//dev_dst2 = NULL;
		return;
	}

	cudaStatus = cudaMalloc(&dev_dst2, size_img);
	if (cudaStatus != cudaSuccess) {
		//printf("Error en reserva de memoria del dev_dst");
		*error = -2;
		//dev_src2 = NULL;
		//dev_dst2 = NULL;
		cudaFree(dev_src2);
		return;
	}
	//}

	cudaStatus = cudaMemcpy(dev_src2, src, size_img, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		//printf("Error en copia de CPU a GPU");
		*error = -3;

		//dev_src2 = NULL;
		cudaFree(dev_src2);

		//dev_dst2 = NULL;
		cudaFree(dev_dst2);
	}

}
/*void reservationMemory_CopyHostToDeviceMulti(byte* src, byte* dev_src,  byte* dev_dst, int size, int* error) {
	
	cudaError_t cudaStatus;

	*error = 1000;

	if (dev_src == NULL) 
	{
		cudaStatus = cudaMalloc(&dev_src, size);
		if (cudaStatus != cudaSuccess) {
			//printf("Error en reserva de memoria del dev_src");
			*error = -1;
			dev_src = NULL;
			dev_dst = NULL;
			return;
		}

		cudaStatus = cudaMalloc(&dev_dst, size);
		if (cudaStatus != cudaSuccess) {
			//printf("Error en reserva de memoria del dev_dst");
			*error = -2;
			dev_src = NULL;
			dev_dst = NULL;
			cudaFree(dev_src);
		}
	}

	cudaStatus = cudaMemcpy(dev_src, src, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		//printf("Error en copia de CPU a GPU");
		*error = -3;

		dev_src = NULL;
		cudaFree(dev_src);

		dev_dst = NULL;
		cudaFree(dev_dst);
	}		

}*/

//***********************************************************************************************************************************//




//************************************ Liberación de memoria en GPU y copia de imagen en CPU ****************************************//

void freeMemory_CopyDeviceToHostOnce(byte* dst, int* error) {

	cudaError_t cudaStatus;

	*error = 0;

	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(dst, dev_src2, size_img, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		*error = -4;
		//dev_src2 = NULL;
		cudaFree(dev_src2);

		//dev_dst2 = NULL;
		cudaFree(dev_dst2);
		return;
	}

	//dev_src2 = NULL;
	cudaFree(dev_src2);

	//dev_dst2 = NULL;
	cudaFree(dev_dst2);

}
/*void freeMemory_CopyDeviceToHostMulti(byte* dst, byte* dev_src, byte* dev_dst, int size, int* error) {

	cudaError_t cudaStatus;

	*error = 1000;

	cudaStatus = cudaMemcpy(dst, dev_src, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		*error = -4;
		dev_src = NULL;
		cudaFree(dev_src);

		dev_dst = NULL;
		cudaFree(dev_dst);
	}

	dev_src = NULL;
	cudaFree(dev_src);

	dev_dst = NULL;
	cudaFree(dev_dst);

}*/

//***********************************************************************************************************************************//




//************************************************ Reserva de memoria en GPU ********************************************************//

void reservationMemoryOnce(int w, int h, int* error) {

	cudaError_t cudaStatus;

	width = w;
	height = h;
	size_img = width * height;

	//if (dev_src2 == NULL)
	//{
	cudaStatus = cudaMalloc(&dev_src2, size_img);
	if (cudaStatus != cudaSuccess)
	{
		//printf("Error en reserva de memoria");
		*error = -1;
		//dev_src2 = NULL;
		cudaFree(dev_src2);
		//dev_dst2 = NULL;
		return;
	}

	cudaStatus = cudaMalloc(&dev_dst2, size_img);
	if (cudaStatus != cudaSuccess)
	{
		//printf("Error en reserva de memoria");
		*error = -2;
		//dev_src2 = NULL;
		cudaFree(dev_src2);
		//dev_dst2 = NULL;
		cudaFree(dev_dst2);
	}	
	//}

	cudaDeviceSynchronize();

}
/*void reservationMemoryMulti(byte* dev_src, byte* dev_dst, int size, string* error) {

	cudaError_t cudaStatus;

	if (dev_src == NULL)
	{
		cudaStatus = cudaMalloc(&dev_src, size);
		if (cudaStatus != cudaSuccess) 
		{
			dev_src = NULL;
			dev_dst = NULL;
			cudaFree(dev_src);
			goto Error;
		}
		
		cudaStatus = cudaMalloc(&dev_dst, size);
		if (cudaStatus != cudaSuccess) 
		{
			dev_src = NULL;
			dev_dst = NULL;
			cudaFree(dev_src);
			cudaFree(dev_dst);
			goto Error;
		}
		*error = cudaGetErrorName(cudaStatus);
	}

Error:

	//printf("Error en reserva de memoria");
	*error = cudaGetErrorName(cudaStatus);
}*/

//***********************************************************************************************************************************//




//********************************************* Liberacion de memoria en la GPU *****************************************************//

void freeMemoryOnce(int* error) {

	cudaError_t cudaStatus;

	*error = 0;

	

	//dev_src2 = NULL;
	cudaStatus = cudaFree(dev_src2);
	if (cudaStatus != cudaSuccess)
		*error = -5;
	else {
		*error = 10;
	}

	//dev_dst2 = NULL;
	cudaStatus = cudaFree(dev_dst2);
	if (cudaStatus != cudaSuccess)
		*error = -6;
	else {
		*error = 11;
	}

	cudaDeviceSynchronize();

}
/*void freeMemoryMulti(byte* dev_src, byte* dev_dst, string* error) {

	cudaError_t cudaStatus;

	dev_src = NULL;
	dev_dst = NULL;

	cudaStatus = cudaFree(dev_src);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaFree(dev_dst);
	if (cudaStatus != cudaSuccess)
		goto Error;

	*error = cudaGetErrorName(cudaStatus);

Error:
	*error = cudaGetErrorName(cudaStatus);
}*/

//***********************************************************************************************************************************//




//******************************************** Copia de memoria de CPU a GPU ********************************************************//

void copyHostToDeviceOnce(byte* src, int* error) {

	cudaError_t cudaStatus;

	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(dev_src2, src, size_img, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		//printf("Error en copia de CPU a GPU");
		*error = -3;

		//dev_src2 = NULL;
		cudaFree(dev_src2);

		//dev_dst2 = NULL;
		cudaFree(dev_dst2);

	}

	cudaDeviceSynchronize();
}
/*void copyHostToDeviceMulti(byte* src, byte* dev_src, byte* dev_dst, int size, string* error) {

	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(dev_src, src, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	*error = cudaGetErrorName(cudaStatus);

Error:

	//printf("Error en copia de host a device");
	*error = cudaGetErrorName(cudaStatus);

	dev_src = NULL;
	dev_dst = NULL;

	cudaFree(dev_src);
	cudaFree(dev_dst);

}*/

//***********************************************************************************************************************************//




//********************************************** Copia de memoria de GPU a CPU ******************************************************//

void copyDeviceToHostOnce(byte *dst, int* error) {

	cudaError_t cudaStatus;

	*error = 0;

	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(dst, dev_src2, size_img, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		//printf("Error en copia de device a host ");
		*error = -4;

		//dev_src2 = NULL;
		cudaFree(dev_src2);

		//dev_dst2 = NULL;
		cudaFree(dev_dst2);
	}

	cudaDeviceSynchronize();
}
/*void copyDeviceToHostMulti(byte *dst, byte* dev_src, byte* dev_dst, int size, string* error) {

	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(dst, dev_src, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

	*error = cudaGetErrorName(cudaStatus);

Error:

	//printf("Error en copia de device a host ");
	*error = cudaGetErrorName(cudaStatus);

	dev_src = NULL;
	dev_dst = NULL;

	cudaFree(dev_src);
	cudaFree(dev_dst);

}*/


// ************************************************* LLAMADAS A THRESHOLD ***********************************************************//


/////////////////////////////////////////// Le indicas los threads que quieres /////////////////////////////////////////

void dev_threshold_manualOnce(byte min, byte max, int threads, int blocks, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(1, threads, 0, blocks, 0, false);
	int stride = threadsInX * blocksInX;

	dev_threshold(dev_src2, dev_dst2, min, max, threadsInX, blocksInX, stride, size_img, error);
	swapBuffers(&dev_src2, &dev_dst2);
}

/*void dev_threshold_manualMulti(byte* dev_src, byte* dev_dst, int size, byte min, byte max, int threads, int blocks) {

	int stride = threads * blocks;

	if (threads == 0 || threads > 1024)
		threads = 1024;
	if (blocks == 0 || blocks > 65535)
		blocks = 65535;

	dev_threshold(dev_src, dev_dst, min, max, threads, blocks, stride, size);
	swapBuffers(&dev_src, &dev_dst);
}*/

/////////////////////////////////////////// Los threads se calculan de forma automatica /////////////////////////////////////////

void dev_threshold_automaticoOnce(byte min, byte max, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(1, 0, 0, 0, 0, true);
	int stride = threadsInX * blocksInX;

	dev_threshold(dev_src2, dev_dst2, min, max, threadsInX, blocksInX, stride, size_img, error);
	swapBuffers(&dev_src2, &dev_dst2);

}

/*void dev_threshold_automaticoMulti(byte* dev_src, byte* dev_dst, int size, byte min, byte max, int* error) {

	int threads = 1024;
	int blocks = 640;  ///////// 65535 / 1024 = 64 * 10 
	int stride = threads * blocks;

	dev_threshold(dev_src, dev_dst, min, max, threads, blocks, stride, size);
	swapBuffers(&dev_src, &dev_dst);
}*/



//****************************************************** LLAMADAS A ERODE ************************************************************//


// ERODE - Optimo MANUAL 
void dev_erode_manualOnce(int radio, int threads, int blocks, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, threads, threads, blocks, blocks, false);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	dev_erode(dev_src2, dev_dst2, width, height, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

}
/*void dev_erode_manualMulti(byte* dev_src, byte* dev_dst, int width, int height, int radio, int threads, int blocks) {

	if (threads == 0 || threads > 1024 )
		threads = 1024;
	if (blocks == 0 || blocks > 65535)
		blocks = 65535;

	dim3 threadsPerBlock(threads, threads);
	dim3 grid(blocks, blocks);

	//dev_erode(dev_src, dev_dst, width, height, radio, threadsPerBlock, grid);
	swapBuffers(&dev_src, &dev_dst);

}*/


// ERODE - Optimo AUTO 
void dev_erode_automaticoOnce(int radio, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, 0, 0, 0, 0, true);
	
	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);
	
	dev_erode(dev_src2, dev_dst2, width, height, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

}
/*void dev_erode_automaticoMulti(byte* dev_src, byte* dev_dst, int width, int height, int radio) {

	int threads = 32;

	dim3 threadsPerBlock(threads, threads);
	dim3 grid(width / threadsPerBlock.x, height / threadsPerBlock.y);

	//dev_erode(dev_src, dev_dst, width, height, radio, threadsPerBlock, grid);
	swapBuffers(&dev_src, &dev_dst);

}*/


// ERODE + Optimo MANUAL 
void dev_erode_twoSteps_manualOnce(int radio, int threads, int blocks, int* error) {

	cudaError_t cudaStatus;

	cudaDeviceSynchronize();

	cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -8;
		return;
	}

	setDimensionNumber_Threads_Blocks(2, threads, threads, blocks, blocks, false);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	dev_erode_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);

}


// ERODE + Optimo AUTO 
void dev_erode_twoSteps_automaticOnce(int radio, int* error) {

	cudaError_t cudaStatus;

	cudaDeviceSynchronize();

	cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -8;
		return;
	}
	
	setDimensionNumber_Threads_Blocks(2, 0, 0, 0, 0, true);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	dev_erode_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);

}



//****************************************************** LLAMADAS A DILATE ************************************************************//


// DILATE - Optimo MANUAL 
void dev_dilate_manualOnce(int radio, int threads, int blocks, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, threads, threads, blocks, blocks, false);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	dev_dilate(dev_src2, dev_dst2, width, height, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);
}


// DILATE - Optimo AUTO 
void dev_dilate_automaticOnce(int radio, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, 0, 0, 0, 0, true);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	dev_dilate(dev_src2, dev_dst2, width, height, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);
}


// DILATE + Optimo MANUAL 
void dev_dilate_twoSteps_manualOnce(int radio, int threads, int blocks, int* error) {

	cudaError_t cudaStatus;
	
	cudaDeviceSynchronize();
	
	cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -13;
		return;
	}

	setDimensionNumber_Threads_Blocks(2, threads, threads, blocks, blocks, false);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	dev_dilate_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);

}


// DILATE + Optimo AUTO 
void dev_dilate_twoSteps_automaticOnce(int radio, int* error) {

	cudaError_t cudaStatus;

	cudaDeviceSynchronize();

	cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -13;
		return;
	}

	setDimensionNumber_Threads_Blocks(2, 0, 0, 0, 0, true);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	dev_dilate_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);

}





//****************************************************** REVERSE THRESHOLD ************************************************************//

// REVERSE THRESHOLD MANUAL
void dev_reverseThreshold_manualOnce(byte min, byte max, int threads, int blocks, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(1, threads, 0, blocks, 0, false);
	int stride = threadsInX * blocksInX;

	dev_reverseThreshold(dev_src2, dev_dst2, min, max, threadsInX, blocksInY, stride, size_img, error);
	swapBuffers(&dev_src2, &dev_dst2);

}


// REVERSE THRESHOLD AUTO 
void dev_reverseThreshold_automaticOnce(byte min, byte max, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(1, 0, 0, 0, 0, true);
	int stride = threadsInX * blocksInX;

	dev_reverseThreshold(dev_src2, dev_dst2, min, max, threadsInX, blocksInX, stride, size_img, error);
	swapBuffers(&dev_src2, &dev_dst2);
}




//****************************************************** LLAMADAS A OPEN ***************************************************************//

// OPEN + Optimo MANUAL 
void dev_open_fast_manualOnce(int radio, int threads, int blocks, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, threads, threads, blocks, blocks, false);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	cudaError_t cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -17;
		return;
	}
	
	dev_erode_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);
	if (error != 0)
		return;

	dev_dilate_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);
}


// OPEN + Optimo AUTO 
void dev_open_fast_automaticOnce(int radio, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, 0, 0, 0, 0, true);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	cudaError_t cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -17;
		return;
	}

	dev_erode_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);
	if (error != 0)
		return;

	dev_dilate_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);

}




//********************************************************** LLAMADAS A CLOSE **************************************************************//

// CLOSE + Optimo MANUAL 
void dev_close_fast_manualOnce(int radio, int threads, int blocks, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, threads, threads, blocks, blocks, false);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	cudaError_t cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -16;
		return;
	}

	dev_dilate_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);
	if (error != 0)
		return;

	dev_erode_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);

}


// CLOSE + Optimo AUTO 
void dev_close_fast_automaticOnce(int radio, int* error) {

	cudaDeviceSynchronize();
	
	setDimensionNumber_Threads_Blocks(2, 0, 0, 0, 0, true);

	dim3 threadsPerBlock(threadsInX, threadsInY);
	dim3 grid(blocksInX, blocksInY);

	cudaError_t cudaStatus = cudaMalloc(&dev_aux2, size_img);
	if (cudaStatus != cudaSuccess) {
		*error = -16;
		return;
	}

	dev_dilate_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);
	if (error != 0)
		return;

	dev_erode_twoSteps(dev_src2, dev_dst2, dev_aux2, radio, threadsPerBlock, grid, error);
	swapBuffers(&dev_src2, &dev_dst2);

	cudaFree(dev_aux2);

}





//************************************************************ AUTOMASK **************************************************************//

void automask(byte *src, byte* dst, byte * dev_src, byte * dev_dst, int size, byte min, byte max, int *error) {

	cudaError_t cudaSTATUS;

	*error = 1000;

	if (dev_src == NULL) {

		cudaSTATUS = cudaMalloc(&dev_src, size);
		if (cudaSTATUS != cudaSuccess) {
			*error = -1;
			dev_src = NULL;
			dev_dst = NULL;
		}

		cudaSTATUS = cudaMalloc(&dev_dst, size);
		if (cudaSTATUS != cudaSuccess) {
			*error = -2;
			dev_src = NULL;
			dev_dst = NULL;
			cudaFree(dev_src);
		}
	}
	

	cudaSTATUS = cudaMemcpy(dev_src, src, size, cudaMemcpyHostToDevice);
	if (cudaSTATUS != cudaSuccess) {
		*error = -3;
		goto Error;
	}

	int threads = 1024;
	int blocks = 640;  ///////// 65535 / 1024 = 64 * 10 
	int stride = threads * blocks;

	dev_threshold(dev_src, dev_dst, min, max, threads, blocks, stride, size, error);
	swapBuffers(&dev_src, &dev_dst);

	cudaSTATUS = cudaMemcpy(dst,dev_src, size, cudaMemcpyDeviceToHost);
	if (cudaSTATUS != cudaSuccess) {
		*error = -4;
		goto Error;
	}

	//dst = dev_src;
	dev_src = NULL;
	cudaFree(dev_src);

	dev_dst = NULL;
	cudaFree(dev_dst);
	
Error:
	dev_src = NULL;
	cudaFree(dev_src);

	dev_dst = NULL;
	cudaFree(dev_dst);

}




//************************************************************ ERRORES **************************************************************//

/*
else if (cudaError == cudaErrorInvalidDevicePointer)
		*error = -3;
	else if (cudaError == cudaErrorInvalidMemcpyDirection)
		*error = -4;
	else if (cudaError == cudaErrorInvalidValue)
		*error = -5;

else if (cudaSTATUS == cudaErrorUnsupportedLimit) {
		*error = -21;
		goto Error;
	}
else if (cudaSTATUS == cudaErrorDuplicateVariableName) {
		*error = -22;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorDuplicateTextureName) {
		*error = -23;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorDuplicateSurfaceName) {
		*error = -24;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorDevicesUnavailable) {
		*error = -25;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorInvalidKernelImage) {
		*error = -26;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorNoKernelImageForDevice) {
		*error = -27;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorIncompatibleDriverContext) {
		*error = -28;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorPeerAccessAlreadyEnabled) {
		*error = -29;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorPeerAccessNotEnabled) {
		*error = -30;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorDeviceAlreadyInUse) {
		*error = -31;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorProfilerDisabled) {
		*error = -32;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorProfilerNotInitialized) {
		*error = -33;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorProfilerAlreadyStarted) {
		*error = -34;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorProfilerAlreadyStopped) {
		*error = -35;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorStartupFailure) {
		*error = -36;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorApiFailureBase) {
		*error = -37;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorInvalidSurface) {
		*error = -38;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorNoDevice) {
		*error = -39;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorECCUncorrectable) {
		*error = -40;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorSharedObjectSymbolNotFound) {
		*error = -41;
		goto Error;
	}
	else if (cudaSTATUS == cudaErrorSharedObjectInitFailed) {
		*error = -42;
		goto Error;
	}
	else
	{
		*error = -1000001;
		goto Error;
	}
*/

