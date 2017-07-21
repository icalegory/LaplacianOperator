/*
 * This is the CUDA implementation of a close approximation of the morphological
 * Laplacian operator edge detection filter, along with other filters discovered
 * by experimentation.  The CUDA SDK sample Box Filter was used as a base to
 * modify and expand on, and the copyright verbage for the code still present
 * is included below as requested by NVIDIA.
 *
 * ——Ian Calegory, 12/20/2016
 */

////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#ifndef _LAPLACIANFILTER_KERNEL_H_
#define _LAPLACIANFILTER_KERNEL_H_

//#include <array>
//#include <cstdlib>
#include "laplacianFilter.h"
#include <helper_math.h>
#include <helper_functions.h>

texture<float, 2> tex;
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
cudaArray *d_array, *d_tempArray;
const int CHANNEL_COUNT = 4;

int disk3x3StructuringElement[] =
{
	0, 1, 0,
	1, 1, 1,
	0, 1, 0
};

int disk5x5StructuringElement[] =
{
	0, 1, 1, 1, 0,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	0, 1, 1, 1, 0
};

int disk7x7StructuringElement[] =
{
	0, 0, 1, 1, 1, 0, 0,
	0, 1, 1, 1, 1, 1, 0,
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1,
	0, 1, 1, 1, 1, 1, 0,
	0, 0, 1, 1, 1, 0, 0
};

int square3x3StructuringElement[] =
{
	1, 1, 1,
	1, 1, 1,
	1, 1, 1
};

int square5x5StructuringElement[] =
{
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1
};

int square7x7StructuringElement[] =
{
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1

};

int ring3x3StructuringElement[] =
{
	0, 1, 0,
	1, 0, 1,
	0, 1, 0
};

int ring5x5StructuringElement[] = 
{
	0, 1, 1, 1, 0,
	1, 0, 0, 0, 1,
	1, 0, 0, 0, 1,
	1, 0, 0, 0, 1,
	0, 1, 1, 1, 0
};

int ring7x7StructuringElement[] =
{
	0, 0, 1, 1, 1, 0, 0,
	0, 1, 0, 0, 0, 1, 0,
	1, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 1,
	0, 1, 0, 0, 0, 1, 0,
	0, 0, 1, 1, 1, 0, 0
};


// C++11 style arrays are not easy to use in device code
//std::array<std::array<int, 3>, 3> disk3x3StructuringElement{ {
//	{ { 0, 1, 0 } },
//	{ { 1, 1, 1 } },
//	{ { 0, 1, 0 } }
//	} };
// Would be nice to be able to use this or something like it:
//auto &structuringElement = disk5x5StructuringElement;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
           ((unsigned int)(rgba.z * 255.0f) << 16) |
           ((unsigned int)(rgba.y * 255.0f) <<  8) |
           ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

extern "C"
void initTexture(int width, int height, void *pImage, bool useRGBA)
{
    int size = width * height * (useRGBA ? sizeof(uchar4) : sizeof(float));

    // copy image data to array
    cudaChannelFormatDesc channelDesc;
    if (useRGBA)
    {
        channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    }
    else
    {
        channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    }
    checkCudaErrors(cudaMallocArray(&d_array, &channelDesc, width, height));
    checkCudaErrors(cudaMemcpyToArray(d_array, 0, 0, pImage, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMallocArray(&d_tempArray,   &channelDesc, width, height));

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = true;

    // Bind the array to the texture
    if (useRGBA)
    {
        checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_array, channelDesc));
    }
    else
    {
        checkCudaErrors(cudaBindTextureToArray(tex, d_array, channelDesc));
    }
}

extern "C"
void freeTextures()
{
    checkCudaErrors(cudaFreeArray(d_array));
    checkCudaErrors(cudaFreeArray(d_tempArray));
}

// This is used to convert tex2D() call results into the byte components
__device__ void convertTexFloatToUChar(uchar4* dst, const float4 src)
{
	//const unsigned int idx = getTextureIndex();
	//_dst[idx].x = (unsigned char)(_src[idx].x * 255.9999f);
	//_dst[idx].y = (unsigned char)(_src[idx].y * 255.9999f);
	//_dst[idx].z = (unsigned char)(_src[idx].z * 255.9999f);
	//_dst[idx].w = (unsigned char)(_src[idx].w * 255.9999f);

	(*dst).x = (unsigned char)(src.x * 255.9999f);
	(*dst).y = (unsigned char)(src.y * 255.9999f);
	(*dst).z = (unsigned char)(src.z * 255.9999f);
	(*dst).w = (unsigned char)(src.w * 255.9999f);
}

/*
Perform 2D morphological Laplacian operator (approximately? along with a number
of variations) on image using CUDA

This works by calculating the dilation and erosion of the image using the structuring
element centered on the current pixel being processed.  It's passed in as the array
d_structuringElement, which is a 2d array flattened into a 1d array for passing into
CUDA with device cudaMemcpyHostToDevice calls.  Dilation is computed by finding the
maximum r, g, and b values for the pixels around the current pixel determined by the
mask of the structuring element.  (If the and of the masking structuring element pixel
and the source image pixel in the corresponding position with the mask overlaid onto
the source image is 1, include that pixel in the source of pixels for choosing maximum
values.)

Erosion is computed similarly, though replacing the source pixel with the components
having the minimum instead of maximum values.

Dilation results in what's called an internal gradient, while erosion results in an
external gradient.  For further reference on computing the internal and external
gradients, see for example http://www.inf.u-szeged.hu/ssip/1996/morpho/morphology.html

The grayscale filter uses the luminosity algorithm for converting to grayscale:
	0.21 R + 0.72 G + 0.07 B

	--Ian Calegory, 12/20/2016

// Comment from original box filter left here for reference--so as a reminder to check
// for coalescence
Note that the x (row) pass suffers from uncoalesced global memory reads,
since each thread is reading from a different row. For this reason it is
better to use texture lookups for the x pass.
The y (column) pass is perfectly coalesced.

Parameters:
id  - pointer to input image in device memory (not used here--texture is used instead)
od  - pointer to destination image in device memory
w   - image width
h   - image height
d_structuringElement - element 0 of the structuring element array
n   - structuring element is nxn matrix

*/
__global__ void
d_laplacianFilter_rgba(unsigned char *id, unsigned char *od, int w, int h, FilterTypeEnum filter, int* d_structuringElement, unsigned int n)
{
	unsigned int colIndex = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int rowIndex = blockIdx.y*blockDim.y + threadIdx.y;

	if (rowIndex < h && colIndex < w) {
		unsigned int index = rowIndex*w*CHANNEL_COUNT + colIndex*CHANNEL_COUNT;
		//if (colIndex > 1085)
		//printf("colIndex, %d, rowIndex, %d, pixelContents, %d, textureContents, %d\n", colIndex, rowIndex, *(id + index), tex2D(rgbaTex, colIndex, rowIndex));
		//printf("w, %d, h, %d, colIndex, %d, rowIndex, %d\n", w, h, colIndex, rowIndex);

		// Convert float4 texture info to uchar4 to extract r, g, b, and a components
		float4 texelCenter = tex2D(rgbaTex, colIndex, rowIndex);
		uchar4 bytesCenterPixel;
		bytesCenterPixel.x = (unsigned char)(texelCenter.x * 255.9999f);
		bytesCenterPixel.y = (unsigned char)(texelCenter.y * 255.9999f);
		bytesCenterPixel.z = (unsigned char)(texelCenter.z * 255.9999f);
		bytesCenterPixel.w = (unsigned char)(texelCenter.w * 255.9999f);
		//printf("r=%d, g=%d, b=%d, a=%d, texel=%d ", bytes.x, bytes.y, bytes.z, bytes.w, texel);
		
		// Now cycle through every pixel of the structuring element, and process
		// both dilation and erosion of the original image.
		unsigned char rMin = 255, gMin = 255, bMin = 255;
		unsigned char rMax = 0, gMax = 0, bMax = 0;
		int maxVert = n / 2;
		// Treat k as the structuring element's x coordinate
		for (int k = -maxVert; k <= maxVert; k++)
		{
			int maxHoriz = n / 2;
			// Treat l as the structuring element's y coordinate
			for (int l = -maxHoriz; l <= maxHoriz; l++)
			{
				// Make sure that the structuring element has a value of 1 in the position being processed,
				// and that the point with the structuring element overlaid is also inside the bounds of the image.
				if (d_structuringElement[(k + maxVert)*n + (l + maxHoriz)] == 1 && rowIndex + k >= 0 && rowIndex + k < h && colIndex + l >= 0 && colIndex + l < w)
				{
					// Determine offset [NOT USED HERE SINCE CUDA VERSION USES TEXTURE INSTEAD OF RAW MEMORY]
					//int offset = k*w*CHANNEL_COUNT + l*CHANNEL_COUNT;

					// Convert float4 texture info to uchar4 to extract r, g, b, and a components
					float4 texel = tex2D(rgbaTex, colIndex + l, rowIndex + k);
					// bytes.x = red, bytes.y = green, bytes.z = blue, bytes.w = alpha
					uchar4 bytes;
					convertTexFloatToUChar(&bytes, texel);
					//printf("r=%d, g=%d, b=%d, a=%d, texel=%d; ", bytes.x, bytes.y, bytes.z, bytes.w, texel);

					// Checks for dilation
					if (bytes.x > rMax)
						rMax = bytes.x;
					if (bytes.y > gMax)
						gMax = bytes.y;
					if (bytes.z > bMax)
						bMax = bytes.z;

					// Checks for erosion
					if (bytes.x < rMin)
						rMin = bytes.x;
					if (bytes.y < gMin)
						gMin = bytes.y;
					if (bytes.z < bMin)
						bMin = bytes.z;

					/*
					// The following method is for raw image memory manipulation by pointers and offsets
					// from the CPU version
					// Checks for dilation
					if ((*(id + index + offset)) > rMax)
						rMax = (unsigned char)(*(id + index + offset));
					if ((*(id + index + offset + 1)) > gMax)
						gMax = (unsigned char)(*(id + index + offset + 1));
					if ((*(id + index + offset + 2)) > bMax)
						bMax = (unsigned char)(*(id + index + offset + 2));

					// Checks for erosion
					if ((*(id + index + offset)) < rMin)
						rMin = (unsigned char)(*(id + index + offset));
					if ((*(id + index + offset + 1)) < gMin)
						gMin = (unsigned char)(*(id + index + offset + 1));
					if ((*(id + index + offset + 2)) < bMin)
						bMin = (unsigned char)(*(id + index + offset + 2));
					*/
				}
			}
		}
		// Filter according to which filter is selected
		switch(filter)
		{
			case(FilterTypeEnum::AlmostAReference):
				// This is very succinct and crisp and clear!  Mostly black, which outlines etched in sharp white
				// THE BEST OUT OF ALL OF THEM -- and, the closest to the reference Laplacian image provided
				*(od + index) = ((rMax + rMin) / 2 - bytesCenterPixel.x) >= 0 ? (unsigned char)((rMax + rMin) / 2 - bytesCenterPixel.x) : 0;
				(*(od + index + 1)) = ((gMax + gMin) / 2 - bytesCenterPixel.y) >= 0 ? (unsigned char)((gMax + gMin) / 2 - bytesCenterPixel.y) : 0;
				(*(od + index + 2)) = ((bMax + bMin) / 2 - bytesCenterPixel.z) >= 0 ? (unsigned char)((bMax + bMin) / 2 - bytesCenterPixel.z) : 0;
				break;

			case(FilterTypeEnum::AlmostFlattened):
				// Looks like very succinct three shades of gray
				// This is a luminosity-type conversion to grayscale
				unsigned char red = (unsigned char)((((rMax + rMin) / 2 - bytesCenterPixel.x)/2 + 255)*0.21);
				unsigned char green = (unsigned char)((((gMax + gMin) / 2 - bytesCenterPixel.y)/2 + 255)*0.72);
				unsigned char blue = (unsigned char)((((bMax + bMin) / 2 - bytesCenterPixel.z)/2 + 255)*0.07);
				//*dst = ((rMax + rMin) / 2 - *index) >= 0 ? red+green+blue : 0;
				//(*(dst + 1)) = ((gMax + gMin) / 2 - (*(index + 1))) >= 0 ? red+green+blue : 0;
				//(*(dst + 2)) = ((bMax + bMin) / 2 - (*(index + 2))) >= 0 ? red+green+blue : 0;
				*(od + index) = red + green + blue;
				(*(od + index + 1)) = red + green + blue;
				(*(od + index + 2)) = red + green + blue;
				break;

			case(FilterTypeEnum::AntiAliasingSmoothFuzz):
				// Excellent and very succinct outlines!  Colorizes to blue and yellow (BUT NOT IN THE
				// CUDA VERSION FOR SOME REASON!!)
				// This is the Laplacian according to http://www.mif.vu.lt/atpazinimas/dip/FIP/fip-Morpholo.html
				// which defines it as ½(dilation+erosion-2*source).  
				// (Wow, the order of operations of the green and blue commands was mistaken in the CPU version,
				// which produced though a really cool filter effect--but oddly does not seem reproducible in
				// this CUDA version!)
				//(*(od + index + 1)) = (unsigned char)(gMax + gMin - 2 * bytesCenterPixel.y / 2);
				//(*(od + index + 2)) = (unsigned char)(bMax + bMin - 2 * bytesCenterPixel.z / 2);
				*(od + index) = (unsigned char)((rMax + rMin - 2*bytesCenterPixel.x)/2);
				(*(od + index + 1)) = (unsigned char)((gMax + gMin - 2* bytesCenterPixel.y)/2);
				(*(od + index + 2)) = (unsigned char)((bMax + bMin - 2* bytesCenterPixel.z)/2);
				break;

			case(FilterTypeEnum::FuzzInWideOutline):
				// This is wrong--used src instead of index, but it produces a unique result--
				// good gray outlines, though rest of image is fuzzy.  Src is the location
				// of the first pixel in the original CPU code, and its behavior is emulated
				// here by getting the texel at the 0,0 position.
				float4 texel2 = tex2D(rgbaTex, 0, 0);
				uchar4 bytes2;
				convertTexFloatToUChar(&bytes2, texel2);
				*(od + index) = ((rMax + rMin) / 2 - bytes2.x) >= 0 ? ((rMax + rMin) / 2 - bytes2.x) : 0;
				(*(od + index + 1)) = ((gMax + gMin) / 2 - bytes2.y) >= 0 ? ((gMax + gMin) / 2 - bytes2.y) : 0;
				(*(od + index + 2)) = ((bMax + bMin) / 2 - bytes2.z) >= 0 ? ((bMax + bMin) / 2 - bytes2.z) : 0;
				break;

			case(FilterTypeEnum::GhostEdges):
				// From imageJ (very similar to the clamping method below found in imageJ)
				*(od + index) = clamp(rMax - rMin + 128, 0, 255);
				(*(od + index + 1)) = clamp(gMax - gMin + 128, 0, 255);
				(*(od + index + 2)) = clamp(bMax - bMin + 128, 0, 255);
				break;

			case(FilterTypeEnum::InvisoWithWideOutlines):
				// Excellent results--mostly black except the outlines
				*(od + index) = ((rMax - rMin) / 2);
				(*(od + index + 1)) = ((gMax - gMin) / 2);
				(*(od + index + 2)) = ((bMax - bMin) / 2);
				break;

			case(FilterTypeEnum::MosaicInGray):
				// Now convert to grayscale using luminosity algorithm.
				// It produces kind of a grayscale mosaic.
				unsigned char red2 = (unsigned char)((rMax + rMin - 2 * bytesCenterPixel.x) / 2) * 0.21;
				// Interesting mistake!! (see order of operations of above compared with below)
				unsigned char green2 = (unsigned char)(gMax + gMin - 2 * bytesCenterPixel.y / 2) * 0.72;
				unsigned char blue2 = (unsigned char)(bMax + bMin - 2 * bytesCenterPixel.z / 2) * 0.07;
				unsigned char gray = red2 + green2 + blue2;
				*(od + index) = gray;
				(*(od + index + 1)) = gray;
				(*(od + index + 2)) = gray;
				break;

			case(FilterTypeEnum::PsychedelicLines):
				// Very similar to psychedelic lines, below
				*(od + index) = (unsigned char)((rMax + rMin) / 2 - bytesCenterPixel.x);
				(*(od + index + 1)) = (unsigned char)((gMax + gMin) / 2 - bytesCenterPixel.y);
				(*(od + index + 2)) = (unsigned char)((bMax + bMin) / 2 - bytesCenterPixel.z);
				break;

			case(FilterTypeEnum::PsychedelicMellowed):
				*(od + index) = ((rMax + rMin) / 2 - bytesCenterPixel.x) >= 0 ? ((rMax + rMin) / 2 - bytesCenterPixel.x) + 128 : 0;
				(*(od + index + 1)) = ((gMax + gMin) / 2 - bytesCenterPixel.y) >= 0 ? ((gMax + gMin) / 2 - bytesCenterPixel.y) + 128 : 0;
				(*(od + index + 2)) = ((bMax + bMin) / 2 - bytesCenterPixel.z) >= 0 ? ((bMax + bMin) / 2 - bytesCenterPixel.z) + 128 : 0;
				break;

			case(FilterTypeEnum::ReliefInGray):
				// Good results, and is very similar to the other SECOND BEST
				*(od + index) = clamp((((rMax + rMin) - 2* bytesCenterPixel.x)/2 + 255)/2, 0, 255);
				(*(od + index + 1)) = clamp((((gMax + gMin) -2* bytesCenterPixel.y)/2 + 255)/2, 0, 255);
				(*(od + index + 2)) = clamp((((bMax + bMin) -2* bytesCenterPixel.z)/2 + 255)/2, 0, 255);
				break;

			// The following filters produce good results, too, but in most cases are similar to the ones above

			// Wow, psychedelic lines!!!
			//*dst = clamp((rMax + rMin) / 2 - *index, 0, 255);
			//(*(dst + 1)) = clamp((gMax + gMin) / 2 - (*(index+1)), 0, 255);
			//(*(dst + 2)) = clamp((bMax + bMin) / 2 - (*(index+2)), 0, 255);

			// Almost a black and white result
			//*dst = (unsigned char)(((rMax + rMin) / 2 - *index) / 2 + 255);
			//(*(dst + 1)) = (unsigned char)(((gMax + gMin) / 2 - (*(index + 1))) / 2 + 255);
			//(*(dst + 2)) = (unsigned char)(((bMax + bMin) / 2 - (*(index + 2))) / 2 + 255);

			// This block will produce a negative of whatever filter is applied before it
			// Now try producing a negative of the Laplacian (or other--whichever is processed immediately
			// before this block), above (should be processed subsequently from it):
			//*dst = 255 - *dst;
			//(*(dst + 1)) = 255 - (*(dst + 1));
			//(*(dst + 2)) = 255 - (*(dst + 2));

			// This clamping mechanism was found in imageJ
			//unsigned char rExternalGradientDilation = clamp(rMax - *index, 0, 255);
			//unsigned char gExternalGradientDilation = clamp(gMax - *(index + 1), 0, 255);
			//unsigned char bExternalGradientDilation = clamp(bMax - *(index + 2), 0, 255);
			//unsigned char rInternalGradientErosion = clamp(rMin - *index, 0, 255);
			//unsigned char gInternalGradientErosion = clamp(gMin - *(index + 1), 0, 255);
			//unsigned char bInternalGradientErosion = clamp(bMin - *(index + 2), 0, 255);
			//*dst = (unsigned char)clamp(rExternalGradientDilation - rInternalGradientErosion + 128, 0, 255);
			//(*(dst + 1)) = (unsigned char)clamp(gExternalGradientDilation - gInternalGradientErosion + 128, 0, 255);
			//(*(dst + 2)) = (unsigned char)clamp(bExternalGradientDilation - bInternalGradientErosion + 128, 0, 255);

			//**** Wow, very good, all gray scale SECOND BEST
			//*dst = ((rMax + rMin) / 2 - *index) / 2 + 128;
			//(*(dst + 1)) = ((gMax + gMin) / 2 - (*(index + 1))) / 2 + 128;
			//(*(dst + 2)) = ((bMax + bMin) / 2 - (*(index + 2))) / 2 + 128;

			// Create luminescent bars
			//*(od + index) = (blockIdx.x*blockDim.x + threadIdx.x) % 256;
			//(*(od + index + 1)) = (blockIdx.x*blockDim.x + threadIdx.x) % 256;
			//(*(od + index + 2)) = (blockIdx.x*blockDim.x + threadIdx.x) % 256;
			//printf("r=%d, g=%d, b=%d; ", rMax, gMax, bMax);

			/*
			unsigned char red = (unsigned char)(((rMax + rMin) / 2 - *index)*0.21);
			unsigned char green = (unsigned char)(((gMax + gMin) / 2 - (*(index + 1)))*0.72);
			unsigned char blue = (unsigned char)(((bMax + bMin) / 2 - (*(index + 2)))*0.07);
			//*dst = ((rMax + rMin) / 2 - *index) >= 0 ? red+green+blue : 0;
			//(*(dst + 1)) = ((gMax + gMin) / 2 - (*(index + 1))) >= 0 ? red+green+blue : 0;
			//(*(dst + 2)) = ((bMax + bMin) / 2 - (*(index + 2))) >= 0 ? red+green+blue : 0;
			*dst = red + green + blue;
			(*(dst + 1)) = red + green + blue;
			(*(dst + 2)) = red + green + blue;
			*/
		}
	}
}

// RGBA version
extern "C"
double laplacianFilterRGBA(unsigned char *d_src, unsigned char *d_temp, unsigned char *d_dest, int width, int height,
	int iterations, int nthreads, StopWatchInterface *timer, StructuringElementEnum element, FilterTypeEnum filter) //int structuringElement[], int size)
{
	// Copy the array containing the structuring element into the device's memory
	// Gotta be an easier way to do this (would be nice if could use C++11 std::array, for example)
	// For some reason passing in the array from the host code doesn't work (see the backtracking
	// involved with the method signature, above)
	unsigned int n = 0;
	int *devArray;
	if (element == StructuringElementEnum::disk3x3)
	{
		n = int(sqrt(sizeof(disk3x3StructuringElement) / sizeof(*disk3x3StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &disk3x3StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	if (element == StructuringElementEnum::disk5x5)
	{
		n = int(sqrt(sizeof(disk5x5StructuringElement) / sizeof(*disk5x5StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &disk5x5StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	if (element == StructuringElementEnum::disk7x7)
	{
		n = int(sqrt(sizeof(disk7x7StructuringElement) / sizeof(*disk7x7StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &disk7x7StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	else if (element == StructuringElementEnum::square3x3)
	{
		n = int(sqrt(sizeof(square3x3StructuringElement) / sizeof(*square3x3StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &square3x3StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	else if (element == StructuringElementEnum::square5x5)
	{
		n = int(sqrt(sizeof(square5x5StructuringElement) / sizeof(*square5x5StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &square5x5StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	else if (element == StructuringElementEnum::square7x7)
	{
		n = int(sqrt(sizeof(square7x7StructuringElement) / sizeof(*square7x7StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &square7x7StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	else if (element == StructuringElementEnum::ring3x3)
	{
		n = int(sqrt(sizeof(ring3x3StructuringElement) / sizeof(*ring3x3StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &ring3x3StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	else if (element == StructuringElementEnum::ring5x5)
	{
		n = int(sqrt(sizeof(ring5x5StructuringElement) / sizeof(*ring5x5StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &ring5x5StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}
	else if (element == StructuringElementEnum::ring7x7)
	{
		n = int(sqrt(sizeof(ring7x7StructuringElement) / sizeof(*ring7x7StructuringElement)));
		checkCudaErrors(cudaMalloc((void**)&devArray, n*n * sizeof(int)));
		checkCudaErrors(cudaMemcpy(devArray, &ring7x7StructuringElement, n*n * sizeof(int), cudaMemcpyHostToDevice));
	}

	checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_array));

	// var for kernel computation timing
	double dKernelTime;

	for (int i = 0; i<iterations; i++)
	{
		// sync host and start kernel computation timer_kernel
		dKernelTime = 0.0;
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);

		// use texture for horizontal pass
		//d_boxfilter_rgba_x << < height / nthreads, nthreads, 0 >> >(d_temp, width, height, 10);
		//d_boxfilter_rgba_y << < width / nthreads, nthreads, 0 >> >(d_temp, d_dest, width, height, 10);

		dim3 dimBlock = dim3(16, 16);
		int yBlocks = width / dimBlock.y + ((width%dimBlock.y) == 0 ? 0 : 1);
		int xBlocks = height / dimBlock.x + ((height%dimBlock.x) == 0 ? 0 : 1);
		dim3 dimGrid = dim3(xBlocks, yBlocks);
		d_laplacianFilter_rgba <<< dimGrid, dimBlock >>>(d_temp, d_dest, width, height, filter, devArray, n);

		// sync host and stop computation timer_kernel
		checkCudaErrors(cudaDeviceSynchronize());
		dKernelTime += sdkGetTimerValue(&timer);

		if (iterations > 1)
		{
			// copy result back from global memory to array
			checkCudaErrors(cudaMemcpyToArray(d_tempArray, 0, 0, d_dest, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_tempArray));
		}
	}

	return ((dKernelTime / 1000.) / (double)iterations);
}

#endif // #ifndef _LAPLACIANFILTER_KERNEL_H_
