/*
 * Morphological Laplacian Operator Edge Detection Filter Example
 *
 * This is based on the Box Filter CUDA SDK sample, which code is copyrighted.
 * See below for the copyright notice.
 *
 * This sample uses CUDA to perform a morphological Laplacian operator 
 * filter on an image and uses OpenGL to display the results.  It
 * processes the image pixel by pixel, in parallel.
 *
 * The image loaded for processing is the file ref_example_upsidedown.ppm
 * in the LaplacianOperator/data folder, so replace the image with another
 * one of the same name to load a custom image.  The image should be
 * a .ppm file, 1920x1080.
 *
 * Ian Calegory, 12/20/2016
 */

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "laplacianFilter.h"

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_cuda_gl.h>    // CUDA device + OpenGL initialization functions

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY     10 //ms

#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10

const static char *sSDKsample = "CUDA Morphological Laplacian Operator Edge Detection Filter";
/*
// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "lenaRGB_14.ppm",
    "lenaRGB_22.ppm",
    NULL
};
const char *sReference[] =
{
    "ref_14.ppm",
    "ref_22.ppm",
    NULL
};
*/

//const char *image_filename = "lenaRGB.ppm";
const char *image_filename = "ref_example_upsidedown.ppm";
FilterTypeEnum currentFilter = FilterTypeEnum::AlmostAReference;
StructuringElementEnum currentStructuringElement = StructuringElementEnum::disk5x5;
int iterations        = 1;
int filter_radius     = 14;
int nthreads          = 64;  // originally 64--can go as high as 1024 (maybe more? but not 2048 apparently)
unsigned int width, height;
unsigned int *h_img  = NULL;
unsigned int *d_img  = NULL;
unsigned int *d_temp = NULL;

GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint texid;   // Texture
GLuint shader;

StopWatchInterface *timer        = NULL,
                    *kernel_timer = NULL;

// Auto-Verification Code
int   fpsCount = 0;        // FPS count for averaging
int   fpsLimit = 8;        // FPS limit for sampling
int   g_Index = 0;
int   g_nFilterSign = 1;
float avgFPS = 0.0f;
unsigned int frameCount     = 0;
unsigned int g_TotalErrors  = 0;
bool         g_bInteractive = false;

int   *pArgc = NULL;
char **pArgv = NULL;

extern "C" int  runSingleTest(char *ref_file, char *exec_path);
extern "C" int  runBenchmark();
extern "C" void loadImageData(int argc, char **argv);
extern "C" void computeGold(float *id, float *od, int w, int h, int n);

// These are CUDA functions to handle allocation and launching the kernels
extern "C" void   initTexture(int width, int height, void *pImage, bool useRGBA);
extern "C" void   freeTextures();

extern "C" double laplacianFilterRGBA(unsigned char *d_src, unsigned char *d_temp, unsigned char *d_dest, int width, int height,
	int iterations, int nthreads, StopWatchInterface *timer, StructuringElementEnum element, FilterTypeEnum filter); //int structuringElement[], int size)

/*
// Function that will take an auto type for an argument.  I created this so that it would
// be possible to pass in a std::array auto variable as an argument to the CUDA call.
// More research is required to effectively use this, since auto can't yet (though perhaps
// will by C++ 17) be passed as a function argument.
template <typename T>
extern void iterateArray(T data, size_t n)
{
	//for (int i = 0; i < data.size(); i++)
	//	for (int j = 0; j < data[i].size(); j++)
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			printf("structuringElement[i][j] = %d\n", data[i][j]);
	d_boxfilter_rgba_y << < width / nthreads, nthreads, 0 >> >(d_temp, d_dest, width, height, radius);

}
// This creates an access exception
void iterateArray(int **a, size_t n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			printf("structuringElement[i][j] = %d\n", a[i][j]);

}
void iterateSingleDimArray(int *a, size_t n)
{
	for (int i = 0; i < n; i++)
		printf("array[i] = %d\n", a[i]);
}
*/

// This varies the filter radius, so we can see automatic animation
void varySigma()
{
    filter_radius += g_nFilterSign;

    if (filter_radius > 64)
    {
        filter_radius = 64; // clamp to 64 and then negate sign
        g_nFilterSign = -1;
    }
    else if (filter_radius < 0)
    {
        filter_radius = 0;
        g_nFilterSign = 1;
    }
}

// Calculate the Frames per second and print in the title bar
void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.0f);
        sdkResetTimer(&timer);
    }

    char fps[256];
    //sprintf(fps, "CUDA Rolling Box Filter <Animation=%s> (radius=%d, passes=%d): %3.1f fps",
    //        (!g_bInteractive ? "ON" : "OFF"), filter_radius, iterations, avgFPS);
	sprintf(fps, "CUDA Morphological Laplacian Operator  Edge Detection Filter, Structuring Element: %s, Filter: %s : %3.1f fps",
		    elementNames[currentStructuringElement], filterTypeNames[currentFilter], avgFPS);
	glutSetWindowTitle(fps);

    if (!g_bInteractive)
    {
        varySigma();
    }
}

// display results using OpenGL
void display()
{
    sdkStartTimer(&timer);

    // execute filter, writing results to pbo
    unsigned int *d_result;


	// Map buffer object for writing from CUDA
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_result, &num_bytes,  cuda_pbo_resource));

	//This is what modifies the PBO:
    //boxFilterRGBA(d_img, d_temp, d_result, width, height, filter_radius, iterations, nthreads, kernel_timer);
	if (true)  // Compute via GPU
	{
		// Call the Laplacian filter
		// Arrays don't seem to pass too well to device code, so pass an enum instead for
		// the structuring element, and deal with it in the device code
		laplacianFilterRGBA((unsigned char*)d_img, (unsigned char*)d_temp, (unsigned char*)d_result, width, height,
			iterations, nthreads, kernel_timer, currentStructuringElement, currentFilter); // , sizeof(disk5x5StructuringElement) / sizeof(*disk5x5StructuringElement));
	}
	else
		// To be able to use the following, d_result will have to be allocated in the host
		// and the result must be transferred to OpenGL.  I have a separate solution that 
		// is computed on the CPU, and time is a constraint, so I won't finish the implementation
		// of having both CPU and GPU methods in one program.  (In this case the output image
		// d_result ought to reside and be displayed from host memory instead of within CUDA.)
		computeGold((float*)h_img, (float*)d_result, width, height, filter_radius);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // OpenGL display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // load texture from pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // fragment program is required to display floating point texture
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(0.0f, 0.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(1.0f, 0.0f);
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(1.0f, 1.0f);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(0.0f, 1.0f);
        }
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
    }

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
	//exit(1);
	//glutDisplayFunc(displayVoid);
}

// Keyboard callback function for OpenGL (GLUT)
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case 27:
            #if defined (__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
            break;

        //case 'a':
        //case 'A':
        //    g_bInteractive = !g_bInteractive;
        //    printf("> Animation is %s\n", !g_bInteractive ? "ON" : "OFF");
        //    break;

        case '=':
        case '+':
            //if (filter_radius < (int)width-1 &&
            //    filter_radius < (int)height-1)
            //{
            //    filter_radius++;
            //}
			currentStructuringElement++;

            break;

        case '-':
		case '_':
            //if (filter_radius > 1)
            //{
            //    filter_radius--;
            //}
			currentStructuringElement--;

            break;

        case ']':
            //iterations++;
			currentFilter++;
            break;

        case '[':
            //if (iterations>1)
            //{
            //    iterations--;
            //}
			currentFilter--;

            break;

        default:
            break;
    }

    //printf("radius = %d, iterations = %d\n", filter_radius, iterations);
	printf(".");
}

// Timer Event so we can refresh the display
void timerEvent(int value)
{
    if(glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

// Resizing the window
void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initCuda(bool useRGBA)
{
    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, (width * height * sizeof(unsigned int))));
    checkCudaErrors(cudaMalloc((void **) &d_temp, (width * height * sizeof(unsigned int))));

    // Refer to boxFilter_kernel.cu for implementation
    initTexture(width, height, h_img, useRGBA);

    sdkCreateTimer(&timer);
    sdkCreateTimer(&kernel_timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
    sdkDeleteTimer(&kernel_timer);

    if (h_img)
    {
        free(h_img);
        h_img=NULL;
    }

    if (d_img)
    {
        cudaFree(d_img);
        d_img=NULL;
    }

    if (d_temp)
    {
        cudaFree(d_temp);
        d_temp=NULL;
    }

    // Refer to boxFilter_kernel.cu for implementation
    freeTextures();

    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texid);
    glDeleteProgramsARB(1, &shader);
}

// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        printf("Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

// This is where we create the OpenGL PBOs, FBOs, and texture resources
void initGLResources()
{
    // create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, h_img, GL_STREAM_DRAW_ARB);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                                 cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    //glutInitWindowSize(768, 768);
	//glutInitWindowSize(1920, 1080);
	//glutInitWindowSize(960, 540);
	glutInitWindowSize(1440, 810);
	glutCreateWindow("CUDA Morphological Laplacian Operator  Edge Detection Filter");
    glutDisplayFunc(display);

    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    if (!isGLVersionSupported(2,0) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        printf("Error: failed to get minimal extensions for demo\n");
        printf("This sample requires:\n");
        printf("  OpenGL version 2.0\n");
        printf("  GL_ARB_vertex_buffer_object\n");
        printf("  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple benchmark test for CUDA
////////////////////////////////////////////////////////////////////////////////
/*
int runBenchmark()
{
    printf("[runBenchmark]: [%s]\n", sSDKsample);

    initCuda(true);

    unsigned int *d_result;
    checkCudaErrors(cudaMalloc((void **)&d_result, width*height*sizeof(unsigned int)));

    // warm-up
    boxFilterRGBA(d_img, d_temp, d_temp, width, height, filter_radius, iterations, nthreads, kernel_timer);
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStartTimer(&kernel_timer);
    // Start round-trip timer and process iCycles loops on the GPU
    iterations = 1;     // standard 1-pass filtering
    const int iCycles = 150;
    double dProcessingTime = 0.0;
    printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);

    for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += boxFilterRGBA(d_img, d_temp, d_img, width, height, filter_radius, iterations, nthreads, kernel_timer);
    }

    // check if kernel execution generated an error and sync host
    getLastCudaError("Error: boxFilterRGBA Kernel execution FAILED");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&kernel_timer);

    // Get average computation time
    dProcessingTime /= (double)iCycles;

    // log testname, throughput, timing and config info to sample and master logs
    printf("boxFilter-texture, Throughput = %.4f M RGBA Pixels/s, Time = %.5f s, Size = %u RGBA Pixels, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * width * height)/dProcessingTime, dProcessingTime,
           (width * height), 1, nthreads);
    printf("\n");

    return 0;
}

// This test specifies a single test (where you specify radius and/or iterations)
int runSingleTest(char *ref_file, char *exec_path)
{
    int nTotalErrors = 0;
    char dump_file[256];

    printf("[runSingleTest]: [%s]\n", sSDKsample);

    initCuda(true);

    unsigned int *d_result;
    unsigned int *h_result = (unsigned int *)malloc(width * height * sizeof(unsigned int));
    checkCudaErrors(cudaMalloc((void **)&d_result, width*height*sizeof(unsigned int)));

    // run the sample radius
    {
        printf("%s (radius=%d) (passes=%d) ", sSDKsample, filter_radius, iterations);
        boxFilterRGBA(d_img, d_temp, d_result, width, height, filter_radius, iterations, nthreads, kernel_timer);

        // check if kernel execution generated an error
        getLastCudaError("Error: boxFilterRGBA Kernel execution FAILED");
        checkCudaErrors(cudaDeviceSynchronize());

        // readback the results to system memory
        cudaMemcpy((unsigned char *)h_result, (unsigned char *)d_result, width*height*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        sprintf(dump_file, "lenaRGB_%02d.ppm", filter_radius);

        sdkSavePPM4ub((const char *)dump_file, (unsigned char *)h_result, width, height);

        if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, 0.15f, false))
        {
            printf("Image is Different ");
            nTotalErrors++;
        }
        else
        {
            printf("Image is Matching ");
        }

        printf(" <%s>\n", ref_file);
    }
    printf("\n");

    free(h_result);
    checkCudaErrors(cudaFree(d_result));

    return nTotalErrors;
}
*/

void loadImageData(int argc, char **argv)
{
    // load image (needed so we can get the width and height before we create the window)
    char *image_path = NULL;

    if (argc >= 1)
    {
        image_path = sdkFindFilePath(image_filename, argv[0]);
    }

    if (image_path == 0)
    {
        printf("Error finding image file '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPPM4(image_path, (unsigned char **) &h_img, &width, &height);

    if (!h_img)
    {
        printf("Error opening file '%s'\n", image_path);
        exit(EXIT_FAILURE);
    }

    printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);
}

bool checkCUDAProfile(int dev, int min_runtime, int min_compute)
{
    int runtimeVersion = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    cudaRuntimeGetVersion(&runtimeVersion);
    fprintf(stderr,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    fprintf(stderr,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

	// Increased printf buffer size to use printf in the kernel
	// to print massive amounts of information to the console for debugging
	// Default printf buffer size is: 1048576 bytes (1MB)
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 534773760);
	// Set buffer size to 10MB
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10485760);
	size_t sizet;
	cudaDeviceGetLimit(&sizet, cudaLimitPrintfFifoSize);
	fprintf(stderr,"  Size of printf buffer    :\t%d\n", sizet);

    if (runtimeVersion >= min_runtime && ((deviceProp.major<<4) + deviceProp.minor) >= min_compute)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findCapableDevice(int argc, char **argv)
{
    int dev;
    int bestDev = -1;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if (deviceCount==0)
    {
        fprintf(stderr,"There are no CUDA capable devices.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        fprintf(stderr,"Found %d CUDA Capable device(s) supporting CUDA\n", deviceCount);
    }

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (checkCUDAProfile(dev, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION))
        {
            fprintf(stderr,"\nFound CUDA Capable Device %d: \"%s\"\n", dev, deviceProp.name);

            if (bestDev == -1)
            {
                bestDev = dev;
                fprintf(stderr, "Setting active device to %d\n", bestDev);
            }
        }
    }

    if (bestDev == -1)
    {
        fprintf(stderr, "\nNo configuration with available capabilities was found.  Test has been waived.\n");
        fprintf(stderr, "The CUDA Sample minimum requirements:\n");
        fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION/16, MIN_COMPUTE_VERSION%16);
        fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION/1000, (MIN_RUNTIME_VERSION%100)/10);
        exit(EXIT_SUCCESS);
    }

    return bestDev;
}

void
printHelp()
{
    printf("boxFilter usage\n");
    printf("    -threads=n (specify the # of of threads to use)\n");
    printf("    -radius=n  (specify the filter radius n to use)\n");
    printf("    -passes=n  (specify the number of passes n to use)\n");
    printf("    -file=name (specify reference file for comparison)\n");
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    int devID = 0;
    char *ref_file = NULL;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    pArgc = &argc;
    pArgv = argv;

    // start logs
    printf("%s Starting...\n\n", argv[0]);

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printHelp();
        exit(EXIT_SUCCESS);
    }

    // load image to process
    loadImageData(argc, argv);

    // Default mode running with OpenGL visualization and in automatic mode
    // the output automatically changes animation
    printf("\n");

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);
    int dev = findCapableDevice(argc, argv);

    if (dev != -1)
    {
        cudaGLSetGLDevice(dev);
    }
    else
    {
        exit(EXIT_SUCCESS);
    }

    // Now we can create a CUDA context and bind it to the OpenGL context
    initCuda(true);
    initGLResources();

    // sets the callback function so it will call cleanup upon exit
#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    printf("Running Standard Demonstration with GLUT loop...\n\n");
    printf("Press '+' and '_' (also '=' and '-') to change structuring element\n"  
            "Press ']' and '[' to change filter type\n"
            //"Press 'a' or  'A' to change animation ON/OFF\n\n"
	);

    // Main OpenGL loop that will run visualization for every vsync
    glutMainLoop();
}
