#include "include/traj.h"
#include <math.h>
#include <complex>
// #include <fftw3.h>
#include "include/vkFFT.h"
#include "utils_VkFFT.h"
// Shorthand for cleaner code
/*const size_t N0 = n_space_divx2, N1 = n_space_divy2, N2 = n_space_divz2,
             N0N1 = N0 * N1, N0N1_2 = N0N1 / 2,
             N2_c = N2 / 2 + 1;         // Dimension to store the complex data, as required by fftw (from their docs)
const size_t n_cells4 = N0 * N1 * N2_c; // NOTE: This is not actually n_cells * 4, there is an additional buffer that fftw requires.
*/
/*
The flow of this function is as follows, note that the FFT we are using converts real to complex, and the IFFT converts complex to real.
This is because we are making use of the fact that our original data is fully reall, and this saves about half the memory/computation
    Precalculation step on first run:
    precalc_r3_base -> FFT -> precalc_r3

    npt (number of particles per cell) -> copy -> FFT(planforE) -> fft_complex
    fft_complex *= precalc_r3, do this three times for three dimensions
    fft_complex -> IFFT(planbacE) -> fft_real
    E (input) = fft_real + Ee (background constant E also input)

    jc -> copy -> FFT(planforB) -> fft_complex
    fft_complex *= precalc_r3
    fft_complex -> IFFT(planbacB) -> fft_real
    B (input) = fft_real + Be (background constant E also input)

    Bonus: V potential. This is ran in parallel with E.
    npt -> copy -> FFT(planforE) -> fft_complex (reuse)
    fft_complex *= precalc_r2
    fft_complex -> IFFT(planbacV) -> fft_real
    V (input) = fft_real

    Arrays for fft, output is multiplied by 2 because the convolution pattern should be double the size
    (a particle can be both 64 cells up, or 64 cells down, so we need 128 cells to calculate this information)
    TODO should use native VkFFT to (i) zero pad, (ii) not re order, (iii) convolve
*/

int calcEBV(fields *fi, par *par)
{
    //  static fftwf_plan planforE, planforB, planbacE, planbacB;
    static int first = 1;
    static auto *fft_real = static_cast<float(*)[n_cells8]>(_aligned_malloc(sizeof(float) * n_cells8 * 4, 4096));                      // fft_real[4][n_cells8]
    static auto *fft_complex = static_cast<complex<float>(*)[n_cells4]>(_aligned_malloc(sizeof(complex<float>) * n_cells4 * 4, 4096)); // fft_complex[4][n_cells4]
    //  pre-calculate 1/ r3 to make it faster to calculate electric and magnetic fields
    static auto *precalc_r3 = static_cast<complex<float>(*)[3][n_cells4]>(_aligned_malloc(sizeof(complex<float>) * 2 * 3 * n_cells4, 4096)); // precalc_r3[2][3][n_cells4]
#ifdef Uon_                                                                                                                                  // similar arrays for U, but kept separately in one ifdef
    static auto *precalc_r2 = static_cast<complex<float>(*)>(_aligned_malloc(sizeof(complex<float>) * n_cells4, 4096));                      // precalc_r3[n_cells4]
#endif
    static const size_t n_4 = n_cells / 4;
    static auto EUtot = new float[n_4];
    static float posL2[3];
    static cl_kernel copyData_kernel;
    static cl_kernel copy3Data_kernel;
    static cl_kernel NxPrecalc_kernel;
    static cl_kernel NxPrecalcnU_kernel;
    static cl_kernel NxPrecalcr2_kernel;
    static cl_kernel jcxPrecalc_kernel;
    static cl_kernel sumFftFieldo_kernel;
    static cl_kernel sumFftField_kernel;
    static cl_kernel sumFftSField_kernel;
    static cl_kernel copyextField_kernel;
    static cl_kernel EUEst_kernel;
    static cl_kernel maxvalf_kernel;

    static VkFFTApplication app1 = {};
    static VkFFTApplication app3 = {};
    static VkFFTApplication appbac3 = {};
    static VkFFTApplication appbac4 = {};
    static VkFFTApplication appfor_k = {};
    static VkFFTApplication appfor_k2 = {};

    static cl_mem r3_buffer = 0;
    static cl_mem r2_buffer = 0;
    static cl_mem fft_real_buffer = 0;
    static cl_mem fft_complex_buffer = 0;
    static cl_mem fft_p_buffer = 0;
    static cl_mem EUtot_buffer = 0;
    static cl_mem maxval_buffer = 0;

    static VkGPU vkGPU = {};
    // vkGPU.device_id = 0; // 0 = use iGPU for FFT
    vkGPU.device_id = device_id_g; // use same GPU as motion code
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;

    static uint64_t bufferSize_R = (uint64_t)sizeof(float) * n_cells8;          // buffer size per batch Real
    static uint64_t bufferSize_C = (uint64_t)sizeof(complex<float>) * n_cells4; // buffer size per batch Complex
    static uint64_t bufferSize_P = (uint64_t)sizeof(float) * n_cells4 * 2;      // buffer size per batch Complex
    static uint64_t bufferSize_R3 = bufferSize_R * 3;
    static uint64_t bufferSize_P3 = bufferSize_P * 3;
    static uint64_t bufferSize_C3 = bufferSize_C * 3;
    static uint64_t bufferSize_R4 = bufferSize_R * 4;
    static uint64_t bufferSize_P4 = bufferSize_P * 4;
    static uint64_t bufferSize_C4 = bufferSize_C * 4;
    static uint64_t bufferSize_R6 = bufferSize_R * 6;
    // static uint64_t bufferSize_P6 = bufferSize_P * 6;
    static uint64_t bufferSize_C6 = bufferSize_C * 6;
    static VkFFTLaunchParams launchParams = {};

    int Nbatch;
    if (first)
    { // allocate and initialize to 0
        int dims[3] = {N0, N1, N2};
        auto precalc_r3_base = new float[2][3][N2][N1][N0];
        fi->precalc_r3 = (reinterpret_cast<float *>(precalc_r3));
#ifdef Uon_ // similar arrays for U, but kept separately in one ifdef
        auto precalc_r2_base = new float[N2][N1][N0];
        fi->precalc_r2 = (reinterpret_cast<float *>(precalc_r2));
#endif

        vkGPU.device = default_device_g();
        vkGPU.context = context_g();
        // vkGPU. = program_g();
        vkGPU.commandQueue = clCreateCommandQueue(vkGPU.context, vkGPU.device, 0, &res);
        launchParams.commandQueue = &vkGPU.commandQueue;
        // Create the OpenCL kernel
        copyData_kernel = clCreateKernel(program_g(), "copyData", NULL);
        copy3Data_kernel = clCreateKernel(program_g(), "copy3Data", NULL);
        NxPrecalc_kernel = clCreateKernel(program_g(), "NxPrecalc", NULL);
        NxPrecalcr2_kernel = clCreateKernel(program_g(), "NxPrecalcr2", NULL);
        jcxPrecalc_kernel = clCreateKernel(program_g(), "jcxPrecalc", NULL);
#ifdef octant
        sumFftField_kernel = clCreateKernel(program_g(), "sumFftFieldo", NULL); // want rollover fields in x,y,z
#else
#ifdef quadrant
        sumFftField_kernel = clCreateKernel(program_g(), "sumFftFieldq", NULL); // want rollover fields in x,y no z
#else
        sumFftField_kernel = clCreateKernel(program_g(), "sumFftField", NULL);
#endif
#endif
        sumFftSField_kernel = clCreateKernel(program_g(), "sumFftSField", NULL);
        copyextField_kernel = clCreateKernel(program_g(), "copyextField", NULL);
        EUEst_kernel = clCreateKernel(program_g(), "EUEst", NULL);
        maxvalf_kernel = clCreateKernel(program_g(), "maxvalf", NULL);

        fft_real_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_R4, 0, &res);
        fft_complex_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_C4, 0, &res);
        fft_p_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_P4, 0, &res);
        cl_mem r3_base_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_R6, 0, &res);
        cl_mem r2_base_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_R, 0, &res);
        r3_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_C6, 0, &res);
        r2_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_C, 0, &res);
        // Create memory buffers on the device for each vector
        EUtot_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, n_4 * sizeof(float), 0, &res);

        // cl::Buffer npt_buffer (context_g, CL_MEM_READ_WRITE, sizeof(float) * n_cells);
        // npt_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, sizeof(float) * n_cells, 0, &res);
        // jc_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, sizeof(float) * n_cells * 3, 0, &res);
        VkFFTConfiguration configuration = {};
        VkFFTApplication appfor_k = {};

        configuration.device = &vkGPU.device;
        configuration.context = &vkGPU.context;

        configuration.FFTdim = 3; // FFT dimension, 1D, 2D or 3D (default 1).4 D configuration.size[0] = Nbatch;

        configuration.size[0] = N0; // Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.
        configuration.size[1] = N1;
        configuration.size[2] = N2;

        configuration.performR2C = true;
        configuration.disableReorderFourStep = true; // disable reordering =true false

        configuration.isInputFormatted = 1; // out-of-place - we need to specify that input buffer is separate from the main buffer

        // Strides for R2C for forward plans
        configuration.inputBufferStride[0] = configuration.size[0];
        configuration.inputBufferStride[1] = configuration.inputBufferStride[0] * configuration.size[1];
        configuration.inputBufferStride[2] = configuration.inputBufferStride[1] * configuration.size[2];

        configuration.bufferStride[0] = (uint64_t)(configuration.size[0] / 2) + 1;
        configuration.bufferStride[1] = configuration.bufferStride[0] * configuration.size[1];
        configuration.bufferStride[2] = configuration.bufferStride[1] * configuration.size[2];

        // plan for precalc
        configuration.makeForwardPlanOnly = true;
        configuration.numberBatches = 6;
        configuration.inputBuffer = &r3_base_buffer;
        configuration.inputBufferSize = &bufferSize_R6;
        configuration.buffer = &r3_buffer;
        configuration.bufferSize = &bufferSize_C6;
        resFFT = initializeVkFFT(&appfor_k, configuration);

#ifdef Uon_
        configuration.makeForwardPlanOnly = true;
        configuration.numberBatches = 1;
        configuration.inputBuffer = &r2_base_buffer;
        configuration.inputBufferSize = &bufferSize_R;
        configuration.buffer = &r2_buffer;
        configuration.bufferSize = &bufferSize_C;
        resFFT = initializeVkFFT(&appfor_k2, configuration);
#endif

        // plan for density (forward R2C FFT scalar)
        configuration.numberBatches = 1;
        configuration.performZeropadding[0] = false;
        configuration.performZeropadding[1] = false;
        configuration.performZeropadding[2] = false;
        // configuration.performZeropadding[0] = true;
        // configuration.performZeropadding[1] = true;
        // configuration.performZeropadding[2] = true;

        configuration.frequencyZeroPadding = false; // true
        configuration.fft_zeropad_left[0] = (uint64_t)ceil(configuration.size[0] / 2.0);
        configuration.fft_zeropad_right[0] = configuration.size[0];
        configuration.fft_zeropad_left[1] = (uint64_t)ceil(configuration.size[1] / 2.0);
        configuration.fft_zeropad_right[1] = configuration.size[1];
        configuration.fft_zeropad_left[2] = (uint64_t)ceil(configuration.size[2] / 2.0);
        configuration.fft_zeropad_right[2] = configuration.size[2];

        configuration.fft_zeropad_left[0] = 0;
        configuration.fft_zeropad_right[0] = (uint64_t)ceil(configuration.size[0] / 2.0);
        configuration.fft_zeropad_left[1] = 0;
        configuration.fft_zeropad_right[1] = (uint64_t)ceil(configuration.size[1] / 2.0);
        configuration.fft_zeropad_left[2] = 0;
        configuration.fft_zeropad_right[2] = (uint64_t)ceil(configuration.size[2] / 2.0);

        configuration.inputBuffer = &fft_real_buffer;
        configuration.inputBufferSize = &bufferSize_R;
        configuration.buffer = &fft_complex_buffer;
        configuration.bufferSize = &bufferSize_C;
        resFFT = initializeVkFFT(&app1, configuration);

        // plan for current density (forward R2C FFT 3-vector)
        configuration.isOutputFormatted = true;
        configuration.inverseReturnToInputBuffer = false;
        configuration.makeForwardPlanOnly = true;

        configuration.performZeropadding[0] = false;
        configuration.performZeropadding[1] = false;
        configuration.performZeropadding[2] = false;
        //  configuration.performZeropadding[0] = true;
        //  configuration.performZeropadding[1] = true;
        //  configuration.performZeropadding[2] = true;

        configuration.outputBufferStride[0] = (uint64_t)(configuration.size[0] / 2) + 1;
        configuration.outputBufferStride[1] = configuration.outputBufferStride[0] * configuration.size[1];
        configuration.outputBufferStride[2] = configuration.outputBufferStride[1] * configuration.size[2];

        configuration.numberBatches = 3;
        configuration.inputBuffer = &fft_real_buffer;
        configuration.inputBufferSize = &bufferSize_R3;
        configuration.buffer = &fft_p_buffer;
        configuration.bufferSize = &bufferSize_P3;
        configuration.outputBuffer = &fft_complex_buffer; // outputBuffer
        configuration.outputBufferSize = &bufferSize_C3;  // outputBufferSize
        resFFT = initializeVkFFT(&app3, configuration);

        // plan for E and B field  inverse C2R FFT (3-vector)
        configuration.isOutputFormatted = true;
        configuration.inverseReturnToInputBuffer = false;
        configuration.makeForwardPlanOnly = false;
        configuration.makeInversePlanOnly = true;
        configuration.performZeropadding[0] = false;
        configuration.performZeropadding[1] = false;
        configuration.performZeropadding[2] = false;
        configuration.inputBufferStride[0] = (uint64_t)(configuration.size[0] / 2) + 1;
        configuration.inputBufferStride[1] = configuration.inputBufferStride[0] * configuration.size[1];
        configuration.inputBufferStride[2] = configuration.inputBufferStride[1] * configuration.size[2];

        configuration.outputBufferStride[0] = configuration.size[0];
        configuration.outputBufferStride[1] = configuration.outputBufferStride[0] * configuration.size[1];
        configuration.outputBufferStride[2] = configuration.outputBufferStride[1] * configuration.size[2];

        configuration.numberBatches = 3;
        configuration.outputBuffer = &fft_real_buffer;
        configuration.outputBufferSize = &bufferSize_R3;
        configuration.buffer = &fft_p_buffer;
        configuration.bufferSize = &bufferSize_P3;
        configuration.inputBuffer = &fft_complex_buffer;
        configuration.inputBufferSize = &bufferSize_C3;
        resFFT = initializeVkFFT(&appbac3, configuration);

#ifdef Uon_
        // Perform ifft on the entire array; the first 3/4 is used for E while the last 1/4 is used for V
        configuration.isOutputFormatted = true;
        configuration.inverseReturnToInputBuffer = false;
        configuration.makeForwardPlanOnly = false;
        configuration.makeInversePlanOnly = true;
        configuration.performZeropadding[0] = false;
        configuration.performZeropadding[1] = false;
        configuration.performZeropadding[2] = false;

        configuration.numberBatches = 4;
        configuration.outputBuffer = &fft_real_buffer;
        configuration.outputBufferSize = &bufferSize_R4;
        configuration.buffer = &fft_p_buffer;
        configuration.bufferSize = &bufferSize_P4;
        configuration.inputBuffer = &fft_complex_buffer;
        configuration.inputBufferSize = &bufferSize_C4;

        resFFT = initializeVkFFT(&appbac4, configuration);
#endif

#pragma omp barrier
        //       cout << "allocate done\n";
        float r3, rx, ry, rz, rx2, ry2, rz2;
        int i, j, k, loc_i, loc_j, loc_k;
        posL2[0] = -par->dd[0] * ((float)n_space_divx - 0.5);
        posL2[1] = -par->dd[1] * ((float)n_space_divy - 0.5);
        posL2[2] = -par->dd[2] * ((float)n_space_divz - 0.5);

// precalculate 1/r^3 (field) and 1/r^2 (energy)
#pragma omp parallel for simd num_threads(nthreads)
        for (k = -n_space_divz; k < n_space_divz; k++)
        {
            loc_k = k + (k < 0 ? n_space_divz2 : 0); // The "logical" array position
            // We wrap around values smaller than 0 to the other side of the array, since 0, 0, 0 is defined as the center of the convolution pattern an hence rz should be 0
            rz = k * par->dd[2]; // The change in z coordinate for the k-th cell.
            rz2 = rz * rz;
            for (j = -n_space_divy; j < n_space_divy; j++)
            {
                loc_j = j + (j < 0 ? n_space_divy2 : 0);
                ry = j * par->dd[1];
                ry2 = ry * ry + rz2;
                for (i = -n_space_divx; i < n_space_divx; i++)
                {
                    loc_i = i + (i < 0 ? n_space_divx2 : 0);
                    rx = i * par->dd[0];
                    rx2 = rx * rx + ry2;
                    r3 = rx2 == 0 ? 0.f : powf(rx2, -1.5);
                    precalc_r3_base[0][0][loc_k][loc_j][loc_i] = r3 * rx;
                    precalc_r3_base[0][1][loc_k][loc_j][loc_i] = r3 * ry;
                    precalc_r3_base[0][2][loc_k][loc_j][loc_i] = r3 * rz;
                    precalc_r3_base[1][0][loc_k][loc_j][loc_i] = precalc_r3_base[0][0][loc_k][loc_j][loc_i];
                    precalc_r3_base[1][1][loc_k][loc_j][loc_i] = precalc_r3_base[0][1][loc_k][loc_j][loc_i];
                    precalc_r3_base[1][2][loc_k][loc_j][loc_i] = precalc_r3_base[0][2][loc_k][loc_j][loc_i];
#ifdef Uon_
                    precalc_r2_base[loc_k][loc_j][loc_i] = rx2 == 0 ? 0.f : powf(rx2, -0.5);
#endif
                }
            }
        }
        // Multiply by the respective constants here, since it is faster to parallelize it
        const float Vconst = kc * e_charge * r_part_spart / n_cells8;
        const float Aconst = 1e-7 * e_charge * r_part_spart / n_cells8;
//    const size_t n_cells4 = n_space_divx2 * n_space_divy2 * (n_space_divz2 / 2 + 1); // NOTE: This is not actually n_cells * 4, there is an additional buffer that fftw requires.
#pragma omp parallel for simd num_threads(nthreads)
        for (size_t i = 0; i < n_cells8 * 3; i++)
            reinterpret_cast<float *>(precalc_r3_base[0])[i] *= Vconst;
#pragma omp parallel for simd num_threads(nthreads)
        for (size_t i = 0; i < n_cells8 * 3; i++)
            reinterpret_cast<float *>(precalc_r3_base[1])[i] *= Aconst;
#ifdef Uon_
#pragma omp parallel for simd num_threads(nthreads)
        for (size_t i = 0; i < n_cells8; i++)
            (reinterpret_cast<float *>(precalc_r2_base))[i] *= Vconst;
#endif
        resFFT = transferDataFromCPU(&vkGPU, precalc_r3_base, &r3_base_buffer, bufferSize_R6);
        resFFT = VkFFTAppend(&appfor_k, -1, &launchParams); //   cout << "forward transform precalc_r3" << endl;
        res = clFinish(vkGPU.commandQueue);
        deleteVkFFT(&appfor_k);
        resFFT = transferDataToCPU(&vkGPU, precalc_r3, &r3_buffer, bufferSize_C6);
        clReleaseMemObject(r3_base_buffer);
        delete[] precalc_r3_base;
#ifdef Uon_
        resFFT = transferDataFromCPU(&vkGPU, precalc_r2_base, &r2_base_buffer, bufferSize_R);
        resFFT = VkFFTAppend(&appfor_k2, -1, &launchParams);
        res = clFinish(vkGPU.commandQueue);
        deleteVkFFT(&appfor_k2);
        resFFT = transferDataToCPU(&vkGPU, precalc_r2, &r2_buffer, bufferSize_C);
        clReleaseMemObject(r2_base_buffer);
        delete[] precalc_r2_base;
#endif

        //      cout << "filter" << endl; // filter
        /*
#pragma omp parallel for simd num_threads(nthreads)
        for (k = 0; k < n_space_divz; k++)
        {
            loc_k = k + (k < 0 ? n_space_divz2 : 0); // The "logical" array position
                                                     //     cout << loc_k << " ";
            // We wrap around values smaller than 0 to the other side of the array, since 0, 0, 0 is defined as the center of the convolution pattern an hence rz should be 0
            rz = k; // The change in z coordinate for the k-th cell.
            rz2 = rz * rz;
            for (j = -n_space_divy; j < n_space_divy; j++)
            {
                loc_j = j + (j < 0 ? n_space_divy2 : 0);
                ry = j;
                ry2 = ry * ry + rz2;
                for (i = 0; i <= n_space_divx; i++)
                {
                    loc_i = i + (i < 0 ? n_space_divx2 : 0);
                    rx = i;
                    rx2 = rx * rx + ry2;
                    float r = pi * sqrt(rx2) / R_s;
                    float w = r > pi / 2 ? 0.f : cos(r);
                    w *= w;
                    for (int c = 0; c < 3; c++)
                    {
                        precalc_r3[0][c][loc_k * N1N0_c + loc_j * N0_c + loc_i] *= w;
                        precalc_r3[1][c][loc_k * N1N0_c + loc_j * N0_c + loc_i] *= w;
                    }
#ifdef Uon_
                    precalc_r2[loc_k * N1N0_c + loc_j * N0_c + loc_i] *= w;
#endif
                }
            }
        }
        */
        // Set the arguments of the kernel
        clSetKernelArg(copyData_kernel, 0, sizeof(cl_mem), &fi->npt_buffer);
        clSetKernelArg(copyData_kernel, 1, sizeof(cl_mem), &fft_real_buffer);

        clSetKernelArg(copy3Data_kernel, 0, sizeof(cl_mem), &fi->jc_buffer);
        clSetKernelArg(copy3Data_kernel, 1, sizeof(cl_mem), &fft_real_buffer);

        clSetKernelArg(NxPrecalc_kernel, 0, sizeof(cl_mem), &r3_buffer);
        clSetKernelArg(NxPrecalc_kernel, 1, sizeof(cl_mem), &fft_complex_buffer);

        clSetKernelArg(NxPrecalcr2_kernel, 0, sizeof(cl_mem), &r2_buffer);
        clSetKernelArg(NxPrecalcr2_kernel, 1, sizeof(cl_mem), &r3_buffer);
        clSetKernelArg(NxPrecalcr2_kernel, 2, sizeof(cl_mem), &fft_complex_buffer);

        clSetKernelArg(jcxPrecalc_kernel, 0, sizeof(cl_mem), &r3_buffer);
        clSetKernelArg(jcxPrecalc_kernel, 1, sizeof(cl_mem), &fft_complex_buffer);

        first = 0; //      cout << "precalc done\n";
    }

#ifdef Eon_
    {
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copyData_kernel, 1, NULL, &n_cells8, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        res = clFinish(vkGPU.commandQueue);
        //  only density arrn1 = fft(arrn) multiply fft charge with fft of kernel(i.e field associated with 1 charge)
        resFFT = VkFFTAppend(&app1, -1, &launchParams); // -1 = forward transform
        res = clFinish(vkGPU.commandQueue);             //  cout << "execute plan for E" << endl;

#ifdef Uon_
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, NxPrecalcr2_kernel, 1, NULL, &n_cells4, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&appbac4, 1, &launchParams);                                                              // cout << "inverse transform to get convolution" << endl;                                                             // 1 = inverse FFT//if (resFFT)                cout << "execute plan bac E resFFT = " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue);                                                                            // cout << "execute plan bac E ,clFinish res = " << res << endl;
        clSetKernelArg(sumFftSField_kernel, 0, sizeof(cl_mem), &fft_real_buffer);                                      // real[0] is V
        clSetKernelArg(sumFftSField_kernel, 1, sizeof(cl_mem), &fi->V_buffer);                                         // copy to ncells8 to ncells
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, sumFftSField_kernel, 1, NULL, &n_cells, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "sumFftSField_kernel res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
#else
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, NxPrecalc_kernel, 1, NULL, &n_cells4, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&appbac3, 1, &launchParams); // 1 = inverse FFT//if (resFFT) // cout << "inverse transform to get convolution" << endl;               cout << "execute plan bac E resFFT = " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue);               // cout << "execute plan bac E ,clFinish res = " << res << endl;

#endif
        clSetKernelArg(sumFftField_kernel, 0, sizeof(cl_mem), &fft_real_buffer); // real[0-2] is E field
        clSetKernelArg(sumFftField_kernel, 1, sizeof(cl_mem), &fi->Ee_buffer);
        clSetKernelArg(sumFftField_kernel, 2, sizeof(cl_mem), &fi->E_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, sumFftField_kernel, 1, NULL, &n_cells, NULL, 0, NULL, NULL); //  kernel is different for octant, ... selected in the beginning
        if (res)
            cout << "sumFftField_kernel E  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
    }
#else
    {
        clSetKernelArg(copyextField_kernel, 1, sizeof(cl_mem), &fi->Ee_buffer);
        clSetKernelArg(copyextField_kernel, 2, sizeof(cl_mem), &fi->E_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copyextField_kernel, 1, NULL, &nc3_16, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        res = clFinish(vkGPU.commandQueue);
    }
#endif
    //                  cout << "E done\n";

#ifdef Bon_
    {
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copy3Data_kernel, 1, NULL, &n_cells8, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&app3, -1, &launchParams); // -1 = forward transform // cout << "execute plan for E resFFT = " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue);             //  cout << "execute plan for E" << endl;

        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, jcxPrecalc_kernel, 1, NULL, &n_cells8, NULL, 0, NULL, NULL);
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&appbac3, 1, &launchParams);                        // 1 = inverse FFT// cout << "execute plan bac E resFFT = " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue);                                      // cout << "execute plan bac E ,clFinish res = " << res << endl;
                                                                                 // resFFT = transferDataToCPU(&vkGPU, &fft_real[0][0], &fft_real_buffer, bufferSize_R3);
        clSetKernelArg(sumFftField_kernel, 0, sizeof(cl_mem), &fft_real_buffer); // real[0-2] is E field
        clSetKernelArg(sumFftField_kernel, 1, sizeof(cl_mem), &fi->Be_buffer);
        clSetKernelArg(sumFftField_kernel, 2, sizeof(cl_mem), &fi->B_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, sumFftField_kernel, 1, NULL, &n_cells, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "sumFftField_kernel B  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        res = clFinish(vkGPU.commandQueue);
    }
#else
    clSetKernelArg(copyextField_kernel, 1, sizeof(cl_mem), &fi->Be_buffer);
    clSetKernelArg(copyextField_kernel, 2, sizeof(cl_mem), &fi->B_buffer);
    res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copyextField_kernel, 1, NULL, &nc3_16, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
    res = clFinish(vkGPU.commandQueue);
#endif
    //                     cout << "B done\n";

#ifdef Uon_
#ifdef Eon_ // if both Uon and Eon are defined
    float EUtot1 = 0.0f;
    clSetKernelArg(EUEst_kernel, 0, sizeof(cl_mem), &fi->V_buffer);
    clSetKernelArg(EUEst_kernel, 1, sizeof(cl_mem), &fi->npt_buffer);
    clSetKernelArg(EUEst_kernel, 2, sizeof(cl_mem), &EUtot_buffer);
    res = clEnqueueNDRangeKernel(vkGPU.commandQueue, EUEst_kernel, 1, NULL, &n_4, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
    res = clEnqueueReadBuffer(vkGPU.commandQueue, EUtot_buffer, CL_TRUE, 0, sizeof(float) * n_4, EUtot, 0, NULL, NULL);
    for (int i = 0; i < n_4; ++i)
        EUtot1 += EUtot[i];
    EUtot1 *= 0.5f; // * e_charge / ev_to_j; <- this is just 1
#endif
#endif

    size_t n = n_cells * 3 / 16;
    maxval_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, n * sizeof(float), 0, &res);
    float *maxval_array = (float *)_aligned_malloc(sizeof(float) * n, par->cl_align);
#ifdef Eon_
    clSetKernelArg(maxvalf_kernel, 0, sizeof(cl_mem), &fi->E_buffer);
    clSetKernelArg(maxvalf_kernel, 1, sizeof(cl_mem), &maxval_buffer);
    res = clEnqueueNDRangeKernel(vkGPU.commandQueue, maxvalf_kernel, 1, NULL, &n, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
    res = clEnqueueReadBuffer(vkGPU.commandQueue, maxval_buffer, CL_TRUE, 0, sizeof(float) * n, maxval_array, 0, NULL, NULL);
    par->Emax = maxvalf(maxval_array, n);
#endif

#ifdef Bon_
    clSetKernelArg(maxvalf_kernel, 0, sizeof(cl_mem), &fi->B_buffer);
    clSetKernelArg(maxvalf_kernel, 1, sizeof(cl_mem), &maxval_buffer);
    res = clEnqueueNDRangeKernel(vkGPU.commandQueue, maxvalf_kernel, 1, NULL, &n, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
    res = clEnqueueReadBuffer(vkGPU.commandQueue, maxval_buffer, CL_TRUE, 0, sizeof(float) * n, maxval_array, 0, NULL, NULL);
    par->Bmax = maxvalf(maxval_array, n);
#endif
    _aligned_free(maxval_array);
    clReleaseMemObject(maxval_buffer);

    int E_exceeds = 0,
        B_exceeds = 0;
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * (par->Bmax + 1e-3f));
    float acc_e = fabsf(e_charge_mass * par->Emax);
    float vel_e = sqrt(kb * Temp_e / e_mass);
    float TE = (sqrt(1 + 2 * a0 * par->a0_f * acc_e / pow(vel_e, 2)) - 1) * vel_e / acc_e; // time for electron to move across 1 cell
    TE = ((TE <= 0) | (isnanf(TE))) ? a0 * par->a0_f / vel_e : TE;                         // if acc is negligible i.e. in square root ~=1, use approximation is more accurate
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance

    TE *= 1; // x times larger try to save time but makes it unstable.

    // cout << "Tcyclotron=" << Tcyclotron << ",Bmax= " << par->Bmax << ", TE=" << TE << ",Emax= " << par->Emax << ",dt= " << par->dt[0] * f1 << endl;
    if (TE < (par->dt[0] * f1)) // if ideal time step is lower than actual timestep
        E_exceeds = 1;
    else if (TE > (par->dt[0] * f2)) // can increase time step
        E_exceeds = 2;
    if (Tcyclotron < (par->dt[0] * 4 * f1))
        B_exceeds = 4;
    else if (Tcyclotron > (par->dt[0] * 4 * f2))
        B_exceeds = 8;
    // cout <<"calcEBV\n";
    return (E_exceeds + B_exceeds);
}