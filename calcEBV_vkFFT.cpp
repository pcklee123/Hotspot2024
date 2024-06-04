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
This is because we are making use of the fact that our original data is fully real, and this saves about half the memory/computation
    Precalculation step only on first run:
    precalc_r3_base (real 3D vector field) -> FFT -> precalc_r3 (complex 3D vector field)

    Electric field E:
    npt (number of particles per cell) -> copy -> FFT -> fft_complex
    fft_complex *= precalc_r3, do this three times for three dimensions i.e scalar mulitpled with vector
    fft_complex -> IFFT -> fft_real
    E = fft_real + Ee (background constant E also input)

    Bonus: electric potential V. This runs in parallel with E.
    npt -> copy -> FFT -> fft_complex (reuse)
    fft_complex *= precalc_r2
    fft_complex -> IFFT -> fft_real
    V (input) = fft_real

    Magnetic field B:
    jc -> copy -> FFT -> fft_complex
    fft_complex *= precalc_r3 , do this as cross product of jc and precalc_r3
    fft_complex -> IFFT -> fft_real
    B (input) = fft_real + Be (background constant E also input)

    Arrays for fft, output are double(x8) in size because the convolution pattern should be double the size
    (i.e. there can be a particle 64 cells up, and another 64 cells down, so we need 128 cells to calculate this information)
    TODO should use native VkFFT to (i) zero pad, (ii) not re order, (iii) convolve
*/

int calcEBV(fields *fi, par *par)
{
    //  static fftwf_plan planforE, planforB, planbacE, planbacB;
    static int first = 1;
#ifdef _WIN32
    static auto *fft_real = static_cast<float(*)[n_cells8]>(_aligned_malloc(sizeof(float) * n_cells8 * 4, 4096));                      // fft_real[4][n_cells8]
    static auto *fft_complex = static_cast<complex<float>(*)[n_cells4]>(_aligned_malloc(sizeof(complex<float>) * n_cells4 * 4, 4096)); // fft_complex[4][n_cells4]
    //  pre-calculate 1/ r3 to make it faster to calculate electric and magnetic fields
#ifdef Uon_ // similar arrays for U, but kept separately in one ifdef
            // static auto *precalc_r2 = static_cast<complex<float>(*)>(_aligned_malloc(sizeof(complex<float>) * n_cells4, 4096));                      // precalc_r3[n_cells4]
#endif
#else
    //   static auto *fft_real = static_cast<float(*)[n_cells8]>(aligned_alloc(par->cl_align,sizeof(float) * n_cells8 * 4));                      // fft_real[4][n_cells8]
    //   static auto *fft_complex = static_cast<complex<float>(*)[n_cells4]>(aligned_alloc(par->cl_align,sizeof(complex<float>) * n_cells4 * 4)); // fft_complex[4][n_cells4]
    //  pre-calculate 1/ r3 to make it faster to calculate electric and magnetic fields
#ifdef Uon_ // similar arrays for U, but kept separately in one ifdef
            // static auto *precalc_r2 = static_cast<complex<float>(*)>(aligned_alloc(par->cl_align,sizeof(complex<float>) * n_cells4));                      // precalc_r3[n_cells4]
#endif
#endif
    static const size_t n_4 = n_cells / 4;
    static auto EUtot = new float[n_4];
    static float posL2[3];

    static cl_kernel copyData_kernel;
    static cl_kernel copy3Data_kernel;

    static cl_kernel jcxPrecalc_kernel;
    static cl_kernel jd_kernel;
    static cl_kernel Bdot_kernel;
    static cl_kernel sumFftField_kernel;
    static cl_kernel sumFftFieldB_kernel;
    static cl_kernel sumFftSField_kernel;
    static cl_kernel copyextField_kernel;
    static cl_kernel EUEst_kernel;
    static cl_kernel maxvalf_kernel;
    static cl_kernel maxval3f_kernel;
#ifdef Uon_
    static cl_kernel NxPrecalc_kernel;
#else
    static cl_kernel NxPrecalcr2_kernel;
#endif
    static VkFFTApplication app1 = {};
    static VkFFTApplication app3 = {};
    static VkFFTApplication appbac3 = {};
    static VkFFTApplication appbac4 = {};
    static VkFFTApplication appfor_k = {};
    static VkFFTApplication appfor_k2 = {};

    static cl_mem EUtot_buffer = 0;

    static VkGPU vkGPU = {};
    vkGPU.device_id = device_id_g; // use same GPU as motion code TODO: use separate iGPU or dGPU for FFT
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;

    static uint64_t bufferSize_R = (uint64_t)sizeof(float) * n_cells8;          // buffer size per batch Real
    static uint64_t bufferSize_C = (uint64_t)sizeof(complex<float>) * n_cells4; // buffer size per batch Complex
    static uint64_t bufferSize_P = (uint64_t)sizeof(float) * n_cells4 * 2;      // buffer size per batch big enough for either Real or Complex
    static uint64_t bufferSize_R3 = bufferSize_R * 3;                           // 3 for 3-vector
    static uint64_t bufferSize_P3 = bufferSize_P * 3;
    static uint64_t bufferSize_C3 = bufferSize_C * 3;
    static uint64_t bufferSize_R4 = bufferSize_R * 4; // 4 for Potential V and 3-vector E
    static uint64_t bufferSize_P4 = bufferSize_P * 4;
    static uint64_t bufferSize_C4 = bufferSize_C * 4;
    static uint64_t bufferSize_R6 = bufferSize_R * 6;
    static uint64_t bufferSize_C6 = bufferSize_C * 6;
    static VkFFTLaunchParams launchParams = {};
    vkGPU.device = default_device_g();
    vkGPU.context = context_g();
    vkGPU.commandQueue = commandQueue_g();
    //        vkGPU.commandQueue = clCreateCommandQueue(vkGPU.context, vkGPU.device, 0, &res);
    launchParams.commandQueue = &vkGPU.commandQueue;
    // Create the OpenCL kernel for copying "density" or "current density" to the fft buffer
    copyData_kernel = clCreateKernel(program_g(), "copyData", NULL);
    copy3Data_kernel = clCreateKernel(program_g(), "copy3Data", NULL);
    jd_kernel = clCreateKernel(program_g(), "jd", NULL);
    Bdot_kernel = clCreateKernel(program_g(), "Bdot", NULL);
#ifdef Uon_
    static cl_kernel NxPrecalcr2_kernel = clCreateKernel(program_g(), "NxPrecalcr2", NULL);
#else
    static cl_kernel NxPrecalc_kernel = clCreateKernel(program_g(), "NxPrecalc", NULL);
#endif

    jcxPrecalc_kernel = clCreateKernel(program_g(), "jcxPrecalc", NULL);

#ifdef octant
    sumFftField_kernel = clCreateKernel(program_g(), "sumFftFieldo", NULL); // want rollover fields in x,y,z
#else
#ifdef quadrant
    sumFftField_kernel = clCreateKernel(program_g(), "sumFftFieldq", NULL);   // want rollover fields in x,y no z
    sumFftFieldB_kernel = clCreateKernel(program_g(), "sumFftFieldBq", NULL); // want rollover fields in x,y no z
#else
    sumFftField_kernel = clCreateKernel(program_g(), "sumFftField", NULL);
    sumFftFieldB_kernel = clCreateKernel(program_g(), "sumFftField", NULL);
#endif
#endif

    sumFftSField_kernel = clCreateKernel(program_g(), "sumFftSField", NULL);
    copyextField_kernel = clCreateKernel(program_g(), "copyextField", NULL);
    EUEst_kernel = clCreateKernel(program_g(), "EUEst", NULL);
    maxvalf_kernel = clCreateKernel(program_g(), "maxvalf", NULL);
    maxval3f_kernel = clCreateKernel(program_g(), "maxval3f", NULL);

    if (first)
    { // allocate and initialize to 0
        auto precalc_r3_base = new float[2][3][N2][N1][N0];
        cl_mem r3_base_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_R6, 0, &res);
#ifdef Uon_ // similar arrays for U, but kept separately in one ifdef
        auto precalc_r2_base = new float[N2][N1][N0];
        cl_mem r2_base_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, bufferSize_R, 0, &res);
#endif

        // Create memory buffers on the device for each vector
        EUtot_buffer = clCreateBuffer(vkGPU.context, CL_MEM_READ_WRITE, n_4 * sizeof(float), 0, &res);

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
        configuration.buffer = &fi->r3_buffer;
        configuration.bufferSize = &bufferSize_C6;
        resFFT = initializeVkFFT(&appfor_k, configuration);

#ifdef Uon_
        configuration.makeForwardPlanOnly = true;
        configuration.numberBatches = 1;
        configuration.inputBuffer = &r2_base_buffer;
        configuration.inputBufferSize = &bufferSize_R;
        configuration.buffer = &fi->r2_buffer;
        configuration.bufferSize = &bufferSize_C;
        resFFT = initializeVkFFT(&appfor_k2, configuration);
#endif

        // plan for density (forward R2C FFT scalar)
        configuration.numberBatches = 1;
        configuration.performZeropadding[0] = false;
        configuration.performZeropadding[1] = false;
        configuration.performZeropadding[2] = false;
        //       configuration.performZeropadding[0] = true;
        //       configuration.performZeropadding[1] = true;
        //      configuration.performZeropadding[2] = true;

        configuration.frequencyZeroPadding = false; // true
        configuration.fft_zeropad_left[0] = (uint64_t)ceil(configuration.size[0] / 2.0);
        configuration.fft_zeropad_right[0] = configuration.size[0];
        configuration.fft_zeropad_left[1] = (uint64_t)ceil(configuration.size[1] / 2.0);
        configuration.fft_zeropad_right[1] = configuration.size[1];
        configuration.fft_zeropad_left[2] = (uint64_t)ceil(configuration.size[2] / 2.0);
        configuration.fft_zeropad_right[2] = configuration.size[2];
        /*
                configuration.fft_zeropad_left[0] = 0;
                configuration.fft_zeropad_right[0] = (uint64_t)ceil(configuration.size[0] / 2.0);
                configuration.fft_zeropad_left[1] = 0;
                configuration.fft_zeropad_right[1] = (uint64_t)ceil(configuration.size[1] / 2.0);
                configuration.fft_zeropad_left[2] = 0;
                configuration.fft_zeropad_right[2] = (uint64_t)ceil(configuration.size[2] / 2.0);
        */
        configuration.inputBuffer = &fi->fft_real_buffer;
        configuration.inputBufferSize = &bufferSize_R;
        configuration.buffer = &fi->fft_complex_buffer;
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
        configuration.inputBuffer = &fi->fft_real_buffer;
        configuration.inputBufferSize = &bufferSize_R3;
        configuration.buffer = &fi->fft_p_buffer;
        configuration.bufferSize = &bufferSize_P3;
        configuration.outputBuffer = &fi->fft_complex_buffer; // outputBuffer
        configuration.outputBufferSize = &bufferSize_C3;      // outputBufferSize
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
        configuration.outputBuffer = &fi->fft_real_buffer;
        configuration.outputBufferSize = &bufferSize_R3;
        configuration.buffer = &fi->fft_p_buffer;
        configuration.bufferSize = &bufferSize_P3;
        configuration.inputBuffer = &fi->fft_complex_buffer;
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
        configuration.outputBuffer = &fi->fft_real_buffer;
        configuration.outputBufferSize = &bufferSize_R4;
        configuration.buffer = &fi->fft_p_buffer;
        configuration.bufferSize = &bufferSize_P4;
        configuration.inputBuffer = &fi->fft_complex_buffer;
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

// precalculate r_vector/r^3 (field) and 1/r^2 (energy)
#pragma omp parallel for num_threads(nthreads)
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
                // #pragma omp simd
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
        const float Vconst = kc * e_charge * r_part_spart / (float)n_cells8;
        const float Aconst = 1e-7f * e_charge * r_part_spart / (float)n_cells8;
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

        resFFT = VkFFTAppend(&appfor_k, -1, &launchParams); //
        if (resFFT)
            cout << "forward transform precalc_r3" << endl;
        res = clFinish(vkGPU.commandQueue);
        deleteVkFFT(&appfor_k);

        clReleaseMemObject(r3_base_buffer);
        delete[] precalc_r3_base;
#ifdef Uon_
        resFFT = transferDataFromCPU(&vkGPU, precalc_r2_base, &r2_base_buffer, bufferSize_R);
        resFFT = VkFFTAppend(&appfor_k2, -1, &launchParams);
        res = clFinish(vkGPU.commandQueue);
        deleteVkFFT(&appfor_k2);
        // resFFT = transferDataToCPU(&vkGPU, precalc_r2, &fi->r2_buffer, bufferSize_C);
        clReleaseMemObject(r2_base_buffer);
        delete[] precalc_r2_base;
#endif

        // cout << "filter" << endl; // filter
        /*
        //Todo convert to opencl code cause result of FFT is in opencl buffer to avoid copyback to cpu and then to gpu again
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
        first = 0; //      cout << "precalc done\n";
    }

#ifdef Eon_
    {                                                                        //  E field due to charges
        clSetKernelArg(copyData_kernel, 0, sizeof(cl_mem), &fi->npt_buffer); // Set the arguments of the kernel must be done every time. not just on first run
        clSetKernelArg(copyData_kernel, 1, sizeof(cl_mem), &fi->fft_real_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copyData_kernel, 1, NULL, &n_cells_2, NULL, 0, NULL, NULL); //  copy density into zero padded double(8x) cube
        if (res)
            cout << "copyData_kernel res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        //   fft(of density)
        resFFT = VkFFTAppend(&app1, -1, &launchParams); // -1 = forward transform
        if (resFFT)
            cout << "app1 resFFT: " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue); //  cout << "execute plan for E" << endl;
// multiply fft charge with fft of kernel(i.e field associated with 1 charge)
#ifdef Uon_
        clSetKernelArg(NxPrecalcr2_kernel, 0, sizeof(cl_mem), &fi->r2_buffer);
        clSetKernelArg(NxPrecalcr2_kernel, 1, sizeof(cl_mem), &fi->r3_buffer);
        clSetKernelArg(NxPrecalcr2_kernel, 2, sizeof(cl_mem), &fi->fft_complex_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, NxPrecalcr2_kernel, 1, NULL, &n_cells4, NULL, 0, NULL, NULL); //  multiply FFT of density with precalc for both E[0-2] and V[3]
        if (res)
            cout << "NxPrecalcr2_kernel res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&appbac4, 1, &launchParams); // cout << "inverse transform to get convolution" << endl;                                                             // 1 = inverse FFT//if (resFFT)                cout << "execute plan bac E resFFT = " << resFFT << endl;
        if (resFFT)
            cout << "appbac4 resFFT: " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue);                                                                            // cout << "execute plan bac E ,clFinish res = " << res << endl;
        clSetKernelArg(sumFftSField_kernel, 0, sizeof(cl_mem), &fi->fft_real_buffer);                                  // real[3] is V
        clSetKernelArg(sumFftSField_kernel, 1, sizeof(cl_mem), &fi->V_buffer);                                         // copy to ncells8 to ncells
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, sumFftSField_kernel, 1, NULL, &n_cells, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "sumFftSField_kernel res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
#else
        res = clSetKernelArg(NxPrecalc_kernel, 0, sizeof(cl_mem), &fi->r3_buffer);
        if (res)
            cout << "clSetKernelArg NxPrecalc_kernel 0 res: " << resFFT << endl;
        res = clSetKernelArg(NxPrecalc_kernel, 1, sizeof(cl_mem), &fi->fft_complex_buffer);
        if (res)
            cout << "clSetKernelArg NxPrecalc_kernel 1 res: " << resFFT << endl;
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, NxPrecalc_kernel, 1, NULL, &n_cells4, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "NxPrecalc_kernel res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&appbac3, 1, &launchParams); // 1 = inverse FFT//if (resFFT) // cout << "inverse transform to get convolution" << endl;               cout << "execute plan bac E resFFT = " << resFFT << endl;
        if (resFFT)
            cout << "appbac3 resFFT: " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue); // cout << "execute plan bac E ,clFinish res = " << res << endl;

#endif
#ifdef dE_dton_
        clEnqueueCopyBuffer(vkGPU.commandQueue, fi->E_buffer, fi->E0_buffer, 0, 0, n_cellsf * 3, 0, NULL, NULL); // store E(t) in E0_buffer will replace this with dE/dt later B calcluation will use most recent dE/dt
        res = clFinish(vkGPU.commandQueue);
#endif
        clSetKernelArg(sumFftField_kernel, 0, sizeof(cl_mem), &fi->fft_real_buffer); // real[0-2] is E field
        clSetKernelArg(sumFftField_kernel, 1, sizeof(cl_mem), &fi->Ee_buffer);
        clSetKernelArg(sumFftField_kernel, 2, sizeof(cl_mem), &fi->E_buffer); // make E_buffer to be E due to charges + External applied E

        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, sumFftField_kernel, 1, NULL, &n_cells, NULL, 0, NULL, NULL); //  kernel is different for octant, ... selected in the beginning
        if (res)
            cout << "sumFftField_kernel E  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
#ifdef dB_dton_
        // calculate E due to dB/dt
        res = clSetKernelArg(copy3Data_kernel, 0, sizeof(cl_mem), &fi->B0_buffer); // this should - 1/mu0 * (B(t+dt)-B(t))/dt B0 is used for both B(t) and dB/dt (0 for t=0) use old value of dB/dt cause B not calculated yet
        if (res)
            cout << "clSetKernelArg copy3Data_kernel 0 res: " << res << endl;
        res = clSetKernelArg(copy3Data_kernel, 1, sizeof(cl_mem), &fi->fft_real_buffer);
        if (res)
            cout << "clSetKernelArg copy3Data_kernel 1 res: " << res << endl;
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copy3Data_kernel, 1, NULL, &n_cells_2, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "copy3Data_kernel dB/dt  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        if (res)
            cout << "copy3Data_kernel dB/dt clFinish  res: " << res << endl;
        resFFT = VkFFTAppend(&app3, -1, &launchParams); // -1 = forward transform // cout << "execute plan for E resFFT = " << resFFT << endl;
        if (resFFT)
            cout << "app3 B resFFT: " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue); //  cout << "execute plan for dB/dt" << endl;
        res = clSetKernelArg(jcxPrecalc_kernel, 0, sizeof(cl_mem), &fi->r3_buffer);
        if (res)
            cout << "clSetKernelArg jcxPrecalc_kernel 0 res: " << res << endl;
        res = clSetKernelArg(jcxPrecalc_kernel, 1, sizeof(cl_mem), &fi->fft_complex_buffer);
        if (res)
            cout << "clSetKernelArg jcxPrecalc_kernel 1 res: " << res << endl;
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, jcxPrecalc_kernel, 1, NULL, &n_cells4, NULL, 0, NULL, NULL);
        if (res)
            cout << "jcxPrecalc_kernel B  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&appbac3, 1, &launchParams); // 1 = inverse FFT// cout << "execute plan bac E resFFT = " << resFFT << endl;
        if (resFFT)
            cout << "appbac3 B resFFT: " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue); // cout << "execute plan bac E ,clFinish res = " << res << endl;
                                            // resFFT = transferDataToCPU(&vkGPU, &fft_real[0][0], &fi->fft_real_buffer, bufferSize_R3);

        clSetKernelArg(sumFftFieldB_kernel, 0, sizeof(cl_mem), &fi->fft_real_buffer); // real[0-2] is E due to dB/dt field
        clSetKernelArg(sumFftFieldB_kernel, 1, sizeof(cl_mem), &fi->E_buffer);
        clSetKernelArg(sumFftFieldB_kernel, 2, sizeof(cl_mem), &fi->E_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, sumFftFieldB_kernel, 1, NULL, &n_cells, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "sumFftFieldB_kernel B  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
#endif
    }
#else
    {
        clSetKernelArg(copyextField_kernel, 0, sizeof(cl_mem), &fi->Ee_buffer);
        clSetKernelArg(copyextField_kernel, 1, sizeof(cl_mem), &fi->E_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copyextField_kernel, 1, NULL, &nc3_16, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        res = clFinish(vkGPU.commandQueue);
    }
#endif

    //                  cout << "E done\n";

#ifdef Bon_
    {
#ifdef dE_dton_
        // add displacement current epsilon0 dE/dt to total current
        float dedtcoeff = epsilon0 * powf(a0 * par->a0_f, 3) / (par->dt[0] * par->ncalcp[0]);
        clSetKernelArg(jd_kernel, 0, sizeof(cl_mem), &fi->E0_buffer); // previous E field
        clSetKernelArg(jd_kernel, 1, sizeof(cl_mem), &fi->E_buffer);  // recently calculated E field
        clSetKernelArg(jd_kernel, 2, sizeof(cl_mem), &(fi->buff_jc[0]()));
        clSetKernelArg(jd_kernel, 3, sizeof(float), &dedtcoeff);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, jd_kernel, 1, NULL, &nc3_16, NULL, 0, NULL, NULL); //
        if (res)
            cout << "jd_kernel res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
#endif
        res = clSetKernelArg(copy3Data_kernel, 0, sizeof(cl_mem), &(fi->buff_jc[0]()));
        if (res)
            cout << "clSetKernelArg copy3Data_kernel 0 res: " << res << endl;
        res = clSetKernelArg(copy3Data_kernel, 1, sizeof(cl_mem), &fi->fft_real_buffer);
        if (res)
            cout << "clSetKernelArg copy3Data_kernel 1 res: " << res << endl;
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copy3Data_kernel, 1, NULL, &n_cells_2, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "copy3Data_kernel B  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        if (res)
            cout << "copy3Data_kernel B clFinish  res: " << res << endl;
        resFFT = VkFFTAppend(&app3, -1, &launchParams); // -1 = forward transform // cout << "execute plan for E resFFT = " << resFFT << endl;
        if (resFFT)
            cout << "app3 B resFFT: " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue); //  cout << "execute plan for E" << endl;
        res = clSetKernelArg(jcxPrecalc_kernel, 0, sizeof(cl_mem), &fi->r3_buffer);
        if (res)
            cout << "clSetKernelArg jcxPrecalc_kernel 0 res: " << res << endl;
        res = clSetKernelArg(jcxPrecalc_kernel, 1, sizeof(cl_mem), &fi->fft_complex_buffer);
        if (res)
            cout << "clSetKernelArg jcxPrecalc_kernel 1 res: " << res << endl;
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, jcxPrecalc_kernel, 1, NULL, &n_cells4, NULL, 0, NULL, NULL);
        if (res)
            cout << "jcxPrecalc_kernel B  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
        resFFT = VkFFTAppend(&appbac3, 1, &launchParams); // 1 = inverse FFT// cout << "execute plan bac E resFFT = " << resFFT << endl;
        if (resFFT)
            cout << "appbac3 B resFFT: " << resFFT << endl;
        res = clFinish(vkGPU.commandQueue); // cout << "execute plan bac E ,clFinish res = " << res << endl;
                                            // resFFT = transferDataToCPU(&vkGPU, &fft_real[0][0], &fi->fft_real_buffer, bufferSize_R3);

#ifdef dB_dton_
        clEnqueueCopyBuffer(vkGPU.commandQueue, fi->B_buffer, fi->B0_buffer, 0, 0, n_cellsf * 3, 0, NULL, NULL); // store B(t) in B0 will replace thist with Bdot later
        res = clFinish(vkGPU.commandQueue);
#endif
        clSetKernelArg(sumFftFieldB_kernel, 0, sizeof(cl_mem), &fi->fft_real_buffer); // real[0-2] is E field
        clSetKernelArg(sumFftFieldB_kernel, 1, sizeof(cl_mem), &fi->Be_buffer);
        clSetKernelArg(sumFftFieldB_kernel, 2, sizeof(cl_mem), &fi->B_buffer);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, sumFftFieldB_kernel, 1, NULL, &n_cells, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
        if (res)
            cout << "sumFftFieldB_kernel B  res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
#ifdef dB_dton_
        // estimate dE/dt and add epsilon0 dE/dt to total current
        float dBdtcoeff = -1.0 * powf(a0 * par->a0_f, 3) / (u0 * par->dt[0] * par->ncalcp[0]);
        clSetKernelArg(Bdot_kernel, 0, sizeof(cl_mem), &fi->B0_buffer); // replace B0_buffer with -1/u0 * dB/dt
        clSetKernelArg(Bdot_kernel, 1, sizeof(cl_mem), &fi->B_buffer);
        clSetKernelArg(Bdot_kernel, 2, sizeof(float), &dBdtcoeff);
        res = clEnqueueNDRangeKernel(vkGPU.commandQueue, Bdot_kernel, 1, NULL, &nc3_16, NULL, 0, NULL, NULL); //
        if (res)
            cout << "Bdot_kernel res: " << res << endl;
        res = clFinish(vkGPU.commandQueue);
#endif
    }
#else
    clSetKernelArg(copyextField_kernel, 0, sizeof(cl_mem), &fi->Be_buffer);
    clSetKernelArg(copyextField_kernel, 1, sizeof(cl_mem), &fi->B_buffer);
    res = clEnqueueNDRangeKernel(vkGPU.commandQueue, copyextField_kernel, 1, NULL, &nc3_16, NULL, 0, NULL, NULL);
    if (res)
        cout << "copyextField_kernel res: " << res << endl;
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
    if (res)
        cout << "EUEst_kernel   res: " << res << endl;
    res = clFinish(commandQueue_g());
    if (!fastIO)
    {
        res = clEnqueueReadBuffer(vkGPU.commandQueue, EUtot_buffer, CL_TRUE, 0, sizeof(float) * n_4, EUtot, 0, NULL, NULL);
        if (res)
            cout << "EUEst_kernel readbuffer res: " << res << endl;
    }
    for (int i = 0; i < n_4; ++i)
        EUtot1 += EUtot[i];
    EUtot1 *= 0.5f; // * e_charge / ev_to_j; <- this is just 1
#endif
#endif

#ifdef Eon_
    clSetKernelArg(maxval3f_kernel, 0, sizeof(cl_mem), &fi->E_buffer);
    clSetKernelArg(maxval3f_kernel, 1, sizeof(cl_mem), &par->maxval_buffer);
    res = clEnqueueNDRangeKernel(vkGPU.commandQueue, maxval3f_kernel, 1, NULL, &n2048, NULL, 0, NULL, NULL);
    if (res)
        cout << "maxval3f_kernel res: " << res << endl;
    res = clFinish(commandQueue_g());
    if (res)
        cout << "maxval3f_kernel clfinish E res: " << res << endl;
    if (!fastIO)
    {
        res = clEnqueueReadBuffer(vkGPU.commandQueue, par->maxval_buffer, CL_TRUE, 0, sizeof(float) * n2048, par->maxval_array, 0, NULL, NULL);
        if (res)
            cout << "maxval3f_kernel readbuffer res: " << res << endl;
    }
    par->Emax = sqrtf(maxvalf(par->maxval_array, 2048));
    // cout << "Emax = " << par->Emax << endl;
#endif

#ifdef Bon_
    clSetKernelArg(maxval3f_kernel, 0, sizeof(cl_mem), &fi->B_buffer);
    clSetKernelArg(maxval3f_kernel, 1, sizeof(cl_mem), &par->maxval_buffer);
    res = clEnqueueNDRangeKernel(vkGPU.commandQueue, maxval3f_kernel, 1, NULL, &n2048, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
    if (res)
        cout << "maxval3f_kernel res: " << res << endl;
    res = clFinish(commandQueue_g());
    if (res)
        cout << "maxval3f_kernel clfinish B res: " << res << endl;
    if (!fastIO)
    {
        res = clEnqueueReadBuffer(vkGPU.commandQueue, par->maxval_buffer, CL_TRUE, 0, sizeof(float) * n2048, par->maxval_array, 0, NULL, NULL);
        if (res)
            cout << "maxval3f_kernel readbuffer res: " << res << endl;
    }
    par->Bmax = sqrtf(maxvalf(par->maxval_array, 2048));
#endif

    int E_exceeds = 0,
        B_exceeds = 0;
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * (par->Bmax + 1e-3f));
    float acc_e = fabsf(e_charge_mass * par->Emax);
    float vel_e = sqrt(kb * Temp_e / e_mass);
    float TE = (sqrt(1 + 2 * a0 * par->a0_f * acc_e / pow(vel_e, 2)) - 1) * vel_e / acc_e; // time for electron to move across 1 cell
    TE = ((TE <= 0) | (isnan(TE))) ? a0 * par->a0_f / vel_e : TE;                          // if acc is negligible i.e. in square root ~=1, use approximation is more accurate
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance

    TE *= 1; // x times larger try to save time but makes it unstable.

    //  cout << "Tcyclotron=" << Tcyclotron << ",Bmax= " << par->Bmax << ", TE=" << TE << ",Emax= " << par->Emax << ",dt= " << par->dt[0] * f1 << endl;
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