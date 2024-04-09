#include "include/traj.h"
#include <math.h>
#include <complex>
#include <fftw3.h>

// Shorthand for cleaner code using FFTW documentation notation
const size_t N0 = n_space_divx2, N1 = n_space_divy2, N2 = n_space_divz2,
             N0N1 = N0 * N1, N0N1_2 = N0N1 / 2,
             N2_c = N2 / 2 + 1;         // Dimension to store the complex data, as required by fftw (from their docs)
const size_t n_cells4 = N0 * N1 * N2_c; // NOTE: This is not actually n_cells * 4, there is an additional buffer that fftw requires.

void save_vti_cc(string filename, int i, int ncomponents, double t, float (*data1)[N2][N1][N0], par *par)
{
    if (ncomponents > 3)
    {
        cout << "Error: Cannot write file " << filename << " - too many components" << endl;
        return;
    }
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New(); // Create the vtkImageData object
    imageData->SetDimensions(N0, N1, N2);                                           // Set the dimensions of the image data

    imageData->SetSpacing(par->dd[0], par->dd[1], par->dd[2]);
    imageData->SetOrigin(-par->dd[0] * ((float)n_space_divx - 0.5), -par->dd[1] * ((float)n_space_divy - 0.5), -par->dd[2] * ((float)n_space_divz - 0.5)); // Set the origin of the image data
    imageData->AllocateScalars(VTK_FLOAT, ncomponents);
    imageData->GetPointData()->GetScalars()->SetName(filename.c_str());
    float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < N2; ++k)
        for (int j = 0; j < N1; ++j)
            for (int i = 0; i < N0; ++i)
                for (int c = 0; c < ncomponents; ++c)
                {
                    data2[(k * N1 + j) * N0 * ncomponents + i * ncomponents + c] = data1[c][k][j][i];
                }

    // Create a vtkDoubleArray to hold the field data
    vtkSmartPointer<vtkDoubleArray> timeArray = vtkSmartPointer<vtkDoubleArray>::New();
    timeArray->SetName("TimeValue");
    timeArray->SetNumberOfTuples(1);
    timeArray->SetValue(0, t);

    // Add the field data to the image data
    vtkSmartPointer<vtkFieldData> fieldData = imageData->GetFieldData();
    fieldData->AddArray(timeArray);

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New(); // Create the vtkXMLImageDataWriter object
    writer->SetFileName((filename + "_" + to_string(i) + ".vti").c_str());                         // Set the output file name                                                                     // Set the time value
    writer->SetDataModeToBinary();
    // writer->SetCompressorTypeToLZ4();
    writer->SetCompressorTypeToZLib(); // Enable compression
    writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
    writer->SetInputData(imageData);   // Set the input image data
                                       // Set the time step value
    writer->Write();                   // Write the output file
}

void save_vti_cR(string filename, int i, int ncomponents, double t, float (*data1)[N2_c][N1][N0][2], par *par)
{
    if (ncomponents > 3)
    {
        cout << "Error: Cannot write file " << filename << " - too many components" << endl;
        return;
    }
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New(); // Create the vtkImageData object
    imageData->SetDimensions(N0, N1, N2_c);                                         // Set the dimensions of the image data

    imageData->SetSpacing(par->dd[0], par->dd[1], par->dd[2]);
    imageData->SetOrigin(-par->dd[0] * ((float)n_space_divx - 0.5), -par->dd[1] * ((float)n_space_divy - 0.5), -par->dd[2] * ((float)n_space_divz - 0.5)); // Set the origin of the image data
    imageData->AllocateScalars(VTK_FLOAT, ncomponents);
    imageData->GetPointData()->GetScalars()->SetName(filename.c_str());
    float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < N2_c; ++k)
        for (int j = 0; j < N1; ++j)
            for (int i = 0; i < N0; ++i)
                for (int c = 0; c < ncomponents; ++c)
                {
                    data2[(k * N1 + j) * N0 * ncomponents + i * ncomponents + c] = powf(powf(data1[c][k][j][i][0], 2) + powf(data1[c][k][j][i][1], 2), 0.5);
                }
    //    data2[(k * N1 + j) * N0 * ncomponents + i * ncomponents + c] = data1[c][k][j][i][0];
    // auto *precalc_r3 = reinterpret_cast<fftwf_complex (&)[2][3][N2_c][N1][N0]>(*fftwf_alloc_complex(2 * 3 * n_cells4));
    // Create a vtkDoubleArray to hold the field data
    vtkSmartPointer<vtkDoubleArray> timeArray = vtkSmartPointer<vtkDoubleArray>::New();
    timeArray->SetName("TimeValue");
    timeArray->SetNumberOfTuples(1);
    timeArray->SetValue(0, t);

    // Add the field data to the image data
    vtkSmartPointer<vtkFieldData> fieldData = imageData->GetFieldData();
    fieldData->AddArray(timeArray);

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New(); // Create the vtkXMLImageDataWriter object
    writer->SetFileName((filename + "_" + to_string(i) + ".vti").c_str());                         // Set the output file name                                                                     // Set the time value
    writer->SetDataModeToBinary();
    // writer->SetCompressorTypeToLZ4();
    writer->SetCompressorTypeToZLib(); // Enable compression
    writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
    writer->SetInputData(imageData);   // Set the input image data
                                       // Set the time step value
    writer->Write();                   // Write the output file
}

void save_vti_ctheta(string filename, int i, int ncomponents, double t, float (*data1)[N2_c][N1][N0][2], par *par)
{
    if (ncomponents > 3)
    {
        cout << "Error: Cannot write file " << filename << " - too many components" << endl;
        return;
    }
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New(); // Create the vtkImageData object
    imageData->SetDimensions(N0, N1, N2_c);                                         // Set the dimensions of the image data

    imageData->SetSpacing(par->dd[0], par->dd[1], par->dd[2]);
    imageData->SetOrigin(-par->dd[0] * ((float)n_space_divx - 0.5), -par->dd[1] * ((float)n_space_divy - 0.5), -par->dd[2] * ((float)n_space_divz - 0.5)); // Set the origin of the image data
    imageData->AllocateScalars(VTK_FLOAT, ncomponents);
    imageData->GetPointData()->GetScalars()->SetName(filename.c_str());
    float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

    for (int k = 0; k < N2_c; ++k)
        for (int j = 0; j < N1; ++j)
            for (int i = 0; i < N0; ++i)
                for (int c = 0; c < ncomponents; ++c)
                {
                    data2[(k * N1 + j) * N0 * ncomponents + i * ncomponents + c] = atan2f(data1[c][k][j][i][0], data1[c][k][j][i][1]);
                }
    //    data2[(k * N1 + j) * N0 * ncomponents + i * ncomponents + c] = data1[c][k][j][i][0];
    // auto *precalc_r3 = reinterpret_cast<fftwf_complex (&)[2][3][N2_c][N1][N0]>(*fftwf_alloc_complex(2 * 3 * n_cells4));
    // Create a vtkDoubleArray to hold the field data
    vtkSmartPointer<vtkDoubleArray> timeArray = vtkSmartPointer<vtkDoubleArray>::New();
    timeArray->SetName("TimeValue");
    timeArray->SetNumberOfTuples(1);
    timeArray->SetValue(0, t);

    // Add the field data to the image data
    vtkSmartPointer<vtkFieldData> fieldData = imageData->GetFieldData();
    fieldData->AddArray(timeArray);

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New(); // Create the vtkXMLImageDataWriter object
    writer->SetFileName((filename + "_" + to_string(i) + ".vti").c_str());                         // Set the output file name                                                                     // Set the time value
    writer->SetDataModeToBinary();
    // writer->SetCompressorTypeToLZ4();
    writer->SetCompressorTypeToZLib(); // Enable compression
    writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
    writer->SetInputData(imageData);   // Set the input image data
                                       // Set the time step value
    writer->Write();                   // Write the output file
}

void vector_muls(float *A, float Bb, int n)
{
    for (int i = 0; i < n; ++i)
        A[i] = Bb * A[i];
}

// Vector multiplication for complex numbers. Note that this is not in-place.

void Time::mark()
{
    marks.push_back(chrono::high_resolution_clock::now());
}

float Time::elapsed()
{
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - marks.back()).count();
    marks.pop_back();
    return (float)time * 1e-6;
}

// Get the same result as elapsed, but also insert the current time point back in
float Time::replace()
{
    auto now = chrono::high_resolution_clock::now();
    auto back = marks.back();
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(now - back).count();
    back = now;
    return (float)time * 1e-6;
}

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
*/

// Arrays for fft, output is multiplied by 2 because the convolution pattern should be double the size
// (a particle can be both 64 cells up, or 64 cells down, so we need 128 cells to calculate this information)
auto *fft_real = reinterpret_cast<float (&)[4][n_cells8]>(*fftwf_alloc_real(n_cells8 * 4));
auto *fft_complex = reinterpret_cast<fftwf_complex (&)[4][n_cells4]>(*fftwf_alloc_complex(4 * n_cells4));

//  pre-calculate 1/ r3 to make it faster to calculate electric and magnetic fields
auto *precalc_r3 = reinterpret_cast<fftwf_complex (&)[2][3][N2_c][N1][N0]>(*fftwf_alloc_complex(2 * 3 * n_cells4));

int calcEBV(par *par)
{
    static int first = 1;
    static fftwf_plan planforE, planforB, planbacE, planbacB;

    static float posL2[3];
    static unsigned int *n_space_div2;
    if (first)
    { // allocate and initialize to 0
        auto precalc_r2_base = new float[N2][N1][N0];
        auto precalc_r3_base = new float[2][3][N2][N1][N0];

        int dims[3] = {N0, N1, N2};
        cout << "omp_get_max_threads " << omp_get_max_threads() << endl;
        fftwf_plan_with_nthreads(omp_get_max_threads() * 1);
        float r3, rx, ry, rz, rx2, ry2, rz2;
        int i, j, k, loc_i, loc_j, loc_k;
        posL2[0] = -par->dd[0] * ((float)n_space_divx - 0.5);
        posL2[1] = -par->dd[1] * ((float)n_space_divy - 0.5);
        posL2[2] = -par->dd[2] * ((float)n_space_divz - 0.5);
        n_space_div2 = new unsigned int[3]{n_space_divx2, n_space_divy2, n_space_divz2};
        // precalculate 1/r^3 (field) and 1/r^2 (energy)
        timer.mark();
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
        cout << "precalc time = " << timer.replace() << "s\n";
        // Multiply by the respective constants here, since it is faster to parallelize it
        const float Vconst = a0 * a0; // kc * e_charge * r_part_spart / n_cells8;
        const float Aconst = 1;       // 1e-7 * e_charge * r_part_spart / n_cells8;

        vector_muls(reinterpret_cast<float *>(precalc_r3_base[0]), Vconst, n_cells8 * 3);
        // vector_muls(reinterpret_cast<float *>(precalc_r3_base[1]), Aconst, n_cells8 * 3);
        // clFFT start
        cl_int err;
        cl_platform_id platform[3];
        cl_device_id device[3];
        cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
        cl_context ctx = 0;
        cl_command_queue queue = 0;
        cl_mem bufX;
        cl_mem bufY;
        float *X;
        cl_event event = NULL;
        int ret = 0;
        int platform_id = 0;
        int device_id = 1;

        // const size_t N0 = 4, N1 = 4, N2 = 4;
        char platform_name[128];
        char device_name[128];

        /* FFT library realted declarations */
        clfftPlanHandle planHandle;
        clfftDim dim = CLFFT_3D;
        size_t clLengths[3] = {N0, N1, N2};

        /* Setup OpenCL environment. */
        err = clGetPlatformIDs(3, platform, NULL);

        size_t ret_param_size = 0;
        err = clGetPlatformInfo(platform[platform_id], CL_PLATFORM_NAME,
                                sizeof(platform_name), platform_name,
                                &ret_param_size);
        printf("Platform found: %s\n", platform_name);
        cl_uint numdevices;
        err = clGetDeviceIDs(platform[platform_id], CL_DEVICE_TYPE_ALL, 5, device, &numdevices);
        cout << numdevices << endl;
        err = clGetDeviceInfo(device[device_id], CL_DEVICE_NAME,
                              sizeof(device_name), device_name,
                              &ret_param_size);
        printf("Device found on the above platform: %s\n", device_name);

        props[1] = (cl_context_properties)platform[0];
        ctx = clCreateContext(props, 1, device, NULL, NULL, &err);
        cout << "ctx err " << err << "," << ctx << endl;
        queue = clCreateCommandQueueWithProperties(ctx, device[0], NULL, &err);
        cout << "queue err " << err << endl;
        /* Setup clFFT. */
        clfftSetupData fftSetup;
        err = clfftInitSetupData(&fftSetup);
        err = clfftSetup(&fftSetup);

        /* Allocate host & initialize data. */
        /* Only allocation shown for simplicity. */
        size_t buffer_size = N0 * N1 * N2 * sizeof(*X);
        size_t buffer_sizeY = N0 * N1 * (N2 / 2 + 1) * sizeof(*X) * 2;
        //        X = (float *)malloc(buffer_size);

        /* Prepare OpenCL memory objects and place data inside them. */
        bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buffer_size * 6, NULL, &err);
        bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, buffer_sizeY * 6, NULL, &err); // CL_MEM_READ_WRITE

        /* Create a default plan for a complex FFT. */
        err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

        /* Set plan parameters. */
        err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
        err = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED); //  CLFFT_HERMITIAN_INTERLEAVED , CLFFT_COMPLEX_INTERLEAVED
        err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);                // CLFFT_INPLACE
        size_t istride[3] = {1, N2, N2 * N1};
        size_t ostride[3] = {1, N2 / 2 + 1, (N2 / 2 + 1) * N1};
        err = clfftSetPlanDim(planHandle, dim);
        err = clfftSetPlanOutStride(planHandle, dim, ostride);
        err = clfftSetPlanInStride(planHandle, dim, istride);
        err = clfftSetPlanBatchSize(planHandle, 6);
        err = clfftSetPlanDistance(planHandle, N0 * N1 * N2, N0 * N1 * (N2 / 2 + 1));
        //  cout << "clFFT bake plan\n ";
        /* Bake the plan. */
        err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

        cout << "clFFT\n ";

        err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, buffer_size * 6, &precalc_r3_base[0][0][0][0][0], 0, NULL, NULL);
        /* Execute the plan. */
        timer.mark();
        err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, &bufY, NULL);
        //     cout << err<<endl;
        /* Wait for calculations to be finished. */
        err = clFinish(queue);

        /* Fetch results of calculations. */
        err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, buffer_sizeY * 6, &precalc_r3[0][0][0][0][0], 0, NULL, NULL);
        cout << "clFFT time = " << timer.replace() << "s\n";
        /* Release OpenCL memory objects. */
        clReleaseMemObject(bufX);
        clReleaseMemObject(bufY);
        /* Release the plan. */
        err = clfftDestroyPlan(&planHandle);
        /* Release clFFT library. */
        clfftTeardown();
        /* Release OpenCL working objects. */
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
#ifdef CheckFFT
        save_vti_cc("precalc_r3_basecl", 0, 3, 0.0, &precalc_r3_base[0][0], par);
        save_vti_cR("precalc_r3Rcl", 0, 3, 0.0, &precalc_r3[0][0], par);
        save_vti_ctheta("precalc_r3thetacl", 0, 3, 0.0, &precalc_r3[0][0], par);
#endif
        // FFTW start
        //  Create fftw plans

        fftwf_plan planfor_k = fftwf_plan_many_dft_r2c(3, dims, 6, reinterpret_cast<float *>(precalc_r3_base[0][0]), NULL, 1, n_cells8, reinterpret_cast<fftwf_complex *>(precalc_r3[0][0]), NULL, 1, n_cells4, FFTW_ESTIMATE);

        cout << "FFTW " << endl;
        timer.mark();
        fftwf_execute(planfor_k); // fft of kernel arr3=fft(arr)
        cout << "FFTW time = " << timer.replace() << "s\n";
        fftwf_destroy_plan(planfor_k);

#ifdef CheckFFT
        save_vti_cc("precalc_r3_base", 0, 3, 0.0, &precalc_r3_base[0][0], par);
        save_vti_cR("precalc_r3R", 0, 3, 0.0, &precalc_r3[0][0], par);
        save_vti_ctheta("precalc_r3theta", 0, 3, 0.0, &precalc_r3[0][0], par);
#endif
        delete[] precalc_r3_base;

        first = 0;
    }
    //    auto *precalc_r3 = reinterpret_cast<fftwf_complex (&)[2][3][N2_c][N1][N0]>(*fftwf_alloc_complex(2 * 3 * n_cells4));
    return 0;
}
