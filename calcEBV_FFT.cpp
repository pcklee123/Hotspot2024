#include "include/traj.h"
#include <math.h>
#include <complex>
#include <fftw3.h>

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
*/

// Arrays for fft, output is multiplied by 2 because the convolution pattern should be double the size
// (a particle can be both 64 cells up, or 64 cells down, so we need 128 cells to calculate this information)
auto *fft_real = reinterpret_cast<float (&)[4][n_cells8]>(*fftwf_alloc_real(n_cells8 * 4));
auto *fft_complex = reinterpret_cast<fftwf_complex (&)[4][n_cells4]>(*fftwf_alloc_complex(4 * n_cells4));
//  pre-calculate 1/ r3 to make it faster to calculate electric and magnetic fields
auto *precalc_r3 = reinterpret_cast<fftwf_complex (&)[2][3][N2_c][N1][N0]>(*fftwf_alloc_complex(2 * 3 * n_cells4));
#ifdef Uon_ // similar arrays for U, but kept separately in one ifdef
auto *precalc_r2 = reinterpret_cast<fftwf_complex (&)[N2_c][N1][N0]>(*fftwf_alloc_complex(n_cells4));
#endif

int calcEBV(fields *fi, par *par)
// float precalc_r3[3][n_space_divz2][n_space_divy2][n_space_divx2],  float Aconst, float Vconst,
{
    static int first = 1;
    static fftwf_plan planforE, planforB, planbacE, planbacB;

    static float posL2[3];
    static unsigned int *n_space_div2;
    fftwf_plan planfor_k, planfor_k2;
    if (first)
    { // allocate and initialize to 0

        int dims[3] = {N0, N1, N2};
        auto precalc_r3_base = new float[2][3][N2][N1][N0];
        fi->precalc_r3 = (reinterpret_cast<float *>(precalc_r3));
#ifdef Uon_ // similar arrays for U, but kept separately in one ifdef
        auto precalc_r2_base = new float[N2][N1][N0];
        fi->precalc_r2 = (reinterpret_cast<float *>(precalc_r2));
#endif
        // Create fftw plans not thread safe
        //        cout << "omp_get_max_threads " << omp_get_max_threads() << endl;
        fftwf_plan_with_nthreads(omp_get_max_threads() * 1);
        planfor_k = fftwf_plan_many_dft_r2c(3, dims, 6, reinterpret_cast<float *>(precalc_r3_base[0][0]), NULL, 1, n_cells8, reinterpret_cast<fftwf_complex *>(precalc_r3[0][0]), NULL, 1, n_cells4, FFTW_ESTIMATE);
        planforE = fftwf_plan_dft_r2c_3d(N0, N1, N2, fft_real[0], fft_complex[3], FFTW_MEASURE);
#ifdef Uon_
        planfor_k2 = fftwf_plan_dft_r2c_3d(N0, N1, N2, reinterpret_cast<float *>(precalc_r2_base), reinterpret_cast<fftwf_complex *>(precalc_r2), FFTW_ESTIMATE);
        // Perform ifft on the entire array; the first 3/4 is used for E while the last 1/4 is used for V
        planbacE = fftwf_plan_many_dft_c2r(3, dims, 4, fft_complex[0], NULL, 1, n_cells4, fft_real[0], NULL, 1, n_cells8, FFTW_MEASURE);
#else
        // Perform ifft on the first 3/4 of the array for 3 components of E field
        planbacE = fftwf_plan_many_dft_c2r(3, dims, 3, fft_complex[0], NULL, 1, n_cells4, fft_real[0], NULL, 1, n_cells8, FFTW_MEASURE);
#endif
        planforB = fftwf_plan_many_dft_r2c(3, dims, 3, fft_real[0], NULL, 1, n_cells8, fft_complex[0], NULL, 1, n_cells4, FFTW_MEASURE);
        planbacB = fftwf_plan_many_dft_c2r(3, dims, 3, fft_complex[0], NULL, 1, n_cells4, fft_real[0], NULL, 1, n_cells8, FFTW_MEASURE);

#pragma omp barrier
        cout << "allocate done\n";
        float r3, rx, ry, rz, rx2, ry2, rz2;
        int i, j, k, loc_i, loc_j, loc_k;
        posL2[0] = -par->dd[0] * ((float)n_space_divx - 0.5);
        posL2[1] = -par->dd[1] * ((float)n_space_divy - 0.5);
        posL2[2] = -par->dd[2] * ((float)n_space_divz - 0.5);
        n_space_div2 = new unsigned int[3]{n_space_divx2, n_space_divy2, n_space_divz2};
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
            reinterpret_cast<float *>(precalc_r3_base[0])[i] *= Vconst; //  vector_muls(reinterpret_cast<float *>(precalc_r3_base[0]), Vconst, n_cells8 * 3);
#pragma omp parallel for simd num_threads(nthreads)
        for (size_t i = 0; i < n_cells8 * 3; i++)
            reinterpret_cast<float *>(precalc_r3_base[1])[i] *= Aconst; //   vector_muls(reinterpret_cast<float *>(precalc_r3_base[1]), Aconst, n_cells8 * 3);

        fftwf_execute(planfor_k); // fft of kernel arr3=fft(arr)
        fftwf_destroy_plan(planfor_k);
#ifdef Uon_
#pragma omp parallel for simd num_threads(nthreads)
        for (size_t i = 0; i < n_cells8; i++)
            (reinterpret_cast<float *>(fi->precalc_r2))[i] *= Vconst; // vector_muls(reinterpret_cast<float *>(precalc_r2_base), Vconst, n_cells8);
        fftwf_execute(planfor_k2);
        fftwf_destroy_plan(planfor_k2);
#endif

        /*
                cout << "filter" << endl; // filter
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
                        for (i = -n_space_divx; i < n_space_divx; i++)
                        {
                            loc_i = i + (i < 0 ? n_space_divx2 : 0);
                            rx = i;
                            rx2 = rx * rx + ry2;
                            float r = pi * sqrt(rx2) / R_s;
                            float w = r > pi / 2 ? 0.f : cos(r);
                            w *= w;
                            precalc_r3[0][0][loc_k][loc_j][loc_i][0] *= w;
                            precalc_r3[0][1][loc_k][loc_j][loc_i][0] *= w;
                            precalc_r3[0][2][loc_k][loc_j][loc_i][0] *= w;
                            /*precalc_r3[0][0][loc_k][loc_j][loc_i][1] *= w;
                                                precalc_r3[0][1][loc_k][loc_j][loc_i][1] *= w;
                                                precalc_r3[0][2][loc_k][loc_j][loc_i][1] *= w;
                                                            precalc_r3[1][0][loc_k][loc_j][loc_i][1] *= w;
                                                precalc_r3[1][1][loc_k][loc_j][loc_i][1] *= w;
                                                precalc_r3[1][2][loc_k][loc_j][loc_i][1] *= w;
                                                //*
                            precalc_r3[1][0][loc_k][loc_j][loc_i][0] *= w;
                            precalc_r3[1][1][loc_k][loc_j][loc_i][0] *= w;
                            precalc_r3[1][2][loc_k][loc_j][loc_i][0] *= w;

        #ifdef Uon_
                            // precalc_r2[loc_k][loc_j][loc_i][0] = r > pi ? 0.f : w;
                            // precalc_r2[loc_k][loc_j][loc_i][1] = r > pi ? 0.f : w;
                            precalc_r2[loc_k][loc_j][loc_i][0] *= w;
                            //     precalc_r2[loc_k][loc_j][loc_i][1] *=  w;
        #endif
                        }
                    }
                }*/
        delete[] precalc_r3_base;
#ifdef Uon_
        delete[] precalc_r2_base;
#endif
        first = 0;
        //        cout<<"precalc done\n";
    }

#ifdef Eon_
    // #pragma omp parallel sections
    {
        fill(&fft_real[0][0], &fft_real[3][n_cells8], 0.f);
        //   #pragma omp section
        {
            // density field
            size_t i, j, k, jj;
            jj = 0;

            for (k = 0; k < n_space_divz; ++k)
            {
                for (j = 0; j < n_space_divy; ++j)
                {
                    // #pragma omp parallel for simd num_threads(nthreads)
                    for (i = 0; i < n_space_divx; ++i)
                    {
                        fft_real[0][jj + i] = fi->npt[k][j][i];
                    }
                    jj += N0;
                }
                jj += N0N1_2;
            }

            fftwf_execute(planforE); // arrn1 = fft(arrn) multiply fft charge with fft of kernel(i.e field associated with 1 charge)
            for (int c = 0; c < 3; c++)
            {
                // vector_muls(fft_complex[c], fft_complex[3], reinterpret_cast<fftwf_complex *>(precalc_r3[0][c]), n_cells4);
                //  ^ read/write is slow, maybe use host pointer iff it is supported
                const auto dst_std = reinterpret_cast<complex<float> *>(fft_complex[c]), src_std = reinterpret_cast<complex<float> *>(fft_complex[3]),
                           precalc_r3_std = reinterpret_cast<complex<float> *>(precalc_r3[0][c]);
#pragma omp parallel for simd num_threads(nthreads)
                for (int i = 0; i < n_cells4; ++i)
                    dst_std[i] = src_std[i] * precalc_r3_std[i];
            }
#ifdef Uon_
            {
                const auto ptr_std = reinterpret_cast<complex<float> *>(fft_complex[3]),
                           precalc_r2_std = reinterpret_cast<complex<float> *>(precalc_r2);
                complex<float> temp;
                // #pragma omp parallel for simd num_threads(nthreads)
                for (int i = 0; i < n_cells4; ++i)
                    ptr_std[i] *= precalc_r2_std[i];
            }
#endif
            fftwf_execute(planbacE);      // inverse transform to get convolution
                                          // #pragma omp parallel for
                                          /*  float s000[3] = {+1, +1, +1}; // c=0 is x,c=1 is y,c=2 is z
                                            float s001[3] = {+1, +1, -1};
                                            float s010[3] = {+1, -1, +1};
                                            float s011[3] = {+1, -1, -1};
                                            float s100[3] = {-1, +1, +1};
                                            float s101[3] = {-1, +1, -1};
                                            float s110[3] = {-1, -1, +1};
                                            float s111[3] = {-1, -1, -1};
                                            /*
                                                        float s000[3] = {+1, +1, +1}; // c=0 is x,c=1 is y,c=2 is z
                                                        float s001[3] = {+1, +1, -1};
                                                        float s010[3] = {+1, -1, +1};
                                                        float s011[3] = {+1, -1, -1};
                                                        float s100[3] = {-1, +1, +1};
                                                        float s101[3] = {-1, +1, -1};
                                                        float s110[3] = {-1, -1, +1};
                                                        float s111[3] = {-1, -1, -1};
                                                        */
            float s000[3] = {+1, +1, +1}; // c=0 is x,c=1 is y,c=2 is z
            float s001[3] = {-1, +1, +1};
            float s010[3] = {+1, -1, +1};
            float s011[3] = {-1, -1, +1};
            float s100[3] = {+1, +1, -1};
            float s101[3] = {-1, +1, -1};
            float s110[3] = {+1, -1, -1};
            float s111[3] = {-1, -1, -1};
            //*/

            for (int c = 0; c < 3; c++)

            { // 3 axis
                const float *fft_real_c = fft_real[c];
                //               cout << "c " << c << ", thread " << omp_get_thread_num() << ", jj " << jj << endl;
#pragma omp parallel for simd num_threads(nthreads)
                for (k = 0; k < n_space_divz; ++k)
                {
                    for (j = 0; j < n_space_divy; ++j)
                    {

                        for (i = 0; i < n_space_divx; ++i)
                        {
#ifdef octant
                            {
                                int idx000 = k * N0N1 + j * N0 + i; // idx_kji
                                int idx001 = k * N0N1 + j * N0;
                                int idx010 = k * N0N1 + i;
                                int idx011 = k * N0N1;
                                int idx100 = j * N0 + i;
                                int idx101 = j * N0;
                                int idx110 = i;
                                int idx111 = 0;

                                int odx000 = 0;                          // odx_kji
                                int odx001 = i == 0 ? 0 : N0 - i;        // iskip
                                int odx010 = j == 0 ? 0 : N0 * (N1 - j); // jskip
                                int odx011 = odx001 + odx010;
                                int odx100 = k == 0 ? 0 : N0 * N1 * (N2 - k); // kskip
                                int odx101 = odx100 + odx001;
                                int odx110 = odx100 + odx010;
                                int odx111 = odx100 + odx011;
                                fi->E[c][k][j][i] = fi->Ee[c][k][j][i];
                                fi->E[c][k][j][i] += s000[c] * fft_real_c[odx000 + idx000]; // main octant
                                fi->E[c][k][j][i] += s001[c] * fft_real_c[odx001 + idx001]; // add minor effects from other octants
                                fi->E[c][k][j][i] += s010[c] * fft_real_c[odx010 + idx010];
                                fi->E[c][k][j][i] += s011[c] * fft_real_c[odx011 + idx011];
                                fi->E[c][k][j][i] += s100[c] * fft_real_c[odx100 + idx100];
                                fi->E[c][k][j][i] += s101[c] * fft_real_c[odx101 + idx101];
                                fi->E[c][k][j][i] += s110[c] * fft_real_c[odx110 + idx110];
                                fi->E[c][k][j][i] += s111[c] * fft_real_c[odx111 + idx111];
                            }
#else
                            fi->E[c][k][j][i] = fft_real_c[k * N0N1 + j * N0 + i] + fi->Ee[c][k][j][i];
#endif
                        }
                    }
                }
            }
#ifdef Uon_
            jj = 0;

            for (k = 0; k < n_space_divz; ++k)
            {
                // #pragma omp parallel for simd num_threads(nthreads)
                for (j = 0; j < n_space_divy; ++j)
                    memcpy(fi->V[0][k][j], &fft_real[3][jj += N0], sizeof(float) * n_space_divx);
                jj += N0N1_2;
            }
#endif
        }
    }
#else
    memcpy(reinterpret_cast<float *>(E), reinterpret_cast<float *>(Ee), 3 * n_cells * sizeof(float));
#endif
//                  cout << "E done\n";
#ifdef Bon_
    {
        fill(&fft_real[0][0], &fft_real[2][n_cells8], 0.f);
        // #pragma omp section
        // #pragma omp parallel for
        for (int c = 0; c < 3; c++)
        { // 3 axis
            size_t i, j, k, jj;
            jj = 0;
            for (k = 0; k < n_space_divz; ++k)
            {
                for (j = 0; j < n_space_divy; ++j)
                    memcpy(&fft_real[c][jj += N0], fi->jc[c][k][j], sizeof(float) * n_space_divx);
                jj += N0N1_2;
            }
        }
        fftwf_execute(planforB); // arrj1 = fft(arrj)

        const auto ptr = reinterpret_cast<complex<float> *>(fft_complex), r3_1d = reinterpret_cast<complex<float> *>(precalc_r3[1]);
        complex<float> temp1, temp2, temp3;
        for (int i = 0, j = n_cells4, k = n_cells4 * 2; i < n_cells4; ++i, ++j, ++k)
        {
            temp1 = ptr[j] * r3_1d[k] - ptr[k] * r3_1d[j];
            temp2 = ptr[k] * r3_1d[i] - ptr[i] * r3_1d[k];
            temp3 = ptr[i] * r3_1d[j] - ptr[j] * r3_1d[i];
            ptr[i] = temp1;
            ptr[j] = temp2;
            ptr[k] = temp3;
        }
        fftwf_execute(planbacB);
        for (int c = 0; c < 3; c++)
        { // 3 axis
            const float *fft_real_c = fft_real[c];
            // size_t i, j, k, jj;
            // jj = 0;
            for (k = 0; k < n_space_divz; ++k)
            {
                for (j = 0; j < n_space_divy; ++j)
                {
                    for (i = 0; i < n_space_divx; ++i)
                    // fi->B[c][k][j][i] = fft_real_c[jj + i] + fi->Be[c][k][j][i];
#ifdef octant
                    {
                        int idx000 = k * N0N1 + j * N0 + i; // idx_kji
                        int idx001 = k * N0N1 + j * N0;
                        int idx010 = k * N0N1 + i;
                        int idx011 = k * N0N1;
                        int idx100 = j * N0 + i;
                        int idx101 = j * N0;
                        int idx110 = i;
                        int idx111 = 0;

                        int odx000 = 0;                          // odx_kji
                        int odx001 = i == 0 ? 0 : N0 - i;        // iskip
                        int odx010 = j == 0 ? 0 : N0 * (N1 - j); // jskip
                        int odx011 = odx001 + odx010;
                        int odx100 = k == 0 ? 0 : N0 * N1 * (N2 - k); // kskip
                        int odx101 = odx100 + odx001;
                        int odx110 = odx100 + odx010;
                        int odx111 = odx100 + odx011;
                        fi->B[c][k][j][i] = fi->Be[c][k][j][i];
                        fi->B[c][k][j][i] += s000[c] * fft_real_c[odx000 + idx000]; // main octant
                        fi->B[c][k][j][i] += s001[c] * fft_real_c[odx001 + idx001]; // add minor effects from other octants
                        fi->B[c][k][j][i] += s010[c] * fft_real_c[odx010 + idx010];
                        fi->B[c][k][j][i] += s011[c] * fft_real_c[odx011 + idx011];
                        fi->B[c][k][j][i] += s100[c] * fft_real_c[odx100 + idx100];
                        fi->B[c][k][j][i] += s101[c] * fft_real_c[odx101 + idx101];
                        fi->B[c][k][j][i] += s110[c] * fft_real_c[odx110 + idx110];
                        fi->B[c][k][j][i] += s111[c] * fft_real_c[odx111 + idx111];
                    }
#else
                        fi->B[c][k][j][i] = fft_real_c[k * N0N1 + j * N0 + i] + fi->Be[c][k][j][i];
#endif

                    //       jj += N0;
                }
                // jj += N0N1_2;
            }
        }
    }
#else
    memcpy(reinterpret_cast<float *>(fi->B), reinterpret_cast<float *>(fi->Be), 3 * n_cells * sizeof(float));
#endif
//                     cout << "B done\n";
#ifdef Uon_
#ifdef Eon_ // if both Uon and Eon are defined
/*
    {
        // Perform estimate of electric potential energy
        size_t i, j, k, jj = 0;
        float EUtot = 0.f;
        const float *V_1d = reinterpret_cast<float *>(fi->V);
        const float *npt_1d = reinterpret_cast<float *>(fi->npt);
        for (int i = 0; i < n_cells; ++i)
        {
            EUtot += V_1d[i] * npt_1d[i];
        }
        EUtot *= 0.5f; // * e_charge / ev_to_j; <- this is just 1
        cout << "Eele (estimate): " << EUtot << ", ";
    }
    */
#endif
#endif

    int E_exceeds = 0, B_exceeds = 0;
#pragma omp parallel sections
    {
#pragma omp section
        par->Emax = maxvalf(reinterpret_cast<float *>(fi->E), n_cells * 3);
#pragma omp section
        par->Bmax = maxvalf(reinterpret_cast<float *>(fi->B), n_cells * 3);
    }

    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * (par->Bmax + 1e-5f));
    float acc_e = fabsf(par->Emax * e_charge_mass);
    float vel_e = sqrt(kb * Temp_e / e_mass);
    float TE = (sqrt(1 + 2 * a0 * par->a0_f * acc_e / vel_e) - 1) * vel_e / acc_e;
    float TE1 = a0 * par->a0_f / par->Emax * (par->Bmax + .00001);
    cout << "Tcyclotron=" << Tcyclotron << ",Bmax= " << par->Bmax << ", TE=" << TE << ", TE1=" << TE1 << ",Emax= " << par->Emax << endl;
    TE = TE > TE1 ? TE : TE1;

    if (TE < (par->dt[0] * 2 * f1 * ncalc0[0])) // if ideal time step is lower than actual timestep
        E_exceeds = 1;
    else if (TE > (par->dt[0] * 2 * f2 * ncalc0[0]))
        E_exceeds = 2;
    if (Tcyclotron < (par->dt[0] * 4 * f1 * ncalc0[0]))
        B_exceeds = 4;
    else if (Tcyclotron > (par->dt[0] * 4 * f2 * ncalc0[0]))
        B_exceeds = 8;
    // cout <<"calcEBV\n";
    return (E_exceeds + B_exceeds);
}