#include "include/traj.h"
int changedt(particles *pt, int cdt, par *par)
{
    float inc = 0;
    //   cout << endl<< cdt << " ";
    switch (cdt)
    {

    case 1: //
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //     cout << "dt decrease E high B OK \n";
        break;

    case 4:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //       cout << "dt decrease B exceeded E OK\n";
        break;
    case 5:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //     cout << "dt decrease B exceeded and E exceeded\n";
        break;
    case 6:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        // cout << "dt decrease B exceeded and E too low\n";
        break;

    case 9:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //    cout << "dt decrease B too low E too high \n";
        break;
    case 10:
        inc = incf;
        par->dt[0] *= incf;
        par->dt[1] *= incf;
        //    cout << "dt: increase B too low E too low\n";
        break;
    default:
        //   cout << "no change dt" << endl;
        return 0;
    }
#pragma omp parallel for simd
    for (int n = 0; n < par->n_part[0] * 3 * 2; n++)
        pt->pos0[n] = pt->pos1[n] - (pt->pos1[n] - pt->pos0[n]) * inc;
    //   cout << "dt changed" << endl;
    return 1;
}

void changedx(fields *fi, par *par)
{
    par->a0_f *= a0_ff; // Lowest position of cells (x,y,z)
    par->posL[0] *= a0_ff;
    par->posL[1] *= a0_ff;
    par->posL[2] *= a0_ff;
    par->posH[0] *= a0_ff; // Highes position of cells (x,y,z)
    par->posH[1] *= a0_ff;
    par->posH[2] *= a0_ff;
    par->posL_1[0] *= a0_ff; // Lowest position of cells (x,y,z)
    par->posL_1[1] *= a0_ff;
    par->posL_1[2] *= a0_ff;
    par->posH_1[0] *= a0_ff; // Highes position of cells (x,y,z)
    par->posH_1[1] *= a0_ff;
    par->posH_1[2] *= a0_ff;
    par->posL_15[0] *= a0_ff; // Lowest position of cells (x,y,z)
    par->posL_15[1] *= a0_ff;
    par->posL_15[2] *= a0_ff;
    par->posH_15[0] *= a0_ff; // Highes position of cells (x,y,z)
    par->posH_15[1] *= a0_ff;
    par->posH_15[2] *= a0_ff;
    par->posL2[0] *= a0_ff; // Lowest position of cells (x,y,z)
    par->posL2[1] *= a0_ff;
    par->posL2[2] *= a0_ff;
    par->dd[0] *= a0_ff;
    par->dd[1] *= a0_ff;
    par->dd[2] *= a0_ff;

    const size_t n_cells4 = n_space_divx2 * n_space_divy2 * (n_space_divz2 / 2 + 1); // NOTE: This is not actually n_cells * 4, there is an additional buffer that fftw requires.
                                                                                     /*
                                                                                 #pragma omp parallel for simd num_threads(nthreads)
                                                                                     for (size_t i = 0; i < n_cells4 * 3 * 2; i++)
                                                                                         (reinterpret_cast<float *>(fi->precalc_r3))[i] /= (a0_ff * a0_ff);
                                                                                 #ifdef Uon_
                                                                                 #pragma omp parallel for simd num_threads(nthreads)
                                                                                     for (size_t i = 0; i < n_cells4 * 2; i++)
                                                                                         (reinterpret_cast<float *>(fi->precalc_r2))[i] /= a0_ff;
                                                                                 
                                                                                 #endif
                                                                                 */
    buffer_muls(fi->r3_buffer, 1 / (a0_ff * a0_ff), n_cells4 * 2 * 3 * 2);           // 2 floats per complex 3 axis, 2 types E and B
    buffer_muls(fi->r2_buffer, 1 / (a0_ff), n_cells4 * 2);                           // 2 floats per complex
    //  cout << "make cells bigger " << par->nt[0] << " " << nt0prev << ",ao_f = " << par->a0_f << endl;
    generateField(fi, par);
}
