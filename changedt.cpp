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