#include "include/traj.h"
int changedt(particles *pt, int cdt, par *par)
{
    float inc = 0;
    //   cout << endl<< cdt << " ";
    switch (par->cdt)
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
    recalcpos(pt, par, inc);
    return 1;
}
void recalcpos(particles *pt, par *par, float inc)
{
    //   cout << "dt changed" << endl;
    static cl::Kernel kernel_recalcposchangedt = cl::Kernel(program_g, "recalcposchangedt");
    kernel_recalcposchangedt.setArg(0, pt->buff_x0_e[0]);    // x0
    kernel_recalcposchangedt.setArg(1, pt->buff_y0_e[0]);    // y0
    kernel_recalcposchangedt.setArg(2, pt->buff_z0_e[0]);    // z0
    kernel_recalcposchangedt.setArg(3, pt->buff_x1_e[0]);    // x1
    kernel_recalcposchangedt.setArg(4, pt->buff_y1_e[0]);    // y1
    kernel_recalcposchangedt.setArg(5, pt->buff_z1_e[0]);    // z1
    kernel_recalcposchangedt.setArg(6, sizeof(float), &inc); // scale factor
    commandQueue_g.enqueueNDRangeKernel(kernel_recalcposchangedt, cl::NullRange, cl::NDRange(par->n_part[0]/16), cl::NullRange);
    commandQueue_g.finish();
    kernel_recalcposchangedt.setArg(0, pt->buff_x0_i[0]);    // x0
    kernel_recalcposchangedt.setArg(1, pt->buff_y0_i[0]);    // y0
    kernel_recalcposchangedt.setArg(2, pt->buff_z0_i[0]);    // z0
    kernel_recalcposchangedt.setArg(3, pt->buff_x1_i[0]);    // x1
    kernel_recalcposchangedt.setArg(4, pt->buff_y1_i[0]);    // y1
    kernel_recalcposchangedt.setArg(5, pt->buff_z1_i[0]);    // z1
    kernel_recalcposchangedt.setArg(6, sizeof(float), &inc); // scale factor
    commandQueue_g.enqueueNDRangeKernel(kernel_recalcposchangedt, cl::NullRange, cl::NDRange(par->n_part[1]/16), cl::NullRange);
    commandQueue_g.finish();
}

void changedx(fields *fi, par *par)
{
    static bool first = true;
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

    cl_int res;
    static cl_kernel kernel_buffer_muls;
    float Bb = 1 / (a0_ff * a0_ff);
    size_t n = n_cells4 * 3 * 2 * 2/16; // 2 floats per complex 3 axis, 2 types E and B float16 vector
    if (first)
    {
        kernel_buffer_muls = clCreateKernel(program_g(), "buffer_muls", &res);
        if (res)
            cout << "create buffer_muls " << res << endl;
    }
    clSetKernelArg(kernel_buffer_muls, 0, sizeof(cl_mem), &fi->r3_buffer);
    clSetKernelArg(kernel_buffer_muls, 1, sizeof(float), &Bb);
    clEnqueueNDRangeKernel(commandQueue_g(), kernel_buffer_muls, 1, NULL, &n, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
    clFinish(commandQueue_g());
#ifdef Uon_
    Bb = 1 / (a0_ff);
    n = n_cells4 * 2/16; // 2 floats per complex
    clSetKernelArg(kernel_buffer_muls, 0, sizeof(cl_mem), &fi->r2_buffer);
    clSetKernelArg(kernel_buffer_muls, 1, sizeof(float), &Bb);
    clEnqueueNDRangeKernel(commandQueue_g(), kernel_buffer_muls, 1, NULL, &n, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
    clFinish(commandQueue_g());
#endif

    generateField(fi, par);
    first = false;
}
