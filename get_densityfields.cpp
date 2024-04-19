#include "include/traj.h"
void get_densityfields(fields *fi, particles *pt, par *par)
{
   static bool first = true;

   cl::Kernel kernel_density = cl::Kernel(program_g, "density"); // select the kernel program to run
   cl::Kernel kernel_df = cl::Kernel(program_g, "df");           // select the kernel program to run
   cl::Kernel kernel_dtotal = cl::Kernel(program_g, "dtotal");

   commandQueue_g.enqueueFillBuffer(fi->buff_npi[0], 0, 0, n_cellsi);
   commandQueue_g.enqueueFillBuffer(fi->buff_cji[0], 0, 0, n_cellsi * 3);

   //  set arguments to be fed into the kernel program
   // cout << "kernel arguments for electron" << endl;

   kernel_density.setArg(0, pt->buff_x0_e[0]);          // x0
   kernel_density.setArg(1, pt->buff_y0_e[0]);          // y0
   kernel_density.setArg(2, pt->buff_z0_e[0]);          // z0
   kernel_density.setArg(3, pt->buff_x1_e[0]);          // x1
   kernel_density.setArg(4, pt->buff_y1_e[0]);          // y1
   kernel_density.setArg(5, pt->buff_z1_e[0]);          // z1
   kernel_density.setArg(6, fi->buff_npi[0]);           // npt
   kernel_density.setArg(7, fi->buff_cji[0]);           // current
   kernel_density.setArg(8, pt->buff_q_e[0]);           // q
   kernel_density.setArg(9, sizeof(float), &par->a0_f); // scale factor

   // cout << "run kernel to get density for electron" << endl;
   commandQueue_g.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n_partd), cl::NullRange);
   // cout << "run kernel for electron done" << endl;
   commandQueue_g.finish();

   kernel_df.setArg(0, fi->buff_np_e[0]);          // np ion
   kernel_df.setArg(1, fi->buff_npi[0]);           // np ion temp integer
   kernel_df.setArg(2, fi->buff_currentj_e[0]);    // current
   kernel_df.setArg(3, fi->buff_cji[0]);           // current
   kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor

   commandQueue_g.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   commandQueue_g.finish();

   commandQueue_g.enqueueFillBuffer(fi->buff_npi[0], 0, 0, n_cellsi);
   commandQueue_g.enqueueFillBuffer(fi->buff_cji[0], 0, 0, n_cellsi * 3);

   kernel_density.setArg(0, pt->buff_x0_i[0]);          // x0
   kernel_density.setArg(1, pt->buff_y0_i[0]);          // y0
   kernel_density.setArg(2, pt->buff_z0_i[0]);          // z0
   kernel_density.setArg(3, pt->buff_x1_i[0]);          // x1
   kernel_density.setArg(4, pt->buff_y1_i[0]);          // y1
   kernel_density.setArg(5, pt->buff_z1_i[0]);          // z1
   kernel_density.setArg(6, fi->buff_npi[0]);           // npt
   kernel_density.setArg(7, fi->buff_cji[0]);           // current
   kernel_density.setArg(8, pt->buff_q_i[0]);           // q
   kernel_density.setArg(9, sizeof(float), &par->a0_f); // scale factor
   // cout << "run kernel for ions" << endl;
   commandQueue_g.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n_partd), cl::NullRange);
   commandQueue_g.finish();                        // wait for the end of the kernel program
   kernel_df.setArg(0, fi->buff_np_i[0]);          // np ion
   kernel_df.setArg(1, fi->buff_npi[0]);           // np ion temp integer
   kernel_df.setArg(2, fi->buff_currentj_i[0]);    // current
   kernel_df.setArg(3, fi->buff_cji[0]);           // current
   kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor
   commandQueue_g.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   commandQueue_g.finish();
   // sum total electron and ion densitiies and current densities for E B calculations

   unsigned long ntemp = n_cells;
   //  cout << "\neions  " << timer.elapsed() << "s, \n";
   // sum total electron and ion densitiies and current densities for E B calculations
   kernel_dtotal.setArg(0, fi->buff_np_e[0]);       // np ion
   kernel_dtotal.setArg(1, fi->buff_np_i[0]);       // np ion
   kernel_dtotal.setArg(2, fi->buff_currentj_e[0]); // current
   kernel_dtotal.setArg(3, fi->buff_currentj_i[0]); // current
   kernel_dtotal.setArg(4, fi->buff_npt[0]);        // total particles density
   kernel_dtotal.setArg(5, fi->buff_jc[0]);         // total current density
   kernel_dtotal.setArg(6, sizeof(unsigned long), &ntemp);
   commandQueue_g.enqueueNDRangeKernel(kernel_dtotal, cl::NullRange, cl::NDRange(n_cells / 16), cl::NullRange);
   
   commandQueue_g.finish();
}