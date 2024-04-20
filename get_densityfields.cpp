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
   cl_int res = 0;
   // uint64_t n = n_cells / 256;
   unsigned int np = n_partd;
   uint64_t n = n_partd / 2048;
   cl_mem nt_buffer = clCreateBuffer(context_g(), CL_MEM_READ_WRITE, n * sizeof(int), 0, &res);
   int *nt_array = (int *)_aligned_malloc(sizeof(int) * n, par->cl_align);
   cl_kernel nsumi_kernel = clCreateKernel(program_g(), "nsumi", NULL);
   clSetKernelArg(nsumi_kernel, 0, sizeof(cl_mem), &(pt->buff_q_e[0]()));
   clSetKernelArg(nsumi_kernel, 1, sizeof(cl_mem), &nt_buffer);
   clSetKernelArg(nsumi_kernel, 2, sizeof(unsigned int), &np);
   res = clEnqueueNDRangeKernel(commandQueue_g(), nsumi_kernel, 1, NULL, &n, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
   res = clFinish(commandQueue_g());
   res = clEnqueueReadBuffer(commandQueue_g(), nt_buffer, CL_TRUE, 0, sizeof(int) * n, nt_array, 0, NULL, NULL);
   int nt = 0;
   for (int i = 0; i < n; ++i)
      nt += nt_array[i];
   par->nt[0] = nt;
   //cout << "nt (e) = " << nt << ", n = " << n << endl;

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
   commandQueue_g.finish(); // wait for the end of the kernel program

   clSetKernelArg(nsumi_kernel, 0, sizeof(cl_mem), &(pt->buff_q_i[0]()));
   clSetKernelArg(nsumi_kernel, 1, sizeof(cl_mem), &nt_buffer);
   res = clEnqueueNDRangeKernel(commandQueue_g(), nsumi_kernel, 1, NULL, &n, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
   res = clFinish(commandQueue_g());
   res = clEnqueueReadBuffer(commandQueue_g(), nt_buffer, CL_TRUE, 0, sizeof(int) * n, nt_array, 0, NULL, NULL);
   nt = 0;
   for (int i = 0; i < n; ++i)
      nt += nt_array[i];
   par->nt[1] = nt;
   //cout << "nt (i) = " << nt << endl;
   _aligned_free(nt_array);
   clReleaseMemObject(nt_buffer);

   kernel_df.setArg(0, fi->buff_np_i[0]);          // np ion
   kernel_df.setArg(1, fi->buff_npi[0]);           // np ion temp integer
   kernel_df.setArg(2, fi->buff_currentj_i[0]);    // current
   kernel_df.setArg(3, fi->buff_cji[0]);           // current
   kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor
   commandQueue_g.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   commandQueue_g.finish();
   // sum total electron and ion densitiies and current densities for E B calculations

   uint64_t ntemp = n_cells;
   //  cout << "\neions  " << timer.elapsed() << "s, \n";
   // sum total electron and ion densitiies and current densities for E B calculations
   kernel_dtotal.setArg(0, fi->buff_np_e[0]);       // np electron
   kernel_dtotal.setArg(1, fi->buff_np_i[0]);       // np ion
   kernel_dtotal.setArg(2, fi->buff_currentj_e[0]); // current
   kernel_dtotal.setArg(3, fi->buff_currentj_i[0]); // current
   kernel_dtotal.setArg(4, fi->buff_npt[0]);        // total particles density
   kernel_dtotal.setArg(5, fi->buff_jc[0]);         // total current density
   kernel_dtotal.setArg(6, sizeof(uint64_t), &ntemp);
   commandQueue_g.enqueueNDRangeKernel(kernel_dtotal, cl::NullRange, cl::NDRange(n_cells / 16), cl::NullRange);
   commandQueue_g.finish();
}