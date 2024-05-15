#include "include/traj.h"
void get_densityfields(fields *fi, particles *pt, par *par)
{
   static bool first = true;
   uint32_t np = n_partd;
   size_t ntry = n_partd / 16;
   cl_int res = 0;

   // static cl::Kernel kernel_density, kernel_df, kernel_dtotal;

   cl::Kernel kernel_density = cl::Kernel(program_g, "density"); // select the kernel program to run
   cl::Kernel kernel_df = cl::Kernel(program_g, "df");           // select the kernel program to run
   cl::Kernel kernel_dtotal = cl::Kernel(program_g, "dtotal");

   cl_kernel nsumi_kernel = clCreateKernel(program_g(), "nsumi", NULL);

   // commandQueue_g.enqueueFillBuffer(fi->buff_npi[0], 0, 0, n_cellsi);
   // commandQueue_g.enqueueFillBuffer(fi->buff_cji[0], 0, 0, n_cellsi * 3);
   //  res = clFinish(commandQueue_g());
   if (res)
      cout << "enqueueFillBuffer e  res: " << res << endl;
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
   res = commandQueue_g.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(ntry), cl::NullRange);
   if (res)
      cout << "kernel_density e  res: " << res << endl; // cout << "run kernel for electron done" << endl;
   commandQueue_g.finish();

   clSetKernelArg(nsumi_kernel, 0, sizeof(cl_mem), &(pt->buff_q_e[0]()));
   clSetKernelArg(nsumi_kernel, 1, sizeof(cl_mem), &par->nt_buffer);
   // clSetKernelArg(nsumi_kernel, 2, sizeof(uint32_t), &np);
   res = clEnqueueNDRangeKernel(commandQueue_g(), nsumi_kernel, 1, NULL, &n2048, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
   if (res)
      cout << "nsumi_kernel e  res: " << res << endl;
   res = clFinish(commandQueue_g());
   if (!fastIO)
   {
      res = clEnqueueReadBuffer(commandQueue_g(), par->nt_buffer, CL_TRUE, 0, sizeof(int) * n2048, par->nt_array, 0, NULL, NULL);
   }
   int nt = 0;
#pragma omp parallel for simd num_threads(nthreads) reduction(+ : nt)
   for (int i = 0; i < n2048; ++i)
      nt += par->nt_array[i];
   par->nt[0] = nt;

   kernel_df.setArg(0, fi->buff_np_e[0]);          // np ion
   kernel_df.setArg(1, fi->buff_npi[0]);           // np ion temp integer
   kernel_df.setArg(2, fi->buff_currentj_e[0]);    // current
   kernel_df.setArg(3, fi->buff_cji[0]);           // current
   kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor
   kernel_df.setArg(5, sizeof(float), &par->dt[0]);
   res = commandQueue_g.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   if (res)
      cout << "kernel_df e  res: " << res << endl;
   commandQueue_g.finish();

   // commandQueue_g.enqueueFillBuffer(fi->buff_npi[0], 0, 0, n_cellsi);
   //  commandQueue_g.enqueueFillBuffer(fi->buff_cji[0], 0, 0, n_cellsi * 3);
   // res = clFinish(commandQueue_g());
   if (res)
      cout << "enqueueFillBuffer i  res: " << res << endl;
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
   res = commandQueue_g.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(ntry), cl::NullRange);
   if (res)
      cout << "kernel_density i  res: " << res << endl;
   commandQueue_g.finish(); // wait for the end of the kernel program

   res = clSetKernelArg(nsumi_kernel, 0, sizeof(cl_mem), &(pt->buff_q_i[0]()));
   if (res)
      cout << "clSetKernelArg nsumi_kernel i 0 res: " << res << endl;
   res = clSetKernelArg(nsumi_kernel, 1, sizeof(cl_mem), &par->nt_buffer);
   if (res)
      cout << "clSetKernelArg nsumi_kernel i 1 res: " << res << endl;
   // res = clSetKernelArg(nsumi_kernel, 2, sizeof(uint32_t), &np);
   //  if (res)
   //   cout << "clSetKernelArg nsumi_kernel i 2 res: " << res << endl;
   res = clEnqueueNDRangeKernel(commandQueue_g(), nsumi_kernel, 1, NULL, &n2048, NULL, 0, NULL, NULL); //  Enqueue NDRange kernel
   if (res)
      cout << "nsumi_kernel i  res: " << res << endl;
   res = clFinish(commandQueue_g());
   if (!fastIO)
   {
      res = clEnqueueReadBuffer(commandQueue_g(), par->nt_buffer, CL_TRUE, 0, sizeof(int) * n2048, par->nt_array, 0, NULL, NULL);
   }
   nt = 0;
#pragma omp parallel for simd num_threads(nthreads) reduction(+ : nt)
   for (int i = 0; i < n2048; ++i)
      nt += par->nt_array[i];
   par->nt[1] = nt;
   // cout << "nt (e) = " << par->nt[0] << ", nt (i) = " << par->nt[1] << ", n = " << n_part_2048 << endl;
   //  cout << "nt (i) = " << nt << endl;

   kernel_df.setArg(0, fi->buff_np_i[0]);          // np ion
   kernel_df.setArg(1, fi->buff_npi[0]);           // np ion temporary integer
   kernel_df.setArg(2, fi->buff_currentj_i[0]);    // current
   kernel_df.setArg(3, fi->buff_cji[0]);           // current
   kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor
   kernel_df.setArg(5, sizeof(float), &par->dt[1]);
   res = commandQueue_g.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   if (res)
      cout << "kernel_df i  res: " << res << endl;
   commandQueue_g.finish();
   // sum total electron and ion densitiies and current densities for E B calculations

   uint32_t ntemp = n_cells;
   //  cout << "\neions  " << timer.elapsed() << "s, \n";
   // sum total electron and ion densitiies and current densities for E B calculations
   kernel_dtotal.setArg(0, fi->buff_np_e[0]);       // np electron
   kernel_dtotal.setArg(1, fi->buff_np_i[0]);       // np ion
   kernel_dtotal.setArg(2, fi->buff_currentj_e[0]); // current
   kernel_dtotal.setArg(3, fi->buff_currentj_i[0]); // current
   kernel_dtotal.setArg(4, fi->buff_npt[0]);        // total particles density
   kernel_dtotal.setArg(5, fi->buff_jc[0]);         // total current density
   kernel_dtotal.setArg(6, sizeof(uint32_t), &ntemp);
   commandQueue_g.enqueueNDRangeKernel(kernel_dtotal, cl::NullRange, cl::NDRange(n_cells_16), cl::NullRange);
   if (res)
      cout << "kernel_dtotal  res: " << res << endl;
   commandQueue_g.finish();
}