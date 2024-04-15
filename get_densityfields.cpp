#include "include/traj.h"
void get_densityfields(fields *fi, particles *pt, par *par)
{
   unsigned int n0 = n_partd;                 // number of particles ci[0];
   unsigned int n = n_partd * 2;              // both electron and ion
   unsigned int n4 = n_partd * sizeof(float); // number of particles * sizeof(float)
   unsigned int n8 = n * sizeof(float);       // number of particles * sizeof(float)
   unsigned int nc = n_cells * ncoeff * 3;    // trilin constatnts have 8 coefficients 3 components
   unsigned int n_cellsi = n_cells * sizeof(int);
   unsigned int n_cellsf = n_cells * sizeof(float);
   static bool fastIO;
   static bool first = true;
   static int ncalc_e = 0, ncalc_i = 0;
   // cout << "check for unified memory " << endl;
   //  create buffers on the device
   /** IMPORTANT: do not use CL_MEM_USE_HOST_PTR if on dGPU **/
   /** HOST_PTR is only used so that memory is not copied, but instead shared between CPU and iGPU in RAM**/
   // Note that special alignment has been given to Ea, Ba, y0, z0, x0, x1, y1 in order to actually do this properly
   // Assume buffers A, B, I, J (Ea, Ba, ci, cf) will always be the same. Then we save a bit of time.

  // cl::Buffer buff_npt = fi->buff_npt[0];
  // cl::Buffer buff_jc = fi->buff_jc[0];

   //  cout << "buffers " << endl;
   static cl::Buffer buff_x0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[0] : NULL); // x0
   static cl::Buffer buff_y0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[0] : NULL); // y0
   static cl::Buffer buff_z0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[0] : NULL); // z0
   static cl::Buffer buff_x1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[0] : NULL); // x1
   static cl::Buffer buff_y1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[0] : NULL); // y1
   static cl::Buffer buff_z1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[0] : NULL); // z1
                                                                                                                                     //  cout << "buffers " << endl;


   static cl::Buffer buff_x0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[1] : NULL); // x0
   static cl::Buffer buff_y0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[1] : NULL); // y0
   static cl::Buffer buff_z0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[1] : NULL); // z0
   static cl::Buffer buff_x1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[1] : NULL); // x1
   static cl::Buffer buff_y1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[1] : NULL); // y1
   static cl::Buffer buff_z1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[1] : NULL); // z1



   // cout << "command q" << endl; //  create queue to which we will push commands for the device.

   static cl::CommandQueue queue(context_g, default_device_g);
   cl::Kernel kernel_density = cl::Kernel(program_g, "density"); // select the kernel program to run
   cl::Kernel kernel_df = cl::Kernel(program_g, "df");           // select the kernel program to run
                                                                 // write input arrays to the device
   cl::Kernel kernel_dtotal = cl::Kernel(program_g, "dtotal");
   if (fastIO)
   { // is mapping required? // Yes we might need to map because OpenCL does not guarantee that the data will be shared, alternatively use SVM
     // auto * mapped_buff_x0_e = (float *)queue.enqueueMapBuffer(buff_x0_e, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * n); queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
   }
   else
   {
      queue.enqueueWriteBuffer(buff_x0_e, CL_TRUE, 0, n4, pt->pos0x[0]);
      queue.enqueueWriteBuffer(buff_y0_e, CL_TRUE, 0, n4, pt->pos0y[0]);
      queue.enqueueWriteBuffer(buff_z0_e, CL_TRUE, 0, n4, pt->pos0z[0]);
      queue.enqueueWriteBuffer(buff_x1_e, CL_TRUE, 0, n4, pt->pos1x[0]);
      queue.enqueueWriteBuffer(buff_y1_e, CL_TRUE, 0, n4, pt->pos1y[0]);
      queue.enqueueWriteBuffer(buff_z1_e, CL_TRUE, 0, n4, pt->pos1z[0]);

      queue.enqueueWriteBuffer(fi->buff_q_e[0], CL_TRUE, 0, n4, pt->q[0]);

      // queue.enqueueReadBuffer(buff_np_e, CL_TRUE, 0, n_cellsf, fi->np[0]);
      // queue.enqueueReadBuffer(buff_currentj_e, CL_TRUE, 0, n_cellsf * 3, fi->currentj[0]);
      //  ions next

      queue.enqueueWriteBuffer(buff_x0_i, CL_TRUE, 0, n4, pt->pos0x[1]);
      queue.enqueueWriteBuffer(buff_y0_i, CL_TRUE, 0, n4, pt->pos0y[1]);
      queue.enqueueWriteBuffer(buff_z0_i, CL_TRUE, 0, n4, pt->pos0z[1]);
      queue.enqueueWriteBuffer(buff_x1_i, CL_TRUE, 0, n4, pt->pos1x[1]);
      queue.enqueueWriteBuffer(buff_y1_i, CL_TRUE, 0, n4, pt->pos1y[1]);
      queue.enqueueWriteBuffer(buff_z1_i, CL_TRUE, 0, n4, pt->pos1z[1]);

      queue.enqueueWriteBuffer(fi->buff_q_i[0], CL_TRUE, 0, n4, pt->q[1]);
   }

   queue.enqueueFillBuffer(fi->buff_npi[0], 0, 0, n_cellsi);
   queue.enqueueFillBuffer(fi->buff_cji[0], 0, 0, n_cellsi * 3);

   //  set arguments to be fed into the kernel program
   // cout << "kernel arguments for electron" << endl;

   kernel_density.setArg(0, buff_x0_e);                 // x0
   kernel_density.setArg(1, buff_y0_e);                 // y0
   kernel_density.setArg(2, buff_z0_e);                 // z0
   kernel_density.setArg(3, buff_x1_e);                 // x1
   kernel_density.setArg(4, buff_y1_e);                 // y1
   kernel_density.setArg(5, buff_z1_e);                 // z1
   kernel_density.setArg(6, fi->buff_npi[0]);                  // npt
   kernel_density.setArg(7, fi->buff_cji[0]);                  // current
   kernel_density.setArg(8, fi->buff_q_e[0]);                  // q
   kernel_density.setArg(9, sizeof(float), &par->a0_f); // scale factor
                                                        // kernel_density.setArg(14, sizeof(int), n_cells);          // ncells
                                                        // cout << "run kernel for electron" << endl;

   // run the kernel
   queue.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n0), cl::NullRange);
   // cout << "run kernel for electron done" << endl;
   queue.finish();

   kernel_df.setArg(0, fi->buff_np_e[0]);                 // np ion
   kernel_df.setArg(1, fi->buff_npi[0]);                  // np ion temp integer
   kernel_df.setArg(2, fi->buff_currentj_e[0]);           // current
   kernel_df.setArg(3, fi->buff_cji[0]);                  // current
   kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor

   queue.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   queue.finish();
   // cout << "read electron density" << endl;

   queue.enqueueFillBuffer(fi->buff_npi[0], 0, 0, n_cellsi);
   queue.enqueueFillBuffer(fi->buff_cji[0], 0, 0, n_cellsi * 3);
   //  set arguments to be fed into the kernel program
   kernel_density.setArg(0, buff_x0_i);                 // x0
   kernel_density.setArg(1, buff_y0_i);                 // y0
   kernel_density.setArg(2, buff_z0_i);                 // z0
   kernel_density.setArg(3, buff_x1_i);                 // x1
   kernel_density.setArg(4, buff_y1_i);                 // y1
   kernel_density.setArg(5, buff_z1_i);                 // z1
   kernel_density.setArg(6, fi->buff_npi[0]);                  // npt
   kernel_density.setArg(7, fi->buff_cji[0]);                  // current
   kernel_density.setArg(8, fi->buff_q_i[0]);                  // q
   kernel_density.setArg(9, sizeof(float), &par->a0_f); // scale factor
                                                        // kernel_density.setArg(14, sizeof(int), &n_cells);          // ncells
                                                        // cout << "run kernel for ions" << endl;
   //  run the kernel
   queue.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n0), cl::NullRange);
   queue.finish();                                 // wait for the end of the kernel program
   kernel_df.setArg(0, fi->buff_np_i[0]);                 // np ion
   kernel_df.setArg(1, fi->buff_npi[0]);                  // np ion temp integer
   kernel_df.setArg(2, fi->buff_currentj_i[0]);           // current
   kernel_df.setArg(3, fi->buff_cji[0]);                  // current
   kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor
   queue.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   queue.finish();

   kernel_dtotal.setArg(0, fi->buff_np_e[0]);       // np ion
   kernel_dtotal.setArg(1, fi->buff_np_i[0]);       // np ion
   kernel_dtotal.setArg(2, fi->buff_currentj_e[0]); // current
   kernel_dtotal.setArg(3, fi->buff_currentj_i[0]); // current
   kernel_dtotal.setArg(4, fi->buff_npt[0]);        // total particles density
   kernel_dtotal.setArg(5, fi->buff_jc[0]);         // total current density
   kernel_dtotal.setArg(6, sizeof(size_t), &n_cells);
   queue.enqueueNDRangeKernel(kernel_dtotal, cl::NullRange, cl::NDRange(n_cells / 16), cl::NullRange);
   queue.finish();
   // read result arrays from the device to main memory
   if (fastIO)
   { // is mapping required?
     // mapped_buff_x0_e = (float *)queue.enqueueMapBuffer(buff_x0_e, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * n); queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
   }
   else
   {
      queue.enqueueReadBuffer(fi->buff_q_e[0], CL_TRUE, 0, n4, pt->q[0]);
      queue.enqueueReadBuffer(fi->buff_q_i[0], CL_TRUE, 0, n4, pt->q[1]);

      queue.enqueueReadBuffer(fi->buff_np_e[0], CL_TRUE, 0, n_cellsf, fi->np[0]);
      queue.enqueueReadBuffer(fi->buff_np_i[0], CL_TRUE, 0, n_cellsf, fi->np[1]);

      queue.enqueueReadBuffer(fi->buff_currentj_e[0], CL_TRUE, 0, n_cellsf * 3, fi->currentj[0]);
      queue.enqueueReadBuffer(fi->buff_currentj_i[0], CL_TRUE, 0, n_cellsf * 3, fi->currentj[1]);

      queue.enqueueReadBuffer(fi->buff_npt[0], CL_TRUE, 0, n_cellsf, fi->npt);
      queue.enqueueReadBuffer(fi->buff_jc[0], CL_TRUE, 0, n_cellsf * 3, fi->jc);
   }

   first = false;
}