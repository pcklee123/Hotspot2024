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
   if (first)
   { // get whether or not we are on an iGPU/similar, and can use certain memmory optimizations
      bool temp;
      default_device_g.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &temp);
      if (temp == true)
         info_file << "Using unified memory: " << temp << " ";
      else
         info_file << "No unified memory: " << temp << " ";
      fastIO = temp;
      fastIO = false;
   }

   //  create buffers on the device
   /** IMPORTANT: do not use CL_MEM_USE_HOST_PTR if on dGPU **/
   /** HOST_PTR is only used so that memory is not copied, but instead shared between CPU and iGPU in RAM**/
   // Note that special alignment has been given to Ea, Ba, y0, z0, x0, x1, y1 in order to actually do this properly

   // Assume buffers A, B, I, J (Ea, Ba, ci, cf) will always be the same. Then we save a bit of time.

   static cl::Buffer buff_np_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->np[0] : NULL);
   static cl::Buffer buff_np_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->np[1] : NULL);

   static cl::Buffer buff_currentj_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->currentj[0] : NULL);
   static cl::Buffer buff_currentj_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->currentj[1] : NULL);

   static cl::Buffer buff_npi(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsi, fastIO ? fi->npi : NULL);
   static cl::Buffer buff_cji(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsi * 3, fastIO ? fi->cji : NULL);

   //  cout << "buffers " << endl;
   static cl::Buffer buff_x0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[0] : NULL); // x0
   static cl::Buffer buff_y0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[0] : NULL); // y0
   static cl::Buffer buff_z0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[0] : NULL); // z0
   static cl::Buffer buff_x1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[0] : NULL); // x1
   static cl::Buffer buff_y1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[0] : NULL); // y1
   static cl::Buffer buff_z1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[0] : NULL); // z1
                                                                                                                                     //  cout << "buffers " << endl;
   static cl::Buffer buff_q_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[0] : NULL);      // q

   static cl::Buffer buff_x0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[1] : NULL); // x0
   static cl::Buffer buff_y0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[1] : NULL); // y0
   static cl::Buffer buff_z0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[1] : NULL); // z0
   static cl::Buffer buff_x1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[1] : NULL); // x1
   static cl::Buffer buff_y1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[1] : NULL); // y1
   static cl::Buffer buff_z1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[1] : NULL); // z1

   static cl::Buffer buff_q_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[1] : NULL); // q

   // cout << "command q" << endl; //  create queue to which we will push commands for the device.

   static cl::CommandQueue queue(context_g, default_device_g);
   cl::Kernel kernel_density = cl::Kernel(program_g, "density"); // select the kernel program to run
   cl::Kernel kernel_df = cl::Kernel(program_g, "df");           // select the kernel program to run
   // write input arrays to the device
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

      queue.enqueueWriteBuffer(buff_q_e, CL_TRUE, 0, n4, pt->q[0]);
   }

   queue.enqueueFillBuffer(buff_npi, 0, 0, n_cellsi);
   queue.enqueueFillBuffer(buff_cji, 0, 0, n_cellsi * 3);

   //  set arguments to be fed into the kernel program
   // cout << "kernel arguments for electron" << endl;

   kernel_density.setArg(0, buff_x0_e); // x0
   kernel_density.setArg(1, buff_y0_e); // y0
   kernel_density.setArg(2, buff_z0_e); // z0
   kernel_density.setArg(3, buff_x1_e); // x1
   kernel_density.setArg(4, buff_y1_e); // y1
   kernel_density.setArg(5, buff_z1_e); // z1
   kernel_density.setArg(6, buff_npi);  // npt
   kernel_density.setArg(7, buff_cji);  // current
   kernel_density.setArg(8, buff_q_e);  // q
                                        // kernel_density.setArg(14, sizeof(int), n_cells);          // ncells
                                        // cout << "run kernel for electron" << endl;

   // run the kernel
   queue.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n0), cl::NullRange);
   // cout << "run kernel for electron done" << endl;
   queue.finish();
   kernel_df.setArg(0, buff_np_e);       // np ion
   kernel_df.setArg(1, buff_npi);        // np ion temp integer
   kernel_df.setArg(2, buff_currentj_e); // current
   kernel_df.setArg(3, buff_cji);        // current
   queue.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   queue.finish();
   // cout << "read electron density" << endl;
   if (fastIO)
   { // is mapping required?
     // mapped_buff_x0_e = (float *)queue.enqueueMapBuffer(buff_x0_e, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * n); queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
   }
   else
   {
      queue.enqueueReadBuffer(buff_np_e, CL_TRUE, 0, n_cellsf, fi->np[0]);
      queue.enqueueReadBuffer(buff_currentj_e, CL_TRUE, 0, n_cellsf * 3, fi->currentj[0]);
      // ions next

      queue.enqueueWriteBuffer(buff_x0_i, CL_TRUE, 0, n4, pt->pos0x[1]);
      queue.enqueueWriteBuffer(buff_y0_i, CL_TRUE, 0, n4, pt->pos0y[1]);
      queue.enqueueWriteBuffer(buff_z0_i, CL_TRUE, 0, n4, pt->pos0z[1]);
      queue.enqueueWriteBuffer(buff_x1_i, CL_TRUE, 0, n4, pt->pos1x[1]);
      queue.enqueueWriteBuffer(buff_y1_i, CL_TRUE, 0, n4, pt->pos1y[1]);
      queue.enqueueWriteBuffer(buff_z1_i, CL_TRUE, 0, n4, pt->pos1z[1]);

      queue.enqueueWriteBuffer(buff_q_i, CL_TRUE, 0, n4, pt->q[1]);
   }
   queue.enqueueFillBuffer(buff_npi, 0, 0, n_cellsi);
   queue.enqueueFillBuffer(buff_cji, 0, 0, n_cellsi * 3);
   //  set arguments to be fed into the kernel program
   kernel_density.setArg(0, buff_x0_i); // x0
   kernel_density.setArg(1, buff_y0_i); // y0
   kernel_density.setArg(2, buff_z0_i); // z0
   kernel_density.setArg(3, buff_x1_i); // x1
   kernel_density.setArg(4, buff_y1_i); // y1
   kernel_density.setArg(5, buff_z1_i); // z1
   kernel_density.setArg(6, buff_npi);  // npt
   kernel_density.setArg(7, buff_cji);  // current
   kernel_density.setArg(8, buff_q_i);  // q
                                        // kernel_density.setArg(14, sizeof(int), &n_cells);          // ncells
                                        // cout << "run kernel for ions" << endl;
   //  run the kernel
   queue.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n0), cl::NullRange);
   queue.finish();                       // wait for the end of the kernel program
   kernel_df.setArg(0, buff_np_i);       // np ion
   kernel_df.setArg(1, buff_npi);        // np ion temp integer
   kernel_df.setArg(2, buff_currentj_i); // current
   kernel_df.setArg(3, buff_cji);        // current
   queue.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
   queue.finish();
   // read result arrays from the device to main memory
   if (fastIO)
   { // is mapping required?
     // mapped_buff_x0_e = (float *)queue.enqueueMapBuffer(buff_x0_e, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * n); queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
   }
   else
   {
      queue.enqueueReadBuffer(buff_q_e, CL_TRUE, 0, n4, pt->q[0]);
      queue.enqueueReadBuffer(buff_q_i, CL_TRUE, 0, n4, pt->q[1]);

      queue.enqueueReadBuffer(buff_np_i, CL_TRUE, 0, n_cellsf, fi->np[1]);
      queue.enqueueReadBuffer(buff_currentj_i, CL_TRUE, 0, n_cellsf * 3, fi->currentj[1]);
#pragma omp parallel for simd num_threads(nthreads)
      for (unsigned int i = 0; i < n_cells; i++)
         (reinterpret_cast<float *>(fi->npt))[i] = (reinterpret_cast<float *>(fi->np[0]))[i] + (reinterpret_cast<float *>(fi->np[1]))[i];

#pragma omp parallel for simd num_threads(nthreads)
      for (unsigned int i = 0; i < n_cells * 3; i++)
         (reinterpret_cast<float *>(fi->jc))[i] = (reinterpret_cast<float *>(fi->currentj[0]))[i] / par->dt[0] + (reinterpret_cast<float *>(fi->currentj[1]))[i] / par->dt[1];
#pragma omp barrier
   }

   first = false;
}