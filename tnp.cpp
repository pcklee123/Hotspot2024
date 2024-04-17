#include "include/traj.h"
void tnp(fields *fi, particles *pt, par *par)
{
   unsigned int n0 = par->n_part[1];       // number of deuteron particles ;
   unsigned int n = par->n_part[2];        // both electron and ion
   unsigned int n4 = n0 * sizeof(float);   // number of particles * sizeof(float)
   unsigned int n8 = n * sizeof(float);    // number of particles * sizeof(float)
   unsigned int nc = n_cells * ncoeff * 3; // trilin constatnts have 8 coefficients 3 components

   static bool fastIO;
   static bool first = true;

   //  create buffers on the device
   /** IMPORTANT: do not use CL_MEM_USE_HOST_PTR if on dGPU **/
   /** HOST_PTR is only used so that memory is not copied, but instead shared between CPU and iGPU in RAM**/
   // Note that special alignment has been given to Ea, Ba, y0, z0, x0, x1, y1 in order to actually do this properly
   // Assume buffers A, B, I, J (Ea, Ba, ci, cf) will always be the same. Then we save a bit of time.
   cl::Buffer buff_E = fi->buff_E[0];
   cl::Buffer buff_B = fi->buff_B[0];
   cl::Buffer buff_Ee = fi->buff_Ee[0];
   cl::Buffer buff_Be = fi->buff_Be[0];

   cl::Buffer buff_npt = fi->buff_npt[0];
   cl::Buffer buff_jc = fi->buff_jc[0];

   cl::Buffer buff_np_e = fi->buff_np_e[0];
   cl::Buffer buff_np_i = fi->buff_np_i[0];
   cl::Buffer buff_currentj_e = fi->buff_currentj_e[0];
   cl::Buffer buff_currentj_i = fi->buff_currentj_i[0];

   cl::Buffer buff_npi = fi->buff_npi[0];
   cl::Buffer buff_cji = fi->buff_cji[0];

   static cl::Buffer buff_Ea(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, sizeof(float) * nc, fastIO ? fi->Ea : NULL);
   static cl::Buffer buff_Ba(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, sizeof(float) * nc, fastIO ? fi->Ba : NULL);

   cl::Buffer buff_x0_e = pt->buff_x0_e[0];
   cl::Buffer buff_y0_e = pt->buff_y0_e[0];
   cl::Buffer buff_z0_e = pt->buff_z0_e[0];
   cl::Buffer buff_x1_e = pt->buff_x1_e[0];
   cl::Buffer buff_y1_e = pt->buff_y1_e[0];
   cl::Buffer buff_z1_e = pt->buff_z1_e[0];

   cl::Buffer buff_q_e = pt->buff_q_e[0];

   cl::Buffer buff_x0_i = pt->buff_x0_i[0];
   cl::Buffer buff_y0_i = pt->buff_y0_i[0];
   cl::Buffer buff_z0_i = pt->buff_z0_i[0];
   cl::Buffer buff_x1_i = pt->buff_x1_i[0];
   cl::Buffer buff_y1_i = pt->buff_y1_i[0];
   cl::Buffer buff_z1_i = pt->buff_z1_i[0];

   cl::Buffer buff_q_i = pt->buff_q_i[0];

   // cout << "command q" << endl; //  create queue to which we will push commands for the device.
   cl::CommandQueue queue = commandQueue_g;

#if defined(sphere)
#if defined(octant)
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicito"); // select the kernel program to run
#else
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicit"); // select the kernel program to run
#endif
#endif

#ifdef cylinder
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicitz"); // select the kernel program to run
#endif
   cl::Kernel kernel_trilin = cl::Kernel(program_g, "trilin_k"); // select the kernel program to run
   cl::Kernel kernel_density = cl::Kernel(program_g, "density"); // select the kernel program to run
   cl::Kernel kernel_df = cl::Kernel(program_g, "df");           // select the kernel program to run
   cl::Kernel kernel_dtotal = cl::Kernel(program_g, "dtotal");
#ifdef BFon_
   // check minus sign
   par->Bcoef[0] = -(float)qs[0] * e_charge_mass / (float)mp[0] * par->dt[0] * 0.5f;
   par->Bcoef[1] = -(float)qs[1] * e_charge_mass / (float)mp[1] * par->dt[1] * 0.5f;
#else
   par->Bcoef[0] = 0;
   par->Bcoef[1] = 0;
#endif
#ifdef EFon_
   par->Ecoef[0] = -(float)qs[0] * e_charge_mass / (float)mp[0] * par->dt[0] * 0.5f * par->dt[0]; // multiply by dt because of the later portion of cl code
   par->Ecoef[1] = -(float)qs[1] * e_charge_mass / (float)mp[1] * par->dt[1] * 0.5f * par->dt[1]; // multiply by dt because of the later portion of cl code
#else
   par->Ecoef[0] = 0;
   par->Ecoef[1] = 0;
#endif
   // cout << " Bconst=" << par->Bcoef[0] << ", Econst=" << par->Ecoef[0] << endl;
   if (first)
   { // get whether or not we are on an iGPU/similar, and can use certain memmory optimizations
      bool temp;
      default_device_g.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &temp);
      if (temp == true)
      { // is mapping required? // Yes we might need to map because OpenCL does not guarantee that the data will be shared, alternatively use SVM
         info_file << "Using unified memory: " << temp << " ";
         //   queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
      }
      else
      {
         info_file << "No unified memory: " << temp << " ";
      }
      fastIO = temp;
      fastIO = false;
   }

   int cdt;
   for (int ntime = 0; ntime < par->nc; ntime++)
   {
      // timer.mark();
      kernel_trilin.setArg(0, buff_Ea);                   // the 1st argument to the kernel program Ea
      kernel_trilin.setArg(1, buff_E);                    // Ba
      kernel_trilin.setArg(2, sizeof(float), &par->a0_f); // scale

      queue.enqueueNDRangeKernel(kernel_trilin, cl::NullRange, cl::NDRange(n_cells), cl::NullRange); // run the kernel
      queue.finish(); // wait for the end of the kernel program

      kernel_trilin.setArg(0, buff_Ba);                   // the 1st argument to the kernel program Ea
      kernel_trilin.setArg(1, buff_B);                    // Ba
      kernel_trilin.setArg(2, sizeof(float), &par->a0_f); // scale
      queue.enqueueNDRangeKernel(kernel_trilin, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
      queue.finish();
      
      queue.enqueueFillBuffer(buff_npi, 0, 0, n_cellsi);
      queue.enqueueFillBuffer(buff_cji, 0, 0, n_cellsi * 3);
      //    queue.finish();
      //   set arguments to be fed into the kernel program
      //   cout << "kernel arguments for electron" << endl;
      queue.finish(); // wait for trilinear to end before startin tnp electron
      //      cout << "\ntrilin " << timer.elapsed() << "s, \n";
      kernel_tnp.setArg(0, buff_Ea);                        // the 1st argument to the kernel program Ea
      kernel_tnp.setArg(1, buff_Ba);                        // Ba
      kernel_tnp.setArg(2, buff_x0_e);                      // x0
      kernel_tnp.setArg(3, buff_y0_e);                      // y0
      kernel_tnp.setArg(4, buff_z0_e);                      // z0
      kernel_tnp.setArg(5, buff_x1_e);                      // x1
      kernel_tnp.setArg(6, buff_y1_e);                      // y1
      kernel_tnp.setArg(7, buff_z1_e);                      // z1
      kernel_tnp.setArg(8, sizeof(float), &par->Bcoef[0]);  // Bconst
      kernel_tnp.setArg(9, sizeof(float), &par->Ecoef[0]);  // Econst
      kernel_tnp.setArg(10, sizeof(float), &par->a0_f);     // scale factor
      kernel_tnp.setArg(11, sizeof(int), &par->n_partp[0]); // npart
      kernel_tnp.setArg(12, sizeof(int), &par->ncalcp[0]);  // ncalc
      kernel_tnp.setArg(13, buff_q_e);                      // q
      // cout << "run kernel_tnp for electron" << endl;
      //  timer.mark();
      queue.enqueueNDRangeKernel(kernel_tnp, cl::NullRange, cl::NDRange(n0), cl::NullRange);
      queue.finish();

      kernel_density.setArg(0, buff_x0_e);                 // x0
      kernel_density.setArg(1, buff_y0_e);                 // y0
      kernel_density.setArg(2, buff_z0_e);                 // z0
      kernel_density.setArg(3, buff_x1_e);                 // x1
      kernel_density.setArg(4, buff_y1_e);                 // y1
      kernel_density.setArg(5, buff_z1_e);                 // z1
      kernel_density.setArg(6, buff_npi);                  // np integer indices temp
      kernel_density.setArg(7, buff_cji);                  // current integer indices temp
      kernel_density.setArg(8, buff_q_e);                  // q
      kernel_density.setArg(9, sizeof(float), &par->a0_f); // scale factor
      //      cout << "\nelectron tnp " << timer.elapsed() << "s, \n";
      // wait for the end of the tnp electron to finish before starting density electron
      // run the kernel to get electron density
      //  timer.mark();
      queue.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n0), cl::NullRange);
      queue.finish();

      kernel_df.setArg(0, buff_np_e);                 // np
      kernel_df.setArg(1, buff_npi);                  // indices
      kernel_df.setArg(2, buff_currentj_e);           // electron current
      kernel_df.setArg(3, buff_cji);                  // current indices
      kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor
      queue.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
      queue.finish();

      //  cout << "\nelectron density " << timer.elapsed() << "s, \n";
      // timer.mark();
      //  set arguments to be fed into the kernel program
      kernel_tnp.setArg(0, buff_Ea);                        // the 1st argument to the kernel program Ea
      kernel_tnp.setArg(1, buff_Ba);                        // Ba
      kernel_tnp.setArg(2, buff_x0_i);                      // x0
      kernel_tnp.setArg(3, buff_y0_i);                      // y0
      kernel_tnp.setArg(4, buff_z0_i);                      // z0
      kernel_tnp.setArg(5, buff_x1_i);                      // x1
      kernel_tnp.setArg(6, buff_y1_i);                      // y1
      kernel_tnp.setArg(7, buff_z1_i);                      // z1
      kernel_tnp.setArg(8, sizeof(float), &par->Bcoef[1]);  // Bconst
      kernel_tnp.setArg(9, sizeof(float), &par->Ecoef[1]);  // Econst
      kernel_tnp.setArg(10, sizeof(float), &par->a0_f);     // scale factor
      kernel_tnp.setArg(11, sizeof(int), &par->n_partp[1]); // npart
      kernel_tnp.setArg(12, sizeof(int), &par->ncalcp[1]);  //
      kernel_tnp.setArg(13, buff_q_i);                      // q
      // cout << "run kernel for ions" << endl;
      queue.enqueueNDRangeKernel(kernel_tnp, cl::NullRange, cl::NDRange(n0), cl::NullRange);
      queue.finish(); // wait for the tnp for ions to finish before

      queue.enqueueFillBuffer(buff_npi, 0, 0, n_cellsi);
      queue.enqueueFillBuffer(buff_cji, 0, 0, n_cellsi * 3);

      queue.finish(); // wait for the tnp for ions to finish before

      kernel_density.setArg(0, buff_x0_i);                 // x0
      kernel_density.setArg(1, buff_y0_i);                 // y0
      kernel_density.setArg(2, buff_z0_i);                 // z0
      kernel_density.setArg(3, buff_x1_i);                 // x1
      kernel_density.setArg(4, buff_y1_i);                 // y1
      kernel_density.setArg(5, buff_z1_i);                 // z1
      kernel_density.setArg(6, buff_npi);                  // np temp integer indices
      kernel_density.setArg(7, buff_cji);                  // current indices
      kernel_density.setArg(8, buff_q_i);                  // q
      kernel_density.setArg(9, sizeof(float), &par->a0_f); // scale factor
      // wait for the end of the tnp ion to finish before starting density ion
      // run the kernel to get ion density
      queue.enqueueNDRangeKernel(kernel_density, cl::NullRange, cl::NDRange(n0), cl::NullRange);
      queue.finish();

      kernel_df.setArg(0, buff_np_i);                 // np ion
      kernel_df.setArg(1, buff_npi);                  // np ion temp integer indices
      kernel_df.setArg(2, buff_currentj_i);           // current
      kernel_df.setArg(3, buff_cji);                  // current indices
      kernel_df.setArg(4, sizeof(float), &par->a0_f); // scale factor
      queue.enqueueNDRangeKernel(kernel_df, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
      queue.finish();
      //  cout << "\neions  " << timer.elapsed() << "s, \n";

      // sum total electron and ion densitiies and current densities for E B calculations
      kernel_dtotal.setArg(0, buff_np_e);       // np ion
      kernel_dtotal.setArg(1, buff_np_i);       // np ion
      kernel_dtotal.setArg(2, buff_currentj_e); // current
      kernel_dtotal.setArg(3, buff_currentj_i); // current
      kernel_dtotal.setArg(4, buff_npt);        // total particles density
      kernel_dtotal.setArg(5, buff_jc);         // total current density
      kernel_dtotal.setArg(6, sizeof(size_t), &n_cells);
      queue.enqueueNDRangeKernel(kernel_dtotal, cl::NullRange, cl::NDRange(n_cells / 16), cl::NullRange);
      queue.finish();

      // timer.mark();
      // set externally applied fields this is inside time loop so we can set time varying E and B field
      // calcEeBe(Ee,Be,t); // find E field must work out every i,j,k depends on charge in every other cell
      // queue.enqueueWriteBuffer(buff_Ee, CL_TRUE, 0, n_cellsf * 3, fi->Ee);
      // queue.enqueueWriteBuffer(buff_Be, CL_TRUE, 0, n_cellsf * 3, fi->Be);
      cdt = calcEBV(fi, par);
      // cout << "\nEBV: " << timer.elapsed() << "s, \n";
   }

   if (fastIO)
   { // is mapping required?
     //    mapped_buff_x0_e = (float *)queue.enqueueMapBuffer(buff_x0_e, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * n);
     //    queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
   }
   else
   { // read buffers to save to disk
      queue.enqueueReadBuffer(buff_npt, CL_TRUE, 0, n_cellsf, fi->npt);
      queue.enqueueReadBuffer(buff_jc, CL_TRUE, 0, n_cellsf * 3, fi->jc);
      queue.enqueueReadBuffer(buff_E, CL_TRUE, 0, n_cellsf * 3, fi->E);
      queue.enqueueReadBuffer(buff_B, CL_TRUE, 0, n_cellsf * 3, fi->B);

      queue.enqueueReadBuffer(buff_x0_e, CL_TRUE, 0, n4, pt->pos0x[0]);
      queue.enqueueReadBuffer(buff_y0_e, CL_TRUE, 0, n4, pt->pos0y[0]);
      queue.enqueueReadBuffer(buff_z0_e, CL_TRUE, 0, n4, pt->pos0z[0]);
      queue.enqueueReadBuffer(buff_x1_e, CL_TRUE, 0, n4, pt->pos1x[0]);
      queue.enqueueReadBuffer(buff_y1_e, CL_TRUE, 0, n4, pt->pos1y[0]);
      queue.enqueueReadBuffer(buff_z1_e, CL_TRUE, 0, n4, pt->pos1z[0]);

      queue.enqueueReadBuffer(buff_x0_i, CL_TRUE, 0, n4, pt->pos0x[1]);
      queue.enqueueReadBuffer(buff_y0_i, CL_TRUE, 0, n4, pt->pos0y[1]);
      queue.enqueueReadBuffer(buff_z0_i, CL_TRUE, 0, n4, pt->pos0z[1]);
      queue.enqueueReadBuffer(buff_x1_i, CL_TRUE, 0, n4, pt->pos1x[1]);
      queue.enqueueReadBuffer(buff_y1_i, CL_TRUE, 0, n4, pt->pos1y[1]);
      queue.enqueueReadBuffer(buff_z1_i, CL_TRUE, 0, n4, pt->pos1z[1]);

      queue.enqueueReadBuffer(buff_q_e, CL_TRUE, 0, n4, pt->q[0]);
      queue.enqueueReadBuffer(buff_q_i, CL_TRUE, 0, n4, pt->q[1]);
   }
   if (changedt(pt, cdt, par))
   {
      queue.enqueueWriteBuffer(buff_x0_e, CL_TRUE, 0, n4, pt->pos0x[0]);
      queue.enqueueWriteBuffer(buff_y0_e, CL_TRUE, 0, n4, pt->pos0y[0]);
      queue.enqueueWriteBuffer(buff_z0_e, CL_TRUE, 0, n4, pt->pos0z[0]);
      queue.enqueueWriteBuffer(buff_x1_e, CL_TRUE, 0, n4, pt->pos1x[0]);
      queue.enqueueWriteBuffer(buff_y1_e, CL_TRUE, 0, n4, pt->pos1y[0]);
      queue.enqueueWriteBuffer(buff_z1_e, CL_TRUE, 0, n4, pt->pos1z[0]);

      queue.enqueueWriteBuffer(buff_x0_i, CL_TRUE, 0, n4, pt->pos0x[1]);
      queue.enqueueWriteBuffer(buff_y0_i, CL_TRUE, 0, n4, pt->pos0y[1]);
      queue.enqueueWriteBuffer(buff_z0_i, CL_TRUE, 0, n4, pt->pos0z[1]);
      queue.enqueueWriteBuffer(buff_x1_i, CL_TRUE, 0, n4, pt->pos1x[1]);
      queue.enqueueWriteBuffer(buff_y1_i, CL_TRUE, 0, n4, pt->pos1y[1]);
      queue.enqueueWriteBuffer(buff_z1_i, CL_TRUE, 0, n4, pt->pos1z[1]);
      //  cout<<"change_dt done"<<endl;
   }
   first = false;
}
