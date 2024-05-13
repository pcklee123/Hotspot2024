#include "include/traj.h"
void tnp(fields *fi, particles *pt, par *par)
{
   //  create buffers on the device
   /** IMPORTANT: do not use CL_MEM_USE_HOST_PTR if on dGPU **/
   /** HOST_PTR is only used so that memory is not copied, but instead shared between CPU and iGPU in RAM**/
   // Note that special alignment has been given to Ea, Ba, y0, z0, x0, x1, y1 in order to actually do this properly
   static int nt0prev;
   static bool first = true;
   cl_int res = 0;
   if (first)
   {
      nt0prev = -(int)n_partd;
      first = false;
   }
#if defined(sphere)
#if defined(octant)
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicito"); // select the kernel program to run
#elif defined(quadrant)
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicitq", &res); // select the kernel program to run
#elif defined(spherez)
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicitz"); // select the kernel program to run
#else
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicit"); // select the kernel program to run
#endif
#endif

   if (res)
   {
      cout << "kernel_tnp progrem  res: " << res << endl;
      exit(1);
   }
#if defined(cylinder)
   cl::Kernel kernel_tnp = cl::Kernel(program_g, "tnp_k_implicitz"); // select the kernel program to run
#endif

   cl::Kernel kernel_trilin = cl::Kernel(program_g, "trilin_k"); // select the kernel program to run
   if (res)
   {
      cout << "kernel_trilin progrem  res: " << res << endl;
      exit(1);
   }

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

   int cdt;
   if (fastIO)
   {
      // commandQueue_g.enqueueUnmapMemObject(pt->buff_x0_e[0], pt->pos0x[0]);
   }
   par->ndeltat = 0;
   for (uint32_t ntime = 0; ntime < par->nc; ntime++)
   {
      // timer.mark();
      kernel_trilin.setArg(0, fi->buff_Ea[0]);            // the 1st argument to the kernel program Ea
      kernel_trilin.setArg(1, fi->buff_E[0]);             // Ba
      kernel_trilin.setArg(2, sizeof(float), &par->a0_f); // scale
      res = commandQueue_g.enqueueNDRangeKernel(kernel_trilin, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
      if (res)
         cout << "kernel_trilin E  res: " << res << endl;
      commandQueue_g.finish(); // wait for the end of the kernel program

      kernel_trilin.setArg(0, fi->buff_Ba[0]);            // the 1st argument to the kernel program Ea
      kernel_trilin.setArg(1, fi->buff_B[0]);             // Ba
      kernel_trilin.setArg(2, sizeof(float), &par->a0_f); // scale
      res = commandQueue_g.enqueueNDRangeKernel(kernel_trilin, cl::NullRange, cl::NDRange(n_cells), cl::NullRange);
      if (res)
         cout << "kernel_trilin B  res: " << res << endl;
      commandQueue_g.finish(); //      cout << "\ntrilin " << timer.elapsed() << "s, \n";

      kernel_tnp.setArg(0, fi->buff_Ea[0]);                      // the 1st argument to the kernel program Ea
      kernel_tnp.setArg(1, fi->buff_Ba[0]);                      // Ba
      kernel_tnp.setArg(2, pt->buff_x0_e[0]);                    // x0
      kernel_tnp.setArg(3, pt->buff_y0_e[0]);                    // y0
      kernel_tnp.setArg(4, pt->buff_z0_e[0]);                    // z0
      kernel_tnp.setArg(5, pt->buff_x1_e[0]);                    // x1
      kernel_tnp.setArg(6, pt->buff_y1_e[0]);                    // y1
      kernel_tnp.setArg(7, pt->buff_z1_e[0]);                    // z1
      kernel_tnp.setArg(8, sizeof(float), &par->Bcoef[0]);       // Bconst
      kernel_tnp.setArg(9, sizeof(float), &par->Ecoef[0]);       // Econst
      kernel_tnp.setArg(10, sizeof(float), &par->a0_f);          // scale factor
      kernel_tnp.setArg(11, sizeof(uint32_t), &par->n_partp[0]); // npart
      kernel_tnp.setArg(12, sizeof(uint32_t), &par->ncalcp[0]);  // ncalc
      kernel_tnp.setArg(13, pt->buff_q_e[0]);                    // q
      // cout << "run kernel_tnp for electron" << endl;
      //  timer.mark();
      res = commandQueue_g.enqueueNDRangeKernel(kernel_tnp, cl::NullRange, cl::NDRange(par->n_part[1]), cl::NullRange);
      if (res)
         cout << "kernel_tnp e  res: " << res << endl;
      commandQueue_g.finish();

      //  set arguments to be fed into the kernel program
      kernel_tnp.setArg(0, fi->buff_Ea[0]);                      // the 1st argument to the kernel program Ea
      kernel_tnp.setArg(1, fi->buff_Ba[0]);                      // Ba
      kernel_tnp.setArg(2, pt->buff_x0_i[0]);                    // x0
      kernel_tnp.setArg(3, pt->buff_y0_i[0]);                    // y0
      kernel_tnp.setArg(4, pt->buff_z0_i[0]);                    // z0
      kernel_tnp.setArg(5, pt->buff_x1_i[0]);                    // x1
      kernel_tnp.setArg(6, pt->buff_y1_i[0]);                    // y1
      kernel_tnp.setArg(7, pt->buff_z1_i[0]);                    // z1
      kernel_tnp.setArg(8, sizeof(float), &par->Bcoef[1]);       // Bconst
      kernel_tnp.setArg(9, sizeof(float), &par->Ecoef[1]);       // Econst
      kernel_tnp.setArg(10, sizeof(float), &par->a0_f);          // scale factor
      kernel_tnp.setArg(11, sizeof(uint32_t), &par->n_partp[1]); // npart
      kernel_tnp.setArg(12, sizeof(uint32_t), &par->ncalcp[1]);  //
      kernel_tnp.setArg(13, pt->buff_q_i[0]);                    // q
      // cout << "run kernel for ions" << endl;
      res = commandQueue_g.enqueueNDRangeKernel(kernel_tnp, cl::NullRange, cl::NDRange(par->n_part[1]), cl::NullRange);
      if (res)
         cout << "kernel_tnp i  res: " << res << endl;
      commandQueue_g.finish();

      get_densityfields(fi, pt, par);
      // check if particles are moving out
      //      cout << "change dx : " << nt0prev << ", " << par->nt[0] - nt0prev << endl;
      if (par->nt[0] > nt0prev)
      {
         //        cout << "change dx : " << nt0prev << ", " << par->nt[0] << endl;
         changedx(fi, par); // particles are moving out of bounds. make cells bigger.
         nt0prev = par->nt[0];
      }

      // timer.mark();
      // set externally applied fields this is inside time loop so we can set time varying E and B field
      /*
      generateField(fi, par); // find E field must work out every i,j,k depends on charge in every other cell
      */
      par->cdt = calcEBV(fi, par);
      par->ndeltat += par->dt[0] * par->ncalcp[0];
      changedt(pt, par->cdt, par);
      // cout << changedt(pt, par->cdt, par) << ", ";
      // cout << "change dt: " << par->cdt << ",  dt= " << par->dt[0] << endl;
      // cout << "\nEBV: " << timer.elapsed() << "s, \n";
   }
   if (!fastIO)
   {
      // cout << "for saving to disk"<<endl;

      commandQueue_g.enqueueReadBuffer(pt->buff_x0_e[0], CL_TRUE, 0, n_partf, pt->pos0x[0]);
      commandQueue_g.enqueueReadBuffer(pt->buff_y0_e[0], CL_TRUE, 0, n_partf, pt->pos0y[0]);
      commandQueue_g.enqueueReadBuffer(pt->buff_z0_e[0], CL_TRUE, 0, n_partf, pt->pos0z[0]);
      commandQueue_g.enqueueReadBuffer(pt->buff_x1_e[0], CL_TRUE, 0, n_partf, pt->pos1x[0]);
      commandQueue_g.enqueueReadBuffer(pt->buff_y1_e[0], CL_TRUE, 0, n_partf, pt->pos1y[0]);
      commandQueue_g.enqueueReadBuffer(pt->buff_z1_e[0], CL_TRUE, 0, n_partf, pt->pos1z[0]);

      commandQueue_g.enqueueReadBuffer(pt->buff_x0_i[0], CL_TRUE, 0, n_partf, pt->pos0x[1]);
      commandQueue_g.enqueueReadBuffer(pt->buff_y0_i[0], CL_TRUE, 0, n_partf, pt->pos0y[1]);
      commandQueue_g.enqueueReadBuffer(pt->buff_z0_i[0], CL_TRUE, 0, n_partf, pt->pos0z[1]);
      commandQueue_g.enqueueReadBuffer(pt->buff_x1_i[0], CL_TRUE, 0, n_partf, pt->pos1x[1]);
      commandQueue_g.enqueueReadBuffer(pt->buff_y1_i[0], CL_TRUE, 0, n_partf, pt->pos1y[1]);
      commandQueue_g.enqueueReadBuffer(pt->buff_z1_i[0], CL_TRUE, 0, n_partf, pt->pos1z[1]);

      commandQueue_g.enqueueReadBuffer(pt->buff_q_e[0], CL_TRUE, 0, n_partf, pt->q[0]);
      commandQueue_g.enqueueReadBuffer(pt->buff_q_i[0], CL_TRUE, 0, n_partf, pt->q[1]);

      commandQueue_g.enqueueReadBuffer(fi->buff_E[0], CL_TRUE, 0, n_cellsf * 3, fi->E);
      commandQueue_g.enqueueReadBuffer(fi->buff_B[0], CL_TRUE, 0, n_cellsf * 3, fi->B);

      commandQueue_g.enqueueReadBuffer(fi->buff_np_e[0], CL_TRUE, 0, n_cellsf, fi->np[0]);
      commandQueue_g.enqueueReadBuffer(fi->buff_np_i[0], CL_TRUE, 0, n_cellsf, fi->np[1]);

      commandQueue_g.enqueueReadBuffer(fi->buff_currentj_e[0], CL_TRUE, 0, n_cellsf * 3, fi->currentj[0]);
      // commandQueue_g.enqueueReadBuffer(fi->buff_jc[0], CL_TRUE, 0, n_cellsf * 3, fi->currentj[0]);
      commandQueue_g.enqueueReadBuffer(fi->buff_currentj_i[0], CL_TRUE, 0, n_cellsf * 3, fi->currentj[1]);
   }
   else
   {
      // pt->pos0x = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(commandQueue_g.enqueueMapBuffer(pt->buff_x0_e[0], CL_TRUE, CL_MAP_READ, 0, n_partf)));
      // pt->pos0y = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(commandQueue_g.enqueueMapBuffer(pt->buff_y0_e[0], CL_TRUE, CL_MAP_READ, 0, n_partf)));
      // pt->pos0z = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(commandQueue_g.enqueueMapBuffer(pt->buff_z0_e[0], CL_TRUE, CL_MAP_READ, 0, n_partf)));
      // pt->pos1x = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(commandQueue_g.enqueueMapBuffer(pt->buff_x1_e[0], CL_TRUE, CL_MAP_READ, 0, n_partf)));
      // pt->pos1y = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(commandQueue_g.enqueueMapBuffer(pt->buff_y1_e[0], CL_TRUE, CL_MAP_READ, 0, n_partf)));
      // pt->pos1z = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(commandQueue_g.enqueueMapBuffer(pt->buff_z1_e[0], CL_TRUE, CL_MAP_READ, 0, n_partf)));
      //  Repeat for other buffers...
   }
}