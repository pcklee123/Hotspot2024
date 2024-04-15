/* TS3.cpp
This contains the main loop for the program. Most of the initialization occurs here, and time steps are iterated through.
For settings (as to what to calculate, eg. E / B field, E / B force) go to the defines in include/traj.h
*/
#include "include/traj.h"
ofstream info_file;
int main()
{
    par par1;
    par *par = &par1;
    float nt0prev;
    cl_int cl_err;
    timer.mark(); // Yes, 3 time marks. The first is for the overall program dt
    timer.mark(); // The second is for compute_d_time
    timer.mark(); // The third is for start up dt
    double t = 0;
    // allocate memory for particles
    particles *pt = alloc_particles(par);
    const unsigned int n_cells = n_space_divx * n_space_divy * n_space_divz;
    fields *fi = alloc_fields(par);
    int total_ncalc[2] = {0, 0}; // particle 0 - electron, particle 1 deuteron

    info_file.open("info.csv");
    info_file << std::scientific;
    info_file.precision(3);

    cin.tie(NULL); // Fast printing
    // ios_base::sync_with_stdio(false);
    cout << std::scientific;
    cout.precision(1);
    cerr << std::scientific;
    cerr.precision(3);

    omp_set_nested(true);
    nthreads = omp_get_max_threads(); // omp_set_num_threads(nthreads);
    cl_set_build_options(par);
    cl_start(fi, par);

    try
    {
        if (!std::filesystem::create_directory(outpath1))
            par->outpath = outpath1;
        else if (!std::filesystem::create_directory(outpath2))
            par->outpath = outpath2;
    }
    catch (const std::filesystem::__cxx11::filesystem_error &e)
    {
        std::cerr << "Error creating output directory: " << e.what() << '\n';
        try
        {
            if (!std::filesystem::create_directory(outpath2))
                par->outpath = outpath2;
        }
        catch (const std::filesystem::__cxx11::filesystem_error &e)
        {
            std::cerr << "Error creating output directory: " << e.what() << '\n';
        }
    }
    cout << "Start up time = " << timer.replace() << "s\n";
    // startup stuff set output path opencl and print initial info
    cout << "Set initial random positions: ";
    timer.mark();
    // estimate dt. needed to set up initial particles with velocity actual value not important
    float vel_e = sqrt(kb * Temp_e / (mp[0] * e_mass) + vz0 * vz0 + v0_r * v0_r);
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * Bmax0);
    float acc_e = e_charge_mass * Emax0;
    float TE = (sqrt(1 + 2 * a0 * par->a0_f * acc_e / pow(vel_e, 2)) - 1) * vel_e / acc_e; // time for electron to move across 1 cell
    // float TEs = a0 * par->a0_f * vel_e;
    TE = TE <= 0 ? a0 * par->a0_f * vel_e : TE; // if acc is negligible
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance
    par->dt[0] = min(Tcyclotron / 4, TE / 8) / f1; // electron should not move more than 1 cell after ncalc*dt and should not make more than 1/4 gyration and must calculate E before the next 1/4 plasma period
    par->dt[1] = par->dt[0] * md_me;
    // cout << "dt = " << par->dt[0] << ", " << par->dt[1] << endl;
#define generateRandom
#ifdef generateRandom
#ifdef sphere
    generate_rand_sphere(pt, par);
#endif // sphere
#ifdef cylinder
    generate_rand_cylinder(pt, par);
#endif // cylinder
#else
    generateParticles(pt, par);
#endif

    // get limits and spacing of Field cells
    generateField(fi, par);

    cout << timer.replace() << "s\n"; // cout << "Set initial random positions: ";

    int i_time = 0;
    // cout << "get_densityfields: ";
    timer.mark();
    cl_int res = 0;
    get_densityfields(fi, pt, par);
    res = clEnqueueReadBuffer(commandQueue_g(), fi->buff_np_e[0](), CL_TRUE, 0, n_cellsf, fi->np[0], 0, NULL, NULL);

    //   cout << "dt = " << par->dt[0] << ", " << par->dt[1] << endl;
    //   float max_jc = maxvalf((reinterpret_cast<float *>(fi->jc)), n_cells * 3);
    //  cout << "max current density  = " << max_jc << endl;
    cout << timer.elapsed() << "s\n "; // density this is incorporated into tnp which also moves particles, but need to work out here to get good estimate of dt
                                       // cout << "calcEBV: ";
    timer.mark();
    int cdt = calcEBV(fi, par); // electric and magnetic fields this is incorporated into tnp which also moves particles. Need here just to estimate dt

    cl_command_queue commandQueue = clCreateCommandQueue(context_g(), default_device_g(), 0, &res);
    res = clEnqueueReadBuffer(commandQueue, fi->E_buffer, CL_TRUE, 0, n_cellsf * 3, fi->E, 0, NULL, NULL);
    res = clEnqueueReadBuffer(commandQueue, fi->B_buffer, CL_TRUE, 0, n_cellsf * 3, fi->B, 0, NULL, NULL);
    cout << timer.elapsed() << "s\n ";
    // int cdt=0;
    // changedt(pt, cdt, par); /* change time step if E or B too big*/

    float max_ne = maxvalf((reinterpret_cast<float *>(fi->np[0])), n_cells);
    float Density_e = max_ne * r_part_spart / powf(a0, 3);
    // float max_ni = maxvalf((reinterpret_cast<float *>(fi->np[1])), n_cells);
    //     max_jc = maxvalf((reinterpret_cast<float *>(fi->jc)), n_cells * 3);
    //    cout << "max current density  = " << max_jc << endl;
    //  cout << "max density electron = " << max_ne << ", " << max_ne * r_part_spart / powf(a0, 3) << "m-3, ion = " << max_ni << ", " << max_ni * r_part_spart / powf(a0, 3) << endl;
    //    cout << "Emax = " << par->Emax << ", " << "Bmax = " << par->Bmax << endl;
    //  calculated plasma parameters
    float Density_e1 = nback * r_part_spart / (powf(n_space * a0, 3));

    info_file << "initial density = " << Density_e << "/m^3,  background density = " << Density_e1 << "/m^3 \n";
    float plasma_freq = sqrt(Density_e * e_charge * e_charge_mass / (mp[0] * epsilon0)) / (2 * pi);
    float plasma_period = 1 / plasma_freq;
    float Debye_Length = sqrt(epsilon0 * kb * Temp_e / (Density_e * e_charge * e_charge));
    info_file << "debyeLength=" << Debye_Length << ", a0 = " << a0 << endl;
    if (Debye_Length < a0)
    {
        cerr << "a0 = " << a0 << " too large for this density Debyle Length = " << Debye_Length << endl;
        // exit(1);
    }
    float TDebye = Debye_Length / vel_e;
    acc_e = fabsf(e_charge_mass * par->Emax);
    TE = (sqrt(1 + 2 * a0 * par->a0_f * acc_e / pow(vel_e, 2)) - 1) * vel_e / acc_e; // time for electron to move across 1 cell
    TE = ((TE <= 0) | (isnanf(TE))) ? a0 * par->a0_f / vel_e : TE;                   // if acc is negligible i.e. in square root ~=1, use approximation is more accurate
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance
    float TExB = a0 * par->a0_f / (par->Emax + .1) * (par->Bmax + .00001);
    info_file << "Tdebye=" << TDebye << ", Tcycloton/4=" << Tcyclotron / 4 << ", plasma period/4=" << plasma_period / 4 << ",TE=" << TE << ",TExB=" << TExB << endl;
    float inc = min(min(min(TDebye, Tcyclotron / 4), plasma_period / 4), TE / 16) / f1 / par->dt[0]; // redo dt
    par->dt[0] *= inc;
    par->dt[1] *= inc;
    //  cout << "dt0 = " << par->dt[0] << endl;
    info_file << "v0 electron = " << vel_e << endl;
// redo initial particle positions to get the correct velocities
#pragma omp parallel for simd
    for (int n = 0; n < par->n_part[0] * 3 * 2; n++)
        pt->pos0[n] = pt->pos1[n] - (pt->pos1[n] - pt->pos0[n]) * inc;
        //   cout << "dt changed" << endl;

#ifdef Uon_
    // cout << "calculate the total potential energy U\n";
    //                  timer.mark();
    calcU(fi, pt, par);
// cout << "U: " << timer.elapsed() << "s, ";
#endif
    // cout << "savefiles" << endl;
    info(par); // printout initial info.csv file re do this with updated info
    save_files(i_time, t, fi, pt, par);

    //    cout << "logentry" << endl;
    log_headers();                             // log file start with headers
    log_entry(0, 0, cdt, total_ncalc, t, par); // Write everything to log
    nt0prev = par->nt[0];
    //  cout << par->nt[0] << " " << nt0prev << endl;
#pragma omp barrier

    cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << 0 << ")\n";

    for (i_time = 1; i_time < ndatapoints; i_time++)
    {
        timer.mark();     // For timestep
        timer.mark();     // Work out motion
        tnp(fi, pt, par); //  calculate the next position par->ncalcp[p] times
                          // float max_jc = maxvalf((reinterpret_cast<float *>(fi->jc)), n_cells * 3);
                          //  cout << "max current density  = " << max_jc << endl;
        for (int p = 0; p < 2; ++p)
            total_ncalc[p] += par->nc * par->ncalcp[p];
        cout << "motion: " << timer.elapsed() << "s, ";
        t += par->dt[0] * par->ncalcp[0] * par->nc;

        cout << i_time << "." << par->nc << " t = " << t << "(compute_time = " << timer.elapsed() << "s) : ";

        timer.mark();                       //      cout << "savefiles" << endl;
        save_files(i_time, t, fi, pt, par); // print out all files for paraview also get number of particles in cells.
        if (par->nt[0] > nt0prev)
        {
            changedx(fi, par); // particles are moving out of bounds. make cells bigger.
            nt0prev = par->nt[0];
        }

#ifdef Uon_
        // cout << "calculate the total potential energy U\n";
        //  timer.mark();// calculate the total potential energy U
        calcU(fi, pt, par);
        // cout << "calculate the total potential energy U done\n";
        //  cout << "U: " << timer.elapsed() << "s, ";
#endif
        //        cout << "logentry" << endl;
        log_entry(i_time, 0, cdt, total_ncalc, t, par); // cout<<"log entry done"<<endl;
        cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << total_ncalc[0] << ")\n";
    }
    cout << "Overall execution time: " << timer.elapsed() << "s";
    logger.close();
    // info_file.close();
    return 0;
}
