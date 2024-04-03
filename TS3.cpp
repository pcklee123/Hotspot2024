/* TS3.cpp
This contains the main loop for the program. Most of the initialization occurs here, and time steps are iterated through.
For settings (as to what to calculate, eg. E / B field, E / B force) go to the defines in include/traj.h
*/
#include "include/traj.h"
// sphere
// 0,number of "super" electrons, electron +deuteriom ions, total
unsigned int n_space_div[3] = {n_space_divx, n_space_divy, n_space_divz};
unsigned int n_space_div2[3] = {n_space_divx2, n_space_divy2, n_space_divz2};
par par1;
par *par = &par1;
float nt0prev;
// particles particl1;
// particles *pt = &particl1; //= alloc_particles( par);
//  string outpath;
ofstream info_file;
int main()
{
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
    cl_start(par);

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
    // info(par);                   // printout initial info.csv file
    cout << "Start up time = " << timer.replace() << "s\n";
    // startup stuff set output path opencl and print initial info
    cout << "Set initial random positions: ";
    timer.mark();
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

    cout << timer.replace() << "s\n";//cout << "Set initial random positions: ";

    fftwf_init_threads();

    int i_time = 0;
    cout << "get_densityfields: ";
    timer.mark();
    get_densityfields(fi, pt, par);
    cout  << timer.elapsed() << "s\n ";
    cout << "calcEBV: ";
    timer.mark();
    int cdt = calcEBV(fi, par);
    cout << timer.elapsed() << "s\n ";
    // int cdt=0;
    changedt(pt, cdt, par); /* change time step if E or B too big*/

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
    cout << par->nt[0] << " " << nt0prev << endl;
#pragma omp barrier

    cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << 0 << ")\n";

    for (i_time = 1; i_time < ndatapoints; i_time++)
    {
        timer.mark(); // For timestep
         timer.mark();     // Work out motion
        tnp(fi, pt, par); //  calculate the next position par->ncalcp[p] times
        for (int p = 0; p < 2; ++p)
            total_ncalc[p] += par->nc * par->ncalcp[p];
                cout << "\nmotion: " << timer.elapsed() << "s, \n";
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
