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
    info(par);                   // printout initial info.csv file
    cout << "Start up time = " << timer.replace() << "s\n";
    // startup stuff set output path opencl and print initial info

    timer.mark();
#define generateRandom
#ifdef generateRandom
#ifdef sphere
    generate_rand_sphere(pt, par);
#endif // sphere
#ifdef impl_sphere
    generate_rand_impl_sphere(pt, par);
#endif // sphere
#ifdef cylinder
    generate_rand_cylinder(pt, par);
#endif // cylinder
#else
    generateParticles(pt, par);
#endif

    // get limits and spacing of Field cells
    generateField(fi, par);

    cout << "Set initial random positions: " << timer.replace() << "s\n";

    fftwf_init_threads();

    int i_time = 0;
    //  cout << "get_densityfields " << endl;
    get_densityfields(fi, pt, par);
    //  cout << "get_densityfields done" << endl;
    int cdt = calcEBV(fi, par);
    // int cdt=0;
    changedt(pt, cdt, par); /* change time step if E or B too big*/

#ifdef Uon_
    // cout << "calculate the total potential energy U\n";
    //                  timer.mark();
    calcU(fi, pt, par);
    // cout << "U: " << timer.elapsed() << "s, ";
#endif
    // cout << "savefiles" << endl;
    save_files(i_time, t, fi, pt, par);
    //    cout << "logentry" << endl;
    log_entry(0, 0, cdt, total_ncalc, t, par); // Write everything to log

#pragma omp barrier

    cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << 0 << ")\n";

    for (i_time = 1; i_time < ndatapoints; i_time++)
    {
        timer.mark();     // For timestep
        //timer.mark();     // Work out motion
        tnp(fi, pt, par); //  calculate the next position par->ncalcp[p] times
        for (int p = 0; p < 2; ++p)
            total_ncalc[p] += par->nc * par->ncalcp[p];
//        cout << "motion: " << timer.elapsed() << "s, ";
        t += par->dt[0] * par->ncalcp[0] * par->nc;

        cout << i_time << "." << par->nc << " t = " << t << "(compute_time = " << timer.elapsed() << "s) : ";

#ifdef Uon_
        //        cout << "calculate the total potential energy U\n";
        // timer.mark();// calculate the total potential energy U
        calcU(fi, pt, par);
        //       cout << "calculate the total potential energy U done\n";
        // cout << "U: " << timer.elapsed() << "s, ";
#endif
        timer.mark();                       //      cout << "savefiles" << endl;
        save_files(i_time, t, fi, pt, par); // print out all files for paraview
        //        cout << "logentry" << endl;
        log_entry(i_time, 0, cdt, total_ncalc, t, par); // cout<<"log entry done"<<endl;
        cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << total_ncalc[0] << ")\n";
    }
    cout << "Overall execution time: " << timer.elapsed() << "s";
    logger.close();
    // info_file.close();
    return 0;
}
