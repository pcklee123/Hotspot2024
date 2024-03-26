#include "traj.h"
void generate_rand_sphere(particles *pt, par *par)
{
    // spherical plasma set plasma parameters
    float Temp[2] = {Temp_e, Temp_d}; // in K convert to eV divide by 1.160451812e4
    // initial bulk electron, ion velocity
    float v0[2][3] = {{0, 0, -vz0}, {0, 0, vz0 / 60}};

    float r0 = r0_f * a0; // if sphere this is the radius
    float area = 4 * pi * r0 * r0;
    float volume = 4 / 3 * pi * r0 * r0 * r0;

    // calculated plasma parameters
    float Density_e = (n_partd - ((n_space_divx - 2) * (n_space_divy - 2) * (n_space_divz - 2) * nback)) / volume * r_part_spart;
    float Density_e1 = nback * r_part_spart / (a0 * a0 * a0);

    info_file << "initial density = " << Density_e << "/m^3,  background density = " << Density_e1 << "/m^3 \n";
    float plasma_freq = sqrt(Density_e * e_charge * e_charge_mass / (mp[0] * epsilon0)) / (2 * pi);
    float plasma_period = 1 / plasma_freq;
    float Debye_Length = sqrt(epsilon0 * kb * Temp[0] / (Density_e * e_charge * e_charge));
    info_file << "debyeLength=" << Debye_Length << ", a0 = " << a0 << endl;
    if (Debye_Length < a0)
    {
        cerr << "a0 = " << a0 << " too large for this density Debyle Length = " << Debye_Length << endl;

        // exit(1);
    }
    float vel_e = sqrt(kb * Temp[0] / (mp[0] * e_mass) + v0[0][0] * v0[0][0] + v0[0][1] * v0[0][1] + v0[0][2] * v0[0][2]);
    // float Tv = a0 / vel_e; // time for electron to move across 1 cell if E=0
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * Bmax0);
    float TDebye = Debye_Length / vel_e;
    float acc_e = e_charge_mass * Emax0;
    float TE = sqrt(vel_e * vel_e / (acc_e * acc_e) + 2 * a0 / acc_e) - vel_e / acc_e; // time for electron to move across 1 cell
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance
    info_file << "Tdebye=" << TDebye << ", Tcycloton/4=" << Tcyclotron / 4 << ", plasma period/3=" << plasma_period / 4 << ",TE/2=" << TE / 2 << endl;
    par->dt[0] = min(min(min(TDebye, Tcyclotron / 4), plasma_period / 4), TE / 2) / ncalc0[0]; // electron should not move more than 1 cell after ncalc*dt and should not make more than 1/4 gyration and must calculate E before the next 1/4 plasma period
    par->dt[1] = par->dt[0] * md_me;
    //  float mu0_4pidt[2]= {mu0_4pi/par->dt[0],mu0_4pi/par->dt[1]};
    info_file << "v0 electron = " << v0[0][0] << "," << v0[0][1] << "," << v0[0][2] << endl;

    // set initial positions and velocity
    float sigma[2] = {sqrt(kb * Temp[0] / (mp[0] * e_mass)), sqrt(kb * Temp[1] / (mp[1] * e_mass))};
    long seed;
    gsl_rng *rng;                        // random number generator
    rng = gsl_rng_alloc(gsl_rng_rand48); // pick random number generator
    time_t myTime;
    seed = time(&myTime);
    info_file << "seed=" << seed << "\n";
    gsl_rng_set(rng, seed); // set seed

    for (int p = 0; p < 2; p++)
    {
        int na = 0;
        for (int n = 0; n < nback; ++n) // set number of particles per cell in background
        {
            for (int k = 2; k < n_space_divz - 2; ++k)
            {
                for (int j = 2; j < n_space_divy - 2; ++j)
                    for (int i = 2; i < n_space_divx - 2; ++i)
                    {
                        pt->pos0x[p][na] = ((float)(i - n_space_divx / 2) + (float)rand() / RAND_MAX) * a0;
                        pt->pos0y[p][na] = ((float)(j - n_space_divy / 2) + (float)rand() / RAND_MAX) * a0;
                        pt->pos0z[p][na] = ((float)(k - n_space_divz / 2) + (float)rand() / RAND_MAX) * a0;

                        pt->pos1x[p][na] = pt->pos0x[p][na];
                        pt->pos1y[p][na] = pt->pos0y[p][na];
                        pt->pos1z[p][na] = pt->pos0z[p][na];
                        pt->q[p][na] = qs[p];
                        pt->m[p][na] = mp[p];
                        na++;
                    }
                //         cout << pt->pos1z[p][na - 1] << " ";
            }
        }

#pragma omp parallel for ordered
        for (int n = na; n < n_partd; n++)
        {
            // float r = r0 * pow(gsl_ran_flat(rng, 0, 1), 0.3333333333);
            float r = gsl_ran_gaussian(rng, r0);
            while (fabs(r)>=((float)n_space-1.0)/4.0*a0) r = gsl_ran_gaussian(rng, r0);
            // float r = r0 * pow(gsl_ran_flat(rng, 0, 1), 0.5);
            double x, y, z;
            gsl_ran_dir_3d(rng, &x, &y, &z);
            pt->pos0x[p][n] = r * x;
            pt->pos1x[p][n] = pt->pos0x[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][0]) * par->dt[p];
            pt->pos0y[p][n] = r * y;
            pt->pos1y[p][n] = pt->pos0y[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][1]) * par->dt[p];
            pt->pos0z[p][n] = r * z;
            pt->pos1z[p][n] = pt->pos0z[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][2]) * par->dt[p];
            pt->q[p][n] = qs[p];
            pt->m[p][n] = mp[p];
            //         nt[p] += q[p][n];
        }
    }
#pragma omp barrier
    gsl_rng_free(rng); // dealloc the rng
}

void generate_rand_impl_sphere(particles *pt, par *par)
{
    // spherical plasma set plasma parameters
    float Temp[2] = {Temp_e, Temp_d}; // in K convert to eV divide by 1.160451812e4
    // initial bulk electron, ion velocity
    float v0[2][3] = {{0, 0, -vz0}, {0, 0, vz0 / 60}};

    float r0 = r0_f * a0; // if sphere this is the radius
    float area = 4 * pi * r0 * r0;
    float volume = 4 / 3 * pi * r0 * r0 * r0;

    // calculated plasma parameters
    float Density_e = (n_partd - ((n_space_divx - 2) * (n_space_divy - 2) * (n_space_divz - 2) * nback)) / volume * r_part_spart;
    float Density_e1 = nback * r_part_spart / (a0 * a0 * a0);

    info_file << "initial density = " << Density_e << "/m^3,  background density = " << Density_e1 << "/m^3 \n";
    float plasma_freq = sqrt(Density_e * e_charge * e_charge_mass / (mp[0] * epsilon0)) / (2 * pi);
    float plasma_period = 1 / plasma_freq;
    float Debye_Length = sqrt(epsilon0 * kb * Temp[0] / (Density_e * e_charge * e_charge));
    info_file << "debyeLength=" << Debye_Length << ", a0 = " << a0 << endl;
    if (Debye_Length < a0)
    {
        cerr << "a0 = " << a0 << " too large for this density Debyle Length = " << Debye_Length << endl;

        // exit(1);
    }
    float vel_e = sqrt(kb * Temp[0] / (mp[0] * e_mass) + v0[0][0] * v0[0][0] + v0[0][1] * v0[0][1] + v0[0][2] * v0[0][2]);
    // float Tv = a0 / vel_e; // time for electron to move across 1 cell if E=0
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * Bmax0);
    float TDebye = Debye_Length / vel_e;
    float acc_e = e_charge_mass * Emax0;
    float TE = sqrt(vel_e * vel_e / (acc_e * acc_e) + 2 * a0 / acc_e) - vel_e / acc_e; // time for electron to move across 1 cell
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance
    info_file << "Tdebye=" << TDebye << ", Tcycloton/4=" << Tcyclotron / 4 << ", plasma period/3=" << plasma_period / 4 << ",TE/2=" << TE / 2 << endl;
    par->dt[0] = min(min(min(TDebye, Tcyclotron / 4), plasma_period / 4), TE / 2) / ncalc0[0]; // electron should not move more than 1 cell after ncalc*dt and should not make more than 1/4 gyration and must calculate E before the next 1/4 plasma period
    par->dt[1] = par->dt[0] * md_me;
    //  float mu0_4pidt[2]= {mu0_4pi/par->dt[0],mu0_4pi/par->dt[1]};
    info_file << "v0 electron = " << v0[0][0] << "," << v0[0][1] << "," << v0[0][2] << endl;

    // set initial positions and velocity
    float sigma[2] = {sqrt(kb * Temp[0] / (mp[0] * e_mass)), sqrt(kb * Temp[1] / (mp[1] * e_mass))};
    long seed;
    gsl_rng *rng;                        // random number generator
    rng = gsl_rng_alloc(gsl_rng_rand48); // pick random number generator
    time_t myTime;
    seed = time(&myTime);
    info_file << "seed=" << seed << "\n";
    gsl_rng_set(rng, seed); // set seed

    for (int p = 0; p < 2; p++)
    {
        int na = 0;
        for (int n = 0; n < nback; ++n) // set number of particles per cell in background
        {
            for (int k = 2; k < n_space_divz - 2; ++k)
            {
                for (int j = 2; j < n_space_divy - 2; ++j)
                    for (int i = 2; i < n_space_divx - 2; ++i)
                    {
                        pt->pos0x[p][na] = ((float)(i - n_space_divx / 2) + (float)rand() / RAND_MAX) * a0;
                        pt->pos0y[p][na] = ((float)(j - n_space_divy / 2) + (float)rand() / RAND_MAX) * a0;
                        pt->pos0z[p][na] = ((float)(k - n_space_divz / 2) + (float)rand() / RAND_MAX) * a0;

                        pt->pos1x[p][na] = pt->pos0x[p][na];
                        pt->pos1y[p][na] = pt->pos0y[p][na];
                        pt->pos1z[p][na] = pt->pos0z[p][na];
                        pt->q[p][na] = qs[p];
                        pt->m[p][na] = mp[p];
                        na++;
                    }
                //         cout << pt->pos1z[p][na - 1] << " ";
            }
        }

#pragma omp parallel for ordered
        for (int n = na; n < n_partd; n++)
        {
            // float r = r0 * pow(gsl_ran_flat(rng, 0, 1), 0.3333333333);
            float r = gsl_ran_gaussian(rng, r0);
            while (fabs(r)>=((float)n_space-1.0)/2.0*a0) r = gsl_ran_gaussian(rng, r0);
            double x, y, z;
            gsl_ran_dir_3d(rng, &x, &y, &z);
            pt->pos0x[p][n] = r * x;
            pt->pos1x[p][n] = pt->pos0x[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][0] - x * v0_r) * par->dt[p];
            pt->pos0y[p][n] = r * y;
            pt->pos1y[p][n] = pt->pos0y[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][1] - y * v0_r) * par->dt[p];
            pt->pos0z[p][n] = r * z;
            pt->pos1z[p][n] = pt->pos0z[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][2] - z * v0_r) * par->dt[p];
            pt->q[p][n] = qs[p];
            pt->m[p][n] = mp[p];
            //         nt[p] += q[p][n];
        }
    }
#pragma omp barrier
    gsl_rng_free(rng); // dealloc the rng
}

void generate_rand_cylinder(particles *pt, par *par)

{
    // cylindrical plasma radius is r0_f*a0 .
    float Temp[2] = {Temp_e, Temp_d}; // in K convert to eV divide by 1.160451812e4
    // initial bulk electron, ion velocity
    float v0[2][3] = {{0, 0, -vz0}, {0, 0, vz0 / 60}}; /*1e6*/

    float r0 = r0_f * a0; // the radius
    float area = pi * r0 * r0;
    float volume = pi * r0 * r0 * n_space * a0;

    // calculated plasma parameters
    info_file << "initial e Temperature, = " << Temp_e / 11600 << "eV, initial d Temperature, = " << Temp_d / 11600 << " eV\n";
    float Density_e = (n_partd - (n_space_divx - 2) * (n_space_divy - 2) * (n_space_divz - 2) * nback) / volume * r_part_spart;
    float Density_e1 = nback * r_part_spart / (a0 * a0 * a0);
    info_file << "initial density = " << Density_e << "background density = " << Density_e1 << endl;
    float initial_current = Density_e * e_charge * v0[0][2] * area;
    info_file << "initial current = " << initial_current << endl;
    float Bmaxi = initial_current * 2e-7 / r0;
    info_file << "initial Bmax = " << Bmaxi << endl;
    float plasma_freq = sqrt(Density_e * e_charge * e_charge_mass / (mp[0] * epsilon0)) / (2 * pi);
    float plasma_period = 1 / plasma_freq;
    float Debye_Length = sqrt(epsilon0 * kb * Temp[0] / (Density_e * e_charge * e_charge));
    info_file << "debyeLength=" << Debye_Length << ", a0 = " << a0 << endl;
    if (Debye_Length < a0)
    {
        cerr << "a0 = " << a0 << " too large for this density Debyle Length = " << Debye_Length << endl;

        // exit(1);
    }
    float vel_e = sqrt(kb * Temp[0] / (mp[0] * e_mass) + v0[0][0] * v0[0][0] + v0[0][1] * v0[0][1] + v0[0][2] * v0[0][2]);
    info_file << "electron velocity due to temp and initial velocity " << vel_e << endl;
    float Tv = a0 / vel_e; // time for electron to move across 1 cell
    info_file << "Tv = " << Tv << endl;
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * Bmax0);
    float TDebye = Debye_Length / vel_e;
    float TE = sqrt(2 * a0 / e_charge_mass / Emax0);
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance

    par->dt[0] = 4 * min(min(min(TDebye, min(Tv / md_me, Tcyclotron) / 4), plasma_period / ncalc0[0] / 4), TE / ncalc0[0]) / 2; // electron should not move more than 1 cell after ncalc*dt and should not make more than 1/4 gyration and must calculate E before the next 1/4 plasma period
    par->dt[1] = par->dt[0] * md_me;
    info_file << "par->dt[0] = " << par->dt[0] << endl;
    //  float mu0_4pidt[2]= {mu0_4pi/par->dt[0],mu0_4pi/par->dt[1]};
    info_file << "v0 electron = " << v0[0][0] << "," << v0[0][1] << "," << v0[0][2] << endl;
    /*
    cout << "electron Temp = " << Temp[0] << " K, electron Density = " << Density_e << " m^-3" << endl;
    cout << "Plasma Frequency(assume cold) = " << plasma_freq << " Hz, Plasma period = " << plasma_period << " s" << endl;
    cout << "Cyclotron period = " << Tcyclotron << " s, Time for electron to move across 1 cell = " << Tv << " s" << endl;
    cout << "Time taken for electron at rest to accelerate across 1 cell due to E = " << TE << " s" << endl;
    cout << "electron thermal velocity = " << vel_e << endl;
    cout << "dt = " << par->dt[0] << " s, Total time = " << par->dt[0] * ncalc[0] * ndatapoints * nc << ", s" << endl;
    cout << "Debye Length = " << Debye_Length << " m, initial dimension = " << a0 << " m" << endl;
    cout << "number of particle per cell = " << n_partd / (n_space * n_space * n_space) * 8 << endl;
*/
    // set initial positions and velocity
    double sigma[2] = {sqrt(kb * Temp[0] / (mp[0] * e_mass)), sqrt(kb * Temp[1] / (mp[1] * e_mass))};
    long seed;
    gsl_rng *rng;                        // random number generator
    rng = gsl_rng_alloc(gsl_rng_rand48); // pick random number generator

    time_t myTime;
    seed = time(&myTime);
    info_file << "seed=" << seed << "\n";
    gsl_rng_set(rng, seed); // set seed

    for (int p = 0; p < 2; p++)
    {
        int na = 0;
        for (int n = 0; n < nback; ++n) // set number of particles per cell in background
        {
            for (int k = 2; k < n_space_divz - 2; ++k)
            {
                for (int j = 2; j < n_space_divy - 2; ++j)
                    for (int i = 2; i < n_space_divx - 2; ++i)
                    {
                        pt->pos0x[p][na] = ((float)(i - n_space_divx / 2) + (float)rand() / RAND_MAX) * a0;
                        pt->pos0y[p][na] = ((float)(j - n_space_divy / 2) + (float)rand() / RAND_MAX) * a0;
                        pt->pos0z[p][na] = ((float)(k - n_space_divz / 2) + (float)rand() / RAND_MAX) * a0;
                        pt->pos1x[p][na] = pt->pos0x[p][na];
                        pt->pos1y[p][na] = pt->pos0y[p][na];
                        pt->pos1z[p][na] = pt->pos0z[p][na];
                        pt->q[p][na] = qs[p];
                        pt->m[p][na] = mp[p];
                        na++;
                    }
                //      cout << pt->pos1z[p][na - 1] << " ";
            }
        }

        // #pragma omp parallel for ordered
        for (int n = na; n < n_partd; n++)
        {
            float r = r0 * pow(gsl_ran_flat(rng, 0, 1), 0.5);
            double x, y, z;
            z = gsl_ran_flat(rng, -1.0, 1.0) * a0 * (n_space - 3) * 0.5;
            gsl_ran_dir_2d(rng, &x, &y);
            pt->pos0x[p][n] = r * x;
            pt->pos1x[p][n] = pt->pos0x[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][0]) * par->dt[p];
            pt->pos0y[p][n] = r * y;
            pt->pos1y[p][n] = pt->pos0y[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][1]) * par->dt[p];
            pt->pos0z[p][n] = z;
            pt->pos1z[p][n] = pt->pos0z[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][2]) * par->dt[p];
            // cout << pt->pos1z[p][n] << " ";
            // if (n==0) cout << "p = " <<p <<", sigma = " <<sigma[p]<<", temp = " << Temp[p] << ",mass of particle = " << mp[p] << par->dt[p]<<endl;
            pt->q[p][n] = qs[p];
            pt->m[p][n] = mp[p];
        }
        //        nt[p] +=  pt->q[p][n];
    }
    // #pragma omp barrier
    gsl_rng_free(rng); // dealloc the rng
}