#include "traj.h"
void generate_rand_sphere(particles *pt, par *par)
{
    // spherical plasma set plasma parameters
    float Temp[2] = {Temp_e, Temp_d}; // in K convert to eV divide by 1.160451812e4
    // initial bulk electron, ion velocity
    float v0[2][3] = {{0, 0, -vz0}, {0, 0, vz0 / 60}};
    float r0[3] = {r0_f[0] * a0, r0_f[1] * a0, r0_f[2] * a0}; // if sphere this is the radius
    // set initial positions and velocity
    float sigma[2] = {sqrt(kb * Temp[0] / (mp[0] * e_mass)), sqrt(kb * Temp[1] / (mp[1] * e_mass))};
    std::random_device rd;  // Seed for the random number engine
    std::mt19937 gen(rd()); // Mersenne Twister random number engine
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::weibull_distribution<float> weibull_dist(weibullb,1.0f); // Weibull distribution 1st parameter is shape

    size_t na = 0;
    for (int p = 0; p < 2; p++)
    {
        // #pragma omp parallel for simd num_threads(nthreads)
        for (na = 0; na < nback; ++na) // set number of particles per cell in background
        {
            float r = r0[2] * pow(uniform_dist(gen), 0.5);
            float x, y, z;
            z = uniform_dist(gen) * (par->posH_1[2] - par->posL_1[2]) + par->posL_1[2];
            float theta = 2 * pi * uniform_dist(gen);
            float phi = acos(2 * uniform_dist(gen) - 1);
            x = sin(phi) * cos(theta);
            y = sin(phi) * sin(theta);
#ifdef octant
            x = abs(x);
            y = abs(y);
#endif
#ifdef quadrant
            x = abs(x);
            y = abs(y);
#endif
            //          cout << r << ", " << x << ", " << y << ", " << z << endl;
            pt->pos0x[p][na] = r * x;
            pt->pos1x[p][na] = pt->pos0x[p][na] + v0[p][0] * par->dt[p];
            pt->pos0y[p][na] = r * y;
            pt->pos1y[p][na] = pt->pos0y[p][na] + v0[p][1] * par->dt[p];
            pt->pos0z[p][na] = z;
            pt->pos1z[p][na] = pt->pos0z[p][na] + v0[p][2] * par->dt[p];
            pt->q[p][na] = qs[p];
        }

        // #pragma omp parallel for simd num_threads(nthreads)
        for (int n = nback; n < n_partd; n++)
        {
#ifdef Weibull
            float r = r0[p] * pow(weibull_dist(gen),0.33333333333);
            //     cout << r << ", ";
            while ((r) >= r0[2]) // Don't allow particles to be generated outside the plasma
            {
                r = r0[p] * pow(weibull_dist(gen),0.33333333333);
            }
#else
            float r = r0 * pow(uniform_dist(gen), 0.3333333333);
#endif
            float x, y, z;
            float theta = 2 * pi * uniform_dist(gen);
            float phi = acos(2 * uniform_dist(gen) - 1);
            x = sin(phi) * cos(theta);
            y = sin(phi) * sin(theta);
            z = cos(phi);
#ifdef octant
            x = abs(x);
            y = abs(y);
            z = abs(z);
#endif
#ifdef quadrant
            x = abs(x);
            y = abs(y);
#endif
            //  cout << r << ", " << x * x + y * y + z * z << ", " << y << ", " << z << endl;
            pt->pos0x[p][n] = r * x;
            pt->pos1x[p][n] = pt->pos0x[p][n] + (normal_dist(gen) * sigma[p] + v0[p][0] + x * v0_r) * par->dt[p];
            pt->pos0y[p][n] = r * y;
            pt->pos1y[p][n] = pt->pos0y[p][n] + (normal_dist(gen) * sigma[p] + v0[p][1] + y * v0_r) * par->dt[p];
            pt->pos0z[p][n] = r * z;
            pt->pos1z[p][n] = pt->pos0z[p][n] + (normal_dist(gen) * sigma[p] + v0[p][2] + z * v0_r) * par->dt[p];
            pt->q[p][n] = qs[p];
         //   cout << pt->pos0x[p][n] - pt->pos1x[p][n] << ", " << normal_dist(gen) << endl;
        }
    }
#pragma omp barrier
    if (!fastIO) // write CPU generated particle positions to opencl buffers
    {            //  electrons
        commandQueue_g.enqueueWriteBuffer(pt->buff_x0_e[0], CL_TRUE, 0, n_partf, pt->pos0x[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y0_e[0], CL_TRUE, 0, n_partf, pt->pos0y[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z0_e[0], CL_TRUE, 0, n_partf, pt->pos0z[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_x1_e[0], CL_TRUE, 0, n_partf, pt->pos1x[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y1_e[0], CL_TRUE, 0, n_partf, pt->pos1y[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z1_e[0], CL_TRUE, 0, n_partf, pt->pos1z[0]);

        commandQueue_g.enqueueWriteBuffer(pt->buff_q_e[0], CL_TRUE, 0, n_partf, pt->q[0]);
        //  ions
        commandQueue_g.enqueueWriteBuffer(pt->buff_x0_i[0], CL_TRUE, 0, n_partf, pt->pos0x[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y0_i[0], CL_TRUE, 0, n_partf, pt->pos0y[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z0_i[0], CL_TRUE, 0, n_partf, pt->pos0z[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_x1_i[0], CL_TRUE, 0, n_partf, pt->pos1x[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y1_i[0], CL_TRUE, 0, n_partf, pt->pos1y[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z1_i[0], CL_TRUE, 0, n_partf, pt->pos1z[1]);

        commandQueue_g.enqueueWriteBuffer(pt->buff_q_i[0], CL_TRUE, 0, n_partf, pt->q[1]);
    }
}

void generate_rand_cylinder(particles *pt, par *par)
{
    // cylindrical plasma radius is r0_f*a0 .
    float Temp[2] = {Temp_e, Temp_d}; // in K convert to eV divide by 1.160451812e4
    // initial bulk electron, ion velocity
    float v0[2][3] = {{0, 0, -vz0}, {0, 0, vz0 / 60}}; /*1e6*/

    float r0[3] = {r0_f[0] * a0, r0_f[1] * a0, r0_f[2] * a0}; // if sphere this is the radius
    float area = pi * r0[0] * r0[0];
    float volume = pi * r0[0] * r0[0] * n_space * a0;

    // calculated plasma parameters
    info_file << "initial e Temperature, = " << Temp_e / 11600 << "eV, initial d Temperature, = " << Temp_d / 11600 << " eV\n";
    // float Density_e = (n_partd - (n_space_divx - 2) * (n_space_divy - 2) * (n_space_divz - 2) * nback) / volume * r_part_spart;
    // float Density_e1 = nback * r_part_spart / (a0 * a0 * a0);
    float Density_e = (n_partd - nback) * r_part_spart / volume;
    float Density_e1 = nback * r_part_spart / (powf(n_space * a0, 3));

    info_file << "initial density = " << Density_e << "background density = " << Density_e1 << endl;
    float initial_current = Density_e * e_charge * v0[0][2] * area;
    info_file << "initial current = " << initial_current << endl;
    float Bmaxi = initial_current * 2e-7 / r0[0];
    info_file << "initial electron Bmax = " << Bmaxi << endl;
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    std::uniform_real_distribution<> dis_minus1_1(-1.0, 1.0);
    std::normal_distribution<> normal_dis(0, 1);

    size_t na = 0;
    for (int p = 0; p < 2; p++)
    {
// #pragma omp parallel for
#pragma omp parallel for simd num_threads(nthreads)
        for (na = 0; na < nback; ++na) // set number of particles per cell in background
        {
            pt->pos1x[p][na] = pt->pos0x[p][na] = dis(gen) * (par->posH_15[0] - par->posL_15[0]) + par->posL_15[0];
            pt->pos1y[p][na] = pt->pos0y[p][na] = dis(gen) * (par->posH_15[1] - par->posL_15[1]) + par->posL_15[1];
            pt->pos1z[p][na] = pt->pos0z[p][na] = dis(gen) * (par->posH_15[2] - par->posL_15[2]) + par->posL_15[2];
            pt->q[p][na] = qs[p];
            // pt->m[p][na] = mp[p];
        }

// #pragma omp parallel for ordered
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = na; n < n_partd; n++)
        {
            float r = r0[p] * pow(dis(gen), 0.5);
            double x, y, z;
            z = dis_minus1_1(gen) * a0 * (n_space - 3) * 0.5;
            double theta = 2 * pi * dis(gen);
            x = cos(theta);
            y = sin(theta);
            pt->pos0x[p][n] = r * x;
            pt->pos1x[p][n] = pt->pos0x[p][n] + (normal_dis(gen) * sigma[p] + v0[p][0]) * par->dt[p];
            pt->pos0y[p][n] = r * y;
            pt->pos1y[p][n] = pt->pos0y[p][n] + (normal_dis(gen) * sigma[p] + v0[p][1]) * par->dt[p];
            pt->pos0z[p][n] = z;
            pt->pos1z[p][n] = pt->pos0z[p][n] + (normal_dis(gen) * sigma[p] + v0[p][2]) * par->dt[p];
            pt->q[p][n] = qs[p];
            //         pt->m[p][n] = mp[p];
        }
        //        nt[p] +=  pt->q[p][n];
    }
    // #pragma omp barrier
    if (!fastIO) // write CPU generated particle positions to opencl buffers
    {            //  electrons
        commandQueue_g.enqueueWriteBuffer(pt->buff_x0_e[0], CL_TRUE, 0, n_partf, pt->pos0x[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y0_e[0], CL_TRUE, 0, n_partf, pt->pos0y[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z0_e[0], CL_TRUE, 0, n_partf, pt->pos0z[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_x1_e[0], CL_TRUE, 0, n_partf, pt->pos1x[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y1_e[0], CL_TRUE, 0, n_partf, pt->pos1y[0]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z1_e[0], CL_TRUE, 0, n_partf, pt->pos1z[0]);

        commandQueue_g.enqueueWriteBuffer(pt->buff_q_e[0], CL_TRUE, 0, n_partf, pt->q[0]);
        //  ions
        commandQueue_g.enqueueWriteBuffer(pt->buff_x0_i[0], CL_TRUE, 0, n_partf, pt->pos0x[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y0_i[0], CL_TRUE, 0, n_partf, pt->pos0y[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z0_i[0], CL_TRUE, 0, n_partf, pt->pos0z[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_x1_i[0], CL_TRUE, 0, n_partf, pt->pos1x[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_y1_i[0], CL_TRUE, 0, n_partf, pt->pos1y[1]);
        commandQueue_g.enqueueWriteBuffer(pt->buff_z1_i[0], CL_TRUE, 0, n_partf, pt->pos1z[1]);

        commandQueue_g.enqueueWriteBuffer(pt->buff_q_i[0], CL_TRUE, 0, n_partf, pt->q[1]);
    }
}