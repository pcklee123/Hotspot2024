#include "include/traj.h"

void id_to_cell(int id, int *x, int *y, int *z)
{
    constexpr size_t xy = n_space_divx * n_space_divy;
    *z = id / xy;
    id = id % xy;
    *y = id / n_space_divx;
    *x = id % n_space_divx;
}

void Time::mark()
{
    marks.push_back(chrono::high_resolution_clock::now());
}

float Time::elapsed()
{
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - marks.back()).count();
    marks.pop_back();
    return (float)time * 1e-6;
}

// Get the same result as elapsed, but also insert the current time point back in
float Time::replace()
{
    auto now = chrono::high_resolution_clock::now();
    auto back = marks.back();
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(now - back).count();
    back = now;
    return (float)time * 1e-6;
}

Log::Log()
{
    if (!log_file.is_open())
        log_file.open("log.csv");
    log_file << setprecision(5);
}

void Log::newline()
{
    log_file << "\n";
    log_file.flush();
    firstEntry = true;
}
void Log::close()
{
    log_file.close();
}

void log_headers()
{
    logger.write("t_large");
    logger.write("t_small");
    logger.write("dt_ch");
    logger.write("nc_ele");
    logger.write("nc_deut");
    logger.write("dt_ele_fs");
    logger.write("dt_deut_fs");
    logger.write("t_sim_ps");
    logger.write("ne");
    logger.write("ni");
    logger.write("KEt_e");
    logger.write("KEt_d");
    logger.write("Ele_pot");
    logger.write("Mag_pot");
    logger.write("E_tot");
    logger.write("Emax");
    logger.write("Bmax");
    logger.write("Ecoeff_e");
    logger.write("Bcoeff_e");
    logger.write("Ecoeff_i");
    logger.write("Bcoeff_i");
    logger.newline();
}

void log_entry(int i_time, int ntime, int cdt, int total_ncalc[2], double t, par *par)
{
    float ntall = par->nt[1] - par->nt[0];
    logger.write(i_time);
    logger.write(ntime);
    logger.write(cdt);
    logger.write(total_ncalc[0]);
    logger.write(total_ncalc[1]);
    logger.write(par->dt[0] * 1e15); // in fs
    logger.write(par->dt[1] * 1e15);
    logger.write(t * 1e12); // in ps
    logger.write(par->nt[0]);
    logger.write(par->nt[1]);
    logger.write(-par->KEtot[0] / par->nt[0]);
    logger.write(par->KEtot[1] / par->nt[1]);
    logger.write(par->UE / ntall);
    logger.write(par->UB / ntall);
    logger.write((par->KEtot[0] + par->KEtot[1] + par->UB + par->UE*0.5) / ntall);
    logger.write(par->Emax);
    logger.write(par->Bmax*1000);
    logger.write(par->Ecoef[0] * 1e21);
    logger.write(par->Bcoef[0] * 1e9);
    logger.write(par->Ecoef[1] * 1e21);
    logger.write(par->Bcoef[1] * 1e9);
    logger.newline();
}
float maxvalf(float *data_1d, int n)
{
    float max = 0;
#pragma omp parallel for reduction(max : max)
    for (unsigned int i = 0; i < n; ++i)
    {
        float absVal = fabs(data_1d[i]);
        max = (absVal > max) ? absVal : max; // use the ternary operator to update the maximum
    }
    return max;
}

void info(par *par)
{
    info_file.open("info.csv");
    omp_set_nested(true);
    nthreads = omp_get_max_threads(); // omp_set_num_threads(nthreads);

    cin.tie(NULL); // Fast printing
    // ios_base::sync_with_stdio(false);
    cout << std::scientific;
    cout.precision(1);
    cerr << std::scientific;
    cerr.precision(3);

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
    info_file << "Output dir: " << par->outpath << "\n";
    cl_set_build_options(par);
    cl_start(par);

    log_headers();

    // print initial conditions
    {
        info_file << "float size=" << sizeof(float) << ", "
                  << "int32_t size=" << sizeof(int32_t) << ", "
                  << "int size=" << sizeof(int) << "(unsigned int) ((int)(-2.5f))" << (unsigned int)((int)(-2.5f)) << endl;
        info_file << "omp_get_max_threads()= " << omp_get_max_threads() << endl;
        info_file << "Data Origin," << par->posL[0] << "," << par->posL[1] << "," << par->posL[0] << endl;
        info_file << "Data Spacing," << par->dd[0] << "," << par->dd[1] << "," << par->dd[2] << endl;
        info_file << "Data extent x, 0," << n_space - 1 << endl;
        info_file << "Data extent y, 0," << n_space - 1 << endl;
        info_file << "Data extent z, 0," << n_space - 1 << endl;
        info_file << "electron Temp+e = ," << Temp_e << ",K" << endl;
        info_file << "Maximum expected B = ," << par->Bmax << endl;
        info_file << "time step between prints = ," << par->dt[0] * par->ncalcp[0] * par->nc << ",s" << endl;
        info_file << "time step between EBcalc = ," << par->dt[0] * par->ncalcp[0] << ",s" << endl;
        info_file << "dt_e = ," << par->dt[0] << ",s" << endl;
        info_file << "dt_i = ," << par->dt[1] << ",s" << endl;
        info_file << "cell size =," << a0 << ",m" << endl;
        info_file << "number of particles per cell = ," << n_partd / (n_space * n_space * n_space) << endl;
        info_file << "time for electrons to leave box = ," << n_space * a0 / sqrt(2 * kb * Temp_e / e_mass) << ",s" << endl;
        info_file << "time for ions to leave box = ," << n_space * a0 * md_me / sqrt(2 * kb * Temp_d / e_mass) << ",s" << endl;
    }
}

particles *alloc_particles(par *par)
{
    auto *pt = (particles *)malloc(sizeof(particles));
    //[pos0,pos1][x,y,z][electrons,ions][n_partd]
    // position of particle and velocity: stored as 2 positions at slightly different times [2 positions previous and current][3 components][2 types of particles][number of particles]
    /** CL: Ensure that pos0/1.. contain multiple of 64 bytes, ie. multiple of 16 floats **/
    //*
    pt->pos = reinterpret_cast<float(&)[2][3][2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * par->n_part[0] * 2 * 3 * 2, par->cl_align)));
    // convenience pointers pos0[3 components][2 types of particles][n-particles] as 1D
    pt->pos0 = reinterpret_cast<float(*)>(pt->pos[0]);
    pt->pos1 = reinterpret_cast<float(*)>(pt->pos[1]);
    pt->pos0x = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pt->pos[0][0]));
    pt->pos0y = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pt->pos[0][1]));
    pt->pos0z = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pt->pos[0][2]));
    pt->pos1x = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pt->pos[1][0]));
    pt->pos1y = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pt->pos[1][1]));
    pt->pos1z = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pt->pos[1][2]));

    //    charge of particles
    auto *q = static_cast<int(*)[n_partd]>(_aligned_malloc(2 * n_partd * sizeof(int), par->cl_align)); // charge of each particle +1 for H,D or T or -1 for electron can also be +2 for He for example
    auto *m = static_cast<int(*)[n_partd]>(_aligned_malloc(2 * n_partd * sizeof(int), par->cl_align)); // mass of of each particle not really useful unless we want to simulate many different types of particles
    pt->q = q;
    pt->m = m;
    return pt;
}
fields *alloc_fields(par *par)
{
    auto *f = (fields *)malloc(sizeof(fields));
    /** CL: Ensure that Ea/Ba contain multiple of 64 bytes, ie. multiple of 16 floats **/

    f->E = reinterpret_cast<float(&)[3][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(3 * n_cells));                                        // selfgenerated E field
    f->Ee = new float[3][n_space_divz][n_space_divy][n_space_divx];                                                                                        // External E field
    f->Ea = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx][ncoeff]>(_aligned_malloc(sizeof(float) * n_cells * 3 * ncoeff, par->cl_align)); // coefficients for Trilinear interpolation Electric field

    f->B = reinterpret_cast<float(&)[3][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(3 * n_cells)); // new float[3][n_space_divz][n_space_divy][n_space_divx];
    f->Be = new float[3][n_space_divz][n_space_divy][n_space_divx];
    f->Ba = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx][ncoeff]>(_aligned_malloc(sizeof(float) * n_cells * 3 * ncoeff, par->cl_align)); // coefficients for Trilinear interpolation Magnetic field

    f->V = reinterpret_cast<float(&)[1][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(n_cells));

    f->np = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * n_cells * sizeof(float), alignment));
    f->npi = static_cast<int(*)[n_space_divy][n_space_divx]>(_aligned_malloc(n_cells * sizeof(int), alignment));
    f->np_centeri = static_cast<int(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(n_cells * 3 * sizeof(int), alignment));
    f->npt = static_cast<float(*)[n_space_divy][n_space_divx]>(_aligned_malloc(n_cells * sizeof(float), alignment));
    f->currentj = static_cast<float(*)[3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * 3 * n_cells * sizeof(float), alignment));
    f->cji = static_cast<int(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(n_cells * sizeof(int) * 3, alignment));
    f->cj_centeri = static_cast<int(*)[3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(n_cells * sizeof(int) * 3 * 3, alignment));
    f->jc = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(3 * n_cells * sizeof(float), alignment));
    return f;
}