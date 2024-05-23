#ifndef TRAJ_H_INCLUDED
#define TRAJ_H_INCLUDED

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
// #include <gsl/gsl_rng.h>//
// #include <gsl/gsl_randist.h>
#include <random>
// #include <cmath>
#include <iostream>
#include <omp.h>
#include <string>
#include <filesystem>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#include <CL/opencl.hpp>
// #include <vtk/vtksys/Configure.hxx>
#ifdef _WIN32
#include <vtk/vtkSmartPointer.h>
#include <vtk/vtkFloatArray.h>
#include <vtk/vtkDoubleArray.h>
#include <vtk/vtkPolyData.h>
#include <vtk/vtkCellData.h>
#include <vtk/vtkSmartPointer.h>

#include <vtk/vtkInformation.h>
#include <vtk/vtkTable.h>

#include <vtk/vtkDelimitedTextWriter.h>

#include <vtk/vtkZLibDataCompressor.h>
#include <vtk/vtkXMLImageDataWriter.h>
#include <vtk/vtkXMLPolyDataWriter.h>
#include <vtk/vtkImageData.h>
#include <vtk/vtkPointData.h>
#else
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkPolyData.h>
#include <vtkCellData.h>
#include <vtkSmartPointer.h>

#include <vtkInformation.h>
#include <vtkTable.h>

#include <vtkDelimitedTextWriter.h>

#include <vtkZLibDataCompressor.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#endif
#include <complex>
// #include <infft.h>

// #include <nfft3.h>
//  #include "nfft3mp.h"
// #include <fftw3_threads.h>
// #include <fftw3.h>

using namespace std;

#include "traj_physics.h"
extern cl::Context context_g;
extern cl::Device default_device_g;
extern cl::Program program_g;
extern cl::CommandQueue commandQueue_g;
extern int device_id_g;
extern cl_bool fastIO;
extern string outpath;

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
/* UNIX-style OS. ------------------------------------------- */
const string outpath1 = std::filesystem::temp_directory_path().string() + "/out/";
const string outpath2 = std::filesystem::temp_directory_path().string() + "/out/";
#else
#ifdef RamDisk // save file info - initialize filepath
const string outpath1 = "R:\\Temp\\out\\";
#else
const string outpath1 = std::filesystem::temp_directory_path().string() + "out/";
#endif
const string outpath2 = std::filesystem::temp_directory_path().string() + "out/";
#endif

static int nthreads;
constexpr int alignment = 64; // 512 bits / 8 bits per byte = 64 bytes

class Time
{
private:
        vector<chrono::_V2::system_clock::time_point> marks;

public:
        void mark();
        float elapsed();
        float replace();
};
class Log
{
private:
        ofstream log_file;
        bool firstEntry = true; // Whether the next item to print is the first item in the line
public:
        Log();
        template <class T>
        void write(T text, bool flush = false)
        {
                if (!firstEntry)
                        log_file << ",";
                firstEntry = false;
                log_file << text;
                if (flush)
                        log_file.flush();
        }
        void newline();
        void close();
};
static Time timer;
static Log logger;

extern ofstream info_file;
void log_entry(int i_time, int ntime, int total_ncalc[2], double t, par *par);
void log_headers();

void cl_start(fields *fi, particles *pt, par *par);
void cl_set_build_options(par *par);

void tnp(fields *fi, particles *pt, par *par);
// void get_precalc_r3(float precalc_r3[3][n_space_divz2][n_space_divy2][n_space_divx2], float dd[3]);
int calcEBV(fields *fi, par *par);

void save_files(int i_time, double t, fields *fi, particles *pt, par *par);
// void sel_part_print(particles *pt, float posp[2][n_output_part][3], float KE[2][n_output_part], par *par);

void get_densityfields(fields *fi, particles *pt, par *par);
void calc_trilin_constants(fields *fi, par *par);

int changedt(particles *p, int inc, par *par);

void calcU(fields *fi, particles *pt, par *par);

void generateParticles(particles *pt, par *par);
void generateField(fields *fi, par *par);
void id_to_cell(int id, int *x, int *y, int *z);
void save_hist(int i_time, double t, int q[2][n_partd], float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], par *par);

void generate_rand_sphere(particles *pt, par *par);
void generate_rand_impl_sphere(particles *pt, par *par);
void generate_rand_cylinder(particles *pt, par *par);
void smoothscalarfield(float f[n_space_divz][n_space_divy][n_space_divx], float ftemp[n_space_divz][n_space_divy][n_space_divx],
                       float fc[n_space_divz][n_space_divy][n_space_divx][3], int s);
float maxvalf(float *data_1d, int n);
void info(par *par);
void changedx(fields *fi, par *par);
particles *alloc_particles(par *par);
fields *alloc_fields(par *par);

void recalcpos(particles *pt, par *par, float inc);

#endif // TRAJ_H_INCLUDED
