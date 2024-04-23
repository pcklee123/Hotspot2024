#include "include/traj.h"
#include <string>
#include <fstream>
#include <streambuf>
cl::Context context_g;
cl::Device default_device_g;
cl::Program program_g;
cl::CommandQueue commandQueue_g;
int device_id_g;
bool fastIO;

stringstream cl_build_options;
void add_build_option(string name, string param)
{
    cl_build_options << "-D" << name << "=" << param << " ";
}
void add_build_option(string name, int param) { add_build_option(name, to_string(param)); }
void add_build_option(string name, float param)
{
    cl_build_options << "-D" << name << "=" << param << "f ";
}

void cl_set_build_options(par *par)
{
    add_build_option("XLOWo", par->posL[0]);
    add_build_option("YLOWo", par->posL[1]);
    add_build_option("ZLOWo", par->posL[2]);
    add_build_option("XHIGHo", par->posH[0]);
    add_build_option("YHIGHo", par->posH[1]);
    add_build_option("ZHIGHo", par->posH[2]);
    add_build_option("DXo", par->dd[0]);
    add_build_option("DYo", par->dd[1]);
    add_build_option("DZo", par->dd[2]);
    add_build_option("NX", (int)par->n_space_div[0]);
    add_build_option("NY", (int)par->n_space_div[1]);
    add_build_option("NZ", (int)par->n_space_div[2]);
    // add_build_option("NC", n_cells);
}

void cl_start(fields *fi, particles *pt, par *par)
{
    int AA[1] = {-1};
#pragma omp target
    AA[0] = omp_is_initial_device();
    if (!AA[0])
        info_file << "Able to use GPU offloading with OMP!\n";
    else
        info_file << "\nNo GPU on OMP\n";
    // get all platforms (drivers)
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::vector<cl::Device> devices;
    int platform_id = 0;
    int device_id;
    cl_int res = 0;

    info_file << "Number of Platforms: " << platforms.size() << std::endl;

    for (cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
    {
        device_id = 0;
        cl::Platform platform(*it);

        info_file << "Platform ID: " << platform_id++ << std::endl;
        info_file << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        info_file << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

        //     platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        //       platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        int i = 0;
        for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
        {
            cl::Device device(*it2);
            info_file << "Number of Devices: " << devices.size() << std::endl;
            info_file << "\tDevice " << device_id++ << ": " << std::endl;
            info_file << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            info_file << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
            info_file << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
            info_file << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            par->maxcomputeunits[i] = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            info_file << "\t\tDevice Max Compute Units: " << par->maxcomputeunits[i] << std::endl;
            info_file << "\t\tDevice Global Memory: MB " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024 << std::endl;
            info_file << "\t\tDevice Max Clock Frequency: MHz " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            info_file << "\t\tDevice Max Allocateable Memory MB: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / 1024 / 1024 << std::endl;
            par->cl_align = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
            info_file << "\t\tDevice addr_align: kB " << device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() << std::endl;
            info_file << "\t\tDevice Local Memory: kB " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << std::endl;
            info_file << "\t\tDevice Available: " << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
            ++i;
        }
        info_file << std::endl;
    }
    // cout << "getplatforms\n";
    cl::Platform::get(&platforms);

    cl::Platform default_platform = platforms[0];
    info_file << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    // cout << "getdevice\n";
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device default_device;
    device_id--;
    device_id = (device_id >= cldevice) ? cldevice : (device_id >= 0 ? device_id : 0); // use dGPU only if available
    cout << "device_id = " << device_id << ", devices.size = " << devices.size() << ", cl_align = " << par->cl_align << endl;
    default_device = devices[device_id];
    info_file << "\t\tDevice Name: " << default_device.getInfo<CL_DEVICE_NAME>() << "\ndevice_id =" << device_id << endl;
    info_file << "OpenCL Version: " << default_device.getInfo<CL_DEVICE_VERSION>() << std::endl;

    cl::Context context({default_device});

    cl::Program::Sources sources;

    // read in kernel code which calculates the next position
    std::ifstream t("cl_kernel_code.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(t)),
                            std::istreambuf_iterator<char>());

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);

    cl_int cl_err = program.build({default_device}, cl_build_options.str().c_str());
    info_file << "building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
    if (cl_err != CL_SUCCESS)
    {
        info_file << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << endl;
        info_file << cl_build_options.str() << endl;
        cerr << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << endl;
        exit(1);
    }
    else
        info_file << " success " << endl;
    context_g = context;
    default_device_g = default_device;
    program_g = program;
    device_id_g = device_id;
    cl::CommandQueue queue(context_g, default_device_g);
    commandQueue_g = queue;

    cout << "allocating buffers\n";

    // cout << "check for unified memory " << endl;
    //  create buffers on the device
    /** IMPORTANT: do not use CL_MEM_USE_HOST_PTR if on dGPU **/
    /** HOST_PTR is only used so that memory is not copied, but instead shared between CPU and iGPU in RAM**/
    // Note that special alignment has been given to Ea, Ba, y0, z0, x0, x1, y1 in order to actually do this properly
    // Assume buffers A, B, I, J (Ea, Ba, ci, cf) will always be the same. Then we save a bit of time.
    // get whether or not we are on an iGPU/similar, and can use certain memmory optimizations
    bool temp;
    default_device_g.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &temp);
    if (temp == true)
        info_file << "Using unified memory: " << temp << " \n";
    else
        info_file << "No unified memory: " << temp << " \n";
    fastIO = temp;
    fastIO = false;

    static cl::Buffer buff_E(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, n_cellsf * 3, fastIO ? fi->E : NULL, &cl_err);
    static cl::Buffer buff_B(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, n_cellsf * 3, fastIO ? fi->B : NULL, &cl_err);
    static cl::Buffer buff_Ee(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, n_cellsf * 3, fastIO ? fi->Ee : NULL, &cl_err);
    static cl::Buffer buff_Be(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, n_cellsf * 3, fastIO ? fi->Be : NULL, &cl_err);
    static cl::Buffer buff_npt(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->npt : NULL, &cl_err); // cannot be static?
    static cl::Buffer buff_jc(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->jc : NULL, &cl_err);

    static cl::Buffer buff_V(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->V : NULL, &cl_err); // cannot be static?

    static cl::Buffer buff_np_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->np[0] : NULL);
    static cl::Buffer buff_np_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->np[1] : NULL);

    static cl::Buffer buff_currentj_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->currentj[0] : NULL);
    static cl::Buffer buff_currentj_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->currentj[1] : NULL);

    static cl::Buffer buff_npi(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsi, fastIO ? fi->npi : NULL);
    static cl::Buffer buff_cji(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsi * 3, fastIO ? fi->cji : NULL);

    unsigned int n4 = n_partd * sizeof(float);                                                                                   // number of particles * sizeof(float)
    static cl::Buffer buff_q_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[0] : NULL); // q
    static cl::Buffer buff_q_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[1] : NULL); // q

    //  cout << "buffers " << endl;
    static cl::Buffer buff_x0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[0] : NULL); // x0
    static cl::Buffer buff_y0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[0] : NULL); // y0
    static cl::Buffer buff_z0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[0] : NULL); // z0
    static cl::Buffer buff_x1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[0] : NULL); // x1
    static cl::Buffer buff_y1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[0] : NULL); // y1
    static cl::Buffer buff_z1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[0] : NULL); // z1
                                                                                                                                      //  cout << "buffers " << endl;

    static cl::Buffer buff_x0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[1] : NULL); // x0
    static cl::Buffer buff_y0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[1] : NULL); // y0
    static cl::Buffer buff_z0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[1] : NULL); // z0
    static cl::Buffer buff_x1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[1] : NULL); // x1
    static cl::Buffer buff_y1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[1] : NULL); // y1
    static cl::Buffer buff_z1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[1] : NULL); // z1

    static cl_mem r3_buffer = clCreateBuffer(context_g(), CL_MEM_READ_WRITE, (uint64_t)sizeof(complex<float>) * n_cells4 * 6, 0, &res);
    if (res)
        cout << "r3_buffer" << endl;
    fi->r3_buffer = r3_buffer;
#ifdef Uon_
    static cl_mem r2_buffer = clCreateBuffer(context_g(), CL_MEM_READ_WRITE, (uint64_t)sizeof(complex<float>) * n_cells4, 0, &res);
    fi->r2_buffer = r2_buffer;
#endif
    static cl_mem fft_real_buffer = clCreateBuffer(context_g(), CL_MEM_READ_WRITE, (uint64_t)sizeof(float) * n_cells8 * 4, 0, &res);
    static cl_mem fft_complex_buffer = clCreateBuffer(context_g(), CL_MEM_READ_WRITE, (uint64_t)sizeof(complex<float>) * n_cells4 * 4, 0, &res);
    static cl_mem fft_p_buffer = clCreateBuffer(context_g(), CL_MEM_READ_WRITE, (uint64_t)sizeof(complex<float>) * n_cells4 * 4, 0, &res);
    fi->fft_real_buffer = fft_real_buffer;
    fi->fft_complex_buffer = fft_complex_buffer;
    fi->fft_p_buffer = fft_p_buffer;

    if (cl_err)
        cout << cl_err << endl;

    fi->buff_E = &buff_E;
    fi->buff_B = &buff_B;
    fi->buff_Ee = &buff_Ee;
    fi->buff_Be = &buff_Be;
    fi->buff_npt = &buff_npt;
    fi->buff_jc = &buff_jc;
    fi->buff_V = &buff_V;

    fi->buff_np_e = &buff_np_e;
    fi->buff_np_i = &buff_np_i;
    fi->buff_currentj_e = &buff_currentj_e;
    fi->buff_currentj_i = &buff_currentj_i;

    fi->buff_npi = &buff_npi;
    fi->buff_cji = &buff_cji;

    pt->buff_q_e = &buff_q_e;
    pt->buff_q_i = &buff_q_i;

    pt->buff_x0_e = &buff_x0_e;
    pt->buff_y0_e = &buff_y0_e;
    pt->buff_z0_e = &buff_z0_e;
    pt->buff_x1_e = &buff_x1_e;
    pt->buff_y1_e = &buff_y1_e;
    pt->buff_z1_e = &buff_z1_e;

    pt->buff_x0_i = &buff_x0_i;
    pt->buff_y0_i = &buff_y0_i;
    pt->buff_z0_i = &buff_z0_i;
    pt->buff_x1_i = &buff_x1_i;
    pt->buff_y1_i = &buff_y1_i;
    pt->buff_z1_i = &buff_z1_i;

    // because some code is in C not C++
    fi->E_buffer = buff_E();
    fi->B_buffer = buff_B();
    fi->Ee_buffer = buff_Ee();
    fi->Be_buffer = buff_Be();
    fi->npt_buffer = buff_npt();
    fi->jc_buffer = buff_jc();
    fi->V_buffer = buff_V();
}
