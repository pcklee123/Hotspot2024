#include "include/traj.h"
#include <string>
#include <fstream>
#include <streambuf>
cl::Context context_g;
cl::Device default_device_g;
cl::Program program_g;
cl::CommandQueue commandQueue_g;
int device_id_g;

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

void cl_start(fields *fi, par *par)
{
    int AA[1] = {-1};
    cl_int cl_err;
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
    cl_int platform_id = 0;
    cl_int device_id;

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
        for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
        {
            cl::Device device(*it2);
            info_file << "Number of Devices: " << devices.size() << std::endl;
            info_file << "\tDevice " << device_id++ << ": " << std::endl;
            info_file << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            info_file << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
            info_file << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
            info_file << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            info_file << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            info_file << "\t\tDevice Global Memory: MB " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024 << std::endl;
            info_file << "\t\tDevice Max Clock Frequency: MHz " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            info_file << "\t\tDevice Max Allocateable Memory MB: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / 1024 / 1024 << std::endl;
            par->cl_align = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
            info_file << "\t\tDevice addr_align: kB " << device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() << std::endl;
            info_file << "\t\tDevice Local Memory: kB " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << std::endl;
            info_file << "\t\tDevice Available: " << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
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
    // cout << "device_id = " << device_id << endl;
    device_id = (device_id >= cldevice) ? cldevice : device_id; // use dGPU only if available
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

    cl_err = program.build({default_device}, cl_build_options.str().c_str());
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
    bool fastIO;
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

    static cl::Buffer buff_np_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->np[0] : NULL);
    static cl::Buffer buff_np_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf, fastIO ? fi->np[1] : NULL);

    static cl::Buffer buff_currentj_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->currentj[0] : NULL);
    static cl::Buffer buff_currentj_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->currentj[1] : NULL);

    static cl::Buffer buff_npi(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsi, fastIO ? fi->npi : NULL);
    static cl::Buffer buff_cji(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsi * 3, fastIO ? fi->cji : NULL);

    static cl::Buffer buff_q_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[0] : NULL); // q
    static cl::Buffer buff_q_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[1] : NULL); // q

    if (cl_err)
        cout << cl_err << endl;

    fi->buff_E = &buff_E;
    fi->buff_B = &buff_B;
    fi->buff_Ee = &buff_Ee;
    fi->buff_Be = &buff_Be;
    fi->buff_npt = &buff_npt;
    fi->buff_jc = &buff_jc;

    fi->buff_np_e = &buff_np_e;
    fi->buff_np_i = &buff_np_i;
    fi->buff_currentj_e = &buff_currentj_e;
    fi->buff_currentj_i = &buff_currentj_i;

    fi->buff_npi = &buff_npi;
    fi->buff_cji = &buff_cji;

    fi->buff_q_e = &buff_q_e;
    fi->buff_q_i = &buff_q_i;

    fi->E_buffer = fi->buff_E[0](); // buff_E();
                                    // cout << fi->E_buffer << ", " << buff_E() << ", " << fi->buff_E[0]() << endl;
    fi->B_buffer = buff_B();
    fi->Ee_buffer = buff_Ee();
    fi->Be_buffer = buff_Be();
    fi->npt_buffer = buff_npt();
    fi->jc_buffer = buff_jc();
}
