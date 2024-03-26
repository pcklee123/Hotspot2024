#include "include/traj.h"
#include <string>
#include <fstream>
#include <streambuf>
cl::Context context_g;
cl::Device default_device_g;
cl::Program program_g;

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
    add_build_option("XLOW", par->posL[0]);
    add_build_option("YLOW", par->posL[1]);
    add_build_option("ZLOW", par->posL[2]);
    add_build_option("XHIGH", par->posH[0]);
    add_build_option("YHIGH", par->posH[1]);
    add_build_option("ZHIGH", par->posH[2]);
    add_build_option("DX", par->dd[0]);
    add_build_option("DY", par->dd[1]);
    add_build_option("DZ", par->dd[2]);
    add_build_option("NX", (int)par->n_space_div[0]);
    add_build_option("NY", (int)par->n_space_div[1]);
    add_build_option("NZ", (int)par->n_space_div[2]);
    // add_build_option("NC", n_cells);
}

void cl_start(par *par)
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
    cout << "getplatforms\n";
    cl::Platform::get(&platforms);

    cl::Platform default_platform = platforms[0];
    info_file << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    cout << "getdevice\n";
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device default_device;
    cout << "device_id = " << device_id << endl;
    device_id = device_id >= cldevice ? cldevice : device_id;
    default_device = devices[device_id];
    info_file << "\t\tDevice Name: " << default_device.getInfo<CL_DEVICE_NAME>() << "device_id =" << device_id << endl;
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
}
