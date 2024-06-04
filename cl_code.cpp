#include "include/traj.h"
#include <string>
#include <fstream>
#include <streambuf>

cl::Context context_g;
cl::Device default_device_g;
cl::Program program_g;
cl::CommandQueue commandQueue_g;
int device_id_g;
cl_bool fastIO;
int platformn = 0; // choose a good one based on info.csv, GPU is usually best.

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
    add_build_option("XLOWo", (float)par->posL[0]);
    add_build_option("YLOWo", (float)par->posL[1]);
    add_build_option("ZLOWo", (float)par->posL[2]);
    add_build_option("XHIGHo", (float)par->posH[0]);
    add_build_option("YHIGHo", (float)par->posH[1]);
    add_build_option("ZHIGHo", (float)par->posH[2]);
    add_build_option("DXo", (float)par->dd[0]);
    add_build_option("DYo", (float)par->dd[1]);
    add_build_option("DZo", (float)par->dd[2]);
    add_build_option("NX", (int)par->n_space_div[0]);
    add_build_option("NY", (int)par->n_space_div[1]);
    add_build_option("NZ", (int)par->n_space_div[2]);
    add_build_option("NXNY", (int)par->n_space_div[0] * (int)par->n_space_div[1]);
    add_build_option("NXNYNZ", (int)n_cells);
    add_build_option("N0", 2 * (int)par->n_space_div[0]);
    add_build_option("N1", 2 * (int)par->n_space_div[1]);
    add_build_option("N2", 2 * (int)par->n_space_div[2]);
    add_build_option("N0N1", 4 * (int)par->n_space_div[0] * (int)par->n_space_div[1]);
    add_build_option("N0N1N2", (int)n_cells8);
    add_build_option("NC4", (int)n_cells4);
    add_build_option("NPART", (int)n_partd);
}
std::pair<int, int> getFastestDevice()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    int max_performance = 0;
    int fastest_device_num = 0;
    int fastest_platform_num = 0;

    for (int platform_num = 0; platform_num < platforms.size(); ++platform_num)
    {
        std::vector<cl::Device> devices;
        platforms[platform_num].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (int device_num = 0; device_num < devices.size(); ++device_num)
        {
            int frequency = devices[device_num].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
            int compute_units = devices[device_num].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

            bool is_cpu = devices[device_num].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU;
            bool is_gpu = devices[device_num].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU;
            string name = devices[device_num].getInfo<CL_DEVICE_NAME>();
            string vendor = devices[device_num].getInfo<CL_DEVICE_VENDOR>();
            //cout << vendor << endl;
            unsigned int ipc = is_gpu ? 2u : 32u; // IPC (instructions per cycle) is 2 for GPUs and 32 for most modern CPUs
            bool intel_16_cores_per_cu = (name.find("gpu max") != std::string::npos);
            float intel = (float)(vendor.find("Intel") != std::string::npos) * (is_gpu ? (intel_16_cores_per_cu ? 16.0f : 8.0f) : 0.5f); // Intel GPUs have 16 cores/CU (PVC) or 8 cores/CU (integrated/Arc), Intel CPUs (with HT) have 1/2 core/CU
            //cout << "intel" << endl;
            bool amd_128_cores_per_dualcu = (name.find("gfx10") != std::string::npos); // identify RDNA/RDNA2 GPUs where dual CUs are reported
            bool amd_256_cores_per_dualcu = (name.find("gfx11") != std::string::npos); // identify RDNA3 GPUs where dual CUs are reported
            float amd = (float)(vendor.find("Advanced") != std::string::npos | vendor.find("AMD") != std::string::npos ) * (is_gpu ? (amd_256_cores_per_dualcu ? 256.0f : amd_128_cores_per_dualcu ? 128.0f
                                                                                                                                                        : 64.0f)                                               
                                                                                               : 0.5f); // AMD GPUs have 64 cores/CU (GCN, CDNA), 128 cores/dualCU (RDNA, RDNA2) or 256 cores/dualCU (RDNA3), AMD CPUs (with SMT) have 1/2 core/CU
            //cout << "amd" << endl;
            int performance = (frequency * compute_units * ipc * (intel + amd)) / 1000;
            cout << "device " << platform_num << ", " << device_num << " perf=" << performance << " ipc=" << ipc;
            if (intel != 0)
                cout << " intel=" << intel << endl;
            if (amd != 0)
                cout << " amd=" << amd << endl;
            if (performance > max_performance)
            {
                max_performance = performance;
                fastest_device_num = device_num;
                fastest_platform_num = platform_num;
            }
        }
    }

    return {fastest_platform_num, fastest_device_num};
}
void cl_start(fields *fi, particles *pt, par *par)
{
    /*
    int AA[1] = {-1};
#pragma omp target
    AA[0] = omp_is_initial_device();
    if (!AA[0])
        info_file << "Able to use GPU offloading with OMP!\n";
    else
        info_file << "\nNo GPU on OMP\n";
        */
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

        platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
        // platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
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
            // par->cl_align = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
            info_file << "\t\tDevice addr_align: kB " << device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() << std::endl;
            info_file << "\t\tDevice Local Memory: kB " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << std::endl;
            info_file << "\t\tDevice Available: " << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
            ++i;
        }
        info_file << std::endl;
    }
    //cout << "getplatforms\n";
    cl::Platform::get(&platforms);

    // choose the fastest device
    pair<int, int> fastest_device = getFastestDevice();
    platformn = fastest_device.first;
    device_id = fastest_device.second;
    cl::Platform default_platform = platforms[platformn];
    info_file << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    //cout << "getdevice\n";
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device default_device;
    cout << "device_id = " << platformn << ", " << device_id << ", devices.size = " << devices.size() << ", cl_align = " << par->cl_align << endl;
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

    std::string deviceVendor = default_device.getInfo<CL_DEVICE_VENDOR>();
    if (deviceVendor.find("Intel") != std::string::npos)
    {
        cout << "Intel" << endl;
        //        cl_build_options << "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero -cl-single-precision-constant";
    }
    else if (deviceVendor.find("Advanced Micro Devices") != std::string::npos)
    {
        cout << "AMD" << endl;
        //     cl_build_options << "-O3 -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero -cl-single-precision-constant";
    }
    cl_int cl_err = program.build({default_device}, cl_build_options.str().c_str());
    cout << "program.build " << cl_err << endl;
    info_file << "building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
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

    cout << "check for unified memory ";
    cl_bool temp;
    default_device_g.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &temp);
    if (temp == true)
    {
        info_file << "Using unified memory: " << ((temp == CL_TRUE) ? "TRUE" : "FALSE") << " \n";
        cout << ((temp == CL_TRUE) ? "TRUE" : "FALSE") << endl;
    }
    else
        info_file << "No unified memory: " << temp << " \n";
    fastIO = temp;
    //    fastIO = CL_FALSE;

    // cout << "allocating buffers\n";
    //  create buffers on the device
    /** IMPORTANT: do not use CL_MEM_USE_HOST_PTR if on dGPU **/
    /** HOST_PTR is only used so that memory is not copied, but instead shared between CPU and iGPU in RAM**/
    // Note that special alignment has been given to Ea, Ba, y0, z0, x0, x1, y1 in order to actually do this properly
    // Assume buffers A, B, I, J (Ea, Ba, ci, cf) will always be the same. Then we save a bit of time.
    // get whether or not we are on an iGPU/similar, and can use certain memmory optimizations
    // Assume buffers A, B, I, J (Ea, Ba, ci, cf) will always be the same. Then we save a bit of time.
    static cl::Buffer buff_Ea(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cells3x8f, fastIO ? fi->Ea : NULL, &cl_err);
    static cl::Buffer buff_Ba(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cells3x8f, fastIO ? fi->Ba : NULL, &cl_err);
    static cl::Buffer buff_E(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->E : NULL, &cl_err);
    static cl::Buffer buff_B(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->B : NULL, &cl_err);
    static cl::Buffer buff_E0(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->E0 : NULL, &cl_err);
    static cl::Buffer buff_B0(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->B0 : NULL, &cl_err);
    static cl::Buffer buff_Ee(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->Ee : NULL, &cl_err);
    static cl::Buffer buff_Be(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n_cellsf * 3, fastIO ? fi->Be : NULL, &cl_err);
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
    //        pt->pos = reinterpret_cast<float(&)[2][3][2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * par->n_part[0] * 2 * 3 * 2, par->cl_align)));

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
    static cl::Buffer buff_maxval(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, sizeof(float) * n2048, fastIO ? par->maxval_array : NULL); // x0
    static cl_mem maxval_buffer = buff_maxval();
    // clCreateBuffer(context_g(), CL_MEM_READ_WRITE, n2048 * sizeof(float), 0, &res);
    par->maxval_buffer = maxval_buffer;
    // static cl_mem nt_buffer = clCreateBuffer(context_g(), CL_MEM_READ_WRITE, n2048 * sizeof(int), 0, &res);
    static cl::Buffer buff_nt(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, sizeof(int) * n2048, fastIO ? par->nt_array : NULL); // x0
    static cl_mem nt_buffer = buff_nt();
    par->nt_buffer = nt_buffer;
    if (res)
        cout << res << endl;
    fi->buff_Ea = &buff_Ea;
    fi->buff_Ba = &buff_Ba;
    fi->buff_E = &buff_E;
    fi->buff_B = &buff_B;
    fi->buff_E0 = &buff_E0;
    fi->buff_B0 = &buff_B0;
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
    fi->E0_buffer = buff_E0();
    fi->B0_buffer = buff_B0();
    fi->Ee_buffer = buff_Ee();
    fi->Be_buffer = buff_Be();
    fi->npt_buffer = buff_npt();
    fi->jc_buffer = buff_jc();
    fi->V_buffer = buff_V();
    clEnqueueFillBuffer(commandQueue_g(), fi->E_buffer, 0, n_cellsf * 3, 0, 0, 0, 0, 0);
    clEnqueueFillBuffer(commandQueue_g(), fi->B_buffer, 0, n_cellsf * 3, 0, 0, 0, 0, 0);
    clEnqueueFillBuffer(commandQueue_g(), fi->E0_buffer, 0, n_cellsf * 3, 0, 0, 0, 0, 0);
    clEnqueueFillBuffer(commandQueue_g(), fi->B0_buffer, 0, n_cellsf * 3, 0, 0, 0, 0, 0);
}
