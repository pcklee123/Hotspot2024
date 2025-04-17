//#define RamDisk // whether to use RamDisk if no ramdisk files will be in temp directory
#define maxcells 32
#define cldevice 0 // 0 usually means integrated GPU
#define sphere     // do hot spot  problem
#define spherez    // but allow particles to rollover in the z direction
                   // #define octant     // do hot spot problem 1/8 sphere. Magnetic fields do not make sense as will break symmetry
                   // #define cylinder //do hot rod problem
//#define quadrant   // do problem 1/4 sphere or cylinder
#define Weibull
constexpr float weibullb = 2; // b factor for weibull distribn. larger means closer to a shell. ~1 means filled more at the center.
#define Temp_e 1e5            // in Kelvin 1e7 ~1keV
#define Temp_d 1e7            // in Kelvin
constexpr int f1 = 1000;      // make bigger to make smaller time steps // 300 is min for sphere slight increase in KE
constexpr int f2 = f1 * 1.2;
constexpr float incf = 1.2f;        // increment
constexpr float decf = 1.0f / incf; // decrement factor

constexpr int n_space = 64; // should be 2 to power of n for faster FFT e.g. 32,64,128,256 (128 is 2 million cells, ~ 1gB of ram, 256 is not practical for systems with 8GB or less GPU ram) dont go below 16. some code use 16vectors

constexpr size_t n_partd =1 * 1024 * 1024; // n_space * n_space * n_space ; // must be 2 to power of n
constexpr size_t n_parte = n_partd;
constexpr size_t nback = n_partd / 2; // background stationary particles distributed over all cells - improves stability

constexpr float R_s = n_space / 1;                                                            // Low Pass Filter smoothing radius. Not in use
constexpr float r0_f[3] = {(float)n_space / 8.0, (float)n_space / 8.0, (float)n_space / 4.0}; //  radius of sphere or cylinder (electron, ion, z-pinch plasma)

constexpr float Bz0 = 0.00001;     // in T, static constant fields
constexpr float Btheta0 = 0.00001; // in T, static constant fields
constexpr float Ez0 = 1.0e1;       // in V/m
constexpr float vz0 = 2.0e7f;
constexpr float a0 = 1e-6;                          // typical dimensions of a cell in m This needs to be smaller than debye length otherwise energy is not conserved if a particle moves across a cell
constexpr float a0_ff = 1.0 + 1.0 / (float)n_space; // rescale cell size, if particles rollover this cannot encrement more than 1 cell otherwise will have fake "waves"
constexpr float target_part = 1e9;                 // 3.5e22 particles per m^3 per torr of ideal gas. 7e22 electrons for 1 torr of deuterium
constexpr float v0_r = 0;                           // initial directed radial velocity outwards is positive

// The maximum expected E and B fields. If fields go beyond this, the the time step, cell size etc will be wrong. Should adjust and recalculate.
//  maximum expected magnetic field
constexpr float Bmax0 = Bz0 + Btheta0 + 0.0001; // in T earth's magnetic field is of the order of ~ 1e-4 T DPF ~ 100T
constexpr float Emax0 = Ez0 + 1;                // 1e11V/m is approximately interatomic E field -extremely large fields implies poor numerical stability

// technical parameters

// Te 1e7,Td 1e7,B 0.1,E 1e8,nback 64, a0 0.1e-3,part 1e10,nspace 32 npartd *4 sphere, r1=1.8
// a sphere 0.4 mm radius with 1e24*4/3*pi()*0.4^3 *1.6e-19C E on surface =2.4e12Vm-1 if all electrons have left.
// r0=8*a0 Te 1e7,Td 1e7,B 100,E 1e10,nback 64, a0 1e-3,part 1e15,nspace 64 npartd *4 cylinder
constexpr unsigned int ncoeff = 8;

constexpr int n_output_part = (n_partd > 9369) ? 9369 : n_partd; // maximum number of particles to output to file
// const int nprtd=floor(n_partd/n_output_part);

constexpr int ndatapoints = 10; // total number of time steps to print
constexpr int nc1 = 100;          // f1 * 1;      // number of times to calculate E and B between printouts total number of electron time steps calculated = ndatapoints *nc1*md_me
constexpr int md_me = 60;       // ratio of electron speed/deuteron speed at the same KE. Used to calculate electron motion more often than deuteron motion

#define Hist_n 512
// #define Hist_max Temp_e / 11600 * 60 // in eV Kelvin to eV is divide by 11600
#define Hist_max 50000 // 50keV
#define trilinon_

#define Eon_ // whether to calculate the internally generated electric (E) field externally applied fields are always on
// #define Uon_     // whether to calculate the electric (V) potential and potential energy (U). Needs Eon to be enabled.
// #define UE_field // whether to calculate the total energy due to electric energy density
// #define UE_cell // whether to calculate the EPE due to particles within a cell
#define Bon_     // whether to calculate the internally generated magnetic (B) field
#define dE_dton_ // whether to calculate the displacement current only usefull if both Eon_ and Bon_
#define dB_dton_ // whether to calculate the displacement current only usefull if both Eon_ and Bon_
// #define UB_field // whether to calculate the total energy due to magnetic energy density
#define EFon_ // whether to apply electric force
#define BFon_ // whether to apply magnetic force
#define printDensity
#define printParticles
// #define printV // print out V
#define printB // print out B field
#define printE // print out E field
// #define FileIn //whether to load from input file (unused)

constexpr float r_part_spart = target_part / n_partd; // 1e12 / n_partd; // ratio of particles per tracked "super" particle
// ie. the field of N particles will be multiplied by (1e12/N), as if there were 1e12 particles

constexpr int n_space_divx = n_space;
constexpr int n_space_divy = n_space;
constexpr int n_space_divz = n_space;
constexpr int n_space_divx2 = n_space_divx * 2;
constexpr int n_space_divy2 = n_space_divy * 2;
constexpr int n_space_divz2 = n_space_divz * 2;
constexpr size_t n_cells = n_space_divx * n_space_divy * n_space_divz; // number of cells
constexpr size_t n_cells8 = n_cells * 8;                               // number of cells * 8 = cells for FFT to prevent rollover fields
constexpr size_t n_cells_2 = n_cells / 2;                              // number of n_cells8/16 for float16
constexpr size_t n_cellsf = n_cells * sizeof(float);                   // number of cells * sizeof(float) (4bytes)
constexpr size_t n_cellsi = n_cells * sizeof(int);
constexpr size_t n_partf = n_partd * sizeof(float); // number of particles * sizeof(float)
constexpr size_t n_part_2048 = 2048;                // number of particles/2048 for  2048 parallel computations and 2048 times smaller buffer to transfer to CPU
constexpr size_t n2048 = 2048;
constexpr size_t n_cells3x8f = n_cells * 3 * 8 * sizeof(float);
constexpr size_t nc3_16 = n_cells * 3 / 16; // number of cells/16 for 3D float16
constexpr size_t n_cells_16 = n_cells / 16; // number of cells/16 for float16
// constexpr size_t n4 = n_partd * sizeof(float);
constexpr size_t N0 = n_space_divx2, N1 = n_space_divy2, N2 = n_space_divz2,
                 N0N1 = N0 * N1, N0N1_2 = N0N1 / 2, N0N1N2 = N0 * N1 * N2,
                 N0_c = N0 / 2 + 1,
                 N1N0_c = N1 * N0_c,
                 N2_c = N2 / 2 + 1; // Dimension to store the complex data, as required by fftw (from their docs)

constexpr size_t n_cells4 = N2 * N1 * N0_c; // n_cells4 is not actually n_cells8/2
// physical "constants"
constexpr float kb = 1.38064852e-23;       // m^2kss^-2K-1
constexpr float e_charge = 1.60217662e-19; // C
constexpr float ev_to_j = e_charge;
constexpr float e_mass = 9.10938356e-31;
constexpr float e_charge_mass = e_charge / e_mass;
constexpr float kc = 8.9875517923e9;         // kg m3 s-2 C-2
constexpr float epsilon0 = 8.8541878128e-12; // F m-1
constexpr float pi = 3.1415926536;
constexpr float u0 = 4e-7 * pi;

constexpr int ncalc0[2] = {md_me, 1};
constexpr int qs[2] = {-1, 1}; // Sign of charge
constexpr int mp[2] = {1, 1835 * 2};

struct par // useful parameters
{
    float dt[2] = {1e-12, 1e-12 / 60}; // time step electron,deuteron
    float ndeltat = 0;
    float Emax = Emax0;
    float Bmax = Bmax0;
    int nt[2];      // total number of particles
    float KEtot[2]; // Total KE of particles
#if defined(octant)
    float posL[3] = {-a0 / 2, -a0 / 2, -a0 / 2};                                                       // Lowest position of cells (x,y,z)
    float posH[3] = {a0 * (n_space_divx - 1.5), a0 *(n_space_divy - 1.5), a0 *(n_space_divz - 1.5)};   // Highest position of cells (x,y,z)
    float posL_1[3] = {a0 / 2, a0 / 2, a0 / 2};                                                        // Lowest position of cells (x,y,z)
    float posH_1[3] = {a0 * (n_space_divx - 2.5), a0 *(n_space_divy - 2.5), a0 *(n_space_divz - 2.5)}; // Highest position of cells (x,y,z)
    float posL_15[3] = {a0 * 1, a0 * 1, a0 * 1};                                                       // Lowest position of cells (x,y,z)
    float posH_15[3] = {a0 * (n_space_divx - 3), a0 *(n_space_divy - 3), a0 *(n_space_divz - 3)};      // Highes position of cells (x,y,z)
    float posL2[3] = {a0 * 1.5, a0 * 1.5, a0 * 1.5};
#else
#if defined(quadrant)
    float posL[3] = {-a0 / 2, -a0 / 2, -a0 *(n_space_divz - 1.0) / 2.0};                                     // Lowest position of cells (x,y,z)
    float posH[3] = {a0 * (n_space_divx - 1.5), a0 *(n_space_divy - 1.5), a0 *(n_space_divz - 1.0) / 2.0};   // Highest position of cells (x,y,z)
    float posL_1[3] = {a0 / 2, a0 / 2, -a0 *(n_space_divz - 3.0) / 2.0};                                     // Lowest position of cells (x,y,z)
    float posH_1[3] = {a0 * (n_space_divx - 2.5), a0 *(n_space_divy - 2.5), a0 *(n_space_divz - 3.0) / 2.0}; // Highest position of cells (x,y,z)
    float posL_15[3] = {a0 * 1, a0 * 1, -a0 *(n_space_divz - 4.0) / 2.0};                                    // Lowest position of cells (x,y,z)
    float posH_15[3] = {a0 * (n_space_divx - 3), a0 *(n_space_divy - 3), a0 *(n_space_divz - 4.0) / 2.0};    // Highes position of cells (x,y,z)
    float posL2[3] = {a0 * 1.5, a0 * 1.5, -a0 *n_space_divz};
#else
    float posL[3] = {-a0 * (n_space_divx - 1) / 2.0f, -a0 *(n_space_divy - 1.0) / 2.0, -a0 *(n_space_divz - 1.0) / 2.0};    // Lowest position of cells (x,y,z)
    float posH[3] = {a0 * (n_space_divx - 1) / 2.0f, a0 *(n_space_divy - 1.0) / 2.0, a0 *(n_space_divz - 1.0) / 2.0};       // Highes position of cells (x,y,z)
    float posL_1[3] = {-a0 * (n_space_divx - 3) / 2.0f, -a0 *(n_space_divy - 3.0) / 2.0, -a0 *(n_space_divz - 3.0) / 2.0};  // Lowest position of cells (x,y,z)
    float posH_1[3] = {a0 * (n_space_divx - 3) / 2.0f, a0 *(n_space_divy - 3.0) / 2.0, a0 *(n_space_divz - 3.0) / 2.0};     // Highes position of cells (x,y,z)
    float posL_15[3] = {-a0 * (n_space_divx - 4) / 2.0f, -a0 *(n_space_divy - 4.0) / 2.0, -a0 *(n_space_divz - 4.0) / 2.0}; // Lowest position of cells (x,y,z)
    float posH_15[3] = {a0 * (n_space_divx - 4) / 2.0f, a0 *(n_space_divy - 4.0) / 2.0, a0 *(n_space_divz - 4.0) / 2.0};    // Highes position of cells (x,y,z)

    float posL2[3] = {-a0 * n_space_divx, -a0 *n_space_divy, -a0 *n_space_divz};
#endif
#endif
    float dd[3] = {a0, a0, a0}; // cell spacing (x,y,z)

    unsigned int n_space_div[3] = {n_space_divx, n_space_divy, n_space_divz};
    unsigned int n_space_div2[3] = {n_space_divx2, n_space_divy2, n_space_divz2};
    unsigned int n_part[3] = {n_parte, n_partd, n_parte + n_partd};
    float UE = 0;
    float UB = 0;
    // for tnp
    float Ecoef[2] = {0, 0};
    float Bcoef[2] = {0, 0};
    uint32_t ncalcp[2] = {md_me, 1};
    uint32_t nc = nc1;
    uint32_t n_partp[2] = {n_parte, n_partd}; // 0,number of "super" electrons, electron +deuteriom ions, total
    unsigned int cl_align = 64;
    std::string outpath;
    float a0_f = 1.0; // factor to scale cell size
    cl_mem maxval_buffer = 0;
    cl_mem nt_buffer = 0;
    int cdt = 0;
    unsigned int maxcomputeunits[10];
    float *maxval_array;
    int *nt_array;
};

struct particles // particles
{
    float (*pos)[3][2][n_partd]; //[2{previous-0,current-1}][3{x,y,z}][2{e,d}][n_partd]
    float *pos0;
    float *pos1;
    float (*pos0x)[n_partd];
    float (*pos0y)[n_partd];
    float (*pos0z)[n_partd];
    float (*pos1x)[n_partd];
    float (*pos1y)[n_partd];
    float (*pos1z)[n_partd];
    int32_t (*q)[n_partd];
    int32_t (*m)[n_partd];

    cl::Buffer *buff_x0_e;
    cl::Buffer *buff_y0_e;
    cl::Buffer *buff_z0_e;
    cl::Buffer *buff_x1_e;
    cl::Buffer *buff_y1_e;
    cl::Buffer *buff_z1_e;

    cl::Buffer *buff_x0_i;
    cl::Buffer *buff_y0_i;
    cl::Buffer *buff_z0_i;
    cl::Buffer *buff_x1_i;
    cl::Buffer *buff_y1_i;
    cl::Buffer *buff_z1_i;

    cl::Buffer *buff_q_e;
    cl::Buffer *buff_q_i;
};

struct fields                                                      // particles
{                                                                  //[{x,y,z}][k][j][i]
    float (*E)[n_space_divz][n_space_divy][n_space_divx];          // selfgenerated E field[3][k][j][i]
    float (*Ee)[n_space_divz][n_space_divy][n_space_divx];         // External E field[3][k][j][i]
    float (*Ea)[n_space_divz][n_space_divy][n_space_divx][ncoeff]; // coefficients for Trilinear interpolation Electric field  Ea[3][k][j][i][8]
    float (*B)[n_space_divz][n_space_divy][n_space_divx];
    float (*Be)[n_space_divz][n_space_divy][n_space_divx];
    float (*Ba)[n_space_divz][n_space_divy][n_space_divx][ncoeff]; // coefficients for Trilinear interpolation Magnetic field
                                                                   //    float (*V)[n_space_divz][n_space_divy][n_space_divx];
    float (*E0)[n_space_divz][n_space_divy][n_space_divx];
    float (*B0)[n_space_divz][n_space_divy][n_space_divx];

    float (*V)[n_space_divy][n_space_divx];
    float (*np)[n_space_divz][n_space_divy][n_space_divx];
    int32_t (*npi)[n_space_divy][n_space_divx];
    int32_t (*np_centeri)[n_space_divz][n_space_divy][n_space_divx];
    float (*npt)[n_space_divz][n_space_divy][n_space_divx];
    float (*currentj)[3][n_space_divz][n_space_divy][n_space_divx];
    int32_t (*cji)[n_space_divz][n_space_divy][n_space_divx]; //[3][z][y][x]
    int32_t (*cj_centeri)[3][n_space_divz][n_space_divy][n_space_divx];
    float (*jc)[n_space_divz][n_space_divy][n_space_divx];

    float *precalc_r3; //  pre-calculate 1/ r3 to make it faster to calculate electric and magnetic fields
#ifdef Uon_
    float *precalc_r2; // similar arrays for U, but kept separately in one ifdef
#endif
    cl_mem r3_buffer = 0;
    cl_mem r2_buffer = 0;
    cl_mem V_buffer = 0;
    cl_mem E_buffer = 0;
    cl_mem B_buffer = 0;
    cl_mem E0_buffer = 0;
    cl_mem B0_buffer = 0;
    cl_mem Ee_buffer = 0;
    cl_mem Be_buffer = 0;
    cl_mem npt_buffer = 0;
    cl_mem jc_buffer = 0;
    cl_mem fft_p_buffer = 0;
    cl_mem fft_real_buffer = 0;
    cl_mem fft_complex_buffer = 0;

    cl::Buffer *buff_E;
    cl::Buffer *buff_B;
    cl::Buffer *buff_E0;
    cl::Buffer *buff_B0;
    cl::Buffer *buff_Ee;
    cl::Buffer *buff_Be;
    cl::Buffer *buff_Ea;
    cl::Buffer *buff_Ba;

    cl::Buffer *buff_V;

    cl::Buffer *buff_npt;
    cl::Buffer *buff_jc;

    cl::Buffer *buff_np_e;
    cl::Buffer *buff_np_i;
    cl::Buffer *buff_currentj_e;
    cl::Buffer *buff_currentj_i;

    cl::Buffer *buff_npi;
    cl::Buffer *buff_cji;
};
