// Preprocessor things for compilation of tnp assume if XLOWo is defined then
// everything else is.Need these defines here to avoid warnings in the editor
#ifndef XLOWo
#define XLOWo -1.0
#define YLOWo -1.0
#define ZLOWo -1.0
#define XHIGHo 1.0
#define YHIGHo 1.0
#define ZHIGHo 1.0
#define DXo 1.0
#define DYo 1.0
#define DZo 1.0
#define NX 128
#define NY 128
#define NZ 128
#define NXNY 16384
#define NXNYNZ 2097152
#define N0 256
#define N1 256
#define N2 256
#define N0N1 65536
#define N0N1N2 16777216
#define NC4 8454144   // N0*N1*(N2/2+1) = 4 * NX * NY * (NZ + 1)
#define NPART 1048576 // number of particles e.g D or e.
#endif
void kernel vector_cross_mul(global float *A0, global const float *B0,
                             global const float *C0, global float *A1,
                             global const float *B1, global const float *C1,
                             global float *A2, global const float *B2,
                             global const float *C2) {
  int i = get_global_id(0); // Get index of the current element to be processed
  A0[i] = B1[i] * C2[i] - B2[i] * C1[i]; // Do the operation
  A1[i] = B2[i] * C0[i] - B0[i] * C2[i];
  A2[i] = B0[i] * C1[i] - B1[i] * C0[i];
}

void kernel vector_mul(global float *A, global const float *B,
                       global const float *C) {
  int i = get_global_id(0); // Get index of the current element to be processed
  A[i] = B[i] * C[i];       // Do the operation
}

void kernel vector_muls_addv(global float *A, global const float *B,
                             global const float *C) {
  float Bb = B[0];
  int i = get_global_id(0); // Get index of current element processed
  A[i] = Bb * A[i] + C[i];  // Do the operation
}

void kernel vector_add(global float *A, global const float *B,
                       global const float *C) {
  int i = get_global_id(0); // Get index of current element processed
  A[i] = B[i] + C[i];       // Do the operation
}

void kernel vector_muls(global float *A, global const float *B) {
  float Bb = B[0];
  int i = get_global_id(0); // Get index of current element processed
  A[i] = Bb * A[i];         // Do the operation
}

void kernel vector_mul_complex(global float2 *A, global float2 *B,
                               global float2 *C) {
  int i = get_global_id(0); // Get index of the current element to be processed
  float2 b = B[i], c = C[i];
  A[i] = (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
}

void kernel copy3Data(global const float *jc, global float *fft_real) {
  // get global indices
  uint idx = get_global_id(0);
  // Compute 3D index for dest array
  uint i = idx % N0;
  uint j = (idx / N0) % N1;
  uint k = (idx / N0N1) % N2;

  uint in = (i < NX) && (j < NY) && (k < NZ); // Check if in range of source

  uint s_idx = (in) ? k * NY * NX + j * NX + i
                    : 0; // Compute global index for source array
  //  Copy element from source to destination array or with zeroes do for each
  //  component
  fft_real[idx] = (in) ? jc[s_idx] : 0;
  fft_real[idx + N0N1N2] = (in) ? jc[s_idx + NXNYNZ] : 0;
  fft_real[idx + N0N1N2 * 2] = (in) ? jc[s_idx + NXNYNZ * 2] : 0;
}

void kernel copyData(global const float *npt, global float *fft_real) {
  // get global indices
  uint idx = get_global_id(0);
  // Compute 3D index for dest array
  uint i = idx % N0;
  uint j = (idx / N0) % N1;
  uint k = (idx / N0N1) % N2;

  // Check if in range of source
  uint in = (i < NX) && (j < NY) && (k < NZ);

  // Compute global index for source array
  uint source_index = (in) ? k * NY * NX + j * NX + i : 0;
  //  Copy element from source to destination array or with zeroes
  fft_real[idx] = (in) ? npt[source_index] : 0;
}
// This does scalar complex multiply the fft of the r_vector/r^3 with the fft of
// the "density" since density is a scalar field it is contained in 1st
// component of fft_complex the final result is a vector field with 3 components
// in the array each with NC4 elements
void kernel NxPrecalc(global const float2 *r3, global float2 *fft_complex) {
  uint i = get_global_id(0), j = i + NC4, k = j + NC4;
  float2 b = fft_complex[i], c = r3[k];
  fft_complex[k] =
      (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
  c = r3[j];
  fft_complex[j] =
      (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
  c = r3[i];
  fft_complex[i] =
      (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
}
// This does complex vector cross multiply the fft of the r_vector/r^3 with the
// fft of the "current density" since current density is a vector field, each
// component is contained  fft_complex x has NC4 elements, y has .... the final
// result is a vector field with 3 components in the array each with NC4
// elements
void kernel jcxPrecalc(global const float2 *r3, global float2 *jc) {
  float2 t1, t2, t3;
  uint x = get_global_id(0), y = x + NC4, z = y + NC4, x1 = z + NC4,
       y1 = x1 + NC4, z1 = y1 + NC4;
  t1 = (float2)(jc[y].s0 * r3[z1].s0 - jc[y].s1 * r3[z1].s1,
                jc[y].s0 * r3[z1].s1 + jc[y].s1 * r3[z1].s0) -
       (float2)(jc[z].s0 * r3[y1].s0 - jc[z].s1 * r3[y1].s1,
                jc[z].s0 * r3[y1].s1 + jc[z].s1 * r3[y1].s0);
  t2 = (float2)(jc[z].s0 * r3[x1].s0 - jc[z].s1 * r3[x1].s1,
                jc[z].s0 * r3[x1].s1 + jc[z].s1 * r3[x1].s0) -
       (float2)(jc[x].s0 * r3[z1].s0 - jc[x].s1 * r3[z1].s1,
                jc[x].s0 * r3[z1].s1 + jc[x].s1 * r3[z1].s0);
  t3 = (float2)(jc[x].s0 * r3[y1].s0 - jc[x].s1 * r3[y1].s1,
                jc[x].s0 * r3[y1].s1 + jc[x].s1 * r3[y1].s0) -
       (float2)(jc[y].s0 * r3[x1].s0 - jc[y].s1 * r3[x1].s1,
                jc[y].s0 * r3[x1].s1 + jc[y].s1 * r3[x1].s0);
  jc[x] = (float2)t1;
  jc[y] = (float2)t2;
  jc[z] = (float2)t3;
}

void kernel NxPrecalcr2(global const float2 *r2, global const float2 *r3,
                        global float2 *fft_complex) {

  uint i = get_global_id(0), j = i + NC4, k = j + NC4;
  float2 b = fft_complex[i], c = r2[i];
  // V is at fft_complex[3] fourth place
  fft_complex[NC4 + k] =
      (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
  c = r3[k];
  // fft of E is at fft_complex[0]-[2]
  fft_complex[k] =
      (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
  c = r3[j];
  fft_complex[j] =
      (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
  c = r3[i];
  fft_complex[i] =
      (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
}

void kernel sumFftFieldo(global const float *fft_real, global const float *Fe,
                         global float *F) {

  const float s000[3] = {+1, +1, +1}; // c=0 is x,c=1 is y,c=2 is z
  const float s001[3] = {-1, +1, +1};
  const float s010[3] = {+1, -1, +1};
  const float s011[3] = {-1, -1, +1};
  const float s100[3] = {+1, +1, -1};
  const float s101[3] = {-1, +1, -1};
  const float s110[3] = {+1, -1, -1};
  const float s111[3] = {-1, -1, -1};

  // get global indices
  uint idx = get_global_id(0);
  // Compute 3D index for dest array
  uint i = idx % NX;
  uint j = (idx / NX) % NY;
  uint k = (idx / NXNY) % NZ;

  uint cdx = 0, cdx8 = 0;

  int idx000 = k * N0N1 + j * N0 + i; // idx_kji
  int idx001 = k * N0N1 + j * N0;
  int idx010 = k * N0N1 + i;
  int idx011 = k * N0N1;
  int idx100 = j * N0 + i;
  int idx101 = j * N0;
  int idx110 = i;
  int idx111 = 0;

  int odx000 = 0;                          // odx_kji
  int odx001 = i == 0 ? 0 : N0 - i;        // iskip
  int odx010 = j == 0 ? 0 : N0 * (N1 - j); // jskip
  int odx011 = odx001 + odx010;
  int odx100 = k == 0 ? 0 : N0 * N1 * (N2 - k); // kskip
  int odx101 = odx100 + odx001;
  int odx110 = odx100 + odx010;
  int odx111 = odx100 + odx011;
  for (int c = 0; c < 3; ++c, cdx += NXNYNZ, cdx8 += N0N1N2) {
    F[cdx + idx] = Fe[cdx + idx];
    F[cdx + idx] += s000[c] * fft_real[cdx8 + odx000 + idx000]; // main octant
    // add minor effects from other octants
    F[cdx + idx] += s001[c] * fft_real[cdx8 + odx001 + idx001];
    F[cdx + idx] += s010[c] * fft_real[cdx8 + odx010 + idx010];
    F[cdx + idx] += s011[c] * fft_real[cdx8 + odx011 + idx011];
    F[cdx + idx] += s100[c] * fft_real[cdx8 + odx100 + idx100];
    F[cdx + idx] += s101[c] * fft_real[cdx8 + odx101 + idx101];
    F[cdx + idx] += s110[c] * fft_real[cdx8 + odx110 + idx110];
    F[cdx + idx] += s111[c] * fft_real[cdx8 + odx111 + idx111];
  }
}

void kernel sumFftFieldq(global const float *fft_real, global const float *Fe,
                         global float *F) {
  const float s000[3] = {+1, +1, +1}; // c=0 is x,c=1 is y,c=2 is z
  const float s001[3] = {-1, +1, +1};
  const float s010[3] = {+1, -1, +1};
  const float s011[3] = {-1, -1, +1};

  uint idx = get_global_id(0); // get global indices
  // Compute 3D index for dest array
  uint i = idx % NX;
  uint j = (idx / NX) % NY;
  uint k = (idx / NXNY) % NZ;

  uint cdx = 0, cdx8 = 0;

  int idx000 = k * N0N1 + j * N0 + i; // idx_kji
  int idx001 = k * N0N1 + j * N0;
  int idx010 = k * N0N1 + i;
  int idx011 = k * N0N1;

  int odx000 = 0;                          // odx_kji
  int odx001 = i == 0 ? 0 : N0 - i;        // iskip
  int odx010 = j == 0 ? 0 : N0 * (N1 - j); // jskip
  int odx011 = odx001 + odx010;

  for (int c = 0; c < 3; ++c, cdx += NXNYNZ, cdx8 += N0N1N2) {
    F[cdx + idx] = Fe[cdx + idx];
    F[cdx + idx] += s000[c] * fft_real[cdx8 + odx000 + idx000]; // main octant
    // add minor effects from other octants
    F[cdx + idx] += s001[c] * fft_real[cdx8 + odx001 + idx001];
    F[cdx + idx] += s010[c] * fft_real[cdx8 + odx010 + idx010];
    F[cdx + idx] += s011[c] * fft_real[cdx8 + odx011 + idx011];
  }
}

void kernel sumFftFieldBq(global const float *fft_real, global const float *Fe,
                          global float *F) {
  const float s000[3] = {+1, +1, +1}; // c=0 is x,c=1 is y,c=2 is z
  const float s001[3] = {+1, -1, +1}; // {-1, +1, +1};
  const float s010[3] = {-1, +1, +1}; // {+1, -1, +1};
  const float s011[3] = {+1, +1, +1}; // {-1, -1, +1};

  uint idx = get_global_id(0); // get global indices
  // Compute 3D index for dest array
  uint i = idx % NX;
  uint j = (idx / NX) % NY;
  uint k = (idx / NXNY) % NZ;

  uint cdx = 0, cdx8 = 0;

  int idx000 = k * N0N1 + j * N0 + i; // idx_kji
  int idx001 = k * N0N1 + j * N0;
  int idx010 = k * N0N1 + i;
  int idx011 = k * N0N1;

  int odx000 = 0;                          // odx_kji
  int odx001 = i == 0 ? 0 : N0 - i;        // iskip
  int odx010 = j == 0 ? 0 : N0 * (N1 - j); // jskip
  int odx011 = odx001 + odx010;

  for (int c = 0; c < 3; ++c, cdx += NXNYNZ, cdx8 += N0N1N2) {
    F[cdx + idx] = Fe[cdx + idx];
    F[cdx + idx] += s000[c] * fft_real[cdx8 + odx000 + idx000]; // main octant
    // add minor effects from other octants
    F[cdx + idx] += s001[c] * fft_real[cdx8 + odx001 + idx001];
    F[cdx + idx] += s010[c] * fft_real[cdx8 + odx010 + idx010];
    F[cdx + idx] += s011[c] * fft_real[cdx8 + odx011 + idx011];
  }
}

void kernel sumFftField(global const float *fft_real, global const float *Fe,
                        global float *F) {
  // get global indices
  uint idx = get_global_id(0);
  // Compute 3D index for dest array
  uint i = idx % NX;
  uint j = (idx / NX) % NY;
  uint k = (idx / NXNY) % NZ;
  uint cdx = 0, cdx8 = 0;
  int idx000 = k * N0N1 + j * N0 + i; // idx_kji
  for (int c = 0; c < 3; ++c, cdx += NXNYNZ, cdx8 += N0N1N2) {
    F[cdx + idx] = Fe[cdx + idx] + fft_real[cdx8 + idx000];
  }
}

void kernel sumFftSField(global const float *fft_real, global float *V) {
  // get global indices
  uint idx = get_global_id(0);
  // Compute 3D index for dest array
  uint i = idx % NX;
  uint j = (idx / NX) % NY;
  uint k = (idx / NXNY) % NZ;

  int idx000 = k * N0N1 + j * N0 + i; // idx_kji
  V[idx] = fft_real[idx000];          // V[idx] = 5.0;
}

void kernel tnp_k_implicit(global const float8 *a1,
                           global const float8 *a2, // E, B coeff
                           global float *x0, global float *y0,
                           global float *z0, // prev pos
                           global float *x1, global float *y1,
                           global float *z1, // current pos
                           float Bcoef,
                           float Ecoef, // Bcoeff, Ecoeff
                           float a0_f, const unsigned int n,
                           const uint ncalc, //
                           global int *q) {

  uint id = get_global_id(0);
  uint prev_idx = UINT_MAX;
  float xprev = x0[id], yprev = y0[id], zprev = z0[id], x = x1[id], y = y1[id],
        z = z1[id];
  float8 temp, pos;
  float r1 = 1.0f;
  float r2 = r1 * r1;
  float8 store0, store1, store2, store3, store4, store5;
  const float Bcoeff = Bcoef / r1;
  const float Ecoeff = Ecoef / r1;
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;
  const float XL = (XLOW + 1.5f * DX), YL = (YLOW + 1.5f * DY),
              ZL = (ZLOW + 1.5f * DZ);
  const float XH = (XHIGH - 1.5f * DX), YH = (YHIGH - 1.5f * DY),
              ZH = (ZHIGH - 1.5f * DZ);

  const float8 ones = (float8)(1, 1, 1, 1, 1, 1, 1, 1);
  for (uint t = 0; t < ncalc; t++) {
    float xy = x * y, xz = x * z, yz = y * z, xyz = x * yz;
    uint idx =
        ((uint)((z - ZLOW) / DZ) * NZ + (uint)((y - YLOW) / DY)) * NY +
        (uint)((x - XLOW) / DX); // round down the cells - this is intentional
    idx *= 3; // find out the index to which cell the particle is in.
    pos = (float8)(1.f, x, y, z, xy, xz, yz, xyz);
    // Is there no better way to do this? Why does float8 not have dot()?
    if (prev_idx != idx) {
      store0 = a1[idx]; // Ex
      store1 = a1[idx + 1];
      store2 = a1[idx + 2];
      store3 = a2[idx]; // Bx
      store4 = a2[idx + 1];
      store5 = a2[idx + 2];
      prev_idx = idx;
    }
    temp = store0 * pos;
    // get interpolated Electric field at particle position
    float xE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store1 * pos;
    float yE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store2 * pos;
    float zE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    // get interpolated Magnetic field at particle position
    temp = store3 * pos;
    float xP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store4 * pos;
    float yP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store5 * pos;
    float zP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;

    xP *= Bcoeff;
    yP *= Bcoeff;
    zP *= Bcoeff;
    xE *= Ecoeff;
    yE *= Ecoeff;
    zE *= Ecoeff;

    float xyP = xP * yP, yzP = yP * zP, xzP = xP * zP;
    float xxP = xP * xP, yyP = yP * yP, zzP = zP * zP;
    // float b_det = 1.f / (1.f + xxP + yyP + zzP);
    float b_det = r2 / (r2 + xxP + yyP + zzP);

    float vx = (x - xprev); // / dt -> cancels out in the end
    float vy = (y - yprev);
    float vz = (z - zprev);

    xprev = x;
    yprev = y;
    zprev = z;

    float vxxe = vx + xE, vyye = vy + yE, vzze = vz + zE;

    x += fma(b_det,
             fma(-vx, yyP + zzP,
                 fma(vyye, zP + xyP, fma(vzze, xzP - yP, fma(xxP, xE, xE)))),
             vx);
    y += fma(b_det,
             fma(vxxe, xyP - zP,
                 fma(-vy, xxP + zzP, fma(vzze, xP + yzP, fma(yyP, yE, yE)))),
             vy);
    z += fma(b_det,
             fma(vxxe, yP + xzP,
                 fma(vyye, yzP - xP, fma(-vz, xxP + yyP, fma(zzP, zE, zE)))),
             vz);
  }

  xprev = x > XL ? xprev : XL;
  xprev = x < XH ? xprev : XH;
  yprev = y > YL ? yprev : YL;
  yprev = y < YH ? yprev : YH;
  zprev = z > ZL ? zprev : ZL;
  zprev = z < ZH ? zprev : ZH;
  q[id] = (x > XL & x<XH & y> YL & y<YH & z> ZL & z < ZH) ? q[id] : 0;
  x = x > XL ? x : XL;
  x = x < XH ? x : XH;
  y = y > YL ? y : YL;
  y = y < YH ? y : YH;
  z = z > ZL ? z : ZL;
  z = z < ZH ? z : ZH;

  x0[id] = xprev;
  y0[id] = yprev;
  z0[id] = zprev;
  x1[id] = x;
  y1[id] = y;
  z1[id] = z;
}

void kernel tnp_k_implicitz(global const float8 *a1,
                            global const float8 *a2, // E, B coeff
                            global float *x0, global float *y0,
                            global float *z0, // prev pos
                            global float *x1, global float *y1,
                            global float *z1, // current pos
                            float Bcoef,
                            float Ecoef, // Bcoeff, Ecoeff
                            float a0_f, const unsigned int n,
                            const uint ncalc, //
                            global int *q) {

  uint id = get_global_id(0);
  uint prev_idx = UINT_MAX;
  float xprev = x0[id], yprev = y0[id], zprev = z0[id], x = x1[id], y = y1[id],
        z = z1[id];
  float8 temp, pos;
  float r1 = 1.0f;
  float r2 = r1 * r1;
  float8 store0, store1, store2, store3, store4, store5;
  const float Bcoeff = Bcoef / r1;
  const float Ecoeff = Ecoef / r1;
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;
  const float XL = (XLOW + 1.5f * DX), YL = (YLOW + 1.5f * DY),
              ZL = (ZLOW + 1.5f * DZ);
  const float XH = (XHIGH - 1.5f * DX), YH = (YHIGH - 1.5f * DY),
              ZH = (ZHIGH - 1.5f * DZ);
  const float ZDZ = ZH - ZL - DZ / 10;
  const float8 ones = (float8)(1, 1, 1, 1, 1, 1, 1, 1);
  for (uint t = 0; t < ncalc; t++) {
    float xy = x * y, xz = x * z, yz = y * z, xyz = x * yz;
    uint idx =
        ((uint)((z - ZLOW) / DZ) * NZ + (uint)((y - YLOW) / DY)) * NY +
        (uint)((x - XLOW) / DX); // round down the cells - this is intentional
    idx *= 3;
    pos = (float8)(1.f, x, y, z, xy, xz, yz, xyz);
    // Is there no better way to do this? Why does float8 not have dot()?
    if (prev_idx != idx) {
      store0 = a1[idx]; // Ex
      store1 = a1[idx + 1];
      store2 = a1[idx + 2];
      store3 = a2[idx]; // Bx
      store4 = a2[idx + 1];
      store5 = a2[idx + 2];
      prev_idx = idx;
    }
    temp = store0 * pos;
    float xE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store1 * pos;
    float yE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store2 * pos;
    float zE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store3 * pos;
    float xP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store4 * pos;
    float yP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store5 * pos;
    float zP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;

    xP *= Bcoeff;
    yP *= Bcoeff;
    zP *= Bcoeff;
    xE *= Ecoeff;
    yE *= Ecoeff;
    zE *= Ecoeff;

    float xyP = xP * yP, yzP = yP * zP, xzP = xP * zP;
    float xxP = xP * xP, yyP = yP * yP, zzP = zP * zP;
    // float b_det = 1.f / (1.f + xxP + yyP + zzP);
    float b_det = r2 / (r2 + xxP + yyP + zzP);

    float vx = (x - xprev); // / dt -> cancels out in the end
    float vy = (y - yprev);
    float vz = (z - zprev);

    xprev = x;
    yprev = y;
    zprev = z;

    float vxxe = vx + xE, vyye = vy + yE, vzze = vz + zE;

    x += fma(b_det,
             fma(-vx, yyP + zzP,
                 fma(vyye, zP + xyP, fma(vzze, xzP - yP, fma(xxP, xE, xE)))),
             vx);
    y += fma(b_det,
             fma(vxxe, xyP - zP,
                 fma(-vy, xxP + zzP, fma(vzze, xP + yzP, fma(yyP, yE, yE)))),
             vy);
    z += fma(b_det,
             fma(vxxe, yP + xzP,
                 fma(vyye, yzP - xP, fma(-vz, xxP + yyP, fma(zzP, zE, zE)))),
             vz);
  }

  xprev = x > XL ? xprev : XL;
  xprev = x < XH ? xprev : XH;
  yprev = y > YL ? yprev : YL;
  yprev = y < YH ? yprev : YH;
  zprev = z > ZL ? zprev : zprev + ZDZ;
  zprev = z < ZH ? zprev : zprev - ZDZ;
  q[id] = (x > XL & x<XH & y> YL & y < YH) ? q[id] : 0;
  x = x > XL ? x : XL;
  x = x < XH ? x : XH;
  y = y > YL ? y : YL;
  y = y < YH ? y : YH;
  z = z > ZL ? z : z + ZDZ;
  z = z < ZH ? z : z - ZDZ;

  x0[id] = xprev;
  y0[id] = yprev;
  z0[id] = zprev;
  x1[id] = x;
  y1[id] = y;
  z1[id] = z;
}

void kernel tnp_k_implicito(global const float8 *a1,
                            global const float8 *a2, // E, B coeff
                            global float *x0, global float *y0,
                            global float *z0, // prev pos
                            global float *x1, global float *y1,
                            global float *z1, // current pos
                            float Bcoef,
                            float Ecoef, // Bcoeff, Ecoeff
                            float a0_f, const unsigned int n,
                            const uint ncalc, // n, ncalc
                            global int *q) {

  uint id = get_global_id(0);
  uint prev_idx = UINT_MAX;
  float xprev = x0[id], yprev = y0[id], zprev = z0[id], x = x1[id], y = y1[id],
        z = z1[id];
  float8 temp, pos;
  float r1 = 1.0f;
  float r2 = r1 * r1;
  float8 store0, store1, store2, store3, store4, store5;
  const float Bcoeff = Bcoef / r1;
  const float Ecoeff = Ecoef / r1;
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;
  const float XL = (XLOW + 0.5f * DX), YL = (YLOW + 0.5f * DX),
              ZL = (ZLOW + 0.5f * DX);
  const float XH = (XHIGH - 1.5f * DX), YH = (YHIGH - 1.5f * DY),
              ZH = (ZHIGH - 1.5f * DZ);
  // const float ZDZ = ZH - ZL - DZ / 10;
  const float8 ones = (float8)(1, 1, 1, 1, 1, 1, 1, 1);
  for (uint t = 0; t < ncalc; t++) {
    float xy = x * y, xz = x * z, yz = y * z, xyz = x * yz;
    uint idx =
        ((uint)((z - ZLOW) / DZ) * NZ + (uint)((y - YLOW) / DY)) * NY +
        (uint)((x - XLOW) / DX); // round down the cells - this is intentional
    idx *= 3;
    pos = (float8)(1.f, x, y, z, xy, xz, yz, xyz);
    // Is there no better way to do this? Why does float8 not have dot()?
    if (prev_idx != idx) {
      store0 = a1[idx]; // Ex
      store1 = a1[idx + 1];
      store2 = a1[idx + 2];
      store3 = a2[idx]; // Bx
      store4 = a2[idx + 1];
      store5 = a2[idx + 2];
      prev_idx = idx;
    }
    temp = store0 * pos;
    float xE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store1 * pos;
    float yE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store2 * pos;
    float zE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store3 * pos;
    float xP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store4 * pos;
    float yP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store5 * pos;
    float zP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;

    xP *= Bcoeff;
    yP *= Bcoeff;
    zP *= Bcoeff;
    xE *= Ecoeff;
    yE *= Ecoeff;
    zE *= Ecoeff;

    float xyP = xP * yP, yzP = yP * zP, xzP = xP * zP;
    float xxP = xP * xP, yyP = yP * yP, zzP = zP * zP;
    // float b_det = 1.f / (1.f + xxP + yyP + zzP);
    float b_det = r2 / (r2 + xxP + yyP + zzP);

    float vx = (x - xprev); // / dt -> cancels out in the end
    float vy = (y - yprev);
    float vz = (z - zprev);

    xprev = x;
    yprev = y;
    zprev = z;

    float vxxe = vx + xE, vyye = vy + yE, vzze = vz + zE;

    x += fma(b_det,
             fma(-vx, yyP + zzP,
                 fma(vyye, zP + xyP, fma(vzze, xzP - yP, fma(xxP, xE, xE)))),
             vx);
    y += fma(b_det,
             fma(vxxe, xyP - zP,
                 fma(-vy, xxP + zzP, fma(vzze, xP + yzP, fma(yyP, yE, yE)))),
             vy);
    z += fma(b_det,
             fma(vxxe, yP + xzP,
                 fma(vyye, yzP - xP, fma(-vz, xxP + yyP, fma(zzP, zE, zE)))),
             vz);
  }

  xprev = x > XL ? xprev : -xprev;
  xprev = x < XH ? xprev : XH;
  yprev = y > YL ? yprev : -yprev;
  yprev = y < YH ? yprev : YH;
  zprev = z > ZL ? zprev : -zprev;
  zprev = z < ZH ? zprev : ZH;
  q[id] = (x < XH & y < YH & z < ZH) ? q[id] : 0;
  x = x > XL ? x : -x;
  x = x < XH ? x : XH;
  y = y > YL ? y : -y;
  y = y < YH ? y : YH;
  z = z > ZL ? z : -z;
  z = z < ZH ? z : ZH;

  x0[id] = xprev;
  y0[id] = yprev;
  z0[id] = zprev;
  x1[id] = x;
  y1[id] = y;
  z1[id] = z;
}

void kernel tnp_k_implicitq(global const float8 *a1,
                            global const float8 *a2, // E, B coeff
                            global float *x0, global float *y0,
                            global float *z0, // prev pos
                            global float *x1, global float *y1,
                            global float *z1, // current pos
                            float Bcoef,
                            float Ecoef, // Bcoeff, Ecoeff
                            float a0_f, const unsigned int n,
                            const uint ncalc, //
                            global int *q) {

  uint id = get_global_id(0);
  uint prev_idx = UINT_MAX;
  float xprev = x0[id], yprev = y0[id], zprev = z0[id], x = x1[id], y = y1[id],
        z = z1[id];
  float8 temp, pos;
  float r1 = 1.0f;
  float r2 = r1 * r1;
  float8 store0, store1, store2, store3, store4, store5;
  const float Bcoeff = Bcoef / r1;
  const float Ecoeff = Ecoef / r1;

  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;
  const float XL = (XLOW + 0.5f * DX), YL = (YLOW + 0.5f * DX),
              ZL = (ZLOW + 0.5f * DX);
  const float XH = (XHIGH - 1.5f * DX), YH = (YHIGH - 1.5f * DY),
              ZH = (ZHIGH - 1.5f * DZ);
  const float ZDZ = ZH - ZL - DZ / 10;
  const float8 ones = (float8)(1, 1, 1, 1, 1, 1, 1, 1);
  for (uint t = 0; t < ncalc; t++) {
    float xy = x * y, xz = x * z, yz = y * z, xyz = x * yz;
    uint idx =
        ((uint)((z - ZLOW) / DZ) * NZ + (uint)((y - YLOW) / DY)) * NY +
        (uint)((x - XLOW) / DX); // round down the cells - this is intentional
    idx *= 3;
    pos = (float8)(1.f, x, y, z, xy, xz, yz, xyz);
    // Is there no better way to do this? Why does float8 not have dot()?
    if (prev_idx != idx) {
      store0 = a1[idx]; // Ex
      store1 = a1[idx + 1];
      store2 = a1[idx + 2];
      store3 = a2[idx]; // Bx
      store4 = a2[idx + 1];
      store5 = a2[idx + 2];
      prev_idx = idx;
    }
    temp = store0 * pos;
    float xE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store1 * pos;
    float yE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store2 * pos;
    float zE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store3 * pos;
    float xP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store4 * pos;
    float yP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store5 * pos;
    float zP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;

    xP *= Bcoeff;
    yP *= Bcoeff;
    zP *= Bcoeff;
    xE *= Ecoeff;
    yE *= Ecoeff;
    zE *= Ecoeff;

    float xyP = xP * yP, yzP = yP * zP, xzP = xP * zP;
    float xxP = xP * xP, yyP = yP * yP, zzP = zP * zP;
    // float b_det = 1.f / (1.f + xxP + yyP + zzP);
    float b_det = r2 / (r2 + xxP + yyP + zzP);

    float vx = (x - xprev); // / dt -> cancels out in the end
    float vy = (y - yprev);
    float vz = (z - zprev);

    xprev = x;
    yprev = y;
    zprev = z;

    float vxxe = vx + xE, vyye = vy + yE, vzze = vz + zE;

    x += fma(b_det,
             fma(-vx, yyP + zzP,
                 fma(vyye, zP + xyP, fma(vzze, xzP - yP, fma(xxP, xE, xE)))),
             vx);
    y += fma(b_det,
             fma(vxxe, xyP - zP,
                 fma(-vy, xxP + zzP, fma(vzze, xP + yzP, fma(yyP, yE, yE)))),
             vy);
    z += fma(b_det,
             fma(vxxe, yP + xzP,
                 fma(vyye, yzP - xP, fma(-vz, xxP + yyP, fma(zzP, zE, zE)))),
             vz);
  }
  float xt[4] = {xprev, yprev, -yprev, -xprev};
  float yt[4] = {yprev, -xprev, xprev, -yprev};
  float xt1[4] = {x, y, -y, -x};
  float yt1[4] = {y, -x, x, -y};
  int idx = (x < XL) + ((y < YL) << 1);
  xprev = xt[idx];
  yprev = yt[idx];
  zprev = z > ZL ? zprev : zprev + ZDZ;
  zprev = z < ZH ? zprev : zprev - ZDZ;
  q[id] = ((x < XH) & (y < YH)) ? q[id] : 0;
  x = xt1[idx];
  y = yt1[idx];
  z = z > ZL ? z : z + ZDZ;
  z = z < ZH ? z : z - ZDZ;

  x0[id] = xprev;
  y0[id] = yprev;
  z0[id] = zprev;
  x1[id] = x;
  y1[id] = y;
  z1[id] = z;
}

void kernel densitybylayer(global const float *x0, global const float *y0,
                           global const float *z0, // prev pos
                           global const float *x1, global const float *y1,
                           global const float *z1, // current pos
                           global int *npi, global int *cji,
                           global const int *q, const float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  int8 f;                         // = (1, 0, 0, 0, 0, 0, 0, 0);
  uint size = get_global_size(0); // set this in the main code to be NZ\3 + 1
  uint id = get_global_id(0);
  // number of iterations ensure that this is an integer from main code
  // int num = NPART / size;

  for (int n = 0; n < 3; ++n) {
    int nn = id * 3 + n;
    if (nn < NZ) {
      for (int ni = 0; ni < NPART; ++ni) {
        float z = z1[ni];
        float fk = (z - ZLOW) * invDZ;
        float frk = round(fk);
        uint k = (uint)frk;
        if (k == nn) {
          float xprev = x0[ni], yprev = y0[ni], zprev = z0[ni], x = x1[ni],
                y = y1[ni];
          // float fk = (z - ZLOW) * invDZ;
          float fj = (y - YLOW) * invDY;
          float fi = (x - XLOW) * invDX;
          // float frk = round(fk);
          float frj = round(fj);
          float fri = round(fi);
          // uint k = (uint)frk;
          uint j = (uint)frj;
          uint i = (uint)fri;
          int ofx = (fi - fri) * 256.0f;
          int ofy = (fj - frj) * 256.0f;
          int ofz = (fk - frk) * 256.0f;
          // oct 000,001,010,011,100,101,110,111
          int odx000 = 0;
          int odx001 = ofx > 0 ? 1 : -1;
          int odx010 = ofy > 0 ? NX : -NX;
          int odx011 = odx001 + odx010;
          int odx100 = ofz > 0 ? NXNY : -NXNY;
          int odx101 = odx100 + odx001;
          int odx110 = odx100 + odx010;
          int odx111 = odx100 + odx011;

          int fx0 = abs(ofx);
          int fy0 = abs(ofy);
          int fz0 = abs(ofz);
          int fx1 = 128 - fx0;
          int fy1 = 128 - fy0;
          int fz1 = 128 - fz0;
          uint idx00 = k * NXNY + j * NX + i;
          uint idx01 = idx00 + NXNYNZ;
          uint idx02 = idx01 + NXNYNZ;
          // arithmetic shift right by 14 equivalent to division by 16384
          f.s0 = ((fz1 * fy1 * fx1) >> 14), f.s1 = ((fz1 * fy1 * fx0) >> 14),
          f.s2 = ((fz1 * fy0 * fx1) >> 14), f.s3 = ((fz1 * fy0 * fx0) >> 14),
          f.s3 = ((fz0 * fy1 * fx1) >> 14), f.s5 = ((fz0 * fy1 * fx0) >> 14),
          f.s6 = ((fz0 * fy0 * fx1) >> 14), f.s7 = ((fz0 * fy0 * fx0) >> 14);
          f = q[id] * f;

          // current x,y,z-component
          int8 vxi = (int)((x - xprev) * 65536.0f * invDX) * f;
          int8 vyi = (int)((y - yprev) * 65536.0f * invDY) * f;
          int8 vzi = (int)((z - zprev) * 65536.0f * invDZ) * f;
          // np density
          npi[idx00] += f.s0;
          npi[idx00 + odx001] += f.s1;
          npi[idx00 + odx010] += f.s2;
          npi[idx00 + odx011] += f.s3;
          npi[idx00 + odx100] += f.s4;
          npi[idx00 + odx101] += f.s5;
          npi[idx00 + odx110] += f.s6;
          npi[idx00 + odx111] += f.s7;
          cji[idx00] += vxi.s0;
          cji[idx00 + odx001] += vxi.s1;
          cji[idx00 + odx010] += vxi.s2;
          cji[idx00 + odx011] += vxi.s3;
          cji[idx00 + odx100] += vxi.s4;
          cji[idx00 + odx101] += vxi.s5;
          cji[idx00 + odx110] += vxi.s6;
          cji[idx00 + odx111] += vxi.s7;
          cji[idx01] += vyi.s0;
          cji[idx01 + odx001] += vyi.s1;
          cji[idx01 + odx010] += vyi.s2;
          cji[idx01 + odx011] += vyi.s3;
          cji[idx01 + odx100] += vyi.s4;
          cji[idx01 + odx101] += vyi.s5;
          cji[idx01 + odx110] += vyi.s6;
          cji[idx01 + odx111] += vyi.s7;
          cji[idx02] += vzi.s0;
          cji[idx02 + odx001] += vzi.s1;
          cji[idx02 + odx010] += vzi.s2;
          cji[idx02 + odx011] += vzi.s3;
          cji[idx02 + odx100] += vzi.s4;
          cji[idx02 + odx101] += vzi.s5;
          cji[idx02 + odx110] += vzi.s6;
          cji[idx02 + odx111] += vzi.s7;
        }
      }
    }
  }
}

void kernel densitynoatomic(global const float *x0, global const float *y0,
                            global const float *z0, // prev pos
                            global const float *x1, global const float *y1,
                            global const float *z1, // current pos
                            global int *npi, global int *cji,
                            global const int *q, const float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  int8 f; // = (1, 0, 0, 0, 0, 0, 0, 0);
  uint size = get_global_size(0);
  uint id = get_global_id(0);
  // number of iterations ensure that this is an integer from main code
  int num = NPART / size;
  // for (int n = 0; n < num; ++n) {
  int nn = id;
  float xprev = x0[nn], yprev = y0[nn], zprev = z0[nn], x = x1[nn], y = y1[nn],
        z = z1[nn];
  float fk = (z - ZLOW) * invDZ;
  float fj = (y - YLOW) * invDY;
  float fi = (x - XLOW) * invDX;
  float frk = round(fk);
  float frj = round(fj);
  float fri = round(fi);
  uint k = (uint)frk;
  uint j = (uint)frj;
  uint i = (uint)fri;
  int ofx = (fi - fri) * 256.0f;
  int ofy = (fj - frj) * 256.0f;
  int ofz = (fk - frk) * 256.0f;
  // oct 000,001,010,011,100,101,110,111
  int odx000 = 0;
  int odx001 = ofx > 0 ? 1 : -1;
  int odx010 = ofy > 0 ? NX : -NX;
  int odx011 = odx001 + odx010;
  int odx100 = ofz > 0 ? NXNY : -NXNY;
  int odx101 = odx100 + odx001;
  int odx110 = odx100 + odx010;
  int odx111 = odx100 + odx011;

  int fx0 = abs(ofx);
  int fy0 = abs(ofy);
  int fz0 = abs(ofz);
  int fx1 = 128 - fx0;
  int fy1 = 128 - fy0;
  int fz1 = 128 - fz0;
  uint idx00 = k * NXNY + j * NX + i;
  uint idx01 = idx00 + NXNYNZ;
  uint idx02 = idx01 + NXNYNZ;
  // arithmetic shift right by 14 equivalent to division by 16384
  f.s0 = ((fz1 * fy1 * fx1) >> 14), f.s1 = ((fz1 * fy1 * fx0) >> 14),
  f.s2 = ((fz1 * fy0 * fx1) >> 14), f.s3 = ((fz1 * fy0 * fx0) >> 14),
  f.s3 = ((fz0 * fy1 * fx1) >> 14), f.s5 = ((fz0 * fy1 * fx0) >> 14),
  f.s6 = ((fz0 * fy0 * fx1) >> 14), f.s7 = ((fz0 * fy0 * fx0) >> 14);
  f = q[nn] * f;

  // current x,y,z-component
  int8 vxi = (int)((x - xprev) * 65536.0f * invDX) * f;
  int8 vyi = (int)((y - yprev) * 65536.0f * invDY) * f;
  int8 vzi = (int)((z - zprev) * 65536.0f * invDZ) * f;
  // np density

  mem_fence(CLK_GLOBAL_MEM_FENCE);
  npi[idx00] += f.s0;
  npi[idx00 + odx001] += f.s1;
  npi[idx00 + odx010] += f.s2;
  npi[idx00 + odx011] += f.s3;
  npi[idx00 + odx100] += f.s4;
  npi[idx00 + odx101] += f.s5;
  npi[idx00 + odx110] += f.s6;
  npi[idx00 + odx111] += f.s7;
  cji[idx00] += vxi.s0;
  cji[idx00 + odx001] += vxi.s1;
  cji[idx00 + odx010] += vxi.s2;
  cji[idx00 + odx011] += vxi.s3;
  cji[idx00 + odx100] += vxi.s4;
  cji[idx00 + odx101] += vxi.s5;
  cji[idx00 + odx110] += vxi.s6;
  cji[idx00 + odx111] += vxi.s7;
  cji[idx01] += vyi.s0;
  cji[idx01 + odx001] += vyi.s1;
  cji[idx01 + odx010] += vyi.s2;
  cji[idx01 + odx011] += vyi.s3;
  cji[idx01 + odx100] += vyi.s4;
  cji[idx01 + odx101] += vyi.s5;
  cji[idx01 + odx110] += vyi.s6;
  cji[idx01 + odx111] += vyi.s7;
  cji[idx02] += vzi.s0;
  cji[idx02 + odx001] += vzi.s1;
  cji[idx02 + odx010] += vzi.s2;
  cji[idx02 + odx011] += vzi.s3;
  cji[idx02 + odx100] += vzi.s4;
  cji[idx02 + odx101] += vzi.s5;
  cji[idx02 + odx110] += vzi.s6;
  cji[idx02 + odx111] += vzi.s7;
  mem_fence(CLK_GLOBAL_MEM_FENCE);
}

// density_interpolated add a fraction of the density to the 8 surrounding cells
void kernel density(global const float4 *x0, global const float4 *y0,
                    global const float4 *z0, // prev pos
                    global const float4 *x1, global const float4 *y1,
                    global const float4 *z1, // current pos
                    global int *npi, global int *cji, global const int4 *q,
                    const float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;

  // const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,          ZHIGH =
  // ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  const float XLOW_DX = -XLOWo * invDX * a0_f, YLOW_DY = -YLOWo * invDY * a0_f,
              ZLOW_DZ = -ZLOWo * invDZ * a0_f;
  // = (1, 0, 0, 0, 0, 0, 0, 0);
  const int8 ones = (int8)(1, 1, 1, 1, 1, 1, 1, 1);
  const uint size = get_global_size(0);
  const uint ss = 4;
  const uint id = get_global_id(0);
  // number of iterations ensure that this is an integer from main code
  const uint num = NPART / (size * ss);
  const uint n0 = id * num;
  const uint n1 = n0 + num;
  for (uint nn = n0; nn < n1; ++nn) {
    __private float4 x1t = x1[nn], y1t = y1[nn], z1t = z1[nn], x0t = x0[nn],
                     y0t = y0[nn], z0t = z0[nn];
    __private int4 qt = q[nn];
    for (uint s = 0; s < ss; ++s) {
      float x = s == 0 ? x1t.s0 : s == 1 ? x1t.s1 : s == 2 ? x1t.s2 : x1t.s3;
      float y = s == 0 ? y1t.s0 : s == 1 ? y1t.s1 : s == 2 ? y1t.s2 : y1t.s3;
      float z = s == 0 ? z1t.s0 : s == 1 ? z1t.s1 : s == 2 ? z1t.s2 : z1t.s3;

      float xprev = s == 0   ? x0t.s0
                    : s == 1 ? x0t.s1
                    : s == 2 ? x0t.s2
                             : x0t.s3;
      float yprev = s == 0   ? y0t.s0
                    : s == 1 ? y0t.s1
                    : s == 2 ? y0t.s2
                             : y0t.s3;
      float zprev = s == 0   ? z0t.s0
                    : s == 1 ? z0t.s1
                    : s == 2 ? z0t.s2
                             : z0t.s3;
      int q = s == 0 ? qt.s0 : s == 1 ? qt.s1 : s == 2 ? qt.s2 : qt.s3;

      float fk = fma(z, invDZ, ZLOW_DZ);
      float fj = fma(y, invDY, YLOW_DY);
      float fi = fma(x, invDX, XLOW_DX);

      int k = (int)fk;
      int j = (int)fj;
      int i = (int)fi;
      int ofx = (int)(fi * 256.0f) - (i << 8);
      int ofy = (int)(fj * 256.0f) - (j << 8);
      int ofz = (int)(fk * 256.0f) - (k << 8);

        // oct 000,001,010,011,100,101,110,111 - 0-7
      int8 odx = (int8)(0, ofx > 127, ofy > 127 ? NX : 0, 0,
                        ofz > 127 ? NXNY : 0, 0, 0, 0);
      odx.s3 = odx.s1 + odx.s2;
      odx.s5 = odx.s4 + odx.s1;
      odx.s6 = odx.s4 + odx.s2;
      odx.s7 = odx.s4 + odx.s3;
      int8 idx = (k * NXNY + j * NX + i) * ones + odx;

      int fx0 = 256 - ofx;
      int fy0 = 256 - ofy;
      int fz0 = 256 - ofz;
      int fx1 = ofx;
      int fy1 = ofy;
      int fz1 = ofz;
      //  arithmetic shift right by 14 equivalent to division by 16384
      int8 f = (int8)(fx0, fx1, fx0, fx1, fx0, fx1, fx0, fx1);
      f *= (int8)(fy0, fy0, fy1, fy1, fy0, fy0, fy1, fy1);
      f *= (int8)(fz0, fz0, fz0, fz0, fz1, fz1, fz1, fz1);
      // sum of the 8 components = 128 *2^17
      f = f >> 17; //normalize to 128
      f *= q;

      // current x,y,z-component
      int8 vxi = (int)((x - xprev) * 65536.0f * invDX) * f;
      int8 vyi = (int)((y - yprev) * 65536.0f * invDY) * f;
      int8 vzi = (int)((z - zprev) * 65536.0f * invDZ) * f;
      // np density
      atomic_add(&npi[idx.s0], f.s0);
      atomic_add(&npi[idx.s1], f.s1);
      atomic_add(&npi[idx.s2], f.s2);
      atomic_add(&npi[idx.s3], f.s3);
      atomic_add(&npi[idx.s4], f.s4);
      atomic_add(&npi[idx.s5], f.s5);
      atomic_add(&npi[idx.s6], f.s6);
      atomic_add(&npi[idx.s7], f.s7);

      atomic_add(&cji[idx.s0], vxi.s0);
      atomic_add(&cji[idx.s1], vxi.s1);
      atomic_add(&cji[idx.s2], vxi.s2);
      atomic_add(&cji[idx.s3], vxi.s3);
      atomic_add(&cji[idx.s4], vxi.s4);
      atomic_add(&cji[idx.s5], vxi.s5);
      atomic_add(&cji[idx.s6], vxi.s6);
      atomic_add(&cji[idx.s7], vxi.s7);
      idx += NXNYNZ * ones;
      atomic_add(&cji[idx.s0], vyi.s0);
      atomic_add(&cji[idx.s1], vyi.s1);
      atomic_add(&cji[idx.s2], vyi.s2);
      atomic_add(&cji[idx.s3], vyi.s3);
      atomic_add(&cji[idx.s4], vyi.s4);
      atomic_add(&cji[idx.s5], vyi.s5);
      atomic_add(&cji[idx.s6], vyi.s6);
      atomic_add(&cji[idx.s7], vyi.s7);
      idx += NXNYNZ * ones;
      atomic_add(&cji[idx.s0], vzi.s0);
      atomic_add(&cji[idx.s1], vzi.s1);
      atomic_add(&cji[idx.s2], vzi.s2);
      atomic_add(&cji[idx.s3], vzi.s3);
      atomic_add(&cji[idx.s4], vzi.s4);
      atomic_add(&cji[idx.s5], vzi.s5);
      atomic_add(&cji[idx.s6], vzi.s6);
      atomic_add(&cji[idx.s7], vzi.s7);
    }
  }
}

// density_simple8 simplest density just add particle to the nearest cell
void kernel density_simple8(global const float8 *x0, global const float8 *y0,
                            global const float8 *z0, // prev pos
                            global const float8 *x1, global const float8 *y1,
                            global const float8 *z1, // current pos
                            global int *npi, global int *cji,
                            global const int8 *qq, const float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  const uint size = get_global_size(0);
  const uint id = get_global_id(0);
  const uint s = 8;
  const uint num = NPART / (size * s);
  //  number of iterations ensure that this is an integer from main code
  const uint n0 = id * num;
  const uint n1 = n0 + num;

  for (uint nn = n0; nn < n1; nn++) {
    __private float8 x04 = x0[nn], y04 = y0[nn], z04 = z0[nn], x14 = x1[nn],
                     y14 = y1[nn], z14 = z1[nn];
    __private int8 f1 = qq[nn] * 128;
    for (uint i = 0; i < s; i++) {
      float x = i == 0   ? x04.s0
                : i == 1 ? x04.s1
                : i == 2 ? x04.s2
                : i == 3 ? x04.s3
                : i == 4 ? x04.s4
                : i == 5 ? x04.s5
                : i == 6 ? x04.s6
                         : x04.s7;
      float y = i == 0   ? y04.s0
                : i == 1 ? y04.s1
                : i == 2 ? y04.s2
                : i == 3 ? y04.s3
                : i == 4 ? y04.s4
                : i == 5 ? y04.s5
                : i == 6 ? y04.s6
                         : y04.s7;
      float z = i == 0   ? z04.s0
                : i == 1 ? z04.s1
                : i == 2 ? z04.s2
                : i == 3 ? z04.s3
                : i == 4 ? z04.s4
                : i == 5 ? z04.s5
                : i == 6 ? z04.s6
                         : z04.s7;
      float xp = i == 0   ? x14.s0
                 : i == 1 ? x14.s1
                 : i == 2 ? x14.s2
                 : i == 3 ? x14.s3
                 : i == 4 ? x14.s4
                 : i == 5 ? x14.s5
                 : i == 6 ? x14.s6
                          : x14.s7;
      float yp = i == 0   ? y14.s0
                 : i == 1 ? y14.s1
                 : i == 2 ? y14.s2
                 : i == 3 ? y14.s3
                 : i == 4 ? y14.s4
                 : i == 5 ? y14.s5
                 : i == 6 ? y14.s6
                          : y14.s7;
      float zp = i == 0   ? z14.s0
                 : i == 1 ? z14.s1
                 : i == 2 ? z14.s2
                 : i == 3 ? z14.s3
                 : i == 4 ? z14.s4
                 : i == 5 ? z14.s5
                 : i == 6 ? z14.s6
                          : z14.s7;
      int f = i == 0   ? f1.s0
              : i == 1 ? f1.s1
              : i == 2 ? f1.s2
              : i == 3 ? f1.s3
              : i == 4 ? f1.s4
              : i == 5 ? f1.s5
              : i == 6 ? f1.s6
                       : f1.s7;

      // current x,y,z-component
      int vxi = (int)((x - xp) * 65536.0f * invDX) * f;
      int vyi = (int)((y - yp) * 65536.0f * invDY) * f;
      int vzi = (int)((z - zp) * 65536.0f * invDZ) * f;

      uint k = (uint)round((z - ZLOW) * invDZ);
      uint j = (uint)round((y - YLOW) * invDY);
      uint i = (uint)round((x - XLOW) * invDX);

      uint idx00 = k * NXNY + j * NX + i;
      // np density
      atomic_add(&npi[idx00], f);
      atomic_add(&cji[idx00], vxi);
      idx00 += NXNYNZ;
      atomic_add(&cji[idx00], vyi);
      idx00 += NXNYNZ;
      atomic_add(&cji[idx00], vzi);
    }
  }
}

// density_simple16 simplest density just add particle to the nearest cell
// size_t ntry = n_partd/16;
void kernel density_simple16(global const float16 *x0, global const float16 *y0,
                             global const float16 *z0, // prev pos
                             global const float16 *x1, global const float16 *y1,
                             global const float16 *z1, // current pos
                             global int *npi, global int *cji,
                             global const int16 *qq, const float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  // const uint size = get_global_size(0);
  const uint nn = get_global_id(0);
  const uint s = 16;
  // const uint num = NPART / (size * s);

  //  number of iterations ensure that this is an integer from main code
  // const uint n0 = id * num;
  // const uint n1 = n0 + num;

  // for (uint nn = n0; nn < n1; nn++)
  // {
  __private float16 x04 = x0[nn], y04 = y0[nn], z04 = z0[nn], x14 = x1[nn],
                    y14 = y1[nn], z14 = z1[nn];
  __private int16 f1 = qq[nn] * 128;
  for (uint i = 0; i < s; i++) {
    float x = i == 0    ? x04.s0
              : i == 1  ? x04.s1
              : i == 2  ? x04.s2
              : i == 3  ? x04.s3
              : i == 4  ? x04.s4
              : i == 5  ? x04.s5
              : i == 6  ? x04.s6
              : i == 7  ? x04.s7
              : i == 8  ? x04.s8
              : i == 9  ? x04.s9
              : i == 10 ? x04.sA
              : i == 11 ? x04.sB
              : i == 12 ? x04.sC
              : i == 13 ? x04.sD
              : i == 14 ? x04.sE
                        : x04.sF;

    float y = i == 0    ? y04.s0
              : i == 1  ? y04.s1
              : i == 2  ? y04.s2
              : i == 3  ? y04.s3
              : i == 4  ? y04.s4
              : i == 5  ? y04.s5
              : i == 6  ? y04.s6
              : i == 7  ? y04.s7
              : i == 8  ? y04.s8
              : i == 9  ? y04.s9
              : i == 10 ? y04.sA
              : i == 11 ? y04.sB
              : i == 12 ? y04.sC
              : i == 13 ? y04.sD
              : i == 14 ? y04.sE
                        : y04.sF;

    float z = i == 0    ? z04.s0
              : i == 1  ? z04.s1
              : i == 2  ? z04.s2
              : i == 3  ? z04.s3
              : i == 4  ? z04.s4
              : i == 5  ? z04.s5
              : i == 6  ? z04.s6
              : i == 7  ? z04.s7
              : i == 8  ? z04.s8
              : i == 9  ? z04.s9
              : i == 10 ? z04.sA
              : i == 11 ? z04.sB
              : i == 12 ? z04.sC
              : i == 13 ? z04.sD
              : i == 14 ? z04.sE
                        : z04.sF;

    float xp = i == 0    ? x14.s0
               : i == 1  ? x14.s1
               : i == 2  ? x14.s2
               : i == 3  ? x14.s3
               : i == 4  ? x14.s4
               : i == 5  ? x14.s5
               : i == 6  ? x14.s6
               : i == 7  ? x14.s7
               : i == 8  ? x14.s8
               : i == 9  ? x14.s9
               : i == 10 ? x14.sA
               : i == 11 ? x14.sB
               : i == 12 ? x14.sC
               : i == 13 ? x14.sD
               : i == 14 ? x14.sE
                         : x14.sF;

    float yp = i == 0    ? y14.s0
               : i == 1  ? y14.s1
               : i == 2  ? y14.s2
               : i == 3  ? y14.s3
               : i == 4  ? y14.s4
               : i == 5  ? y14.s5
               : i == 6  ? y14.s6
               : i == 7  ? y14.s7
               : i == 8  ? y14.s8
               : i == 9  ? y14.s9
               : i == 10 ? y14.sA
               : i == 11 ? y14.sB
               : i == 12 ? y14.sC
               : i == 13 ? y14.sD
               : i == 14 ? y14.sE
                         : y14.sF;

    float zp = i == 0    ? z14.s0
               : i == 1  ? z14.s1
               : i == 2  ? z14.s2
               : i == 3  ? z14.s3
               : i == 4  ? z14.s4
               : i == 5  ? z14.s5
               : i == 6  ? z14.s6
               : i == 7  ? z14.s7
               : i == 8  ? z14.s8
               : i == 9  ? z14.s9
               : i == 10 ? z14.sA
               : i == 11 ? z14.sB
               : i == 12 ? z14.sC
               : i == 13 ? z14.sD
               : i == 14 ? z14.sE
                         : z14.sF;

    int f = i == 0    ? f1.s0
            : i == 1  ? f1.s1
            : i == 2  ? f1.s2
            : i == 3  ? f1.s3
            : i == 4  ? f1.s4
            : i == 5  ? f1.s5
            : i == 6  ? f1.s6
            : i == 7  ? f1.s7
            : i == 8  ? f1.s8
            : i == 9  ? f1.s9
            : i == 10 ? f1.sA
            : i == 11 ? f1.sB
            : i == 12 ? f1.sC
            : i == 13 ? f1.sD
            : i == 14 ? f1.sE
                      : f1.sF;

    // current x,y,z-component
    int vxi = (int)((x - xp) * 65536.0f * invDX) * f;
    int vyi = (int)((y - yp) * 65536.0f * invDY) * f;
    int vzi = (int)((z - zp) * 65536.0f * invDZ) * f;

    uint kk = (uint)round((z - ZLOW) * invDZ);
    uint jj = (uint)round((y - YLOW) * invDY);
    uint ii = (uint)round((x - XLOW) * invDX);

    uint idx00 = kk * NXNY + jj * NX + ii;
    // np density
    atomic_add(&npi[idx00], f);
    atomic_add(&cji[idx00], vxi);
    idx00 += NXNYNZ;
    atomic_add(&cji[idx00], vyi);
    idx00 += NXNYNZ;
    atomic_add(&cji[idx00], vzi);
    // }
  }
}
// density_simplearray16 wrong?
void kernel density_simplearray16(global const float *x0,
                                  global const float *y0,
                                  global const float *z0, // prev pos
                                  global const float *x1,
                                  global const float *y1,
                                  global const float *z1, // current pos
                                  global int *npi, global int *cji,
                                  global const int *qq, const float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  // int f; // = (1, 0, 0, 0, 0, 0, 0, 0);
  const uint size = get_global_size(0);
  const uint id = get_global_id(0);
  const uint s = 16;
  const uint num = NPART / (size * s);
  // const uint num = 16;
  //  number of iterations ensure that this is an integer from main code
  const uint n0 = id * num;
  const uint n1 = n0 + num;
  __private float xp[16], yp[16], zp[16], x[16], y[16], z[16];
  __private int f1[16];
  for (uint nn = n0; nn < n1; nn++) {
    for (uint ss = 0; ss < s; ss++)
      xp[ss] = x0[nn + ss];
    for (uint ss = 0; ss < s; ss++)
      yp[ss] = y0[nn + ss];
    for (uint ss = 0; ss < s; ss++)
      zp[ss] = z0[nn + ss];
    for (uint ss = 0; ss < s; ss++)
      x[ss] = x1[nn + ss];
    for (uint ss = 0; ss < s; ss++)
      y[ss] = y1[nn + ss];
    for (uint ss = 0; ss < s; ss++)
      z[ss] = z1[nn + ss];
    for (uint ss = 0; ss < s; ss++)
      f1[ss] = qq[nn + ss];

    for (uint ss = 0; ss < s; ss++) {
      int f = f1[ss] * 128;
      // current x,y,z-component
      int vxi = (int)((x[ss] - xp[ss]) * 65536.0f * invDX) * f;
      int vyi = (int)((y[ss] - yp[ss]) * 65536.0f * invDY) * f;
      int vzi = (int)((z[ss] - zp[ss]) * 65536.0f * invDZ) * f;

      uint k = (uint)round((z[ss] - ZLOW) * invDZ);
      uint j = (uint)round((y[ss] - YLOW) * invDY);
      uint i = (uint)round((x[ss] - XLOW) * invDX);

      uint idx00 = k * NXNY + j * NX + i;
      // np density
      atomic_add(&npi[idx00], f);
      atomic_add(&cji[idx00], vxi);
      idx00 += NXNYNZ;
      atomic_add(&cji[idx00], vyi);
      idx00 += NXNYNZ;
      atomic_add(&cji[idx00], vzi);
    }
  }
}
// density_simplevector4 simplest density just add particle to the nearest cell
void kernel density_simplevector4(global const float4 *x0,
                                  global const float4 *y0,
                                  global const float4 *z0, // prev pos
                                  global const float4 *x1,
                                  global const float4 *y1,
                                  global const float4 *z1, // current pos
                                  global int *npi, global int *cji,
                                  global const int *qq, const float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  // int f; // = (1, 0, 0, 0, 0, 0, 0, 0);
  const uint size = get_global_size(0);
  const uint id = get_global_id(0);
  const uint num = NPART / (size * 4);
  // const uint num = 4;
  //  number of iterations ensure that this is an integer from main code
  const uint n0 = id * num;
  const uint n1 = n0 + num;
  for (uint nn = n0; nn < n1; ++nn) {
    __private float4 x = x1[nn], y = y1[nn], z = z1[nn], xp = x0[nn],
                     yp = y0[nn], zp = z0[nn];
    __private int4 f = qq[nn] * 128;
    __private float4 f1 = (float4)(f.s0, f.s1, f.s2, f.s3);
    // current x,y,z-component
    float4 vxi = ((x - xp) * 65536.0f * invDX) * f1;
    float4 vyi = ((y - yp) * 65536.0f * invDY) * f1;
    float4 vzi = ((z - zp) * 65536.0f * invDZ) * f1;

    float4 frk = round((z - ZLOW) * invDZ);
    float4 frj = round((y - YLOW) * invDY);
    float4 fri = round((x - XLOW) * invDX);

    int4 k = (int4)((int)frk.s0, (int)frk.s1, (int)frk.s2, (int)frk.s3);
    int4 j = (int4)((int)frj.s0, (int)frj.s1, (int)frj.s2, (int)frj.s3);
    int4 i = (int4)((int)fri.s0, (int)fri.s1, (int)fri.s2, (int)fri.s3);

    int4 idx00 = k * NXNY + j * NX + i;
    // np density
    atomic_add(&npi[idx00.s0], f.s0);
    atomic_add(&npi[idx00.s1], f.s1);
    atomic_add(&npi[idx00.s2], f.s2);
    atomic_add(&npi[idx00.s3], f.s3);

    atomic_add(&cji[idx00.s0], (int)vxi.s0);
    atomic_add(&cji[idx00.s1], (int)vxi.s1);
    atomic_add(&cji[idx00.s2], (int)vxi.s2);
    atomic_add(&cji[idx00.s3], (int)vxi.s3);

    idx00 += NXNYNZ;
    atomic_add(&cji[idx00.s0], (int)vyi.s0);
    atomic_add(&cji[idx00.s1], (int)vyi.s1);
    atomic_add(&cji[idx00.s2], (int)vyi.s2);
    atomic_add(&cji[idx00.s3], (int)vyi.s3);

    idx00 += NXNYNZ;
    atomic_add(&cji[idx00.s0], (int)vzi.s0);
    atomic_add(&cji[idx00.s1], (int)vzi.s1);
    atomic_add(&cji[idx00.s2], (int)vzi.s2);
    atomic_add(&cji[idx00.s3], (int)vzi.s3);
  }
}

// convert integer density to floating point format multiply in time step and
// cell size
void kernel df(global float *np, global int *npi, global float *currentj,
               global int *cji, const float a0_f, const float dt) {
  const float dx = DXo * a0_f * 1.1920929e-7f / dt,
              dy = DYo * a0_f * 1.1920929e-7f / dt,
              dz = DZo * a0_f * 1.1920929e-7f / dt;
  const float dn = 0.0078125f;
  uint idx00 = get_global_id(0);
  uint idx01 = idx00 + NZ * NY * NX;
  uint idx02 = idx01 + NZ * NY * NX;
  np[idx00] = dn * npi[idx00];
  currentj[idx00] = dx * cji[idx00];
  currentj[idx01] = dy * cji[idx01];
  currentj[idx02] = dz * cji[idx02];
  npi[idx00] = 0;
  cji[idx00] = 0;
  cji[idx01] = 0;
  cji[idx02] = 0;
}

void kernel trilin_k(
    global float8 *Ea, // E, B coeff Ea[k][j][i][3][8] according to tnp_k
    global const float *E_flat, // E or B 3 components per cell E[3][k][j][i]
    float a0_f) {
  // return;
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float dV = DX * DY * DZ;
  const float dV1 = 1.0f / dV;
  const float dx2 = DX * DX;
  const float dy2 = DY * DY;
  const float dz2 = DZ * DZ;
  const float dxdy = DX * DY;
  const float dydz = DY * DZ;
  const float dzdx = DZ * DX;

  const unsigned int n_cells = NX * NY * NZ;
  int offset = get_global_id(0);
  int co = 0;

  for (int c = 0; c < 3; ++c, co += n_cells) {
    unsigned int k = (offset / (NX * NY)) % NZ;
    unsigned int j = (offset / NX) % NY;
    unsigned int i = offset % NX;
    const int odx000 = 0;
    const int odx001 = i < NX ? 1 : 0;  // iskip
    const int odx010 = j < NY ? NX : 0; // jskip
    const int odx011 = odx001 + odx010;
    const int odx100 = k < NZ ? NY * NX : 0;
    const int odx101 = odx100 + odx001;
    const int odx110 = odx100 + odx010;
    const int odx111 = odx100 + odx011;

    const float z0 = k * DZ + ZLOW;
    const float z1 = z0 + DZ;
    const float y0 = j * DY + YLOW;
    const float y1 = y0 + DY;
    const float x0 = i * DX + XLOW;
    const float x1 = x0 + DX;

    const float x0y0 = x0 * y0, x0y1 = x0 * y1, x1y0 = x1 * y0, x1y1 = x1 * y1;
    const float y0z0 = y0 * z0, y0z1 = y0 * z1, y1z0 = y1 * z0, y1z1 = y1 * z1;
    const float x0z0 = x0 * z0, x0z1 = x0 * z1, x1z0 = x1 * z0, x1z1 = x1 * z1;

    const float x0y0z0 = x0 * y0z0, x0y0z1 = x0 * y0z1, x0y1z0 = x0 * y1z0,
                x0y1z1 = x0 * y1z1;
    const float x1y0z0 = x1 * y0z0, x1y0z1 = x1 * y0z1, x1y1z0 = x1 * y1z0,
                x1y1z1 = x1 * y1z1;

    const float c000 = E_flat[offset + co + odx000]; // E[c][k][j][i];
    const float c001 = E_flat[offset + co + odx100]; // E[c][k1][j][i];
    const float c010 = E_flat[offset + co + odx010]; // E[c][k][j1][i];
    const float c011 = E_flat[offset + co + odx110]; // E[c][k1][j1][i];
    const float c100 = E_flat[offset + co + odx001]; // E[c][k][j][i1];
    const float c101 = E_flat[offset + co + odx101]; // E[c][k1][j][i1];
    const float c110 = E_flat[offset + co + odx011]; // E[c][k][j1][i1];
    const float c111 = E_flat[offset + co + odx111]; // E[c][k1][j1][i1];

    int oa = (offset * 3 + c);
    Ea[oa].s0 =
        (-c000 * x1y1z1 + c001 * x1y1z0 + c010 * x1y0z1 - c011 * x1y0z0 +
         c100 * x0y1z1 - c101 * x0y1z0 - c110 * x0y0z1 + c111 * x0y0z0) *
        dV1;
    Ea[oa].s1 = ((c000 - c100) * y1z1 + (-c001 + c101) * y1z0 +
                 (-c010 + c110) * y0z1 + (c011 - c111) * y0z0) *
                dV1;
    Ea[oa].s2 = ((c000 - c010) * x1z1 + (-c001 + c011) * x1z0 +
                 (-c100 + c110) * x0z1 + (c101 - c111) * x0z0) *
                dV1;
    Ea[oa].s3 = ((c000 - c001) * x1y1 + (-c010 + c011) * x1y0 +
                 (-c100 + c101) * x0y1 + (c110 - c111) * x0y0) *
                dV1;
    Ea[oa].s4 =
        ((-c000 + c010 + c100 - c110) * z1 + (c001 - c011 - c101 + c111) * z0) *
        dV1;
    Ea[oa].s5 =
        ((-c000 + c001 + c100 - c101) * y1 + (c010 - c011 - c110 + c111) * y0) *
        dV1;
    Ea[oa].s6 =
        ((-c000 + c001 + c010 - c011) * x1 + (c100 - c101 - c110 + c111) * x0) *
        dV1;
    Ea[oa].s7 = (c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111) * dV1;
  }
}

void kernel EUEst(global const float4 *V, global const float4 *n,
                  global float *EUtot) {
  int i = get_global_id(0);
  // Compute dot product for the given gid
  EUtot[i] = dot(V[i], n[i]);
}

void kernel dtotal(global const float16 *ne, global const float16 *ni,
                   global const float16 *je, global const float16 *ji,
                   global float16 *nt, global float16 *jt, const uint n0) {
  const uint n1 = NXNYNZ / 16;
  const uint n2 = n1 + n1;
  const uint i = get_global_id(0); // Get index of current element processed
  nt[i] = ne[i] + ni[i];           // Do the operation
  jt[i] = je[i] + ji[i];
  jt[n1 + i] = je[n1 + i] + ji[n1 + i];
  jt[n2 + i] = je[n2 + i] + ji[n2 + i];
}

void kernel nsumi(global const int16 *npi, global int *n0) {
  const uint n1 = get_global_size(0); // n_part_2048=2048 work items
  const uint n2 = NPART / (2048 * 16);
  // const uint n2 = 128;
  // make sure n is divisible by n1*16 from calling code
  const uint i = get_global_id(0); // Get index of current element processed
  const uint j0 = i * n2;
  const uint j1 = j0 + n2;
  int16 sum = 0;
  // Use local memory to reduce global memory access
  //  int16 local_npi[16];

  // Load data into local memory
  // for (uint j = 0; j < n2; ++j) {
  //  local_npi[j] = npi[j0 + j];
  // }
  // Ensure all work-items have finished loading data into local memory
  // barrier(CLK_LOCAL_MEM_FENCE);

  // Perform computation using data in local memory
  for (uint j = j0; j < j1; ++j) {
    sum += npi[j];
  }
  n0[i] = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 +
          sum.s7 + sum.s8 + sum.s9 + sum.sA + sum.sB + sum.sC + sum.sD +
          sum.sE + sum.sF;
}

void kernel copyextField(global const float16 *Fe, global float16 *F) {
  // get global indices
  uint idx = get_global_id(0);
  F[idx] = Fe[idx];
}

void kernel maxvalf(global const float16 *In, global float *Ou) {
  // get global indices
  uint i = get_global_id(0);
  float m = 0;
  float a, v;
  v = In[i].s0;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s1;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s2;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s3;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s4;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s5;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s6;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s7;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s8;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].s9;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].sA;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].sB;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].sC;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].sD;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].sE;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;
  v = In[i].sF;
  a = v > 0 ? v : -v;
  m = m > a ? m : a;

  Ou[i] = m;
}

void kernel maxval3f(global const float16 *In, global float *Ou) {
  // get global indices
  uint i = get_global_id(0);
  uint n = get_global_size(0);
  uint n2 = n + n;
  float m = 0;
  float a;
  float4 v3;

  v3 = (float4)(In[i].s0, In[i + n].s0, In[i + n2].s0, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s1, In[i + n].s1, In[i + n2].s1, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s2, In[i + n].s2, In[i + n2].s2, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s3, In[i + n].s3, In[i + n2].s3, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s4, In[i + n].s4, In[i + n2].s4, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s5, In[i + n].s5, In[i + n2].s5, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s6, In[i + n].s6, In[i + n2].s6, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s7, In[i + n].s7, In[i + n2].s7, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s8, In[i + n].s8, In[i + n2].s8, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].s9, In[i + n].s9, In[i + n2].s9, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].sA, In[i + n].sA, In[i + n2].sA, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].sB, In[i + n].sB, In[i + n2].sB, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].sC, In[i + n].sC, In[i + n2].sC, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].sD, In[i + n].sD, In[i + n2].sD, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].sE, In[i + n].sE, In[i + n2].sE, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  v3 = (float4)(In[i].sF, In[i + n].sF, In[i + n2].sF, 0.0f);
  a = dot(v3, v3);
  m = m > a ? m : a;

  Ou[i] = m;
}

void kernel buffer_muls(global float *A, const float Bb) {
  int i = get_global_id(0); // Get index of current element processed
  A[i] = Bb * A[i];         // Do the operation
}

void kernel recalcposchangedt(global float *x0, global float *y0,
                              global float *z0, // prev pos
                              global const float *x1, global const float *y1,
                              global const float *z1, // current pos
                              float const inc         // increment
) {
  int n = get_global_id(0);
  x0[n] = x1[n] - (x1[n] - x0[n]) * inc;
  y0[n] = y1[n] - (y1[n] - y0[n]) * inc;
  z0[n] = z1[n] - (z1[n] - z0[n]) * inc;
}
