// Preprocessor things for compilation of tnp assume if XLOWo is defined then
// everything else is.Need these defines here to avoid warnings in the editor
#ifndef XLOWo
#define XLOWo 0.0
#define YLOWo 0.0
#define ZLOWo 0.0
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
#define NC4 8454144 // N0*N1*(N2/2+1) = 4 * NX * NY * (NZ + 1)
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

// find the particle and current density convert from floating point to integer
// to use atomic_add smoothly assign a fraction of the density to "cell"
// depending on "center of density"
void kernel density(global const float *x0, global const float *y0,
                    global const float *z0, // prev pos
                    global const float *x1, global const float *y1,
                    global const float *z1, // current pos
                    global int *npi, global int *cji, global int *q,
                    float a0_f) {
  const float DX = DXo * a0_f, DY = DYo * a0_f, DZ = DZo * a0_f;
  const float XLOW = XLOWo * a0_f, YLOW = YLOWo * a0_f, ZLOW = ZLOWo * a0_f;
  const float XHIGH = XHIGHo * a0_f, YHIGH = YHIGHo * a0_f,
              ZHIGH = ZHIGHo * a0_f;

  const float invDX = 1.0f / DX, invDY = 1.0f / DY, invDZ = 1.0f / DZ;
  int8 f; // = (1, 0, 0, 0, 0, 0, 0, 0);
  uint id = get_global_id(0);
  float xprev = x0[id], yprev = y0[id], zprev = z0[id], x = x1[id], y = y1[id],
        z = z1[id];

  uint k = round((z - ZLOW) * invDZ);
  uint j = round((y - YLOW) * invDY);
  uint i = round((x - XLOW) * invDX);
  int ofx = ((x - XLOW) * invDX - i) * 256.0f;
  int ofy = ((y - YLOW) * invDY - j) * 256.0f;
  int ofz = ((z - ZLOW) * invDZ - k) * 256.0f;
  // oct 000,001,010,011,100,101,110,111
  int odx000 = 0;
  int odx001 = ofx > 0 ? 1 : -1;
  int odx010 = ofy > 0 ? NX : -NX;
  int odx011 = odx001 + odx010;
  int odx100 = ofz > 0 ? NX * NY : -NX * NY;
  int odx101 = odx100 + odx001;
  int odx110 = odx100 + odx010;
  int odx111 = odx100 + odx011;

  int fx0 = abs(ofx);
  int fy0 = abs(ofy);
  int fz0 = abs(ofz);
  int fx1 = 128 - fx0;
  int fy1 = 128 - fy0;
  int fz1 = 128 - fz0;
  uint idx00 = k * NY * NX + j * NX + i;
  uint idx01 = idx00 + NZ * NY * NX;
  uint idx02 = idx01 + NZ * NY * NX;

  f.s0 = ((fz1 * fy1 * fx1) >> 14),
  f.s1 = ((fz1 * fy1 * fx0) >>
          14), // arithmetic shift right by 14 equivalent to division by 16384
      f.s2 = ((fz1 * fy0 * fx1) >> 14), f.s3 = ((fz1 * fy0 * fx0) >> 14),
  f.s3 = ((fz0 * fy1 * fx1) >> 14), f.s5 = ((fz0 * fy1 * fx0) >> 14),
  f.s6 = ((fz0 * fy0 * fx1) >> 14), f.s7 = ((fz0 * fy0 * fx0) >> 14);
  f = q[id] * f;
  // np density
  atomic_add(&npi[idx00 + odx000], f.s0);
  atomic_add(&npi[idx00 + odx001], f.s1);
  atomic_add(&npi[idx00 + odx010], f.s2);
  atomic_add(&npi[idx00 + odx011], f.s3);
  atomic_add(&npi[idx00 + odx100], f.s4);
  atomic_add(&npi[idx00 + odx101], f.s5);
  atomic_add(&npi[idx00 + odx110], f.s6);
  atomic_add(&npi[idx00 + odx111], f.s7);
  /*

    npi[idx00 + odx000] += f.s0;
  npi[idx00 + odx001] += f.s1;
  npi[idx00 + odx010] += f.s2;
  npi[idx00 + odx011] += f.s3;
  npi[idx00 + odx100] += f.s4;
  npi[idx00 + odx101] += f.s5;
  npi[idx00 + odx110] += f.s6;
  npi[idx00 + odx111] += f.s7; */

  // current x-component
  int8 vxi = (int)((x - xprev) * 65536.0f * invDX) * f;
  atomic_add(&cji[idx00 + odx000], vxi.s0);
  atomic_add(&cji[idx00 + odx001], vxi.s1);
  atomic_add(&cji[idx00 + odx010], vxi.s2);
  atomic_add(&cji[idx00 + odx011], vxi.s3);
  atomic_add(&cji[idx00 + odx100], vxi.s4);
  atomic_add(&cji[idx00 + odx101], vxi.s5);
  atomic_add(&cji[idx00 + odx110], vxi.s6);
  atomic_add(&cji[idx00 + odx111], vxi.s7);

  int8 vyi = (int)((y - yprev) * 65536.0f * invDY) * f;
  atomic_add(&cji[idx01 + odx000], vyi.s0);
  atomic_add(&cji[idx01 + odx001], vyi.s1);
  atomic_add(&cji[idx01 + odx010], vyi.s2);
  atomic_add(&cji[idx01 + odx011], vyi.s3);
  atomic_add(&cji[idx01 + odx100], vyi.s4);
  atomic_add(&cji[idx01 + odx101], vyi.s5);
  atomic_add(&cji[idx01 + odx110], vyi.s6);
  atomic_add(&cji[idx01 + odx111], vyi.s7);

  int8 vzi = (int)((z - zprev) * 65536.0f * invDZ) * f;
  atomic_add(&cji[idx02 + odx000], vzi.s0);
  atomic_add(&cji[idx02 + odx001], vzi.s1);
  atomic_add(&cji[idx02 + odx010], vzi.s2);
  atomic_add(&cji[idx02 + odx011], vzi.s3);
  atomic_add(&cji[idx02 + odx100], vzi.s4);
  atomic_add(&cji[idx02 + odx101], vzi.s5);
  atomic_add(&cji[idx02 + odx110], vzi.s6);
  atomic_add(&cji[idx02 + odx111], vzi.s7);
}

// convert integer density to floating point format multiply in time step and
// cell size
void kernel df(global float *np, global const int *npi, global float *currentj,
               global const int *cji, const float a0_f, const float dt) {
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

void kernel nsumi(global const int *npi, global int *n0, const uint npart) {
  const uint n1 = get_global_size(0); // n_part_2048
  const uint n2 =
      npart / n1; // make sure n is divisible by n1 from calling code
  const uint i = get_global_id(0); // Get index of current element processed
  const uint j0 = i * n2;
  n0[i] = 0;
  for (uint j = 0; j < n2; ++j)
    n0[i] += npi[j0 + j]; // Do the operation
  // n0[i] = n0[i];
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
