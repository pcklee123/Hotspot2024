// Preprocessor things for compilation of tnp
#ifndef XLOW
#define XLOW 0.f
#endif
#ifndef YLOW
#define YLOW 0.f
#endif
#ifndef ZLOW
#define ZLOW 0.f
#endif
#ifndef XHIGH
#define XHIGH 0.f
#endif
#ifndef YHIGH
#define YHIGH 0.f
#endif
#ifndef ZHIGH
#define ZHIGH 0.f
#endif
#ifndef DX
#define DX 0
#endif
#ifndef DY
#define DY 0
#endif
#ifndef DZ
#define DZ 0
#endif
#ifndef NX
#define NX 0
#endif
#ifndef NY
#define NY 0
#endif
#ifndef NZ
#define NZ 0
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

void kernel tnp_k_implicit(global const float8 *a1,
                           global const float8 *a2, // E, B coeff
                           global float *x0, global float *y0,
                           global float *z0, // prev pos
                           global float *x1, global float *y1,
                           global float *z1, // current pos
                           float Bcoef,
                           float Ecoef, // Bcoeff, Ecoeff
                           const unsigned int n,
                           const unsigned int ncalc, // n, ncalc
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
  const float XL = XLOW + 1.5f * DX, YL = YLOW + 1.5f * DY,
              ZL = ZLOW + 1.5f * DZ;
  const float XH = XHIGH - 1.5f * DX, YH = YHIGH - 1.5f * DY,
              ZH = ZHIGH - 1.5f * DZ;
  const float8 ones = (float8)(1, 1, 1, 1, 1, 1, 1, 1);
  for (int t = 0; t < ncalc; t++) {

    // if (x <= XLOW || x >= XHIGH || y <= YLOW || y >= YHIGH || z <= ZLOW ||
    //  z >= ZHIGH)
    // break;
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
    //get interpolated Electric field at particle position
    float xE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store1 * pos;
    float yE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store2 * pos;
    float zE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    //get interpolated Magnetic field at particle position
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
                            const unsigned int n,
                            const unsigned int ncalc, // n, ncalc
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
  const float XL = XLOW + 1.5f * DX, YL = YLOW + 1.5f * DY,
              ZL = ZLOW + 1.5f * DZ;
  const float XH = XHIGH - 1.5f * DX, YH = YHIGH - 1.5f * DY,
              ZH = ZHIGH - 1.5f * DZ;
  const float ZDZ = ZH - ZL - DZ/10;
  const float8 ones = (float8)(1, 1, 1, 1, 1, 1, 1, 1);
  for (int t = 0; t < ncalc; t++) {

    // if (x <= XLOW || x >= XHIGH || y <= YLOW || y >= YHIGH || z <= ZLOW ||
    //  z >= ZHIGH)
    // break;
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

void kernel density(global const float *x0, global const float *y0,
                    global const float *z0, // prev pos
                    global const float *x1, global const float *y1,
                    global const float *z1, // current pos
                    global int *npi, global int *cji, global int *q) {
  const float XL = XLOW + 1.5f * DX, YL = YLOW + 1.5f * DY,
              ZL = ZLOW + 1.5f * DZ;
  const float XH = XHIGH - 1.5f * DX, YH = YHIGH - 1.5f * DY,
              ZH = ZHIGH - 1.5f * DZ;
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

  f.s0 = ((fz1 * fy1 * fx1) >> 14), f.s1 = ((fz1 * fy1 * fx0) >> 14),
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

void kernel df(global float *np, global const int *npi, global float *currentj,
               global const int *cji) {
  float dx = DX * 1.1920929e-7f, dy = DY * 1.1920929e-7f,
        dz = DZ * 1.1920929e-7f;
  float dn = 0.0078125f;
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
    global const float *E_flat // E or B 3 components per cell E[3][k][j][i]
) {
  // return;
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
