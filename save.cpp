#include "include/traj.h"
// #include <iostream>
// #include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
// #include <vtkPoints.h>
// #include <vtkDoubleArray.h>
#include <vtkXMLStructuredGridWriter.h>

void save_hist(int i_time, double t, particles *pt, par *par)
{
  // cout << "save_hist"<<endl;
  long KEhist[2][Hist_n];
  memset(KEhist, 0, sizeof(KEhist));
  float coef[2];
  for (int p = 0; p < 2; ++p)
  {
    float KE = 0;
    long nt = 0;
    coef[p] = 0.5 * (float)mp[p] * (float)Hist_n / (e_charge_mass * par->dt[p] * par->dt[p] * (float)Hist_max);
    for (int i = 0; i < par->n_part[p]; ++i)
    {
      float dx = pt->pos1x[p][i] - pt->pos0x[p][i];
      float dy = pt->pos1y[p][i] - pt->pos0y[p][i];
      float dz = pt->pos1z[p][i] - pt->pos0z[p][i];
      float v2 = (dx * dx + dy * dy + dz * dz);
      unsigned int index = (int)floor(coef[p] * v2);
      KE += v2;
      nt += pt->q[p][i];

      if (index >= Hist_n)
        index = Hist_n - 1;
      //   if (index < 0) cout << "error index<0"<<(0.5 * (float)mp[p] * (dx * dx + dy * dy + dz * dz) * (float)Hist_n/ (e_charge_mass * par->dt[p] * par->dt[p]*(float)Hist_max))<< endl;
      KEhist[p][index]++;
    }
    par->KEtot[p] = KE * 0.5 * mp[p] / (e_charge_mass * par->dt[p] * par->dt[p]) * r_part_spart; // as if these particles were actually samples of the greater thing
                                                                                                 // par->nt[p] = nt;// * r_part_spart;
    //   cout << "p = " << p << ", KE = " << par->KEtot[p] << ", npart[p]" << par->n_part[p] << endl;
  }

  // Create a vtkPolyData object
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

  // Add the FieldData to the PolyData
  vtkSmartPointer<vtkFieldData> fieldData = polyData->GetFieldData();
  vtkSmartPointer<vtkDoubleArray> timevalue = vtkSmartPointer<vtkDoubleArray>::New();
  timevalue->SetName("TimeValue");
  timevalue->InsertNextValue(t);
  fieldData->AddArray(timevalue);

  // Create a vtkPoints object to store the bin centers
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  // Create a vtkDoubleArray object to store the bin counts
  vtkSmartPointer<vtkDoubleArray> ecounts = vtkSmartPointer<vtkDoubleArray>::New();
  ecounts->SetName("ecounts");
  // ecounts->SetNumberOfComponents(1);

  vtkSmartPointer<vtkDoubleArray> icounts = vtkSmartPointer<vtkDoubleArray>::New();
  icounts->SetName("icounts");
  // icounts->SetNumberOfComponents(1);

  // Fill the points array with data
  for (int i = 0; i < Hist_n; ++i)
  {
    double z = ((double)(i + 0.5) * (double)Hist_max) / (double)(Hist_n); // Calculate the center of the i-th bin
    points->InsertNextPoint(0.0, 0.0, z);                                 // Set the i-th point to the center of the i-th bin
    ecounts->InsertNextValue((double)(log(KEhist[0][i] + 1)));
    icounts->InsertNextValue((double)(log(KEhist[1][i] + 1)));
  }

  // Set the arrays as the data for the polyData object
  polyData->SetPoints(points);
  polyData->GetPointData()->AddArray(ecounts);
  polyData->GetPointData()->AddArray(icounts);

  // Write the polyData object to a file using VTK's XML file format
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName((par->outpath + "KEhist_" + to_string(i_time) + ".vtp").c_str());
  writer->SetDataModeToBinary();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(polyData);
  writer->Write();
}

void save_vti_c(string filename, int i,
                int ncomponents, double t,
                float (*data1)[n_space_divz][n_space_divy][n_space_divx], par *par)
{
  // Create structured grid
  vtkSmartPointer<vtkStructuredGrid> structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();
  int xi = (par->n_space_div[0] - 1) / maxcells + 1;
  int yj = (par->n_space_div[0] - 1) / maxcells + 1;
  int zk = (par->n_space_div[0] - 1) / maxcells + 1;
  int nx = par->n_space_div[0] / xi;
  int ny = par->n_space_div[1] / yj;
  int nz = par->n_space_div[2] / zk;
  // Set dimensions
  structuredGrid->SetDimensions(nx + 1, ny + 1, nz + 1);
  // Set points at is one more vertex than cell
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  for (int k = 0; k <= nz; ++k)
  {
    for (int j = 0; j <= ny; ++j)
    {
      for (int i = 0; i <= nx; ++i)
      {
        double point[3];
        point[0] = par->posL[0] + i * par->dd[0] * xi;
        point[1] = par->posL[1] + j * par->dd[1] * yj;
        point[2] = par->posL[2] + k * par->dd[2] * zk;
        points->InsertNextPoint(point);
      }
    }
  }
  structuredGrid->SetPoints(points);
  // Set field vector data
  vtkSmartPointer<vtkDoubleArray> FieldVectorArray = vtkSmartPointer<vtkDoubleArray>::New();
  FieldVectorArray->SetName(filename.c_str());             // cout << filename << ", " <<nx*ny*nz << endl;
  FieldVectorArray->SetNumberOfComponents(ncomponents);    // Three components (Ex, Ey, Ez)
  FieldVectorArray->SetNumberOfTuples((nx) * (ny) * (nz)); // average cells shifted by half cell?
  for (int k = 0; k < nz; ++k)
  {
    for (int j = 0; j < ny; ++j)
    {
      for (int i = 0; i < nx; ++i)
      {
        int index = k * ny * nx + j * nx + i;
        double data[3] = {0, 0, 0};
        for (int c = 0; c < ncomponents; ++c)
        {
          for (int kk = 0; kk < zk; ++kk)
          {
            for (int jj = 0; jj < yj; ++jj)
            {
              for (int ii = 0; ii < xi; ++ii)
              {
                data[c] += (double)data1[c][k * zk + kk][j * yj + jj][i * xi + ii];
              }
            }
          }
          data[c] /= xi * yj * zk;
        }
        if (ncomponents == 3)
          FieldVectorArray->SetTuple3(index, data[0], data[1], data[2]); //(index,x,y,z)
        if (ncomponents == 1)
          FieldVectorArray->SetTuple1(index, data[0]);
      }
    }
  }
  structuredGrid->GetCellData()->AddArray(FieldVectorArray);
  // Create a vtkDoubleArray to hold the field data
  vtkSmartPointer<vtkDoubleArray> timeArray = vtkSmartPointer<vtkDoubleArray>::New();
  timeArray->SetName("TimeValue");
  timeArray->SetNumberOfTuples(1);
  timeArray->SetValue(0, t);

  // Add the field data to the FieldVectorArray data
  vtkSmartPointer<vtkFieldData> fieldData = structuredGrid->GetFieldData();
  fieldData->AddArray(timeArray);
  // Write to XML file
  vtkSmartPointer<vtkXMLStructuredGridWriter> writer = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();

  writer->SetFileName((par->outpath + filename + "_" + to_string(i) + ".vts").c_str());
  writer->SetDataModeToBinary();
  // writer->SetCompressorTypeToLZ4();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(structuredGrid);
  writer->Write();
}

void save_vtp(string filename, int i, uint64_t num, double t, int p, particles *pt, par *par)
{
  // cout << "save_vti_p"<<endl;
  static int first = 1;
  static int nr[n_output_part];
  int nprtd = floor(par->n_part[p] / n_output_part);
  if (first)
  {
    for (int i = 0; i < n_output_part; i++)
      nr[i] = i * nprtd + rand() % nprtd;
  }
  // Create a polydata object
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
  // Add the FieldData to the PolyData
  vtkSmartPointer<vtkFieldData> fieldData = polyData->GetFieldData();
  vtkSmartPointer<vtkDoubleArray> timeArray = vtkSmartPointer<vtkDoubleArray>::New();
  timeArray->SetName("TimeValue");
  timeArray->SetNumberOfTuples(1);
  timeArray->SetValue(0, t);
  fieldData->AddArray(timeArray);

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkFloatArray> kineticEnergy = vtkSmartPointer<vtkFloatArray>::New();
  kineticEnergy->SetName("KE");

  // #pragma omp parallel for simd
  //  #pragma omp distribute parallel for simd
  for (int nprt = 0; nprt < n_output_part; nprt++)
  {
    int n = nr[nprt];
    float KE, dpos, dpos2 = 0;
    dpos = (pt->pos1x[p][n] - pt->pos0x[p][n]);
    dpos *= dpos;
    dpos2 += dpos;
    dpos = (pt->pos1y[p][n] - pt->pos0y[p][n]);
    dpos *= dpos;
    dpos2 += dpos;
    dpos = (pt->pos1z[p][n] - pt->pos0z[p][n]);
    dpos *= dpos;
    dpos2 += dpos;
    KE = 0.5 * mp[p] * (dpos2) / (e_charge_mass * par->dt[p] * par->dt[p]);
    if (KE >= 0)
    {
      kineticEnergy->InsertNextValue(KE);
      // in units of eV
      points->InsertNextPoint(pt->pos1x[p][n], pt->pos1y[p][n], pt->pos1z[p][n]);
    }
  }

  polyData->SetPoints(points);
  polyData->GetPointData()->AddArray(kineticEnergy);
  // Write the output file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName((par->outpath + filename + "_" + to_string(i) + ".vtp").c_str());
  writer->SetDataModeToBinary();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(polyData);
  writer->Write();
}

void save_files(int i_time, double t, fields *fi, particles *pt, par *par)
{
#pragma omp parallel sections
  {
#pragma omp section
    save_hist(i_time, t, pt, par);
#ifdef printDensity
#pragma omp section
    save_vti_c("Ne", i_time, 1, t, &fi->np[0], par);
#pragma omp section
    save_vti_c("je", i_time, 3, t, fi->currentj[0], par);
#endif
#ifdef printV
#pragma omp section
    save_vti_c("V", i_time, 1, t, fi->V, par);
#endif
#ifdef printE
#pragma omp section
    save_vti_c("E", i_time, 3, t, fi->E, par);
#endif
#ifdef printB
#pragma omp section
    save_vti_c("B", i_time, 3, t, fi->B, par);
#endif
#ifdef printParticles
#pragma omp section
    save_vtp("e", i_time, n_output_part, t, 0, pt, par);
#pragma omp section
    save_vtp("d", i_time, n_output_part, t, 1, pt, par);
#endif
  }
}