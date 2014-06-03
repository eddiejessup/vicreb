import numpy as np
import matplotlib.pyplot as pp
import vtk
from vtk.util import numpy_support
import utils
import butils
import argparse
import os
import time

def vis(dyns):
    dirname = os.path.dirname(os.path.commonprefix(dyns))
    stat = np.load(os.path.join(dirname, 'static.npz'))
    L = stat['L']
    try:
        As = stat['As']
    except KeyError:
        As = np.ones(len(stat['r0']), dtype=np.bool)

    ren = vtk.vtkRenderer()
    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(ren)

    cube = vtk.vtkCubeSource()
    cube.SetXLength(L)
    cube.SetYLength(L)
    cube.SetZLength(L)
    sysmap = vtk.vtkPolyDataMapper()
    sysmap.SetInputConnection(cube.GetOutputPort())
    sysact = vtk.vtkActor()
    sysact.GetProperty().SetOpacity(0.2)
    sysact.SetMapper(sysmap)
    ren.AddActor(sysact)

    Apoints = vtk.vtkPoints()
    Apolyd = vtk.vtkPolyData()
    Apolyd.SetPoints(Apoints)
    Aglyph = vtk.vtkGlyph3D()
    arrow = vtk.vtkArrowSource()
    Aglyph.SetSourceConnection(arrow.GetOutputPort())
    Aglyph.SetInputData(Apolyd)
    Aglyph.SetScaleFactor(0.2)
    Amap = vtk.vtkPolyDataMapper()
    Amap.SetInputConnection(Aglyph.GetOutputPort())
    Aact = vtk.vtkActor()
    Aact.SetMapper(Amap)
    Aact.GetProperty().SetColor(0, 1, 0)
    ren.AddActor(Aact)

    Bpoints = vtk.vtkPoints()
    Bpolyd = vtk.vtkPolyData()
    Bpolyd.SetPoints(Bpoints)
    Bglyph = vtk.vtkGlyph3D()
    arrow = vtk.vtkArrowSource()
    Bglyph.SetSourceConnection(arrow.GetOutputPort())
    Bglyph.SetInputData(Bpolyd)
    Bglyph.SetScaleFactor(0.2)
    Bmap = vtk.vtkPolyDataMapper()
    Bmap.SetInputConnection(Bglyph.GetOutputPort())
    Bact = vtk.vtkActor()
    Bact.SetMapper(Bmap)
    Bact.GetProperty().SetColor(1, 0, 0)
    ren.AddActor(Bact)

    for fname in dyns:
        dyn = np.load(fname.strip())
        r = butils.pad_to_3d(dyn['r'])
        u = butils.pad_to_3d(dyn['u'])

        rA = r[As]
        uA = u[As]
        rB = r[np.logical_not(As)]
        uB = u[np.logical_not(As)]
        Apoints.SetData(numpy_support.numpy_to_vtk(rA))
        Apolyd.GetPointData().SetVectors(numpy_support.numpy_to_vtk(uA))
        Bpoints.SetData(numpy_support.numpy_to_vtk(rB))
        Bpolyd.GetPointData().SetVectors(numpy_support.numpy_to_vtk(uB))
        renwin.Render()
        time.sleep(0.1)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)
    iren.Initialize()
    iren.Start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise porous states using VTK')
    parser.add_argument('dyns', nargs='*',
        help='npz files containing dynamic states')
    args = parser.parse_args()

    vis(args.dyns)
