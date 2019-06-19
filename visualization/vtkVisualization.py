import numpy as np
import vtk

from abc import ABC


class VTKEntity3D(ABC):
    def __init__(self, mapper: vtk.vtkMapper):
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(mapper)

    @property
    def actor(self):
        return self._actor


class VTKPointCloud(VTKEntity3D):
    def __init__(self, points: np.ndarray=None, colors: np.ndarray=None):
        assert (points is None and colors is None) or (points is not None and colors is not None)

        self.num_points = 0

        # VTK geometry representation
        self._points = vtk.vtkPoints()

        # VTK color representation
        self._colors = vtk.vtkUnsignedCharArray()
        self._colors.SetName("Colors")
        self._colors.SetNumberOfComponents(3)

        # Visualization pipeline
        # - Data source
        point_data = vtk.vtkPolyData()
        point_data.SetPoints(self._points)
        point_data.GetPointData().SetScalars(self._colors)

        # - Automatically generate topology cells from points
        mask_points = vtk.vtkMaskPoints()
        mask_points.SetInputData(point_data)
        mask_points.GenerateVerticesOn()
        mask_points.SingleVertexPerCellOn()
        mask_points.Update()

        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mask_points.GetOutputPort())

        super().__init__(mapper)

        self.add_points(points, colors)

    def add_points(self, points: np.ndarray, colors: np.ndarray): #points, colors Nx3
        assert len(points.shape) == 2
        assert len(colors.shape) == 2
        assert points.shape[0] == colors.shape[0]
        assert points.shape[1] == 3
        assert colors.shape[1] == 3

        [num_new_points, _] = points.shape

        # Allocate additional memory
        self._points.Resize(self.num_points + num_new_points * 2)
        self._colors.Resize(self.num_points + num_new_points * 2)

        # Add points
        for point_idx in range(num_new_points):
            for _ in range(2):  # add every point twice due to vtk bug
                self._points.InsertNextPoint(points[point_idx, :])
                self._colors.InsertNextTuple(colors[point_idx, :])

        self._points.Modified()
        self._colors.Modified()

    def set_point_size(self, size: int):
        self.actor.GetProperty().SetPointSize(size)


class VTKVectorField(VTKEntity3D):
    def __init__(self, positions: np.ndarray, vectors: np.ndarray):
        self.num_vectors = 0

        # VTK position representation
        self._positions = vtk.vtkPoints()

        # VTK vector representation
        self._vectors = vtk.vtkFloatArray()
        self._vectors.SetName("Vector Field")
        self._vectors.SetNumberOfComponents(3)

        # Visualization Pipeline
        # - Data source
        position_data = vtk.vtkPolyData()
        position_data.SetPoints(self._positions)
        position_data.GetPointData().AddArray(self._vectors)
        position_data.GetPointData().SetActiveVectors("Vector Field")

        # - Add the vector arrays as 3D Glyphs
        arrow_source = vtk.vtkArrowSource()

        add_arrows = vtk.vtkGlyph3D()
        add_arrows.SetInputData(position_data)
        add_arrows.SetSourceConnection(arrow_source.GetOutputPort())
        add_arrows.Update()

        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(add_arrows.GetOutputPort())

        super().__init__(mapper)

        self.add_vectors(positions, vectors)

    def add_vectors(self, positions: np.ndarray, vectors: np.ndarray):
        assert len(positions.shape) == 2
        assert len(vectors.shape) == 2
        assert positions.shape[0] == vectors.shape[0]
        assert positions.shape[1] == 3
        assert vectors.shape[1] == 3

        [num_new_vectors, _] = vectors.shape

        # Allocate additional memory
        self._positions.Resize(self.num_vectors + num_new_vectors)
        self._vectors.Resize(self.num_vectors + num_new_vectors)

        # Add points
        for vector_idx in range(num_new_vectors):
            self._positions.InsertNextPoint(positions[vector_idx, :])
            self._vectors.InsertNextTuple(vectors[vector_idx, :])

        self._positions.Modified()
        self._vectors.Modified()


class VTKVisualization(object):
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.5, 0.5, 0.5)
        self.renderer.ResetCamera()

        axes_actor = vtk.vtkAxesActor()
        axes_actor.AxisLabelsOff()
        self.renderer.AddActor(axes_actor)

        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.SetRenderWindow(self.window)

        self.camera = vtk.vtkCamera()
        self.camera.SetViewUp(0.0, -1.0, 0.0)
        self.camera.SetPosition(0.0, 0.0, -5)
        self.camera.SetFocalPoint(0.0, 0.0, 0.0)
        # self.camera.SetClippingRange(0.0, 100000)

        self.renderer.SetActiveCamera(self.camera)

    def add_entity(self, entity: VTKEntity3D):
        self.renderer.AddActor(entity.actor)

    def add_image(self, image):
        img_mapper = vtk.vtkImageMapper()
        img_actor = vtk.vtkActor2D()
        img_data = vtk.vtkImageData()
        img_data.SetDimensions(image.shape[0], image.shape[1], 1)
        img_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                pixel = img_data.GetScalarPointer(x, y, 0)
                pixel = np.array(image[x, y, :])
        img_mapper.SetInputData(img_data)
        img_mapper.SetColorWindow(255)
        img_mapper.SetColorLevel(127.5)
        img_actor.SetMapper(img_mapper)
        self.renderer.AddActor(img_actor)

    def init(self):
        self.window.Render()
        self.window.SetSize(1200, 800)

        self.interactor.Initialize()

    def show(self):
        self.window.Render()
        self.window.SetSize(1200, 800)

        self.interactor.Start()
