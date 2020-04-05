from bokeh import models
from bokeh import plotting
from bokeh.palettes import Category10
from bokeh.transform import linear_cmap
from bqplot import *
from psm.geometry import polygon_area, point_in_polygon
from psm.graph import stable_delaunay_faces
from psm.representation import order_adjacency_clockwise, faces_to_dual_adjacency, faces_to_edges, \
    faces_to_faces_around_node, faces_to_quadedge
from psm.tools import extract_defects, enclosing_path, reference_path_from_path
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.transform import downscale_local_mean
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


class Playback:

    def __init__(self, num_items=0, **kwargs):
        self._previous_button = models.Button(label='Previous')
        self._next_button = models.Button(label='Next')
        self._slider = models.Slider(start=0, end=1, value=0, step=1)

        self.num_items = num_items

        def next_item(*args):
            self._slider.value = (self._slider.value + 1) % (self._slider.end + 1)

        self._next_button.on_click(next_item)

        def previous_item(*args):
            self._slider.value = (self._slider.value - 1) % (self._slider.end + 1)

        self._previous_button.on_click(previous_item)

    @property
    def num_items(self):
        return self._slider.end + 1

    @num_items.setter
    def num_items(self, value):
        if value < 2:
            value = 2
            self._slider.disabled = True

        self._slider.end = value - 1

    @property
    def value(self):
        return self._slider.value

    @property
    def widgets(self):
        return models.Column(self._slider, models.Row(self._previous_button, self._next_button))


class ViewerExtension:

    def __init__(self, widgets):
        self._widgets = widgets
        self._viewer = None

    def set_viewer(self, viewer):
        self._viewer = viewer

    @property
    def widgets(self):
        return self._widgets


class BinningSlider(ViewerExtension):

    def __init__(self, max=4, value=1):
        self._slider = models.Slider(start=1, end=max, value=value, step=1, title='Binning')
        super().__init__(self._slider)

    def modify_image(self, image):
        return downscale_local_mean(image, (self._slider.value,) * 2)

    def set_viewer(self, viewer):
        self._slider.on_change('value', lambda attr, old, new: viewer.set_image())


class GaussianFilterSlider(ViewerExtension):

    def __init__(self, min=0, max=10, value=0):
        self._slider = models.Slider(start=min, end=max, value=value, step=1, title='Gaussian filter')
        super().__init__(self._slider)

    def modify_image(self, image):
        return gaussian_filter(image, self._slider.value)

    def set_viewer(self, viewer):
        self._slider.on_change('value', lambda attr, old, new: viewer.update_image())


class ClipSlider(ViewerExtension):

    def __init__(self, value=None, step=0.001, title='Clip', **kwargs):
        if value is None:
            value = (0, 1)
        self._slider = models.RangeSlider(start=0, end=1, step=step, value=value, title=title)
        super().__init__(self._slider)

    def modify_image(self, image):
        ptp = image.ptp()
        clip_min = image.min() + self._slider.value[0] * ptp
        clip_max = image.min() + self._slider.value[1] * ptp
        image = np.clip(image, a_min=clip_min, a_max=clip_max)
        return image

    def set_viewer(self, viewer):
        self._slider.on_change('value', lambda attr, old, new: viewer.update_image())


class Annotator(ViewerExtension):

    def __init__(self, widgets):
        super().__init__(widgets)

    def annotate_current_image(self):
        pass

    def set_viewer(self, viewer):
        self._viewer = viewer


class FileAnnotator(Annotator):

    def __init__(self, annotations):
        self._annotations = annotations
        super().__init__()

    def annotate_current_image(self):
        npzfile = np.load(self._annotations[self._viewer._image_files[self._viewer._image_playback.value]])
        return list(npzfile['x']), list(npzfile['y']), list(npzfile['labels'])


def analyse_points(points, labels, shape):
    contamination = points[labels == 1]
    # print(contamination)

    if len(contamination) > 0:
        nbrs = NearestNeighbors(radius=20, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.radius_neighbors(contamination, 20)
        indices = np.unique(np.concatenate(indices))
        labels[indices] = 1

    l = (labels != 1) & (labels != 3)
    lattice_points = points[l]

    faces = stable_delaunay_faces(lattice_points, 2, 50)
    edges = faces_to_edges(faces)

    area = 0
    for face in faces:
        p = lattice_points[face]
        if np.all(p[:,0] > -20) and np.all(p[:,0] < 1044) and np.all(p[:,1] > -20) and np.all(p[:,1] < 1044):
            area += polygon_area(p)

    faces_around_node = faces_to_faces_around_node(faces)
    dopant = np.where(labels[l] == 2)[0]

    dual_points = np.array([lattice_points[face].mean(axis=0) for face in faces])
    dual_degree = np.array([len(face) for face in faces])
    dual_adjacency = order_adjacency_clockwise(dual_points, faces_to_dual_adjacency(faces))

    dual_degree_w_dopant = np.array(dual_degree).copy()

    for i in dopant:
        dual_degree_w_dopant[faces_around_node[i]] = -1

    defects = extract_defects(faces, labels, dual_degree_w_dopant, dual_adjacency)

    quad_edge = faces_to_quadedge(faces)
    outer_adjacent = [faces[0] for faces in quad_edge.values() if len(faces) == 1]
    dual_degree[outer_adjacent] = 6

    column_names = ['include', 'num_missing', 'enclosed_area', 'center_x', 'center_y', 'num_enclosed', 'dopant',
                    'stable', 'contamination', 'closed', 'outside_image']
    defect_data = dict(zip(column_names, [[] for i in range(len(column_names))]))
    polygons = []

    contamination_area = 0
    for i, defect in enumerate(defects):
        path = enclosing_path(dual_points, defect, dual_adjacency)

        if enclosing_path is None:
            continue

        polygon = dual_points[np.concatenate((path, [path[0]]))]
        reference_path = reference_path_from_path(path, dual_adjacency)

        density = 6 / (3 * np.sqrt(3) / 2)
        inside = np.array([point_in_polygon(point, polygon) for point in lattice_points])

        outside = np.ceil(np.abs(min(0., np.min(polygon), np.min(np.array(shape) - polygon))))

        center = np.mean(polygon, axis=0)

        if np.any(center < -20.) or np.any(center > 1044):
            continue

        inside_all = np.array([point_in_polygon(point, polygon) for point in points])
        is_contamination = np.any(labels[inside_all] == 1)

        if is_contamination:
            contamination_area += polygon_area(polygon)
            continue

        polygons.append(polygon)

        defect_data['num_missing'].append(int(np.round(polygon_area(reference_path) * density - np.sum(inside))))
        defect_data['enclosed_area'].append(round(polygon_area(polygon), 3))

        defect_data['center_x'].append(round(center[0], 3))
        defect_data['center_y'].append(round(center[1], 3))
        defect_data['num_enclosed'].append(np.sum(inside))

        defect_data['contamination'].append(is_contamination)
        defect_data['dopant'].append(np.any(labels[inside_all] == 2))
        defect_data['stable'].append(not np.any(labels[inside_all] == 3))

        defect_data['closed'].append(np.all(np.isclose(reference_path[0], reference_path[-1])))
        defect_data['outside_image'].append(outside)

        defect_data['include'].append(defect_data['closed'][-1] * (defect_data['outside_image'][-1] < 10) *
                                      defect_data['stable'][-1] * (not defect_data['contamination'][-1]))

    area_used = area - contamination_area
    #print(np.abs(area_used-1024*1024) / (1024*1024))
    if np.abs(area_used-1024*1024) /(1024*1024) < .04:
        area_used = 1024*1024

    return dict(defect_data), edges, faces, dual_points, dual_degree, polygons


class ModelAnnotator(Annotator):

    def __init__(self, model, path):
        self._model = model

        self._checkbox_button_group = models.CheckboxButtonGroup(labels=['Points',
                                                                         'Dual Points',
                                                                         'Graph'], active=[0, 1, 2])

        def on_change(attr, old, new):
            self._viewer._point_glyph.visible = 0 in self._checkbox_button_group.active
            self._dual_glyph.visible = 1 in self._checkbox_button_group.active
            self._graph_glyph.visible = 2 in self._checkbox_button_group.active

        self._checkbox_button_group.on_change('active', on_change)

        column_names = ['include', 'num_missing', 'enclosed_area', 'center_x', 'center_y', 'num_enclosed', 'dopant',
                        'stable', 'contamination', 'closed', 'outside_image']

        data = dict(zip(column_names, [] * len(column_names)))
        self._table_source = models.ColumnDataSource(data)
        columns = [models.TableColumn(field=name, title=name) for name in column_names]
        self._data_table = models.DataTable(source=self._table_source, columns=columns)

        self._path = path
        self._button = models.Button(label='Write annotation')

        def on_click(event):
            x = self._viewer._point_source.data['x']
            y = self._viewer._point_source.data['y']
            labels = self._viewer._point_source.data['labels']

            image_name = os.path.split(self._viewer._image_files[self._viewer._image_playback.value])[-1]
            write_path = os.path.join(self._path, '.'.join(image_name.split('.')[:-1]) + '.npz')

            data = dict(self._table_source.data)
            np.savez(write_path, x=x, y=y, labels=labels, **data)

        self._button.on_click(on_click)

        super().__init__(models.Column(self._checkbox_button_group, self._button, self._data_table))

    def update_graph(self):
        x = self._viewer._point_source.data['x']
        y = self._viewer._point_source.data['y']
        labels = np.array(self._viewer._point_source.data['labels'])
        points = np.stack((x, y), axis=-1)

        shape = self._viewer._image.shape
        data, edges, dual_points, dual_degree, polygons = analyse_points(points, labels, shape)

        xs = [[edge[0, 0], edge[1, 0]] for edge in edges]
        ys = [[edge[0, 1], edge[1, 1]] for edge in edges]
        self._graph_source.data = {'xs': xs, 'ys': ys}

        self._dual_source.data = {'x': list(dual_points[:, 0]), 'y': list(dual_points[:, 1]),
                                  'labels': list(dual_degree)}

        self._table_source.data = data

        top = [polygon[:, 1].max() for polygon in polygons]
        bottom = [polygon[:, 1].min() for polygon in polygons]
        left = [polygon[:, 0].min() for polygon in polygons]
        right = [polygon[:, 0].max() for polygon in polygons]

        self._rect_source.data = {'top': top, 'bottom': bottom, 'left': left, 'right': right}

    def set_viewer(self, viewer):
        mapper = linear_cmap(field_name='labels', palette=Category10[9], low=0, high=8)

        self._graph_source = models.ColumnDataSource(dict(xs=[], ys=[]))
        self._graph_model = models.MultiLine(xs='xs', ys='ys', line_width=2)
        self._graph_glyph = viewer._figure.add_glyph(self._graph_source, glyph=self._graph_model)

        self._dual_source = models.ColumnDataSource(data=dict(x=[], y=[], labels=[]))
        self._dual_model = models.Circle(x='x', y='y', fill_color=mapper, radius=8)
        self._dual_glyph = viewer._figure.add_glyph(self._dual_source, glyph=self._dual_model)

        self._rect_source = models.ColumnDataSource(data=dict(top=[], bottom=[], left=[], right=[]))
        self._rect_model = models.Quad(top='top', bottom='bottom', left='left', right='right', line_color="#B3DE69",
                                       fill_color=None, line_width=3)
        self._rect_glyph = viewer._figure.add_glyph(self._rect_source, glyph=self._rect_model)

        def on_change(attr, old, new):
            self.update_graph()

        viewer._point_source.on_change('data', on_change)
        self._viewer = viewer

    def annotate(self):
        image = self._viewer._image
        points, labels, marker_mean, marker_variance, a = self._model(image)

        # self._viewer._image_source
        self._viewer._point_source.data = {'x': list(points[:, 0]), 'y': list(points[:, 1]), 'labels': list(labels)}


class LabelEditor(ViewerExtension):

    def __init__(self):
        self._text_input = models.TextInput(value='0', title='')
        self._button = models.Button(label='Label selected')

        def label_selected(event):
            indices = self._viewer._point_source.selected.indices
            new_data = dict(self._viewer._point_source.data)
            for i in indices:
                new_data['labels'][i] = int(self._text_input.value)
            self._viewer._point_source.data = new_data

        self._button.on_click(label_selected)

        def on_change(attr, old, new):
            self._viewer._point_draw_tool.empty_value = int(self._text_input.value)

        self._text_input.on_change('value', on_change)

        super().__init__(models.Row(self._text_input, self._button))


class VisibilityCheckboxes(ViewerExtension):

    def __init__(self):
        self._checkbox_button_group = models.CheckboxButtonGroup(labels=['Image', 'Points', 'Edges'], active=[0, 1, 2])

        def on_change(attr, old, new):
            self._viewer._image_glyph.visible = 0 in self._checkbox_button_group.active
            self._viewer._circle_glyph.visible = 1 in self._checkbox_button_group.active
            self._viewer._multi_line_glyph.visible = 2 in self._checkbox_button_group.active

        self._checkbox_button_group.on_change('active', on_change)

        super().__init__(self._checkbox_button_group)


class WriteAnnotationButton(ViewerExtension):

    def __init__(self, path):
        self._path = path
        self._button = models.Button(label='Write annotation')

        def on_click(event):
            x = self._viewer._circle_source.data['x']
            y = self._viewer._circle_source.data['y']
            labels = self._viewer._circle_source.data['labels']

            image_name = os.path.split(self._viewer._image_files[self._viewer._image_playback.value])[-1]
            write_path = os.path.join(self._path, '.'.join(image_name.split('.')[:-1]) + '.npz')
            np.savez(write_path, x=x, y=y, labels=labels)

        self._button.on_click(on_click)

        super().__init__(self._button)


class Viewer:

    def __init__(self, image_files, extensions=None, annotator=None, ):
        self._image_files = image_files
        self._extensions = extensions
        self._annotator = annotator

        self._figure = plotting.Figure(plot_width=800, plot_height=800, title=None, toolbar_location='below',
                                       tools='lasso_select,box_select,pan,wheel_zoom,box_zoom,reset')

        self._image_source = models.ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self._image_model = models.Image(image='image', x='x', y='y', dw='dw', dh='dh', palette='Greys256')
        self._image_glyph = self._figure.add_glyph(self._image_source, glyph=self._image_model)

        mapper = linear_cmap(field_name='labels', palette=Category10[9], low=0, high=8)

        self._point_source = models.ColumnDataSource(data=dict(x=[], y=[], labels=[]))
        self._point_model = models.Square(x='x', y='y', fill_color=mapper, size=8)

        self._image_playback = Playback(len(image_files))
        self._image_playback._slider.on_change('value_throttled', lambda attr, old, new: self.update())
        self._image_playback._next_button.on_click(lambda event: self.update())
        self._image_playback._previous_button.on_click(lambda event: self.update())

        for extension in self._extensions:
            extension.set_viewer(self)

        if annotator is not None:
            annotator.set_viewer(self)

        self._point_glyph = self._figure.add_glyph(self._point_source, glyph=self._point_model)
        self._point_draw_tool = models.PointDrawTool(renderers=[self._point_glyph], empty_value=0)
        self._figure.add_tools(self._point_draw_tool)

        self.update()

    def update_image(self):

        image = self._image.copy()
        for extension in self._extensions:
            if hasattr(extension, 'modify_image'):
                image = extension.modify_image(image)

        self._image_source.data = {'image': [image],
                                   'x': [0], 'y': [0], 'dw': [image.shape[0]],
                                   'dh': [image.shape[1]]}

    def update(self):
        self._image = downscale_local_mean(imread(self._image_files[self._image_playback.value]), (2, 2))

        self.update_image()

        if self._annotator is not None:
            self._annotator.annotate()

    @property
    def widgets(self):
        extension_widgets = [module.widgets for module in self._extensions]
        column = models.Column(self._image_playback.widgets, *extension_widgets, self._annotator.widgets)
        return models.Row(self._figure, column)
