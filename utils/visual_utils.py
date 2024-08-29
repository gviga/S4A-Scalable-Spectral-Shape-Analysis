import plotly.graph_objects as go
import numpy as np
def plot_mesh(mesh,color=None, colormap='Sunset',reverse=True):
    '''
    Input:  v: vertices of the mesh
            f: faces
            colormap: type of coloration
    '''
    shape=mesh.vertices
    face=mesh.faces
    x, y, z = shape[:,0],shape[:,1],shape[:,2]
    f1,f2,f3= face[:,0], face[:,1], face[:,2]
    #project the error on the lbo basis
    fig = go.Figure(data=[go.Mesh3d(x=x,y=y,z=z, i=f1, j=f2, k=f3,
                                    intensity = color,  # set color to an array/list of desired values
                                    colorscale=colormap,
                                    reversescale = reverse,
                                    opacity=1#
    )])
    fig.show()

def plot_pointcloud(shape,color=None, colormap='Sunset',reverse=True):
    
    """
    Visualize a 3D point cloud with optional coloring using Plotly.

    This function takes a 3D point cloud and optionally a color array, then 
    generates an interactive 3D scatter plot using Plotly. The points can be 
    colored based on the provided color array, and various colormap options 
    are available.

    Parameters:
    ----------
    shape : numpy.ndarray
        A 2D array with shape (n, 3) representing the 3D coordinates of the 
        point cloud, where n is the number of points.
    color : array-like, optional
        An array of values used to color the points. The length of this array 
        should match the number of points in the point cloud. If None, the 
        points will not be colored based on any values.
    colormap : str, optional
        The colormap to be used for coloring the points. Default is 'Sunset'.
    reverse : bool, optional
        Whether to reverse the colormap. Default is True.

    Returns:
    -------
    None
        This function does not return any value. It creates and shows an 
        interactive Plotly plot in the default web browser.

    Example:
    -------
    >>> import numpy as np
    >>> shape = np.random.rand(100, 3)
    >>> color = np.random.rand(100)
    >>> plot_pointcloud(shape, color)
    """

    x, y, z = shape[:,0],shape[:,1],shape[:,2]

    #project the error on the lbo basis
    fig = go.Figure(data=[go.Scatter3d(x=x,y=y,z=z,
                                    mode='markers',
                                    marker=dict(
                                    color=color,
                                    size=2,
                                    colorscale=colormap,
                                    reversescale = reverse,
                                    opacity=1#
                                    ),
    )])
    fig.show()


def pick_points(pointcloud):

    """
    Visualize a 3D point cloud using Plotly and display point indices on hover.

    This function takes a point cloud (a list of 3D coordinates) as input and 
    generates an interactive 3D scatter plot using Plotly. The plot displays 
    each point in the cloud with markers, colored according to their z-coordinate.
    When hovering over a point, its index within the input list is shown.

    Parameters:
    ----------
    pointcloud : list of tuple
        A list of 3D coordinates, where each coordinate is a tuple (x, y, z).

    Returns:
    -------
    None
        This function does not return any value. It creates and shows an 
        interactive Plotly plot in the default web browser.

    Example:
    -------
    >>> pointcloud = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    >>> pick_points(pointcloud)
    """

    # Extract x, y, z coordinates from the point cloud
    x, y, z = zip(*pointcloud)

    # Create a scatter plot
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=z,  # Use z-coordinate for color gradient
            colorscale='Viridis',  # Choose a colorscale
            opacity=0.8
        )
    )

    # Create layout
    layout = go.Layout(scene=dict(aspectmode="data"))

    # Create figure
    fig = go.Figure(data=[scatter], layout=layout)

    # Add hover info with point indices
    hover_text = [f'Index: {index}' for index in range(len(pointcloud))]
    fig.data[0]['text'] = hover_text

    # Show the interactive plot
    
    fig.show()



def visu(vertices):
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return cmap
