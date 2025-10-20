import numpy as np
import xarray as xr
import os
from mpas_tools.mesh.mask import compute_mpas_region_masks
from geometric_features import read_feature_collection
import geojson

# ***************************************************************************************

# --------- HELPER FUNCTIONS --------------


def n_to_xr_idx(n):
    """Convert from 1-indexed mesh ID to 0-indexed Python index"""
    return n-1

def xr_to_n_idx(xr):
    """Convert from 0-indexed Python index to 1-indexed mesh ID"""
    return xr+1

def xr_inside_mask_info(mesh,mask):
    """
    Find xr indices of the cells, edges, and vertices in a defined mask.
    This includes the edges and vertices on the border of the mask.

    Parameters
    ----------
    mesh: xarray.core.dataset.Dataset
        Contains information about ocean model grid coordinates.
        
    mask: xarray.core.dataset.Dataset
        Contains RegionCellMasks created from mpas_tools compute_mpas_region_masks

    Returns 
    -------
    xr_cells_inside: numpy.ndarray
        xr indices of the cells inside the mask
        
    xr_edges_inside: numpy.ndarray
        xr indices of the edges inside the mask

    xr_vertices_inside: numpy.ndarray
        xr indices of the vertices inside the mask
    """
    # STEP 1: Select all of the cells inside the mask
    # create mask of cells so that whole cells are included in the mask
    cellmask = mask.regionCellMasks.isel(nRegions=0).astype(bool)
    
    # --------- GET CELLS IN XR COORDINATES --------------
    # apply the mask to the mesh 
    # this returns cells in the xr coordinate
    xr_cells_inside = np.int32(mesh.nCells.where(cellmask,drop=True))
    
    # --------- GET VERTICES IN XR COORDINATES --------------
    
    # we also need all the vertices associated with the cells_inside masked mesh
    # this returns vertices in the n coordinate
    n_vertices_inside = mesh.verticesOnCell.isel(nCells=xr_cells_inside)
    
    # we want the vertices in the xr coordinate
    xr_vertices_inside_raw = n_to_xr_idx(n_vertices_inside)
    
    # remove repeat vertex values, remove -1 values (these were originally 0s and represented "blank" array spaces where there were fewer than 7 vertices
    xr_vertices_inside = np.delete(np.unique(xr_vertices_inside_raw), np.unique(xr_vertices_inside_raw)==-1)
    
    # --------- GET EDGES IN XR COORDINATES --------------
    
    # we also need all the edges associated with the cells_inside masked mesh
    # this returns edges in the n coordinate
    n_edges_inside = mesh.edgesOnCell.isel(nCells=xr_cells_inside)
    
    # we want  the edges in the xr coordinate
    xr_edges_inside_raw = n_to_xr_idx(n_edges_inside)
    
    # remove repeat edge values, remove -1 values (these were originally 0s and represented "blank" array spaces where there were fewer than 7 edges
    xr_edges_inside = np.delete(np.unique(xr_edges_inside_raw), np.unique(xr_edges_inside_raw)==-1)

    return xr_cells_inside, xr_edges_inside, xr_vertices_inside

def distance_on_unit_sphere(lon1, lat1, lon2, lat2, R=6.371e6, method="vincenty"):
    """
    Calculate geodesic arc distance between points (lon1, lat1) and (lon2, lat2).

    PARAMETERS:
    -----------
        lon1 : float
            Start longitude(s), in degrees
        lat1 : float
            Start latitude(s), in degrees
        lon2 : float
            End longitude(s), in degrees
        lat2 : float
            End latitude(s), in degrees
        R : float
            Radius of sphere. Default: 6.371e6 (realistic Earth value). Set to 1 for
            arc distance in radius.
        method : str
            Name of method. Supported methods: ["vincenty", "haversine", "law of cosines"].
            Default: "vincenty", which is the most robust. Note, however, that it still can result in
            vanishingly small (but crucially non-zero) errors; such as that the distance between (0., 0.)
            and (360., 0.) is 1.e-16 meters when it should be identically zero.

    RETURNS:
    --------

    dist : float
        Geodesic distance between points (lon1, lat1) and (lon2, lat2).
    """
    
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.abs(phi2-phi1)
    
    lam1 = np.deg2rad(lon1)
    lam2 = np.deg2rad(lon2)
    dlam = np.abs(lam2-lam1)
    
    if method=="vincenty":
        numerator = np.sqrt(
            (np.cos(phi2)*np.sin(dlam))**2 +
            (np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam))**2
        )
        denominator = np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(dlam)
        arc = np.arctan2(numerator, denominator)
        
    elif method=="haversine":
        arc = 2*np.arcsin(np.sqrt(
            np.sin(dphi/2.)**2 + (1. - np.sin(dphi/2.)**2 - np.sin((phi1+phi2)/2.)**2)*np.sin(dlam/2.)**2
        ))
    
        
    elif method=="law of cosines":
        arc = np.arccos(
            np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(dlam)
        )

    return R * arc

## LAZY ATTEMPTS TO OPEN MESHES (WILL GENERALIZE LATER)

def open_transect_from_alg(mesh):
    # get edge and vertex indices
    LS_lats= np.array([54, 60, 66, 64, 58])   
    LS_lons= np.array([302, 315, 310, 295, 296])
    
    LS_lats = np.append(LS_lats, LS_lats[0])
    LS_lons = np.append(LS_lons, LS_lons[0])
    
    # # calculate transects from algorithm, sort vertices & edges to be in consecutive order
    test_edges, test_verts = calculate_transects_multiple_pts(LS_lons, LS_lats, mesh)
    
    # from the transect, create a mask to capture the entire region specified by the transects
    # this will also output lats and lons corresponding to test_verts
    path = './'
    filename = 'LS_test'
    geojson_file_name = 'Labrador Sea from transect algorithm'
    tags = "Labrador_Sea;Davis_Strait"
    author = "Kaila Uyeda"
    
        
    test_verts_lats, test_verts_lons, dsMasks = transect_from_alg_create_nc(test_verts, 
                                                                                mesh, 
                                                                                path,
                                                                                filename, 
                                                                                geojson_file_name,
                                                                                tags, 
                                                                                author)
    
    # use the dsMasks file to get the ACTUAL working vertices that you will need
    # this eliminates duplicate vertices that would occur if the transect moves back on itself
    # check that the vertices and edges from the mask are actually at the same plaaces...
    # in other words, we have the correct cells for the mask but extra edges and vertices because of how the transect is created.
    # we have to do this after we create a mask with the cells. Otherwise, we don't know what side of the boundary
    # is considered "inside the mask"
    
    alg_edges, alg_vertices = find_and_sort_transect_edges_and_vertices(mesh,dsMasks)

    return alg_edges, alg_vertices, dsMasks

def open_from_mask(mesh):
    # open mask of desired region (this is to find transects from a pre-existing mask)
    path = './'
    filename = 'LS_test_transect_from_mask'
    
    check_nc_existence = os.path.isfile(path + filename + '.nc')
    
    # check if .nc mask file exists
    if check_nc_existence == True:
        print(f'Opening {filename}.nc file as mask')
        mask = xr.open_dataset(path + filename + '.nc')
    else: 
        print('Creating .nc file')
        check_geojson_existence = os.path.isfile(path + filename + '.geojson')
    
        # convert LS_test.geojson to LS_test.nc mask file
        if check_geojson_existence == True:
            print(f'Using {filename}.geojson to create .nc file')
            fcMask = read_feature_collection(path + filename + '.geojson')
            # pool = create_pool(process_count=8)
            pool=None
            dsMasks = compute_mpas_region_masks(mesh, fcMask, maskTypes =('cell',), pool=pool)
            dsMasks.to_netcdf(path + filename + '.nc', format='NETCDF4', mode='w')
            mask = xr.open_dataset(path + filename + '.nc')
            print(f'{filename}.nc created and opened as masks')
        else:
            print(f'{filename}.geojson does NOT exist!')
    
    mask_edges, mask_vertices = find_and_sort_transect_edges_and_vertices(mesh,mask)

    return mask_edges, mask_vertices, mask
    
# ***************************************************************************************

# --------- FUNCTIONS TO GET TRANSECT EDGES AND VERTICES FROM A MASK --------------

def sorted_transect_edges_and_vertices(mesh, xr_mask_transect_edges, xr_mask_transect_vertices):
    """
    Given transect edges and vertices, sort them to be in consecutive order.
    This function is used when transects are created from a mask.
    Includes edges and vertices that border land.
    Called in the find_and_sort_transect_edges_and_vertices function.

    Parameters
    ----------
    mesh: xarray.core.dataset.Dataset
        Contains information about ocean model grid coordinates.
        
    xr_mask_transect_edges: numpy.ndarray
        xr indices of the edges of that define a transect

    xr_mask-transect_vertices: numpy.ndarray
        xr indices of the vertices on the edges the define a transect

    Returns
    -------
    np.int32(next_edges): numpy.ndarray
        xr indices of the edges that define a transect now sorted to be in consecutive order

    np.int32(next_vertices): numpy.ndarray
        xr indices of the edges that define a transect now sorted to be in consecutive order
    """
    
    # ----------- SORT THE EDGES IN XR_MASK_EDGES -----------
    xr_startEdge = np.int32(xr_mask_transect_edges[0])
    n_startVertex = mesh.verticesOnEdge.isel(nEdges=xr_startEdge)[0]
    xr_startVertex = n_to_xr_idx(n_startVertex)
    
    remaining_edges = xr_mask_transect_edges[~np.isin(xr_mask_transect_edges, xr_startEdge)]
    remaining_vertices = xr_mask_transect_vertices[~np.isin(xr_mask_transect_vertices, xr_startVertex)]
    
    next_edges = np.array([xr_startEdge])
    next_vertices = np.array([xr_startVertex])
    
    while len(remaining_edges)>0:
        # from the start vertex, find the edge attached to it s.t. the edge is also part of xr_mask_edges
        n_edgesOnStartVertex = mesh.edgesOnVertex.isel(nVertices = xr_startVertex)
        xr_edgesOnStartVertex = n_to_xr_idx(n_edgesOnStartVertex)
        
        xr_nextEdge = np.intersect1d(xr_edgesOnStartVertex, remaining_edges)
        
        # get the vertex that is not the previous vertex
        n_nextVertices = mesh.verticesOnEdge.isel(nEdges = np.int32(xr_nextEdge))
        xr_nextVertices_raw = n_to_xr_idx(n_nextVertices)
        xr_nextVertices = np.int32(xr_nextVertices_raw)
        
        xr_nextVertex_raw = xr_nextVertices[np.isin(xr_nextVertices, remaining_vertices)]
        xr_nextVertex = np.int32(xr_nextVertex_raw)
        
        # update arrays
        remaining_edges = remaining_edges[remaining_edges != xr_nextEdge]
        remaining_vertices = remaining_vertices[remaining_vertices != xr_nextVertex]
        next_edges = np.append(next_edges, xr_nextEdge)
        next_vertices = np.append(next_vertices, xr_nextVertex)
        
        xr_startVertex = xr_nextVertex
    
    # add the start vertex (which was used twice as the start and end) onto the end as well
    next_vertices = np.append(next_vertices,n_to_xr_idx(n_startVertex)) 
    next_edges = np.append(next_edges, np.int32(xr_mask_transect_edges[0]))

    return np.int32(next_edges), np.int32(next_vertices)
    
# def xr_sorted_transect_edges_and_vertices(mesh,mask):
def find_and_sort_transect_edges_and_vertices(mesh,mask):
    """
    Find vertices and edges that are on the edge of a mask (aka part of the transect). Then sort them to be in consecutive order.
    Calls the sorted_transect_edges_and_vertices function.

    Parameters
    ----------
    mesh: xarray.core.dataset.Dataset
        Contains information about ocean model grid coordinates.

    mask: xarray.core.dataset.Dataset
        Contains RegionCellMasks created from mpas_tools compute_mpas_region_masks

    Returns
    -------
    next_edges: numpy.ndarray
        xr indices of the edges that define a transect now sorted to be in consecutive order

    next_vertices: numpy.ndarray
        xr indices of the edges that define a transect now sorted to be in consecutive order    
    """
    # collect all cells, vertices, and edges in the mask
    xr_cells_inside, xr_edges_inside, xr_vertices_inside = xr_inside_mask_info(mesh,mask)

    # ----- MASK EDGES ON LAND -----
    # find edges where one of the cells on edge is land
    all_edgesOnLand_TWO0 = mesh.nEdges.where(np.isin(mesh.cellsOnEdge.isel(TWO=0),0))
    all_edgesOnLand_TWO1 = mesh.nEdges.where(np.isin(mesh.cellsOnEdge.isel(TWO=1),0))
    all_edgesOnLand = np.union1d(all_edgesOnLand_TWO0, all_edgesOnLand_TWO1)

    # then get all the edges inside the mask
    # xr_edges_inside

    # take the intersection of edges inside the mask and all edges on land
    # give mask edges on land
    mask_edgesOnLand = np.intersect1d(xr_edges_inside, all_edgesOnLand)

    # ----- MASK EDGES ON OPEN OCEAN -----
    # identify cells NOT in the mask
    xr_cells_outside = mesh.nCells[~np.isin(mesh.nCells, xr_cells_inside)]
    n_cells_outside = xr_to_n_idx(xr_cells_outside)

    n_cells_inside = xr_to_n_idx(xr_cells_inside)

    # condition where one of the cells is inside the mask and the other is outside the mask
    # this gives cells on the border of the mask
    
    condition = (np.isin(mesh.cellsOnEdge.isel(TWO=0),n_cells_outside)) & (np.isin(mesh.cellsOnEdge.isel(TWO=1),n_cells_inside)) | \
            (np.isin(mesh.cellsOnEdge.isel(TWO=0),n_cells_inside)) & (np.isin(mesh.cellsOnEdge.isel(TWO=1), n_cells_outside))

    all_edgesOnMask = mesh.nEdges.where(condition)

    # take the intersection of edges that border the mask and the edges inside the mask 
    # (this prevents edges on the border outside of the mask from being counted)
    mask_edgesOnOcean = np.intersect1d(xr_edges_inside, all_edgesOnMask)

    # combine the edges on land in the mask with the edges of the mask in the open ocean
    xr_mask_transect_edges = np.union1d(mask_edgesOnLand, mask_edgesOnOcean)

    # ----------- FIND ALL VERTICES ON EDGES -----------
    n_mask_transect_vertices = mesh.verticesOnEdge.isel(nEdges = np.int32(xr_mask_transect_edges))
    xr_mask_transect_vertices = np.unique(n_to_xr_idx(n_mask_transect_vertices))

    next_edges, next_vertices = sorted_transect_edges_and_vertices(mesh, xr_mask_transect_edges, xr_mask_transect_vertices)

    return next_edges, next_vertices

def transect_from_mask_create_nc(path,filename):
    """
    Open a mask file that was created from a geojson file. Create mask if none exists. 

    Parameters
    ----------
    path: str
        path to file location

    filename: str
        Prefix that filenames will begin with. Convention is location_transect_from_{mask or alg}
        Do not include the ".nc"

    Returns
    -------
    mask: xarray.core.dataset.Dataset
        Contains RegionCellMasks created from mpas_tools compute_mpas_region_masks   
    """
    
    check_nc_existence = os.path.isfile(path + filename + '.nc')
    
    # check if .nc mask file exists
    if check_nc_existence == True:
        print(f'Opening {filename}.nc file as mask')
        mask = xr.open_dataset(path + filename + '.nc')
    else: 
        print('Creating .nc file')
        check_geojson_existence = os.path.isfile(path + filename + '.geojson')
    
        # convert LS_test.geojson to LS_test.nc mask file
        if check_geojson_existence == True:
            print(f'Using {filename}.geojson to create .nc file')
            fcMask = read_feature_collection(path + filename + '.geojson')
            pool = create_pool(process_count=8)
            dsMasks = compute_mpas_region_masks(mesh, fcMask, maskTypes =('cell',), pool=pool)
            dsMasks.to_netcdf(path + filename + '.nc', format='NETCDF4', mode='w')
            mask = xr.open_dataset(path + filename + '.nc')
            print(f'{filename}.nc created and opened as masks')
        else:
            print(f'{filename}.geojson does NOT exist!')

    return mask


# ***************************************************************************************

# --------- FUNCTIONS TO GET TRANSECT EDGES AND VERTICES FROM AN ALGORITHM --------------

# function to calculate transect given a target start point, target end point, and mesh
def calculate_transects(target_start_lat, target_start_lon, target_end_lat, target_end_lon, mesh):
    """
    Calculate transects given a defined target start and end point using a nearest-neighbors algorithm.
    Includes edges and vertices that border land.
    Called in calculate_transects_multiple_pts

    PARAMETERS:
    -----------
        target_start_lat : float
            Target start latitude, in degrees
        target_start_lon : float
            Target start longitude, in degrees
        target_end_lat : float
            Target end latitude, in degrees
        target_end_lon : float
            Target end longitude, in degrees
        mesh: xarray.core.dataset.Dataset
            Mesh dataset containing lat/lon Cell/Edge/Vertex
            
    RETURNS:
    -----------
        next_vertices: np.ndarray
            xr indices of nVertices that define the transect
        xr_transect_edges: np.ndarray
            xr indices of edges that define the transect
        
    """
    # ---------- INITIATE START VERTEX ----------------
    # of these transect cells, select the one that is closest to the desired starting point.
    # desired values in deg
    # distance_on_unit_sphere(lon1, lat1, lon2, lat 2)
    # find the shortest path between the two points
    # of all of the points, find the vertex that is closest to the desired start point
    distance = distance_on_unit_sphere(mesh.lonVertex * 180/np.pi, mesh.latVertex * 180/np.pi, target_start_lon, target_start_lat)
    xr_start_vertex = distance.argmin()
    n_start_vertex = xr_to_n_idx(xr_start_vertex)
    
    # repeat to find the vertex that is closest to the desired end point
    dist_to_end = distance_on_unit_sphere(mesh.lonVertex * 180/np.pi, mesh.latVertex * 180/np.pi, target_end_lon, target_end_lat)

    # get the vertex closest to the target end lat and lon
    xr_end_vertex = dist_to_end.argmin()
    end_lon = mesh.isel(nVertices = xr_end_vertex).lonVertex * 180/np.pi
    end_lat = mesh.isel(nVertices = xr_end_vertex).latVertex * 180/np.pi
    
    # get an array of the start and end points (this is useful if transects are broken up by land)
    xr_start_end_vertices = np.array([xr_start_vertex, xr_end_vertex])
    
    # ---------- FIND NEXT VERTEX ----------------
    start_vertices = np.array([])
    next_vertices = np.array([])
    
    while distance.min() > 10000:
        # get the edges attached to the start vertex
        n_edgesOnStartVertex = mesh.edgesOnVertex.isel(nVertices = xr_start_vertex)
        xr_edgesOnStartVertex = n_to_xr_idx(n_edgesOnStartVertex)
        
        # check that the edges you selected are connected to the start vertex (returns in n indices)
        # mesh.verticesOnEdge.isel(nEdges = xr_edgesOnStartVertex[0])
        
        # for each of these edges, find the vertices they are connected to and then remove the start_vertex (we don't want to "travel back" to that vertex)
        n_vertices_nextToStartVertex = np.unique(mesh.verticesOnEdge.isel(nEdges = np.int32(xr_edgesOnStartVertex)))
        xr_vertices_nextToStartVertex = n_to_xr_idx(n_vertices_nextToStartVertex)
        # print(xr_vertices_nextToStartVertex)
    
        used_vertices = np.union1d(start_vertices, xr_start_vertex)
        
        xr_vertices_nextToStartVertex_Use = np.delete(xr_vertices_nextToStartVertex, np.where(np.isin(xr_vertices_nextToStartVertex, used_vertices)))
        # print(xr_vertices_nextToStartVertex_Use)
        # calculate the distance from these new vertices to the desired end point
            # retrieve the lat and lon of the vertex
        ds_vertices_nextLatLon = mesh[['lonVertex','latVertex']].where(mesh.nVertices.isin(xr_vertices_nextToStartVertex_Use))
        ds_vertices_nextLatLon['lonVertex'] = ds_vertices_nextLatLon.lonVertex * 180 / np.pi
        ds_vertices_nextLatLon['latVertex'] = ds_vertices_nextLatLon.latVertex * 180 / np.pi
        
        # calculate the distance between the next vertices and the target end
        distance = distance_on_unit_sphere(ds_vertices_nextLatLon.lonVertex, ds_vertices_nextLatLon.latVertex, target_end_lon, target_end_lat)
        
        # select the nVertex that is the shortest distance from the end point
        xr_chosen_nextVertex = distance.argmin()
        
        # ---------- UPDATE ARRAYS ----------------
        # store vertices
        start_vertices = np.append(start_vertices, xr_start_vertex)
        next_vertices = np.append(next_vertices, xr_chosen_nextVertex)
    
        xr_start_vertex = xr_chosen_nextVertex 

        # break code if the next vertex is the vertex closest to the target end lat and lon
        if xr_chosen_nextVertex == xr_end_vertex:
            break

    # ---------- FIND EDGES OF TRANSECT ---------------- 
    # We want to identify the edges that connect the vertices. The vertices are already ordered consecutively (because the transects are built from an algorithm)
    # We will take advantage of this fact using a for loop to extract the edges that are shared between vertices next to each other
    
    # modify next_vertices to also include the start vertex
    next_vertices = np.insert(next_vertices, 0, n_to_xr_idx(n_start_vertex))
    
    
    # next vertices are in xr indices
    int_next_vertices = np.int32(next_vertices)
    n_transect_edges = np.array([])
    
    for i in range(0,len(int_next_vertices)-1):
        edgesOnVertex0 = mesh.edgesOnVertex.isel(nVertices = int_next_vertices[i]).values
        edgesOnVertex1 = mesh.edgesOnVertex.isel(nVertices = int_next_vertices[i+1]).values
        shared_edge = np.intersect1d(edgesOnVertex0, edgesOnVertex1)
        n_transect_edges = np.append(n_transect_edges, shared_edge)
    
    xr_transect_edges = n_to_xr_idx(n_transect_edges)

        
    
    return xr_transect_edges, next_vertices


        
# calculate transects using multiple points

def calculate_transects_multiple_pts(segment_lons,segment_lats,mesh):
    """
    Calculate transects given the longitude and latitude vertices in a polygon.
    Calls calculate_transects

    Parameters
    ----------
    segment_lons: numpy.ndarray
        Longitude, in degrees, of consecutive vertices making up a polygon

    segment_lats: numpy.ndarray
        Latitude, in degrees, of consecutive vertices making up a polygon

    mesh: xarray.core.dataset.Dataset
        Contains information about ocean model grid coordinates.

    Returns
    -------
    all_xr_transect_vertices: numpy.ndarray
        xr indices of vertices in transect, sorted in consecutive order

    all_xr_transect_edges: numpy.ndarray
        xr indices of edges in transect, sorted in consecutive order
    """
    all_xr_transect_vertices = np.array([])
    all_xr_transect_edges = np.array([])

    for i in range(0,len(segment_lons)-1):
        
        # set start and end target points based on segment lons and lats
        target_start_lat = segment_lats[i]
        target_start_lon = segment_lons[i]

        target_end_lat = segment_lats[i+1]
        target_end_lon = segment_lons[i+1]
        
        xr_transect_edges_segment, xr_next_vertices = calculate_transects(target_start_lat, target_start_lon, target_end_lat, target_end_lon, mesh)

        # update all_xr_transect_ arrays
        all_xr_transect_vertices = np.concatenate((all_xr_transect_vertices, xr_next_vertices))
        all_xr_transect_edges = np.concatenate((all_xr_transect_edges, xr_transect_edges_segment))

    return all_xr_transect_edges, all_xr_transect_vertices
        

# get a .nc and .geojson mask from the region bordered by the transects created by the algorithm
def transect_from_alg_create_nc(test_verts,mesh,path,filename,geojson_file_name,tags,author):
    """
    Get a .nc and .geojson mask from the region bordered by the transects created by the nearest-neighbors algorithm. 
    ** NOTE ** 
    The returned .nc file may contain extra vertices and edges that do not cells in the mask (i.e., borders land)
    The post-processing step of calling the find_and_sort_transect_edges_and_vertices function must be used because it explicitly defines the mask boundaries based on cellsOnEdge (see "condition =" in find_and_sort_transect_edges_and_vertices function)

    Parameters
    ----------
    test_verts: numpy.ndarray
        Initial xr indices of vertices in transect (may contain duplicate vertex indices)

    mesh: xarray.core.dataset.Dataset
        Contains information about ocean model grid coordinates.

    path: str
        Path to desired file location

    filename: str
        Prefix that filenames will begin with. Convention is location_transect_from_{mask or alg}

    geojson_file_name: str
        Longer name/description of mask. Convention is "{Location} from transect {mask or alg}"

    tags: str
        geojson location tags

    author: str
        Author name

    Returns
    -------
    test_verts_lats: xarray.core.dataarray.DataArray
        Latitudes, in degrees, of vertices in the transect

    test_verts_lons: xarray.core.dataarray.DataArray
        Longitudes, in degrees, of vertices in the transect

    dsMasks: xarray.core.dataset.Dataset
        Contains RegionCellMasks created from mpas_tools compute_mpas_region_masks
    
    """
    
    # get the lats and lons of the test_verts to use for creation of a geojson file
    test_verts_lats = mesh.latVertex.isel(nVertices = np.int32(test_verts)) * 180 / np.pi 
    test_verts_lons = mesh.lonVertex.isel(nVertices = np.int32(test_verts)) * 180 / np.pi - 360
    
    test_verts_lonslats = np.array([test_verts_lons,test_verts_lats]).T
    list_test_verts_lonslats = test_verts_lonslats.tolist()

    # ----------- CREATE GEOJSON FILE -----------
    # check if the geojson mask file created from a transect algorithm exists
    alg_filename = filename + '_transect_from_alg'
    
    check_alg_geojson_existence = os.path.isfile(path + alg_filename + '.geojson')
    
    if check_alg_geojson_existence == True:
        print(f'{alg_filename}.geojson exists!')
    else:
        print('Creating geojson file from vertices identified with transect algorithm')
        
        transect_from_alg = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties":{
                    "name":geojson_file_name,
                    "tags":"Labrador_Sea;Davis_Strait",
                    "object":"region",
                    "component":"ocean",
                    "author":"Kaila Uyeda"
                },
                "geometry": {
                    "coordinates":[list_test_verts_lonslats],
                    "type": "Polygon"
                }
            }]
        }
    
        # save to a geojson file
        # with = open and then close
        # w = write mode
        
        with open(path + f'{alg_filename}.geojson','w') as f:
            geojson.dump(transect_from_alg, f, indent=2)

    # ----------- CREATE NETCDF FILE -----------
    
    # check if the .nc mask file created from a transect algorithm exists
    check_alg_nc_existence = os.path.isfile(path + alg_filename + '.nc')
    check_alg_nc_existence == False
    
    if check_alg_nc_existence == True:
        print(f'Opening {alg_filename}.nc as dsMasks')
        dsMasks = xr.open_dataset(path + alg_filename + '.nc')
    else:
        print('Creating netcdf mask file from geojson file (vertices identified from transect algorithm)')
        fcMask = read_feature_collection(path + alg_filename + '.geojson')
        # pool = create_pool(process_count=8)
        pool = None
        dsMasks = compute_mpas_region_masks(mesh, fcMask, maskTypes=('cell',), pool=pool)
        dsMasks.to_netcdf(path + alg_filename + '.nc', format='NETCDF4', mode='w')
        dsMasks = xr.open_dataset(path + alg_filename + '.nc')
        print(f'{alg_filename}.nc created and opened as dsMasks')

    return test_verts_lats, test_verts_lons, dsMasks























