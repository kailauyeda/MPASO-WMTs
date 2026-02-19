import numpy as np
import xarray as xr
import numpy as np

import mpasregions.sections as mps


# create dataset

lonCells = np.array([180.,  90., 270.])
latCells = np.array([-36.,  36.,  36.])

lonEdges = np.array([135., 235.,  90., 270.,  45., 135., 225., 315.,   0., 180., 45., 135., 225., 315.])
latEdges = np.array([-72., -72., -36., -36.,   0.,   0.,   0.,   0.,  36.,  36., 72.,  72.,  72.,  72.])

lonVertices = np.array([180.,  90., 270.,  90., 270.,   0., 180.,   0., 180., 90., 270.])
latVertices = np.array([-90., -54., -54., -18., -18.,  18.,  18.,  54.,  54., 90.,  90.])

ds = xr.Dataset({},coords={
    'nCells':xr.DataArray(np.arange(0,len(lonCells)), dims='nCells'),
    'nEdges': xr.DataArray(np.arange(0,len(lonEdges)), dims='nEdges'),
    'nVertices':xr.DataArray(np.arange(0,len(lonVertices)), dims='nVertices')
})

ds['lonCell'] = xr.DataArray(np.deg2rad(lonCells), dims='nCells')
ds['latCell'] = xr.DataArray(np.deg2rad(latCells), dims='nCells')
ds['lonEdge'] = xr.DataArray(np.deg2rad(lonEdges), dims='nEdges')
ds['latEdge'] = xr.DataArray(np.deg2rad(latEdges), dims='nEdges')
ds['lonVertex'] = xr.DataArray(np.deg2rad(lonVertices), dims='nVertices')
ds['latVertex'] = xr.DataArray(np.deg2rad(latVertices), dims='nVertices')

xr_edgesOnVertex =  np.array([
    [0,1,np.nan], # on vertex 0
    [0,2,np.nan], # on vertex 1
    [1,3,np.nan], # on vertex 2
    [2,4,5], # on vertex 3
    [3,6,7], # on vertex 4
    [4,7,8], # on vertex 5
    [5,6,9], # on vertex 6
    [8,10,13], # on vertex 7
    [9,11,12], # on vertex 8
    [10,11,np.nan], # on vertex 9
    [12,13,np.nan] # on vertex 10
])

n_edgesOnVertex = mps.xr_to_n_idx(xr_edgesOnVertex)
n_edgesOnVertex[np.isnan(n_edgesOnVertex)] = 0 # extra vertices that were once nans are now zeros

ds['edgesOnVertex'] = xr.DataArray(np.int32(n_edgesOnVertex), dims = ('nVertices','vertexDegree'))

xr_verticesOnEdge = np.array([
    [0,1], # on edge 0
    [0,2], # on edge 1
    [1,3], # on edge 2
    [2,4], # on edge 3
    [3,5], # on edge 4
    [3,6], # on edge 5
    [4,6], # on edge 6
    [4,5], # on edge 7
    [5,7], # on edge 8
    [6,8], # on edge 9
    [7,9], # on edge 10
    [8,9], # on edge 11
    [8,10], # on edge 12
    [7,10] # on edge 13
    
])

n_verticesOnEdge = mps.xr_to_n_idx(xr_verticesOnEdge)
ds['verticesOnEdge'] = xr.DataArray(np.int32(n_verticesOnEdge), dims=('nEdges', 'TWO'))

xr_cellsOnEdge = np.array([
    [0,np.nan], # on edge 0
    [0,np.nan], # on edge 1
    [0,np.nan], # on edge 2
    [0,np.nan], # on edge 3
    [1,np.nan], # on edge 4
    [0,1],  # on edge 5
    [0,2],  # on edge 6
    [2,np.nan], # on edge 7
    [1,2],  # on edge 8
    [1,2],  # on edge 9
    [1,np.nan], # on edge 10
    [1,np.nan], # on edge 11
    [2,np.nan], # on edge 12
    [2,np.nan] #on edge 13
])

n_cellsOnEdge = mps.xr_to_n_idx(xr_cellsOnEdge)
n_cellsOnEdge[np.isnan(n_cellsOnEdge)] = 0
ds['cellsOnEdge'] = xr.DataArray(np.int32(n_cellsOnEdge), dims=('nEdges', 'TWO'))

xr_verticesOnCell = np.array([
                    [0,1,3,6,4,2,np.nan], # on cell 0
                    [3,5,7,9,8,6,np.nan], # on cell 1
                    [4,6,8,10,7,5,np.nan] # on cell 2
])

n_verticesOnCell = mps.xr_to_n_idx(xr_verticesOnCell)
n_verticesOnCell[np.isnan(n_verticesOnCell)] = 0
ds['verticesOnCell'] = xr.DataArray(np.int32(n_verticesOnCell), dims=('nCells','maxEdges'))

xr_edgesOnCell = np.array([
                [0,2,5,6,3,1,np.nan], # on cell 0
                [4,8,10,11,9,5,np.nan], # on cell 1
                [6,9,12,13,8,7,np.nan] # on cell 2
])

n_edgesOnCell = mps.xr_to_n_idx(xr_edgesOnCell)
n_edgesOnCell[np.isnan(n_edgesOnCell)] = 0
ds['edgesOnCell'] = xr.DataArray(np.int32(n_edgesOnCell), dims=('nCells','maxEdges'))

# 0 thickness is like a nan. represents a land cell.
xr_layerThickness = np.array([
                    [1, 1, 0], # cell 0. 1 at surface, 0 at bottom
                    [0.5,0,0], # cell 1. 0.5 at surface, 0 at bottom
                    [1, 1, 0] # cell 2. 1 at surface, 0 at bottom
])

ds['timeMonthly_avg_layerThickness'] = xr.DataArray(xr_layerThickness, dims=('nCells','nVertLevels'))

# add time
ds['xtime_startMonthly'] = xr.DataArray(np.array([b'0063-12-01_00:30:00'], dtype='|S64'), dims='Time')
ds = ds.swap_dims({'Time':'xtime_startMonthly'})

# add normal velocity
xr_normalVelocity = np.array([[
    [0,0,0], # on edge 0. 0 for all levels
    [0,0,0], # on edge 1. 0 for all levels
    [0,0,0], # on edge 2. 0 for all levels
    [0,0,0], # on edge 3. 0 for all levels
    [0,0,0], # on edge 4. 0 for all levels
    [-1,0,0], # on edge 5. from cell 1 (B) to cell 0 (A) at surface.
    [0.75,0.5,0], # on edge 6. from cell 0 (A) to cell (B) at surface (0.75) and level 2 (0.5)
    [0,0,0], # on edge 7. 0 for all levels
    [-1,0,0], # on edge 8. from cell 2 (B) to cell 1 (A) at surface
    [0,0,0], # on edge 9. 0 for all levels
    [0,0,0], # on edge 10. 0 for all levels
    [0,0,0], # on edge 11. 0 for all levels
    [0,0,0], # on edge 12. 0 for all levels
    [0,0,0] # on edge 13. 0 for all levels.
]])

ds['timeMonthly_avg_normalVelocity'] = xr.DataArray(xr_normalVelocity, dims=('xtime_startMonthly','nEdges','nVertLevels'))

# add dvEdge (calculated as the distance between vertices)
dvEdge = np.array([188079.69356148, 188079.69356148,  69865.8329326 ,  69865.8329326 ,
       188118.69974708, 188118.69974708, 188118.69974708, 528628.34731413,
        69865.8329326 ,  69865.8329326 , 188079.69356148, 188079.69356148,
       188079.69356148, 528503.3555391 ])

ds['dvEdge'] = xr.DataArray(dvEdge, dims=('nEdges'))

ds['VertexID'] = mps.xr_to_n_idx(ds.nVertices)
ds = ds.assign_coords({'VertexID': mps.xr_to_n_idx(ds.nVertices)})


# test transport creation, finding, sorting functions

# **** TEST TRANSECT CALCULATION ALGORTHMS ****
# test transect calculation functions ```calculate_transect``` and ```calculate_transect_multiple_pts```

def test_calculate_transects():
    from mpasregions.sections import calculate_transects
    # test if basic algorithm works
    target_start_lon, target_start_lat = 90, -30
    target_end_lon, target_end_lat = 180, 60
    test_edges, test_vertices = calculate_transects(target_start_lat, target_start_lon, target_end_lat, target_end_lon, ds)
    assert (test_vertices == np.array([3,6,8])).all()
    assert(test_edges == np.array([5,9])).all()

    # test periodicity in x-direction
    target_start_lon, target_start_lat = 0, 18
    target_end_lon, target_end_lat = 270, 18
    test_edges, test_vertices = calculate_transects(target_start_lat, target_start_lon, target_end_lat, target_end_lon, ds)
    assert (test_vertices == np.array([5,4])).all()
    assert(test_edges == 7)

    # test periodicity in y-direction
        # the transect should move south to north (90,-54) does not connect to (90,54) directly.
    target_start_lon, target_start_lat = 90, -54
    target_end_lon, target_end_lat = 90, 90
    test_edges, test_vertices = calculate_transects(target_start_lat, target_start_lon, target_end_lat, target_end_lon, ds)
    assert(test_vertices == np.array([1,3,5,7,9])).all()
    assert(test_edges == np.array([2,4,8,10])).all()

def test_calculate_transects_multiple_pts():
    from mpasregions.sections import calculate_transects_multiple_pts
    # calculate_transects_multiple_pts(segment_lons,segment_lats,ds)

    # test if basic algorithm works
    target_lons = np.array([90,90,180])
    target_lats = np.array([-18,90,18])
    test_edges, test_vertices = calculate_transects_multiple_pts(target_lons, target_lats, ds)
    assert(test_vertices == np.array([3,5,7,9,8,6])).all()
    assert(test_edges == np.array([4,8,10,11,9,5])).all()

    # test periodicity in x-direction
    target_lons = np.array([270,360,180]) 
    target_lats = np.array([0,0,0])
    test_edges, test_vertices = calculate_transects_multiple_pts(target_lons, target_lats, ds)
    assert(test_vertices == np.array([4,5,3,6])).all()
    assert(test_edges == np.array([7,4,5,6])).all()

    # test periodicity in y-direction
    target_lons = np.array([90,180,270])
    target_lats = np.array([-54,-90,90])
    test_edges, test_vertices = calculate_transects_multiple_pts(target_lons, target_lats, ds)
    assert(test_vertices == np.array([1,0,1,3,5,7,10,7,5,3])).all()


# test mask-creating function ```test_transect_from_alg_create_nc```


def test_transect_from_alg_create_nc():
    from mpasregions.sections import transect_from_alg_create_nc
    
    # test basic algorithm
    test_vertices = np.array([3, 5, 7, 9, 8, 6])
    path = './'
    filename = 'test_basic_alg'
    geojson_file_name = 'basic test if transect_from_alg_create_nc works'
    tags = 'unstructured_hexagonal_grid;test_basic'
    author = 'Kaila Uyeda'
    geojson_vertexLats, geojson_vertexLons, dsMasks = transect_from_alg_create_nc(test_vertices, ds, path, filename, geojson_file_name, tags, author)
    assert(geojson_vertexLats == np.array([-18.,  18.,  54.,  90.,  54.,  18.])).all()
    assert(geojson_vertexLons == np.array([ 90.,   0.,   0.,  90., 180., 180.])).all()
           
    # test transects that cross over themselves
    # geojson file should maintain vertices and edges that are not connected to cells inside the desired region
    # mask that is created should only include cells inside the desired region
    test_vertices = np.array([9, 8, 6, 4, 6, 3, 5, 7])
    path = './'
    filename = 'test_crossing_transect'
    geojson_file_name = 'test if geojson file keeps vertices from transect'
    tags = 'unstructured_hexagonal_grid;test_crossing_transect'
    author = 'Kaila Uyeda'
    geojson_vertexLats, geojson_vertexLons, dsMasks = transect_from_alg_create_nc(test_vertices, ds, path, filename, geojson_file_name, tags, author)
    assert(geojson_vertexLats == np.array([ 90.,  54.,  18., -18.,  18., -18.,  18.,  54.])).all()
    assert(geojson_vertexLons == np.array([ 90., 180., 180., 270., 180.,  90.,   0.,   0.])).all()
    assert(dsMasks.isel(nRegions=0).regionCellMasks.values == np.array([0,1,0])).all()


# test ```find_transect_edges_and_vertices``` function


def test_find_transect_edges_and_vertices():
    from mpasregions.sections import find_transect_edges_and_vertices

    # test basic algorithm around a single cell
    # should return edges and vertices that surround that cell
    regionCellMasks_array = np.int32(np.array([[0],[1],[0]]))
    ds_mask = xr.Dataset(
        data_vars = dict(
            regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
            regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
    )) 
    xr_mask_transect_edges, xr_mask_transect_vertices = find_transect_edges_and_vertices(ds, ds_mask)
    assert(xr_mask_transect_edges == np.array([ 4,  5,  8,  9, 10, 11])).all()
    assert(xr_mask_transect_vertices == np.array([3, 5, 6, 7, 8, 9])).all()

    # test basic algorithm around multiple cells
    # edge between the two cells should not be included
    regionCellMasks_array = np.int32(np.array([[1],[1],[0]]))
    ds_mask = xr.Dataset(
        data_vars = dict(
            regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
            regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
    ))
    xr_mask_transect_edges, xr_mask_transect_vertices = find_transect_edges_and_vertices(ds, ds_mask)
    assert(xr_mask_transect_edges == np.array([ 0,  1,  2,  3,  4,  6,  8,  9, 10, 11])).all()
    assert(xr_mask_transect_vertices == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).all()

    # test basic algorithm that requires crossing a periodic boundary (cells on either side of the boundary are included in the mask)
    # edge between the two cells that border the boundary (0/360) should not be included
    regionCellMasks_array = np.int32(np.array([[0],[1],[1]]))
    ds_mask = xr.Dataset(
        data_vars = dict(
            regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
            regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
    ))
    xr_mask_transect_edges, xr_mask_transect_vertices = find_transect_edges_and_vertices(ds, ds_mask)
    assert(xr_mask_transect_edges == np.array([ 4,  5,  6,  7, 10, 11, 12, 13])).all()
    assert(xr_mask_transect_vertices == np.array([ 3,  4,  5,  6,  7,  8,  9, 10])).all()



# test ```sorted_transect_edges_and_vertices``` function

def test_sorted_transect_edges_and_vertices():
    from mpasregions.sections import sorted_transect_edges_and_vertices
    # test basic algorithm (vertices and edges around a single cell)
    xr_mask_transect_edges = np.array([ 4,  5,  8,  9, 10, 11])
    xr_mask_transect_vertices = np.array([3, 5, 6, 7, 8, 9])
    sorted_edges, sorted_vertices = sorted_transect_edges_and_vertices(ds, xr_mask_transect_edges, xr_mask_transect_vertices)
    assert(sorted_edges == np.array([ 4,  5,  9, 11, 10,  8])).all()
    assert(sorted_vertices == np.array([3, 6, 8, 9, 7, 5])).all()
    
    # test basic algorithm aroudn 2 cells
    xr_mask_transect_edges = np.array([ 0,  1,  2,  3,  4,  6,  8,  9, 10, 11])
    xr_mask_transect_vertices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    sorted_edges, sorted_vertices = sorted_transect_edges_and_vertices(ds, xr_mask_transect_edges, xr_mask_transect_vertices)
    assert(sorted_edges == np.array([ 0,  1,  3,  6,  9, 11, 10,  8,  4,  2])).all()
    assert(sorted_vertices == np.array([0, 2, 4, 6, 8, 9, 7, 5, 3, 1])).all()
    
    # test algorithm for mask bounded with complete lines of latitude
    # should have a broken transect that "switches" from the lower boundary to the upper boundary
    xr_mask_transect_edges = np.array([ 4,  5,  6,  7, 10, 11, 12, 13])
    xr_mask_transect_vertices = np.array([ 3,  4,  5,  6,  7,  8,  9, 10])
    sorted_edges, sorted_vertices = sorted_transect_edges_and_vertices(ds, xr_mask_transect_edges, xr_mask_transect_vertices)
    assert(sorted_edges == np.array([ 4,  5,  6,  7, 10, 13, 12, 11])).all()
    assert(sorted_vertices == np.array([ 3,  6,  4,  5,  7, 10,  8,  9])).all()



# ### test combined find and sort functions ```find_and_sort_transect_edges_and_vertices```

def test_find_and_sort_transect_edges_and_vertices():
    from mpasregions.sections import find_and_sort_transect_edges_and_vertices
    # test basic algorithm 
    # mask around a single cell
    regionCellMasks_array = np.int32(np.array([[0],[1],[0]]))
    ds_mask = xr.Dataset(
            data_vars = dict(
                regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
                regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
        ))
    next_edges, next_vertices = find_and_sort_transect_edges_and_vertices(ds, ds_mask)
    assert(next_edges == np.array([ 4,  5,  9, 11, 10,  8])).all()
    assert(next_vertices == np.array([3, 6, 8, 9, 7, 5])).all()

    # test basic algorithm
    # mask around two cells
    regionCellMasks_array = np.int32(np.array([[1],[1],[0]]))
    ds_mask = xr.Dataset(
        data_vars = dict(
            regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
            regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
    ))
    next_edges, next_vertices = find_and_sort_transect_edges_and_vertices(ds, ds_mask)
    assert(next_edges == np.array([ 0,  1,  3,  6,  9, 11, 10,  8,  4,  2])).all()
    assert(next_vertices == np.array([0, 2, 4, 6, 8, 9, 7, 5, 3, 1])).all()

    # test algorithm for mask bounded with complete lines of latitude
    # should have a broken transect that "switchees from the lower boundary to the upper boundary
    regionCellMasks_array = np.int32(np.array([[0],[1],[1]]))
    ds_mask = xr.Dataset(
        data_vars = dict(
            regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
            regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
    ))
    next_edges, next_vertices = find_and_sort_transect_edges_and_vertices(ds, ds_mask)
    assert(next_edges == np.array([ 4,  5,  6,  7, 10, 13, 12, 11])).all()
    assert(next_vertices == np.array([ 3,  6,  4,  5,  7, 10,  8,  9])).all()



# test transport calculations

def scenarios_mask(test_type):
    if test_type == 'basic_single':
        # copy and paste these datasets for whatever scenario you want to test
        # basic algorithm around a single cell
        regionCellMasks_array = np.int32(np.array([[0],[1],[0]]))
        ds_mask = xr.Dataset(
                data_vars = dict(
                    regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
                    regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
            ))
        next_edges = np.array([ 4,  5,  9, 11, 10,  8])
        next_vertices = np.array([3, 6, 8, 9, 7, 5])
        
    if test_type == 'basic_double':
        # basic algorithm around two cells
        regionCellMasks_array = np.int32(np.array([[1],[1],[0]]))
        ds_mask = xr.Dataset(
            data_vars = dict(
                regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
                regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
        ))
        next_edges = np.array([ 0,  1,  3,  6,  9, 11, 10,  8,  4,  2])
        next_vertices = np.array([0, 2, 4, 6, 8, 9, 7, 5, 3, 1])
        
    if test_type == 'latitude_lines':
        # for a mask bounded by complete lines of latitude
        # zero clue how to ~create~ a mask for a transect like this using an algorithm, but maybe from a mask?
        regionCellMasks_array = np.int32(np.array([[0],[1],[1]]))
        ds_mask = xr.Dataset(
            data_vars = dict(
                regionCellMasks=(['nCells','nRegions'],regionCellMasks_array),
                regionNames =(['nRegions'], np.array([b'test from transect algorithm'], dtype='|S64'))
        ))
        next_edges = np.array([ 4,  5,  6,  7, 10, 13, 12, 11])
        next_vertices = np.array([ 3,  6,  4,  5,  7, 10,  8,  9])

    return next_edges, next_vertices, ds_mask


def scenarios_transect(test_type):
    from mpasregions.sections import format_transect_data
    
    if test_type == 'basic_single':
        edges, vertices, ds_mask = scenarios_mask('basic_single')
        ds_transect_cellsOnEdge, ds_transect_edges = format_transect_data(ds, edges)

    if test_type == 'basic_double':
        edges, vertices, ds_mask = scenarios_mask('basic_double')
        ds_transect_cellsOnEdge, ds_transect_edges = format_transect_data(ds, edges)

    if test_type == 'latitude_lines':
        edges, vertices, ds_mask = scenarios_mask('latitude_lines')
        ds_transect_cellsOnEdge, ds_transect_edges = format_transect_data(ds, edges)

    return edges, ds_mask, ds_transect_cellsOnEdge, ds_transect_edges


# test `format_transect_data`

def test_format_transect_data():
    from mpasregions.sections import format_transect_data

    # test basic algorithm
    edges, vertices, ds_mask = scenarios_mask('basic_single')
    ds_transect_cellsOnEdge, ds_transect_edges = format_transect_data(ds, edges)
    
    assert(ds_transect_edges.transect_edgesOrdered == np.arange(0,len(edges))).all()
    np.testing.assert_equal(ds_transect_cellsOnEdge.values, np.array([[1, np.nan],[0, 1], [1, 2],[1, np.nan],[1, np.nan],[1, 2]]))

    # test basic algorithm around two cells
    edges, vertices, ds_mask = scenarios_mask('basic_double')
    ds_transect_cellsOnEdge, ds_transect_edges = format_transect_data(ds, edges)

    assert(ds_transect_edges.transect_edgesOrdered == np.arange(0,len(edges))).all()
    np.testing.assert_equal(ds_transect_cellsOnEdge.values, np.array([[0,np.nan],[0,np.nan],[0,np.nan],[0,2],[1,2],[1,np.nan],[1,np.nan],[1,2],[1,np.nan],[0,np.nan]]))

    # test a mask bounded by complete lines of latitude
    edges, vertices, ds_mask = scenarios_mask('latitude_lines')
    ds_transect_cellsOnEdge, ds_transect_edges = format_transect_data(ds, edges)

    assert(ds_transect_edges.transect_edgesOrdered == np.arange(0,len(edges))).all()
    np.testing.assert_equal(ds_transect_cellsOnEdge.values, np.array([[1,np.nan],[0,1],[0,2],[2,np.nan],[1,np.nan],[2,np.nan],[2,np.nan],[1,np.nan]]))



# ### test ```calculate_velo_into_mask```

def test_calculate_velo_into_mask():
    from mpasregions.sections import calculate_velo_into_mask
    
    # test basic algorithm around one cell
    # get the unit test, pre-formatted datasets
    edges, ds_mask, ds_transect_cellsOnEdge, ds_transect_edges = scenarios_transect('basic_single')

    # run the test
    ds_transect_edges = calculate_velo_into_mask(ds_transect_edges, ds, ds_mask, edges)
    assert(ds_transect_edges.veloIntoMask == np.array([[[-0., -0., -0.],
                                                        [-1.,  0.,  0.],
                                                        [-0., -0., -0.],
                                                        [-0., -0., -0.],
                                                        [-0., -0., -0.],
                                                        [ 1., -0., -0.]]
                                                      ])).all()
    
    # test basic algorithm around two cells
    # get the unit test, pre-formatted datasets
    edges, ds_mask, ds_transect_cellsOnEdge, ds_transect_edges = scenarios_transect('basic_double')
    
    # run the test
    ds_transect_edges = calculate_velo_into_mask(ds_transect_edges, ds, ds_mask, edges)
    assert(ds_transect_edges.veloIntoMask == np.array([[[-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.75, -0.5 , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [ 1.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ]]
                                                        ])).all()

    # test a mask bounded by complete lines of latitude 
    # get the unit test, pre-formatted datasets
    edges, ds_mask, ds_transect_cellsOnEdge, ds_transect_edges = scenarios_transect('latitude_lines')

    # run the test
    ds_transect_edges = calculate_velo_into_mask(ds_transect_edges, ds, ds_mask, edges)
    assert(ds_transect_edges.veloIntoMask == np.array([[[-0.  , -0.  , -0.  ],
                                                        [-1.  ,  0.  ,  0.  ],
                                                        [ 0.75,  0.5 ,  0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ],
                                                        [-0.  , -0.  , -0.  ]]
                                                      ])).all()


# test ```calculate_transport_into_mask```

def test_calculate_transport_into_mask():
    from mpasregions.sections import calculate_transport_into_mask

    # test basic algorithm around one cell
    # get the unit test, pre-formatted datasets
    edges, ds_mask, ds_transect_cellsOnEdge, ds_transect_edges = scenarios_transect('basic_single')

    # add the veloIntoMask datavariable
    veloIntoMask = np.array([[[-0., -0., -0.],
                              [-1.,  0.,  0.],
                              [-0., -0., -0.],
                              [-0., -0., -0.],
                              [-0., -0., -0.],
                              [ 1., -0., -0.]]])
    ds_transect_edges['veloIntoMask'] = xr.DataArray(veloIntoMask, dims=('xtme_startMonthly', 'nEdges', 'nVertLevels'))

    # run the test
    ds_transect_edges = calculate_transport_into_mask(ds_transect_edges)

    # make the known answer array
    basic_single = np.zeros((6,3,1))
    basic_single[1,:,:] = np.array([[-0.14108902],[0],[0]])
    basic_single[5,:,:] = np.array([[0.05239937],[0],[0]])

    # check equality
    assert(np.allclose(ds_transect_edges['transportIntoMask_Sv'], basic_single))


    # test basic algorithm around two cells
    # get the unit test, pre-formatted datasets
    edges, ds_mask, ds_transect_cellsOnEdge, ds_transect_edges = scenarios_transect('basic_double')

    # add the veloIntoMask datavariable
    veloIntoMask = np.array([[[-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.75, -0.5 , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [ 1.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ]]])
    ds_transect_edges['veloIntoMask'] = xr.DataArray(veloIntoMask, dims=('xtme_startMonthly', 'nEdges', 'nVertLevels'))

    # run the test
    ds_transect_edges = calculate_transport_into_mask(ds_transect_edges)

    # make the known answer array
    basic_double = np.zeros((10,3,1))
    basic_double[3,:,:] = np.array([[-0.14108902], [-0.09405935], [0]])
    basic_double[7,:,:] = np.array([[0.05239937], [0], [0]])

    # check equality
    assert(np.allclose(ds_transect_edges['transportIntoMask_Sv'], basic_double))

    # test a mask bounded by complete lines of latitude
    # get the unit test, pre-formatted datasets
    edges, ds_mask, ds_transect_cellsOnEdge, ds_transect_edges = scenarios_transect('latitude_lines')

    # add the veloIntoMask datavariable
    veloIntoMask = np.array([[[-0.  , -0.  , -0.  ],
                              [-1.  ,  0.  ,  0.  ],
                              [ 0.75,  0.5 ,  0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ],
                              [-0.  , -0.  , -0.  ]]])
    ds_transect_edges['veloIntoMask'] = xr.DataArray(veloIntoMask, dims=('xtme_startMonthly', 'nEdges', 'nVertLevels'))

    # run the test
    ds_transect_edges = calculate_transport_into_mask(ds_transect_edges)

    # make the known answer array
    latitude_lines = np.zeros((8,3,1))
    latitude_lines[1,:,:] = np.array([[-0.14108902],[0],[0]])
    latitude_lines[2,:,:] = np.array([[0.14108902],[0.09405935],[0]])

    # check equality
    assert(np.allclose(ds_transect_edges['transportIntoMask_Sv'], latitude_lines))

