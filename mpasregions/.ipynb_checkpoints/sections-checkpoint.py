import numpy as np
import xarray as xr

def n_to_xr_idx(n):
    return n-1

def xr_to_n_idx(xr):
    return xr+1

def xr_inside_mask_info(mesh,mask):
    # STEP 1: Select all of the cells inside teh mask
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


def sorted_transect_edges_and_vertices(mesh, xr_mask_transect_edges, xr_mask_transect_vertices):
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
    
def xr_sorted_transect_edges_and_vertices(mesh,mask):
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