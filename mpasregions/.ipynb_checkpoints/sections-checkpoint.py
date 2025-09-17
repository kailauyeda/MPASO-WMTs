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

    return xr_cells_inside, xr_vertices_inside, xr_edges_inside