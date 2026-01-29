import numpy as np
from napari_hydra.utils.well_processing import get_well_centers, sort_wells_grid, calculate_well_stats, calculate_well_diameters

def test_get_well_centers():
    # Create a simple 10x10 array with one "well" labeled 1 at center (5,5)
    wells_2d = np.zeros((10, 10), dtype=int)
    wells_2d[4:6, 4:6] = 1 # 2x2 square at center
    
    centers = get_well_centers(wells_2d)
    assert len(centers) == 1
    wid, (cx, cy) = centers[0]
    assert wid == 1
    assert 4 <= cx <= 5
    assert 4 <= cy <= 5

def test_sort_wells_grid():
    # 6 wells in 2x3 grid
    # Row 0: (10,10), (30,10), (50,10)
    # Row 1: (10,30), (30,30), (50,30)
    wells = [
        (1, (10, 10)), (2, (30, 10)), (3, (50, 10)),
        (4, (10, 30)), (5, (30, 30)), (6, (50, 30))
    ]
    # Expected:
    # 0->1, 1->2, 2->3
    # 3->4, 4->5, 5->6
    
    ordered = sort_wells_grid(wells)
    assert ordered[0] == 1
    assert ordered[2] == 3
    assert ordered[5] == 6

def test_calculate_well_stats():
    # 2 wells, one with 1 plaque, one with 0
    wells_2d = np.zeros((20, 20), dtype=int)
    wells_2d[0:10, 0:10] = 1 # well 1 top-left
    
    plaque_2d = np.zeros((20, 20), dtype=int)
    plaque_2d[2:4, 2:4] = 1 # plaque 1 inside well 1
    
    ordered_wells = {0: 1} # Bin 0 is well 1
    
    counts, areas = calculate_well_stats(wells_2d, plaque_2d, ordered_wells)
    
    assert counts[0] == 1
    # Plaque is 2x2 = 4 pixels
    assert areas[0] == 4.0
    
    # Other bins empty
    assert counts[1] == 0

def test_calculate_well_diameters():
    # One well, circular approximation
    wells_2d = np.zeros((10, 10), dtype=int)
    # Center (5,5)
    # Point at (5,8) -> dist 3
    # Point at (5,2) -> dist 3
    # Diameter ~ 6
    wells_2d[5, 8] = 1
    wells_2d[5, 2] = 1
    
    centers = [(1, (5, 5))]
    
    dia = calculate_well_diameters(wells_2d, centers)
    # Max dist is 3, diameter = 2*max_dist = 6
    assert abs(dia - 6.0) < 0.001
