import numpy as np

def get_well_centers(wells_2d):
    """
    Extract unique well IDs and their centroids from a 2D label array.
    """
    well_ids = np.unique(wells_2d)
    well_ids = well_ids[well_ids != 0]
    well_centers = []
    for wid in well_ids:
        ys, xs = np.where(wells_2d == wid)
        if len(xs) == 0 or len(ys) == 0:
            continue
        center_x = np.mean(xs)
        center_y = np.mean(ys)
        well_centers.append((wid, (center_x, center_y)))
    return well_centers

def sort_wells_grid(well_centers):
    """
    Sort well centers into a 2x3 grid and return a map of {bin_idx: well_id}.
    """
    if not well_centers:
        return {}
        
    # If more than 6 wells, select 6 closest to centroid
    if len(well_centers) > 6:
        centroid_x = np.mean([cx for _, (cx, cy) in well_centers])
        centroid_y = np.mean([cy for _, (cx, cy) in well_centers])
        well_centers = sorted(
            well_centers,
            key=lambda item: np.sqrt((item[1][0] - centroid_x)**2 + (item[1][1] - centroid_y)**2)
        )[:6]

    min_x, max_x = min(cx for _, (cx, _) in well_centers), max(cx for _, (cx, _) in well_centers)
    x_dist = max_x - min_x
    min_y, max_y = min(cy for _, (_, cy) in well_centers), max(cy for _, (_, cy) in well_centers)
    y_dist = max_y - min_y
    ordered_wells = {}
    for wid, (cx, cy) in well_centers:
        x_d = (cx - min_x) / x_dist if x_dist > 0 else 0
        if x_d < 0.25:
            x_i = 0
        elif x_d < 0.75:
            x_i = 1
        else:
            x_i = 2
        y_d = (cy - min_y) / y_dist if y_dist > 0 else 0
        if y_d < 0.5:
            y_i = 0
        else:
            y_i = 1
        bin_idx = x_i + 3 * y_i
        ordered_wells[bin_idx] = wid
    return ordered_wells

def calculate_well_stats(wells_2d, plaque_2d, ordered_wells):
    """
    Calculate plaque counts and average areas for each well in the grid.
    Returns lists of length 6.
    """
    plaque_counts = [0] * 6
    avg_areas = [0.0] * 6
    for bin_idx in range(6):
        wid = ordered_wells.get(bin_idx, None)
        if wid is None:
            continue
        mask_well = (wells_2d == wid)
        plaque_labels_in_well = plaque_2d[mask_well]
        unique_plaques = np.unique(plaque_labels_in_well)
        unique_plaques = unique_plaques[unique_plaques != 0]
        plaque_counts[bin_idx] = len(unique_plaques)
        # For each plaque, compute area
        areas = []
        for pid in unique_plaques:
            area = np.sum((plaque_2d == pid) & mask_well)
            areas.append(area)
        avg_areas[bin_idx] = np.mean(areas) if areas else 0.0
    return plaque_counts, avg_areas

def calculate_well_diameters(wells_2d, well_centers):
    """
    Calculate the average well diameter in pixels.
    """
    well_diameters = []
    for wid, (cx, cy) in well_centers:
        yx = np.column_stack(np.where(wells_2d == wid))
        if yx.shape[0] == 0:
            continue
        dists = np.sqrt((yx[:, 1] - cx)**2 + (yx[:, 0] - cy)**2)
        diameter = 2 * np.max(dists)
        well_diameters.append(diameter)
    
    avg_well_diameter_px = np.mean(well_diameters) if well_diameters else 0.0
    return avg_well_diameter_px
