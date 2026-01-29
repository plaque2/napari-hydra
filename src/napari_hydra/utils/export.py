def write_prediction_summary(save_path, well_diameter_mm, px_per_mm, counts_per_frame, avg_areas_per_frame):
    """
    Write the prediction summary to a text file.

    Format includes header, settings (diameter, scale), and a per-frame table
    of plaque counts and average areas for each well.

    Args:
        save_path (str): File path to save the summary to.
        well_diameter_mm (float): Assumed well diameter in millimeters.
        px_per_mm (float): Calculated pixel-to-millimeter scale factor.
        counts_per_frame (list): List of lists, where each inner list contains plaque counts for the 6 wells.
        avg_areas_per_frame (list): List of lists, where each inner list contains average plaque areas (in mm^2).

    Returns:
        None
    """
    with open(save_path, "w") as f:
        f.write(f"PREDICTION SUMMARY\n")
        f.write("="*40 + "\n")
        f.write(f"Assumed Well Diameter: {well_diameter_mm} mm\n")
        f.write(f"Pixel-to-mm Scale: {px_per_mm:.2f} px/mm\n")

        for frame_idx, (counts, avg_areas) in enumerate(zip(counts_per_frame, avg_areas_per_frame)):
            f.write(f"\nFRAME {frame_idx + 1}\n")
            f.write("Plaque Count per Well      Average Plaque Area (mm²)\n")
            f.write("-------------------        ----------------------------\n")
            f.write("| {:03d} | {:03d} | {:03d} |        | {:06.2f} | {:06.2f} | {:06.2f} |\n".format(*counts[:3], *avg_areas[:3]))
            f.write("-------------------        ----------------------------\n")
            f.write("| {:03d} | {:03d} | {:03d} |        | {:06.2f} | {:06.2f} | {:06.2f} |\n".format(*counts[3:], *avg_areas[3:]))
            f.write("-------------------        ----------------------------\n")
