import os
from napari_hydra.utils.export import write_prediction_summary

def test_write_prediction_summary(tmp_path):
    save_path = tmp_path / "summary.txt"
    
    well_diameter_mm = 35.0
    px_per_mm = 10.0
    
    counts_per_frame = [[10, 20, 30, 40, 50, 60]]
    avg_areas_per_frame = [[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
    
    write_prediction_summary(
        str(save_path), 
        well_diameter_mm, 
        px_per_mm, 
        counts_per_frame, 
        avg_areas_per_frame
    )
    
    assert save_path.exists()
    content = save_path.read_text()
    
    assert "PREDICTION SUMMARY" in content
    assert "Assumed Well Diameter: 35.0 mm" in content
    assert "Pixel-to-mm Scale: 10.00 px/mm" in content
    assert "FRAME 1" in content
    assert "| 010 | 020 | 030 |" in content
    assert "| 040 | 050 | 060 |" in content
