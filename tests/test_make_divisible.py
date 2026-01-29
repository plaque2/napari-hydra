from napari_hydra.utils.make_divisible import make_divisible

def test_make_divisible():
    assert make_divisible(17, 16) == 16
    assert make_divisible(32, 16) == 32
    assert make_divisible(15, 16) == 0
    assert make_divisible(100, 10) == 100
    assert make_divisible(105, 10) == 100
