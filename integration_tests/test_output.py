import pytest

# test cases
# 1: num_input_steps > 1, num_steps > 1 (1527)
# => what sources get written (multiple sources without targets?) => test with offset=1
# 2: inference on jepa (1759) (jepa_wael)
# 3: jepa pretraining on era5, finetuning/inference on synop (1736)
# 4: guarantee outputput item is never overwritten (1575)
# 5: allow subsetting (for io) channels used as targets (1705)
# forecast offset 0/1
# predictions without targets/sources
# allow incremental writes
# allow non continouos fsteps
# always have source at fstep=0




@pytest.fixture
def output_items():
    pass

def test_target_source_identity(offset):
    pass

def test_time_coordinates(output_items):
    pass

def test_spatial_coordinates(output_items):
    pass