import numpy as np
import dask.array as da
import pytest

from hyperspy.misc.io.fei_stream_readers import (
    array_to_stream, stream_to_array, stream_to_sparse_COO_array,
    stream_to_array_vector, stream_to_sparse_COO_array_vector,
    stream_to_sparse_COO_array_vector_numba,
    stream_to_array_vector_numba,
    )

spa_sh = (3, 4)
ch = 5
arr = np.random.randint(0, 65535, size=(2, spa_sh[0],
                            spa_sh[1], ch)).astype("uint16")
stream = array_to_stream(arr)


@pytest.mark.parametrize("summed", (True, False))
def test_benchmark_stream_to_array_vector(benchmark, summed):
    benchmark(stream_to_array_vector, stream, spatial_shape=spa_sh,
              sum_frames=summed, channels=ch, last_frame=2)


@pytest.mark.parametrize("summed", (True, False))
def test_benchmark_stream_to_sparse_COO_array_vector(benchmark, summed):
    benchmark(stream_to_sparse_COO_array_vector, stream, spatial_shape=spa_sh,
              sum_frames=summed, channels=ch, last_frame=2)


@pytest.mark.parametrize("summed", (True, False))
def test_benchmark_stream_to_array_vector_numba(benchmark, summed):
    benchmark(stream_to_array_vector_numba, stream, spatial_shape=spa_sh,
              sum_frames=summed, channels=ch, last_frame=2)


@pytest.mark.parametrize("summed", (True, False))
def test_benchmark_stream_to_sparse_COO_array_vector_numba(benchmark, summed):
    benchmark(stream_to_sparse_COO_array_vector_numba, stream, spatial_shape=spa_sh,
              sum_frames=summed, channels=ch, last_frame=2)


@pytest.mark.parametrize("summed", (True, False))
def test_benchmark_stream_to_array(benchmark, summed):
    benchmark(stream_to_array, stream, spatial_shape=spa_sh,
              sum_frames=summed, channels=ch, last_frame=2)


@pytest.mark.parametrize("summed", (True, False))
def test_benchmark_stream_to_sparse_COO_array(benchmark, summed):
    benchmark(stream_to_sparse_COO_array, stream, spatial_shape=spa_sh,
              sum_frames=summed, channels=ch, last_frame=2)