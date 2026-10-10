"""Complete Torch route metadata; no pinned CPU solver or GPU execution."""
from types import SimpleNamespace
import torch
from fudan_wgs_toolkit import _precision_eigen as precision


def test_complete_native_route_and_detached_metadata():
    precision.precision_eigen_execution_metadata(reset=True)
    precision.record_gpu_eigen_route(SimpleNamespace(is_cuda=True,ndim=3,shape=(4,8,8),dtype=torch.float32))
    saved=precision.precision_eigen_execution_metadata(reset=True)
    assert saved['gpu_eigen_matrices']==4
    assert saved['gpu_eigen_routes']=={'torch.linalg.eigvalsh':4}
    assert saved['driver_traced'] is False
    assert saved['eigen_dtype_calls']=={'torch.float32':4}
    assert saved['cpu_eigen_calls']==0 and saved['library_sha256'] is None
    saved['gpu_eigen_routes']['changed']=1
    current=precision.precision_eigen_execution_metadata()
    assert current['gpu_eigen_matrices']==0 and current['gpu_eigen_routes']=={}


def test_cpu_control_not_reported_as_cuda():
    precision.precision_eigen_execution_metadata(reset=True)
    precision.record_gpu_eigen_route(torch.eye(2,dtype=torch.float64))
    assert precision.precision_eigen_execution_metadata()['gpu_eigen_calls']==0
