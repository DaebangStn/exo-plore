from abc import ABC, abstractmethod
from python.gait_generator.util import *


class ParameterBinBase(ABC):
    """Base class for parameter bin trackers."""
    def __init__(self, name, param_cfg, adaptive_sampling_cfg):
        """
        Initialize the parameter bin tracker.
        
        Args:
            name: Name of the parameter
            param_cfg: Configuration for. parameter binning from metadata
            adaptive_sampling_cfg: Configuration for adaptive sampling from metadata
        """
        self.param_cfg = param_cfg
        self._param_name = [name]
        if "paired_to" in param_cfg.keys():
            self._param_name.extend(param_cfg["paired_to"])
        self._bin_dim = len(self._param_name)
        
        self.adaptive_sampling_cfg = adaptive_sampling_cfg
        self._temperature = adaptive_sampling_cfg["temperature"]
        self._min_count = adaptive_sampling_cfg["min_count"]
        self._accum_lpf = adaptive_sampling_cfg.get("accum_lpf", False)
        self._accum_lpf_alpha = adaptive_sampling_cfg.get("accum_lpf_alpha", 0.01)
        self._accum_count = 0
        
        if isinstance(param_cfg["bins"], int):
            self._bin_count = [param_cfg["bins"]]
        else:
            self._bin_count = param_cfg["bins"]
        self._total_bin_num = np.prod(self._bin_count)
        self._bin_acculumator = np.zeros(self._bin_count)
        self._sampling_weights = np.ones_like(self._bin_acculumator) / self._total_bin_num
        
    @property
    def param_name(self) -> List[str]:
        return self._param_name
    
    @property
    def bin_acculumator(self) -> np.ndarray:
        return self._bin_acculumator
    
    @property
    def bin_count(self) -> np.ndarray:
        return self._bin_count
    
    @staticmethod
    def _update_accum_ma(bin_accumulator: np.ndarray, index: List[int], acculumator: float, alpha: float):
        prev_val = bin_accumulator[tuple(index)]
        if prev_val == 0:
            bin_accumulator[tuple(index)] = acculumator
        else:
            bin_accumulator[tuple(index)] = bin_accumulator[tuple(index)] * (1 - alpha) + acculumator * alpha
        return bin_accumulator
    
    def accumulate(self, params: Dict[str, float], acculumator: float):
        bin_idx = self._get_bin_index(params)
        if self._accum_lpf:
            self._update_accum_ma(self._bin_acculumator, bin_idx, acculumator, self._accum_lpf_alpha)
        else:
            self._bin_acculumator[tuple(bin_idx)] += acculumator
        self._accum_count += 1
        
    def sample(self) -> Optional[np.ndarray]:
        """
        Get sampling weights for each bin based on episode performance. anti-proportional to the bin count.
        (1 - bin_count / total_bin_count)

        Returns:
            np.ndarray: sampling weights. if None, use the default value.
        """
        if self._accum_count < self._min_count or self._min_count == -1:
            return None
        
        total_bin_count = np.sum(self._bin_acculumator)
        if total_bin_count == 0:
            print(f"[Warning] Total bin count is 0 for {self._param_name}")
            return None
        
        # Calculate weights - higher accumulator values get lower weights
        bin_weights = 1 - self._bin_acculumator / total_bin_count
        
        # Apply temperature to weights
        if self._temperature != 1.0:
            # For low temperature: raise to a power > 1 to make peaks more extreme
            # For high temperature: raise to a power < 1 to make distribution more uniform
            bin_weights = bin_weights ** (1.0 / self._temperature)
            
        # Renormalize after applying temperature
        weight_sum = np.sum(bin_weights)
        if weight_sum == 0:
            print(f"[Warning] Weight sum is 0 for {self._param_name}")
            return None
        self._sampling_weights = bin_weights / weight_sum

        return self._sampling_weights
    
    def get_statistics(self) -> dict:
        """
        Get the statistics of the parameter bin.
        
        Returns:
            dict: Tensorboard-friendly statistics
        """
        # Sum the sampling weights except the first dimension
        if self._sampling_weights.ndim == 1:
            sampling_weights_1d = self._sampling_weights
            bin_acculumator_1d = self._bin_acculumator
        else:
            sampling_weights_1d = np.sum(self._sampling_weights, axis=tuple(range(1, self._sampling_weights.ndim)))
            bin_acculumator_1d = np.mean(self._bin_acculumator, axis=tuple(range(1, self._bin_acculumator.ndim)))
        result = {}
        for i, weight in enumerate(sampling_weights_1d):
            result[f'bin_weight_{i}'] = weight
            result[f'bin_acculumator_{i}'] = bin_acculumator_1d[i]
        result['total_accumulator'] = np.sum(self._bin_acculumator)
        result['accum_count'] = self._accum_count
        return result
            
    @abstractmethod
    def _get_bin_index(self, params: Dict[str, float]) -> List[int]:
        """
        Get the bin index for a set of parameters.
        
        Args:
            params: Dictionary of parameter values.
        """
    
class ParameterBinUniform(ParameterBinBase):
    def __init__(self, param_name: str, param_cfg: Dict, adaptive_sampling_cfg: Dict):
        """
        Initialize the parameter bin.

        Args:
            param_name: Name of the parameter
            param_cfg: Parameter configuration
            adaptive_sampling_cfg: Adaptive sampling configuration
        """
        # Call the parent class initializer
        super().__init__(param_name, param_cfg, adaptive_sampling_cfg)
        
        # Handle parameter ranges
        self._bin_partitions = []
        
        # Check if value is a nested list (for paired parameters)
        if isinstance(param_cfg["value"][0], list):
            # Multi-dimensional case
            for i, value_range in enumerate(param_cfg["value"]):
                self._bin_partitions.append(np.linspace(value_range[0], value_range[1], self._bin_count[i] + 1))
        else:
            # Single parameter case
            self._bin_partitions.append(np.linspace(param_cfg["value"][0], param_cfg["value"][1], self._bin_count[0] + 1))

    def _get_bin_index(self, params: Dict[str, float]) -> List:
        bin_idx = []
        for idx, pname in enumerate(self._param_name):
            if not (self._bin_partitions[idx][0] <= params[pname] <= self._bin_partitions[idx][-1]):
                print(f"Warning: Parameter {pname} is out of range. Got {params[pname]} range [{self._bin_partitions[idx][0]}, {self._bin_partitions[idx][-1]}]")
            found_idx = np.searchsorted(self._bin_partitions[idx], params[pname], side='right') - 1
            if found_idx == len(self._bin_partitions[idx]):
                found_idx -= 1      
            bin_idx.append(found_idx)
        return bin_idx

class ParameterBinDiagonal(ParameterBinBase):
    def __init__(self, param_name: str, param_cfg: Dict, adaptive_sampling_cfg: Dict):
        super().__init__(param_name, param_cfg, adaptive_sampling_cfg)
        self._bin_dim = 1
        assert len(self._bin_count) == 1, "Bin count must be a single integer in the case of diagonal sampling. got {}".format(self._bin_count)
        
        if not isinstance(param_cfg["value"][0], list):
            sample_range = [param_cfg["value"]]
        else:
            sample_range = param_cfg["value"]
        
        self._sample_range = sample_range
        self._first_param_partition = np.linspace(sample_range[0][0], sample_range[0][1], self._bin_count[0] + 1)
        
    def _get_bin_index(self, params: Dict[str, float]) -> List:
        bin_idx = []
        found_idx = np.searchsorted(self._first_param_partition, params[self._param_name[0]], side='right') - 1
        if found_idx == len(self._first_param_partition):
            found_idx -= 1
        bin_idx.append(found_idx)
        
        for i in range(1, len(self._param_name)):
            if not (self._sample_range[i][0] <= params[self._param_name[i]] <= self._sample_range[i][1]):
                print(f"Warning: Parameter {self._param_name[i]} is out of range. Got {params[self._param_name[i]]} range [{self._sample_range[i][0]}, {self._sample_range[i][1]}]")        
        return bin_idx