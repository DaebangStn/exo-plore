# Find the minimum value within the Phase x Stride = Constant
# In this case, center for the dataX is fixed

from python.plot.util import *


def compute_bound(dataX: np.array, dataY: np.array, range_init: float, param_indices: np.array, margin: float = None,
                  per_velocity: bool = False, range_centers: List[float] = None, members_per_cluster: int = 1) -> Union[List[int], List[List[int]]]:
    """
    :param dataX: Usually velocity, in m/s
    :param dataY:
    :param range_init: Initial range to find the cluster
    :param param_indices: the return value is a list of indices for each velocity
    :param margin: If not None, selected Y is the [min ~ min + margin(in ratio, i.e. 0.1)] range
    :param per_velocity: If True, the return value is a list of indices for each velocity
    :param range_centers: If not None, use this value as the center of the cluster
    :param members_per_cluster: Number of members to select per cluster
    :return: Index of the lower bound point
    """
    if range_centers is None:
        range_centers = np.array([
            1, 2, 3, 4, 5
        ]) / 3.6

    return_buffer = []
    for center in range_centers:
        range_iter = range_init
        cluster_members = []
        while len(cluster_members) < members_per_cluster:
            cluster_members = np.where(np.abs(dataX - center) < range_iter)[0]
            range_iter += range_init / 20
            if range_iter > range_init * 3:
                raise ValueError(f"Cannot find a cluster for center {3.6 * center:.2f}kmh with range {3.6 * range_iter:.3f}kmh")

        if margin is None:
            min_idx_in_cluster = int(cluster_members[np.argmin(dataY[cluster_members])])
            min_idx_in_cluster = param_indices[min_idx_in_cluster]

            if per_velocity:
                return_buffer.append([min_idx_in_cluster])
            else:
                return_buffer.append(min_idx_in_cluster)
        else:
            if margin < 0:
                raise ValueError(f"Margin must be non-negative, but got {margin:.2f}")
            min_y = np.min(dataY[cluster_members])
            up_bound_y = (1 + margin) * min_y
            selected_members = np.where(dataY[cluster_members] < up_bound_y)[0]

            if len(selected_members) == 0:
                raise ValueError(f"Cannot find a member in the range of 1 ~ {1 + margin} from the minimum value {min_y:.2f}")

            cluster_members = list(map(int, cluster_members[selected_members]))
            if param_indices is not None:
                cluster_members = param_indices[cluster_members]
                cluster_members = list(map(int, cluster_members))

            if per_velocity:
                return_buffer.append(cluster_members)
            else:
                return_buffer.extend(cluster_members)
    return return_buffer
