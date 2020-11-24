from python.plot.util import *
import scipy.spatial


def is_pretty(first: float, second: float, third: float) -> bool:
    diff1 = abs(third - first)
    diff2 = min(abs(third - second), abs(first - second))
    return (2 * diff1) > diff2


def find_min_indices(data: np.array, target: np.array, intervals: np.array, margin: float = 0.0) -> List[int]:
    """
    Finds the indices of the minimum values of `target` within each interval of `data`.

    Parameters:
    - data: numpy array containing the values to define the intervals.
    - target: numpy array containing the values where the minimum will be found.
    - intervals: list or array defining the boundaries of the intervals for `data`.
    - margin: float value to set a margin from the minimum value. If margin > 0, 
              only considers points with target values within (min_value, min_value*(1+margin)).

    Returns:
    - min_indices: list of indices corresponding to the minimum values of `target` within each interval of `data`.
    """
    min_indices = []

    for i in range(len(intervals) - 1):
        lower_bound, upper_bound = intervals[i], intervals[i + 1]
        mask = (data >= lower_bound) & (data <= upper_bound)
        candid_indices = np.where(mask)[0]
        if candid_indices.size > 0:
            if margin > 0:
                min_value = np.min(target[candid_indices])
                # Apply margin filter - only consider points within margin of minimum
                margin_mask = target[candid_indices] <= min_value * (1 + margin)
                filtered_indices = candid_indices[margin_mask]
                min_indices.extend(filtered_indices)
            else:
                # Original behavior - just take the minimum
                min_idx = candid_indices[np.argmin(target[candid_indices])]
                min_indices.append(int(min_idx))
    return min_indices


def reject_outliers(target: np.array, indices: np.array):
    left_idx = None
    for i in range(len(indices)):
        if i == 0 or i == len(indices) - 1:
            continue
        if indices[i - 1] != -1:
            left_idx = indices[i - 1]
        x_idx = indices[i]
        right_idx = indices[i + 1]
        if not is_pretty(target[left_idx], target[x_idx], target[right_idx]):
            indices[i] = -1


def reject_distant_points(dataX: np.array, dataY: np.array, param_indices: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    Reject points that are too far from their nearest neighbors, with normalization for different scales.
    
    Parameters:
    - dataX, dataY: coordinate arrays
    - param_indices: original indices of the points
    
    Returns:
    - Filtered dataX, dataY, and param_indices
    """
    # Normalize X and Y coordinates by their respective scales
    x_scale = np.std(dataX) if np.std(dataX) > 0 else 1.0
    y_scale = np.std(dataY) if np.std(dataY) > 0 else 1.0
    
    # Combine normalized X and Y coordinates
    points = np.column_stack((dataX / x_scale, dataY / y_scale))
    
    # Compute pairwise distances on normalized coordinates
    distances = scipy.spatial.distance.pdist(points)
    distances = scipy.spatial.distance.squareform(distances)
    
    # Get minimum distance for each point (excluding self)
    min_distances = np.array([np.min(dist[dist > 0]) for dist in distances])
    
    # Calculate threshold using median
    threshold = 2.5 * np.median(min_distances)
    
    # Create mask for points to keep
    mask = min_distances <= threshold
    
    return dataX[mask], dataY[mask], param_indices[mask]


def compute_bound(dataX: np.array, dataY: np.array, param_indices: np.array, num_interval: int = 10,
                  x_dir_only: bool = True, margin: float = 0.0) -> List[int]:
    """
    Compute the lower bound of XY data.
        1. Reject points that are too distant from their neighbors
        2. Split the X range into N intervals
            - For each interval, find the point with the smallest Y value
            - Remove the points that does not satisfy the pretty(is_pretty) condition for Y
        3. For the M interval, split the Y range also into N / K intervals, (M=2, K=3, normally)
            - For each interval, find the point with the smallest X value
            - Remove the points that does not satisfy the convexity condition for X

    :param dataX:
    :param dataY:
    :param param_indices:
    :param num_interval:
    :param x_dir_only: If True, only the X direction is considered
    :param margin: margin for the minimum value of Y
    :return: Index of the boundary point
    """
    NUM_Y_SPLIT_DIV = 8
    NUM_X_INTERVAL_INIT = 2

    assert len(dataX) == len(dataY), "DataX and DataY must have the same length"
    
    # First reject distant points
    dataX, dataY, param_indices = reject_distant_points(dataX, dataY, param_indices)

    # Continue with existing boundary detection logic
    x_interval = np.linspace(dataX.min(), dataX.max(), num_interval + 1)
    bound_indices_x = find_min_indices(dataX, dataY, x_interval, margin)
    reject_outliers(dataY, bound_indices_x)
    bound_indices = set(bound_indices_x)

    if not x_dir_only:
        x_range_for_y = x_interval[0:NUM_X_INTERVAL_INIT + 1]
        mask_x = (dataX >= x_range_for_y[0]) & (dataX <= x_range_for_y[-1])
        y_range_min = dataY[mask_x].min()
        y_range_max = dataY[mask_x].max()
        y_interval = np.linspace(y_range_min, y_range_max, num_interval // NUM_Y_SPLIT_DIV + 1)

        bound_indices_y = find_min_indices(dataY, dataX, y_interval, margin)
        reject_outliers(dataX, bound_indices_y)

        bound_indices = bound_indices.union(bound_indices_y)
    bound_indices.discard(-1)
    bound_indices = list(bound_indices)
    return param_indices[bound_indices]

