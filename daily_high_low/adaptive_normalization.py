"""
This code attempts to reproduce:
Adaptive Normalization: A novel data normalization approach for non-stationary time series

Conference: International Joint Conference on Neural Networks, IJCNN 2010, Barcelona, Spain, 18-23 July, 2010
Eduardo Ogasawara et Al.

"""



import numpy as np
import matplotlib.pyplot as plt


def calculate_weighted_moving_average(arr, window, alpha=0.333):
    # EMA[1] = SMA[1]
    ma = [np.sum(arr[:window]) / window]

    for idx in range(1, arr.shape[0]):
        if idx + window - 1 > arr.shape[0] - 1:
            ma.append(None)
            continue

        # (1-a) * MA[t-1] + a * S[t + window - 1]
        ma.append((1 - alpha) * ma[-1] + alpha * arr[idx + window - 1])
    return np.array(ma, dtype=np.float64)


def transform_to_sliding_windows(arr, moving_average, window):
    result = None
    for idx in range(moving_average.shape[0]):
        if len(arr[idx: idx + window]) < window:
            break

        window_values = arr[idx: idx + window].copy()
        window_values /= moving_average[idx]
        result = window_values if result is None else np.vstack([result, window_values])

    return result


def discard_outliers(arr, threshold=1.5):
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    l_lim = q1 - threshold * (q3 - q1)
    g_lim = q3 + threshold * (q3 - q1)
    greater_than = arr > g_lim
    less_than = arr < l_lim

    valid = np.invert(
        np.any(
            np.logical_or(greater_than, less_than), axis=1
        )
    )

    return arr[valid], {
        'valid': valid,
        'has_less': np.any(less_than),
        'has_greater': np.any(greater_than),
        'lower_limit': l_lim,
        'upper_limit': g_lim
    }


def normalize(arr, minimum=None, maximum=None):
    _min = minimum if minimum else np.min(arr)
    _max = maximum if maximum else np.max(arr)
    return (arr - _min) / (_max - _min), _min, _max


def denormalize(arr, minimum, maximum):
    return arr * (maximum - minimum) + minimum


def detransform(arr, ma):
    _arr = arr[:, -1]
    _first = arr[0, :-1]

    means = ma[:len(_arr)]
    _arr_m = _arr * means
    _first_m = _first * ma[0]
    return np.concatenate([_first_m, _arr_m])


def discard_limits(s, k, w):
    to_discard = k - (w - 1)

    if to_discard <= 0:
        return s, None

    print(f"\nTo discard: {to_discard}")
    return s[to_discard:], s[:to_discard]


def calc_level_adjustment(s, ma, w, i):

    adjustments = []

    for j in range(i, i + w):
        adjustments.append(np.square(s[j - 1] - ma[i - 1]))

    return sum(adjustments) / w


def calc_level_adjustments_ma(s, ma, w):

    phi = len(s) - w + 1
    adjustments = []

    for i in range(1, phi + 1):
        adjustments.append(calc_level_adjustment(s, ma, w, i))

    return sum(adjustments) / phi


def plot_filtered(arr, org, valid, ax, c):
    _row = 0
    for index in range(0, len(org)):
        if valid[index]:
            ax.plot(arr[_row], color=c[index])
            _row += 1
    return ax


def review_window_lengths(arr, dsw_window, window_range=(2, 8)):
    print(f'DSW window length: {dsw_window}.\nProcessing moving average windows:')
    for length in range(*window_range):
        filtered = calculate_weighted_moving_average(arr, length)
        adjustment = calc_level_adjustments_ma(arr, filtered, dsw_window)
        print(f"Window length: {length} | Adjustment: {adjustment}")


def process(arr, ma_window, dsw_window):

    moving_average = calculate_weighted_moving_average(arr, ma_window)
    filtered_data, discarded = discard_limits(arr, ma_window, dsw_window)
    windowed_data = transform_to_sliding_windows(filtered_data, moving_average, dsw_window)
    filtered_windowed_data, outlier_info = discard_outliers(windowed_data, threshold=100)

    normalized_data, norm_minimum, norm_maximum = normalize(
        filtered_windowed_data,
        minimum=outlier_info['lower_limit'] if outlier_info['has_less'] else None,
        maximum=outlier_info['upper_limit'] if outlier_info['has_greater'] else None
    )

    denormalized_data = denormalize(normalized_data, norm_minimum, norm_maximum)
    reconstructed_data = detransform(denormalized_data, moving_average)

    if discarded:
        reconstructed_data = np.concatenate([discarded, reconstructed_data])

    return normalized_data, reconstructed_data, filtered_windowed_data


def plot(original, windowed, normalized, reconstructed):
    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(original)
    axs[0][0].set_title('Original')

    axs[0][1].plot(windowed)
    axs[0][1].set_title('Transformed Windows')

    axs[1][0].plot(normalized)
    axs[1][0].set_title('Filtered Normalized Windows')

    axs[1][1].plot(reconstructed)
    axs[1][1].set_title('Reconstructed')
    plt.show()


if __name__ == "__main__":
    data = np.array(
        [
            1.734,
            1.720,
            1.707,
            1.708,
            1.735,
            1.746,
            1.744,
            1.759,
            1.751,
            1.749,
            1.763,
            1.753,
            1.774,
        ]
    )

    norm, recon, wind = process(data.copy(), 5, 6)
    plot(data, wind, norm, recon)