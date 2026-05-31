import numpy as np
from scipy.interpolate import CubicSpline


def mixup(x1, y1, x2, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2

    return x_mix, y_mix

def time_warp(signal, sigma=0.2, knot=4):
    signal = signal.reshape(-1)         
    orig_steps = np.arange(len(signal))

    warp_steps = np.linspace(0, len(signal)-1, knot + 2)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=knot + 2)

    warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
    warp = (warp - warp.min()) / (warp.max() - warp.min()) * (len(signal)-1)
    warped_signal = np.interp(orig_steps, warp, signal)
    return warped_signal.reshape(-1, 1)

def timewarp_mixup(x1, y1, x2, y2,
                   sigma=0.2,
                   knot=4,
                   alpha=0.2):

    x1_w = time_warp(x1, sigma=sigma, knot=knot)
    x2_w = time_warp(x2, sigma=sigma, knot=knot)

    return mixup(x1_w, y1, x2_w, y2, alpha=alpha)

def augment(
    X,
    y,
    target_counts,
    mode="timewarp_mixup",
    random_state=42
):
    """
    X: (N, 200, 1)
    y: one-hot (N, n_classes)

    target_counts:
    {
        0: 5000,
        1: 5000,
        2: 5000
    }

    Jeśli:
    - target > current -> oversampling
    - target < current -> undersampling
    """

    np.random.seed(random_state)

    X_result = []
    y_result = []

    classes = np.unique(np.argmax(y, axis=1))

    for cls in classes:

        idx = np.where(np.argmax(y, axis=1) == cls)[0]

        current_count = len(idx)

        if cls not in target_counts:

            X_result.append(X[idx])
            y_result.append(y[idx])

            continue

        desired_count = target_counts[cls]

        if desired_count < current_count:

            selected_idx = np.random.choice(
                idx,
                size=desired_count,
                replace=False
            )

            X_result.append(X[selected_idx])
            y_result.append(y[selected_idx])

        elif desired_count == current_count:

            X_result.append(X[idx])
            y_result.append(y[idx])

        else:

            n_to_generate = desired_count - current_count

            class_x = [X[idx]]
            class_y = [y[idx]]

            new_x = []
            new_y = []

            for _ in range(n_to_generate):

                i1, i2 = np.random.choice(
                    idx,
                    2,
                    replace=True
                )

                x1, y1 = X[i1], y[i1]
                x2, y2 = X[i2], y[i2]

                if mode == "timewarp":

                    x_new = time_warp(x1)
                    y_new = y1

                elif mode == "mixup":

                    x_new, y_new = mixup(
                        x1, y1,
                        x2, y2
                    )

                elif mode == "timewarp_mixup":

                    x_new, y_new = timewarp_mixup(
                        x1, y1,
                        x2, y2
                    )

                else:
                    raise ValueError("Unknown mode")

                new_x.append(x_new)
                new_y.append(y_new)

            class_x.append(np.array(new_x))
            class_y.append(np.array(new_y))

            X_result.append(np.concatenate(class_x, axis=0))
            y_result.append(np.concatenate(class_y, axis=0))

    X_final = np.concatenate(X_result, axis=0)
    y_final = np.concatenate(y_result, axis=0)

    shuffle_idx = np.random.permutation(len(X_final))
    X_final = X_final[shuffle_idx]
    y_final = y_final[shuffle_idx]

    return X_final, y_final



from scipy.signal import butter, filtfilt
def bandpass(signal, low=0.5, high=40.0, fs=360, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def compute_fft(segment):
    fft = np.fft.rfft(segment)
    mag = np.abs(fft)
    return mag