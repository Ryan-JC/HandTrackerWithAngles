import numpy as np
from filterpy.kalman import KalmanFilter


# Initialize Kalman Filter for each finger
def initialize_kalman():
    kf = KalmanFilter(dim_x=2, dim_z=1)  # 2 state variables (angle + velocity), 1 measurement
    kf.x = np.array([[0], [0]])  # Initial state: angle = 0, velocity = 0
    kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement function
    kf.P *= 1000  # Covariance matrix
    kf.R = 10  # Measurement noise
    kf.Q = np.array([[1, 0], [0, 1]])  # Process noise
    return kf

def apply_kalman_filter(kf, angle_measurement):
    """ Applies Kalman filter to smooth the angle measurement """
    kf.predict()
    kf.update([angle_measurement])
    return kf.x[0, 0]  # Return the smoothed angle

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points using the dot product formula.
    a, b, c are tuples (x, y, z) representing joint positions.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Create vectors
    ba = a - b  # Vector from joint b to a
    bc = c - b  # Vector from joint b to c

    # Compute dot product and magnitudes
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Prevent division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    # Compute angle (in degrees)
    angle = np.arccos(dot_product / (magnitude_ba * magnitude_bc))
    return np.degrees(angle)