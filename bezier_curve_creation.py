from enum import Enum
import numpy as np
import scipy as sp

# ENUM for sampling types
class SamplingType(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"

# BezierCurve class for generating and sampling Bezier curves
class BezierCurve:
    def __init__(self):
        self.control_points = []
        self.sampled_points = []

    # generates Bezier curve control points of the specified degree
    def generate_curve(self, degree):
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        self.control_points = np.random.rand(degree + 1, 2)
        self.control_points = self.control_points * 100

        # print(f"Generated {degree}-degree Bezier curve control points:")
        # print(self.control_points)

    # helper function that calculates the Bezier value for a given t-value and control points
    def calc_bezier_value(self, t, control_points):

        result = np.zeros(2, dtype=float)

        n = control_points.shape[0] - 1

        for i in range(len(control_points)):
            bernstein_poly = sp.special.binom(n, i) * ((1-t)**(n-i)) * (t**i)
            result += bernstein_poly * control_points[i]

        return result

    # samples and returns a given number of points on the Bezier curve - either uniformly or randomly (t-values)
    def sample_curve(self, num_points, sampling_type=SamplingType.UNIFORM, curve_to_sample = None):
        if curve_to_sample is None:
            curve_to_sample = self.control_points

        self.sampled_points = []

        if sampling_type == SamplingType.UNIFORM:
            t_values = np.linspace(0, 1, num_points)

        # likely does not use to contain the start point and will certainly not contain the end point (np.random is in the range [0,1))
        elif sampling_type == SamplingType.RANDOM:
            t_values = np.random.rand(num_points)
            t_values.sort()
        else:
            raise ValueError("Invalid sampling type. Use SamplingType.UNIFORM or SamplingType.RANDOM.")

        for t_value in t_values:
            point = self.calc_bezier_value(t_value, curve_to_sample)
            self.sampled_points.append(point)

        self.sampled_points = np.array(self.sampled_points)

        return self.sampled_points

    def get_control_points(self):
        return self.control_points



# Example usage
# bezier_curve = BezierCurve()
# Generate a 10th degree Bezier curve
# bezier_curve.generate_curve(10)
# Sample 5 points uniformly on the Bezier curve
# bezier_curve.sample_curve(5, SamplingType.UNIFORM)