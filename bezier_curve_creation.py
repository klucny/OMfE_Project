from enum import Enum
import numpy as np
import scipy as sp
import os
import threading
from threading import Thread
import time
import multiprocessing as mp


# ENUM for sampling types
class SamplingType(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"

# Custom Thread class to return results from a thread
class ReturnThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        # Initializing the Thread class
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # Running the target function and storing the result
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        # Joining the thread and returning the result
        super().join(*args)
        return self._return




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
        start_time = time.time()
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

        time_before_sampling = time.time()
        for t_value in t_values:
            point = self.calc_bezier_value(t_value, curve_to_sample)
            self.sampled_points.append(point)

        self.sampled_points = np.array(self.sampled_points)
        end_time = time.time()
        # print(f"Time taken to sample curve: {end_time - start_time:.4f} seconds")
        # print("Time taken to sample points: ", end_time - time_before_sampling)


        return self.sampled_points

    def get_control_points(self):
        return self.control_points

    def set_control_points(self, control_points):
        self.control_points = control_points

# BezierCurve class for generating and sampling Bezier curves
class BezierCurveMultiProcess:
    def __init__(self, num_processes=4):
        self.control_points = []
        self.sampled_points = []
        self.num_processes = num_processes

    # generates Bezier curve control points of the specified degree
    def generate_curve(self, degree):
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        self.control_points = np.random.rand(degree + 1, 2)
        self.control_points = self.control_points * 100 # Scale the control points for better visibility -> range [0, 100)

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

    def calc_bezier_value_threaded(self, t_values, control_points, process_id, num_points, queue):
        thread_time_start = time.time()

        # This function is intended to be run in a separate process
        sampled_points = []

        for t_value in t_values:
            point = self.calc_bezier_value(t_value, control_points)
            sampled_points.append(point)

        queue.put((process_id, sampled_points))



        thread_time_end = time.time()

        # print(f"Process {process_id} finished in {thread_time_end - thread_time_start:.4f} seconds")


    # samples and returns a given number of points on the Bezier curve - either uniformly or randomly (t-values)
    def sample_curve(self, num_points, sampling_type=SamplingType.UNIFORM, curve_to_sample = None):
        start_time = time.time()

        sampled_parts_queue = mp.Queue(maxsize=199)

        if curve_to_sample is None:
            curve_to_sample = self.control_points

        self.sampled_points = np.zeros((num_points, 2), dtype=float)  # Initialize as an empty array
        processes = []

        if sampling_type == SamplingType.UNIFORM:
            t_values = np.linspace(0, 1, num_points)

        # likely does not use to contain the start point and will certainly not contain the end point (np.random is in the range [0,1))
        elif sampling_type == SamplingType.RANDOM:
            t_values = np.random.rand(num_points)
            t_values.sort()
        else:
            raise ValueError("Invalid sampling type. Use SamplingType.UNIFORM or SamplingType.RANDOM.")

        start_index = 0
        for i in range(self.num_processes):
            # start_index = i * self.num_processes # inclusive
            end_index = start_index + num_points // self.num_processes # exclusive

            if i == self.num_processes - 1:
                end_index = num_points

            if end_index > num_points:
                end_index = num_points

            time_before_thread_start = time.time()

            processes.append(
                mp.Process(target=self.calc_bezier_value_threaded, args=(t_values[start_index:end_index], curve_to_sample, i, num_points, sampled_parts_queue)))

            # print(f"processs {i} will process t_values from index {start_index} to {end_index}")
            start_index = end_index


        for i in range(self.num_processes):
            processes[i].start()

        time_after_start = time.time()

        # Collect results from all processes
        results_received = 0

        while results_received < self.num_processes:
            process_id, sampled_points = sampled_parts_queue.get()
            print(f"Process {process_id} got from queue")

            if process_id == (self.num_processes - 1):
                self.sampled_points[process_id * (num_points // self.num_processes):] = np.array(sampled_points)
            else:
                self.sampled_points[
                process_id * (num_points // self.num_processes):(process_id + 1) * (
                        num_points // self.num_processes)] = np.array(sampled_points)
            results_received += 1


        # if threading.current_thread().name == "MainThread":
        # if __name__ == "__main__":
        for i in range(self.num_processes):
            # print(f"sampled: {sampled_points}")
            processes[i].join()
            # print(f"Process {i} joined.")




        end_time = time.time()
        # print(f"Time taken to sample curve: {end_time - start_time:.4f} seconds")
        # print(f"Time taken to start processes: {time_after_start - start_time:.4f} seconds")
        # print(f"Time taken to join processes: {end_time - time_after_start:.4f} seconds")

        # if process_id == (self.num_processes - 1):
        #     self.sampled_points[process_id * (num_points // self.num_processes):] = np.array(sampled_points)
        # else:
        #     self.sampled_points[
        #     process_id * (num_points // self.num_processes):(process_id + 1) * (
        #                 num_points // self.num_processes)] = np.array(sampled_points)



        sampled_parts_queue.close()
        return self.sampled_points

    # samples and returns a given number of points on the Bezier curve - either uniformly or randomly (t-values)
    def sample_curve_sequential(self, num_points, sampling_type=SamplingType.UNIFORM, curve_to_sample=None):
        start_time = time.time()

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

        time_before_sampling = time.time()
        for t_value in t_values:
            point = self.calc_bezier_value(t_value, curve_to_sample)
            self.sampled_points.append(point)

        self.sampled_points = np.array(self.sampled_points)
        end_time = time.time()
        # print(f"Time taken to sample curve: {end_time - start_time:.4f} seconds")
        # print("Time taken to sample points: ", end_time - time_before_sampling)

        return self.sampled_points

        # for t_value in t_values:
        #     point = self.calc_bezier_value(t_value, curve_to_sample)
        #     self.sampled_points.append(point)
        #
        # self.sampled_points = np.array(self.sampled_points)
        #
        # return self.sampled_points

    def get_control_points(self):
        return self.control_points

    def set_control_points(self, control_points):
        self.control_points = control_points



if __name__ == "__main__":

    num_points = 1001
    degree = 5

    start_time = time.time()

    bezier_curve_multi_threaded = BezierCurveMultiProcess(num_processes=3)
    # Generate a 10th degree Bezier curve
    bezier_curve_multi_threaded.generate_curve(degree)
    # Sample 5 points uniformly on the Bezier curve
    sampled_points_multi_threaded = bezier_curve_multi_threaded.sample_curve(num_points, SamplingType.UNIFORM)

    end_time = time.time()
    print(f"Time taken for multi-threaded Bezier curve sampling: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    # Example usage of the BezierCurve class
    bezier_curve = BezierCurve()
    # Generate a 10th degree Bezier curve
    bezier_curve.generate_curve(degree)
    # Sample 5 points uniformly on the Bezier curve
    sampled_points = bezier_curve.sample_curve(num_points, SamplingType.UNIFORM)
    end_time = time.time()
    print(f"Time taken for single-threaded Bezier curve sampling: {end_time - start_time:.4f} seconds")



    # start_time = time.time()
    #
    # # bezier_curve_multi_threaded.generate_curve(degree)
    # # Sample 5 points uniformly on the Bezier curve
    # sampled_points = bezier_curve_multi_threaded.sample_curve_sequential(num_points, SamplingType.UNIFORM)
    #
    # end_time = time.time()
    # print(f"Time taken for single-threaded Bezier curve sampling: {end_time - start_time:.4f} seconds")
    #
    # # print("Sampled points:", sampled_points)
    # # print("Sampled points multi-threaded:", sampled_points_multi_threaded)
    #
    # print((sampled_points_multi_threaded == sampled_points).all())


