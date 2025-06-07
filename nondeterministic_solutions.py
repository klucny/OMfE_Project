from bezier_curve_creation import BezierCurve
import scipy as sp
import math
import numpy as np
import matplotlib.pyplot as plt


# class that executes the simulated annealing algorithm to reconstruct a Bezier curve
# it takes samples from a Bezier curve (provides by BezierCurve class) and tries to find the control points of the Bezier curve
# by minimizing the squared l2-norm between the sampled points and the points sampled from the created control points
class SimulatedAnnealing:
    def __init__(self, degree, num_points_sampled):
        self.degree = degree
        self.num_points_sampled = num_points_sampled

        # Create a BezierCurve instance and generate the curve
        self.bezier_curve = BezierCurve()
        self.bezier_curve.generate_curve(degree)

        self.sampled_points = self.bezier_curve.sample_curve(num_points_sampled)

        self.max_range = 4 # maximum range for parameter changes (results in changes of min -30 to max +30)

        self.fitness_values = []
        self.par_end = None



    # takes control points which are the parameters in the simulated annealing algorithm -
    # compares the current parameters with the actual control points of the Bezier curve based on the squared l2-norm
    def calc_fitness(self, parameters):
        sample_from_par = self.bezier_curve.sample_curve(self.num_points_sampled, curve_to_sample=parameters)
        fitness = np.sum(np.square(self.sampled_points - sample_from_par))
        return fitness


    def plot_points(self, points, title="Points"):
        plt.scatter(points[:, 0], points[:, 1], label=title)
        plt.title("Sampled Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def simulated_annealing(self):
        # initialize parameters with random control points
        par_cur = np.random.rand(self.bezier_curve.get_control_points().shape[0],self.bezier_curve.get_control_points().shape[1])

        # scale the parameters to a reasonable range based on the sampled points
        x_max_with_buffer = np.max(self.sampled_points[:, 0]) * 1.2 # 20% buffer
        y_max_with_buffer = np.max(self.sampled_points[:, 1]) * 1.2 # 20% buffer

        par_cur[:,0] *= (x_max_with_buffer)
        par_cur[:,1] *= (y_max_with_buffer)

        self.plot_points(par_cur)

        par_best = par_cur.copy()

        fit_cur = self.calc_fitness(par_cur)
        fit_best = fit_cur

        self.fitness_values.append(fit_cur)

        for i in range(10000):
            # set new temperature based on the current iteration
            temperature = math.e ** (-0.001 * (i - 1000))
            if i % 500 == 0:
                print(f"Iteration {i}, Fit best: {fit_best}, Fit current: {fit_cur}, Temperature: {temperature}")


            # randomly change the parameters within the defined range
            parameter_change_x = (np.random.rand() * self.max_range) - (self.max_range / 2)
            parameter_change_y = (np.random.rand() * self.max_range) - (self.max_range / 2)

            par_new = par_cur + np.array([parameter_change_x, parameter_change_y])
            par_new[par_new < 0] = 0  # ensure parameters are non-negative

            par_new[:,0 > x_max_with_buffer] = x_max_with_buffer  # ensure x parameters do not exceed the max range
            par_new[:,1 > y_max_with_buffer] = y_max_with_buffer  # ensure y parameters do not exceed the max range
            par_new[par_new > 100] = 100  # ensure parameters do not exceed 100

            fit_new = self.calc_fitness(par_new)
            self.fitness_values.append(fit_new)

            if fit_new < fit_cur:

                par_best = par_new
                fit_best = fit_new

                par_cur = par_new
                fit_cur = fit_new

            elif (math.e ** (abs(fit_new - fit_cur) / temperature)) > np.random.rand():
                par_cur = par_new

        self.par_end = par_best


    def plot_fitness(self):
        plt.plot(self.fitness_values)
        plt.title("Fitness Values Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Value")
        plt.show()

    def plot_control_points(self):
        original_control_points = self.bezier_curve.get_control_points()
        reconstructed_control_points = self.par_end
        print("Original Control Points:\n", original_control_points)
        print("Reconstructed Control Points:\n", reconstructed_control_points)

        plt.scatter(original_control_points[:, 0], original_control_points[:, 1], color='blue',
                    label='Original Control Points')
        plt.scatter(reconstructed_control_points[:, 0], reconstructed_control_points[:, 1], color='red',
                    label='Reconstructed Control Points')
        plt.legend()
        plt.title("Original vs Reconstructed Control Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    # this function plots the reconstructed Bezier curve and the original Bezier curve (based on sampled points)
    def plot_bezier_curve(self):
        t_values = np.linspace(0, 1, 10000)
        reconstructed_bezier_curve = np.array([self.bezier_curve.calc_bezier_value(t, self.par_end) for t in t_values])
        original_curve = np.array([self.bezier_curve.calc_bezier_value(t, self.bezier_curve.control_points) for t in t_values])


        plt.scatter(reconstructed_bezier_curve[:, 0], reconstructed_bezier_curve[:, 1], label='Reconstructed Bezier Curve')
        # plt.scatter(self.sampled_points[:, 0], self.sampled_points[:, 1], color='orange', label='Sampled Points')
        plt.scatter(original_curve[:, 0], original_curve[:, 1], label='Original Bezier Curve')

        plt.title("Reconstructed Bezier Curve")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()


sa = SimulatedAnnealing(degree=4, num_points_sampled=1000)
sa.simulated_annealing()
sa.plot_fitness()
sa.plot_control_points()
sa.plot_bezier_curve()

