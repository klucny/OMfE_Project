from functools import total_ordering

from bezier_curve_creation import BezierCurve, BezierCurveMultiProcess
import scipy as sp
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os


# class that executes the simulated annealing algorithm to reconstruct a Bezier curve
# it takes samples from a Bezier curve (provides by BezierCurve class) and tries to find the control points of the Bezier curve
# by minimizing the squared l2-norm between the sampled points and the points sampled from the created control points
class SimulatedAnnealing:
    def __init__(self, degree, num_points_sampled, iterations= 5000, given_curve=None):
        # degree of the Bezier curve to reconstruct
        self.degree = degree

        self.num_points_sampled = num_points_sampled # number of points sampled from the Bezier curve to reconstruct the control points
        self.iterations = iterations # number of iterations for the simulated annealing algorithm

        # Create a BezierCurve instance and generate the curve
        # self.bezier_curve = BezierCurveMultiProcess(6)
        self.bezier_curve = BezierCurve()

        # using the given_curve feature, you can provide a specific set of control points to reconstruct e.g. so that multiple runs can be executed on the same curve
        if given_curve is not None:
            self.bezier_curve.set_control_points(given_curve)
        else:
            self.bezier_curve.generate_curve(degree)

        # sample points from the Bezier curve to reconstruct the control points
        self.sampled_points = self.bezier_curve.sample_curve(num_points_sampled)

        # maximum range for parameter changes (i.e. value of  60 will result in changes of min -30 to max +30)
        self.max_range = 4

        # variables to store the final parameters and fitness values for plotting
        self.fitness_values = []
        self.par_end = None
        self.fitness_end = None
        self.probabilities = np.zeros((3,2,), dtype=float)  # probabilities for the look-ahead parameter change


    # function that calculates the squared distance between the original and reconstructed Bezier curve
    # this function can only be used after the simulated annealing algorithm has been executed
    def calc_real_error(self):
        return np.sum(np.abs(self.bezier_curve.get_control_points() - self.par_end))


    # takes control points which are the parameters in the simulated annealing algorithm -
    # compares the current parameters with the actual control points of the Bezier curve based on the squared l2-norm
    def calc_fitness(self, parameters):
        points_same_coordinates_penality = 0

        threshold = 1 # threshold below which points that are considered too close to each other

        amount_too_close = 0 # amount of points that are too close to each other -> will be penalized in the fitness function

        for i in range(parameters.shape[0]-1):
            amount_too_close += parameters[np.sum(np.abs(parameters-parameters[i])) < threshold].shape[0]

        # if there are points that are too close to each other, add a penalty to the fitness function, the weight is currently set to 1000000 without much thought
        points_same_coordinates_penality += amount_too_close * 1000000



        sample_from_par = self.bezier_curve.sample_curve(self.num_points_sampled, curve_to_sample=parameters)


        assert(sample_from_par.shape == self.sampled_points.shape), "Sampled points from the Bezier curve and sampled points from the parameters must have the same shape"

        # try to keep fitness function as low as possible
        fitness = np.sum(np.square(self.sampled_points - sample_from_par)) + points_same_coordinates_penality

        return fitness


    # cooling schedule function for the simulated annealing algorithm
    def cosine_cooldown(self, i):
        return 0.5*math.cos((math.pi * i )/ self.iterations) + 0.5

    def exponential_cooldown(self, i):
        return math.e ** (-(10/self.iterations) * i)


    def look_ahead_parameter_change(self, par_cur, update_fitness=True):
        # adapt step size to be in the three separate range
        third_max_range = self.max_range / 3
        x_steps = np.random.rand(par_cur.shape[0]) * third_max_range - (third_max_range / 2)
        y_steps= np.random.rand(par_cur.shape[0]) * third_max_range - (third_max_range / 2)

        # compute weighted probabilities to move in certain direction
        for index, parameter in enumerate(par_cur):
            if index == 0 or index == par_cur.shape[0] - 1:
                # first and last point should not be changed
                continue

            changed_parameters = par_cur

            # fitness_x = np.zeros(3, dtype=float)
            # fitness_y = np.zeros(3, dtype=float)

            # format (x1,y1), (x2,y2), (x3,y3)

            if update_fitness:
                for fit_idx, direction in enumerate([-1, 0, 1]):
                    changed_parameters[index][0] += direction * third_max_range * 0.5
                    self.probabilities[fit_idx][0] = self.calc_fitness(changed_parameters[index])
                    changed_parameters[index][0] -= direction * third_max_range * 0.5


                self.probabilities[0:3,0]= self.probabilities[0:3,0] / np.sum(self.probabilities[0:3,0])

            choose_rand_x_interval = np.random.rand()

            if choose_rand_x_interval < self.probabilities[0,0]:
                par_cur[index][0] += x_steps[index] -third_max_range
            elif choose_rand_x_interval < self.probabilities[0,0] + self.probabilities[1,0]:
                par_cur[index][0] += x_steps[index]
            else:
                par_cur[index][0] += x_steps[index] + third_max_range

            if update_fitness:
                # y parameter change
                for fit_idx, direction in enumerate([-1, 0, 1]):
                    changed_parameters[index][1] += direction * third_max_range * 0.5
                    self.probabilities[fit_idx][1] = self.calc_fitness(changed_parameters[index])
                    changed_parameters[index][1] -= direction * third_max_range * 0.5

                self.probabilities[0:3, 1] = self.probabilities[0:3, 1] / np.sum(self.probabilities[0:3, 1])

            choose_rand_y_interval = np.random.rand()

            if choose_rand_y_interval < self.probabilities[0,1]:
                par_cur[index][1] += y_steps[index] - third_max_range
            elif choose_rand_y_interval < self.probabilities[0,1] + self.probabilities[1,1]:
                par_cur[index][1] += y_steps[index]
            else:
                par_cur[index][1] += y_steps[index] + third_max_range

        return par_cur



    # start_point = first_sampled_point
    # end_point = last_sampled_point
    # inbetween points are evenly spaced points on the Bezier curve
    def start_points_on_curve(self):
        par_initial = self.sampled_points[::self.num_points_sampled // (self.degree+1)]
        assert(par_initial.shape[0]== self.degree + 1), "Number of initial parameters must match degree + 1"
        assert np.all(par_initial[0]== self.sampled_points[0]), "First point must be the first sampled point"
        par_initial[-1]  = self.sampled_points[-1]  # ensure last point is the last sampled point

        return par_initial


       # actual simulated annealing algorithm
    def simulated_annealing(self):
        # initialize parameters with random control points
        # randomly generate control points within a reasonable range
        # par_cur = np.random.rand(self.bezier_curve.get_control_points().shape[0],self.bezier_curve.get_control_points().shape[1])

        # fixed control points starting all at (0,0)
        # par_cur = np.zeros((self.bezier_curve.get_control_points().shape[0], self.bezier_curve.get_control_points().shape[1]), dtype=float)

        # fixed control points starting all at (50,50)
        # par_cur = np.full((self.bezier_curve.get_control_points().shape[0], self.bezier_curve.get_control_points().shape[1]), 50, dtype=float)

        # fixed example control points for testing degree 4 Bezier curve
        # par_cur = np.array([[0,0], [100,0],[100,100],[0, 100], [50,50]], dtype=float)

        # points evenly spaced on sampled points
        par_cur = self.start_points_on_curve()

        # self.plot_starting_and_og_sampled(par_cur)

        # scale the parameters to a reasonable range based on the sampled points
        # comment out if starting points already scaled
        x_max_with_buffer = np.max(self.sampled_points[:, 0]) * 1.2 # 20% buffer
        y_max_with_buffer = np.max(self.sampled_points[:, 1]) * 1.2 # 20% buffer
        # par_cur[:,0] *= (x_max_with_buffer)
        # par_cur[:,1] *= (y_max_with_buffer)


        par_best = par_cur.copy()

        fit_cur = self.calc_fitness(par_cur)
        fit_best = fit_cur

        self.fitness_values.append(fit_cur)

        # Sanity Check: ensure the first and last points of the reconstructed curve match the sampled points
        assert np.all( par_cur[0] == self.sampled_points[0]), f"First point must be the first sampled point"
        assert np.all(par_cur[-1] == self.sampled_points[-1]), f"Last point must be the last sampled point"


        for i in range(self.iterations):
            # self.plot_starting_and_og_sampled(par_cur)
            # assert np.all(par_cur[0] == self.sampled_points[0]), f"Iteration {i}:First point must be the first sampled point"
            # assert np.all(par_cur[-1] == self.sampled_points[-1]), f"Iteration {i}:Last point must be the last sampled point"

            if fit_cur > 30000:
                self.max_range = 4
            elif fit_cur < 10000:
                self.max_range = 2
            else:
                self.max_range = 2
            # set new temperature based on the current iteration
            temperature = self.exponential_cooldown(i)

            # randomly change the parameters within the defined range

            # straight-forward version that changes all parameters at once
            # parameter_change = np.random.rand(self.degree - 1, 2) * self.max_range - (self.max_range / 2)
            # par_new = par_cur
            # par_new[1:par_new.shape[0]-1] = par_cur[1:par_new.shape[0]-1] + parameter_change

            # straight-forward version that changes one parameter at a time
            parameter_change = np.random.rand(1, 2) * self.max_range - (self.max_range / 2)
            par_new = par_cur.copy()
            index_to_change = np.random.randint(1, par_cur.shape[0] - 1)  # do not change first and last point
            par_new[index_to_change] = par_cur[index_to_change] + parameter_change


            # lookahead version
            # if i % 10:
            #     par_new = self.look_ahead_parameter_change(par_cur)
            # else:
            #     par_new = self.look_ahead_parameter_change(par_cur, update_fitness=False)

            par_new[par_new < -20] = -20  # ensure parameters are > -20

            par_new[:,0 > x_max_with_buffer] = x_max_with_buffer  # ensure x parameters do not exceed the max range
            par_new[:,1 > y_max_with_buffer] = y_max_with_buffer  # ensure y parameters do not exceed the max range
            par_new[par_new > 120] = 120  # ensure parameters do not exceed 120

            fit_new = self.calc_fitness(par_new)
            self.fitness_values.append(fit_new)

            if fit_new < fit_cur:
                par_best = par_new
                fit_best = fit_new

                par_cur = par_new
                fit_cur = fit_new

            elif (math.e ** (abs(fit_new - fit_cur) / temperature)) > np.random.rand():
                par_cur = par_new

            # print update
            if i % (self.iterations/10) == 0:
                print(f"Iteration {i}, Fit best: {fit_best}, Fit new: {fit_new}, Temperature: {temperature}")


        self.par_end = par_best
        # Sanity Check: ensure the first and last points of the reconstructed curve match the sampled points
        assert np.all(self.par_end[0] == self.sampled_points[0]), "First point must be the first sampled point"
        assert np.all(self.par_end[-1] == self.sampled_points[-1]), "Last point must be the last sampled point"

        self.fitness_end = fit_best


# ------------ PLOTTING FUNCTIONS ---------------

    def plot_starting_and_og_sampled(self, starting_points):
        plt.scatter(self.sampled_points[:, 0], self.sampled_points[:, 1], color='blue', label='Original Curve: Sampled Points')
        plt.scatter(starting_points[:, 0], starting_points[:, 1], color='red', label='Reconstruction Starting Points', alpha=0.7)
        plt.title("Sampled Points and Starting Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()


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


    def plot_all(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))

        # Plot fitness values
        axs[0].plot(self.fitness_values[20:])
        axs[0].set_title("Fitness Values Over Iterations")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Fitness Value")

        # Plot control points
        original_control_points = self.bezier_curve.get_control_points()
        reconstructed_control_points = self.par_end
        axs[1].scatter(original_control_points[:, 0], original_control_points[:, 1], color='blue',
                       label='Original Control Points')
        axs[1].scatter(reconstructed_control_points[:, 0], reconstructed_control_points[:, 1], color='red',
                       label='Reconstructed Control Points')
        axs[1].legend()
        axs[1].set_title("Original vs Reconstructed Control Points")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")

        # Plot Bezier curves
        t_values = np.linspace(0, 1, 20000)
        reconstructed_bezier_curve = np.array([self.bezier_curve.calc_bezier_value(t, self.par_end) for t in t_values])
        original_curve = np.array(
            [self.bezier_curve.calc_bezier_value(t, self.bezier_curve.control_points) for t in t_values])
        axs[2].scatter(reconstructed_bezier_curve[:, 0], reconstructed_bezier_curve[:, 1],
                       label='Reconstructed Bezier Curve', color="red")
        axs[2].scatter(original_curve[:, 0], original_curve[:, 1], label='Original Bezier Curve', color="blue")
        axs[2].set_title("Reconstructed Bezier Curve")
        axs[2].set_xlabel("X")
        axs[2].set_ylabel("Y")
        axs[2].legend()

        plt.tight_layout()
        plt.figtext(0.25, 0.01, f"Final Fitness: {self.fitness_end:.4f}", ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.7, "pad":5})
        plt.figtext(0.75, 0.01, f"Control Point Error: {self.calc_real_error():.4f}", ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.7, "pad":5})

        plt.show()



def run_sa_plot(sa):
    sa.simulated_annealing()
    sa.plot_all()

if __name__ == '__main__':

    # degree of curve to reconstruct
    degree = 8

    # Create a BezierCurve instance and generate the curve
    bezier_curve = BezierCurve()
    bezier_curve.generate_curve(degree)

    # Get the control points of the generated Bezier curve, they will be passed to the SimulatedAnnealing instance executed by the multiprocess pool
    curve_to_reconstruct = bezier_curve.get_control_points()

    # Number of runs on the same curve and number of processes that will execute the specified number of runs
    runs_on_same_curve = 10
    num_processes = 10

    # create a list of SimulatedAnnealing instances, each initialized with the same curve to reconstruct
    # sa_instances = [
    #     SimulatedAnnealing(degree=degree, num_points_sampled=207, iterations=50000, given_curve=curve_to_reconstruct) for _ in
    #     range(runs_on_same_curve)]

    # currently only num_points_sampled % (degree + 1) == 0 is supported, because of the way the initial parameters are set on the curve
    sa_instances =  [SimulatedAnnealing(degree=degree, num_points_sampled=108, iterations=20000, given_curve=curve_to_reconstruct) for _ in
        range(runs_on_same_curve)]


    # Use multiprocessing to run the simulated annealing algorithm on multiple instances in parallel
    with mp.Pool(processes=num_processes) as pool:
        pool.map(run_sa_plot, sa_instances)

    # with mp.Pool(processes=num_processes) as pool:
    #     pool.map(run_sa_plot, sa_instances_2)


        # sa.plot_fitness()
        # sa.plot_control_points()
        # sa.plot_bezier_curve()

