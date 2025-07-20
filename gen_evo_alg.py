from bezier_curve_creation import BezierCurve, BezierCurveMultiProcess
import scipy as sp
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

class GeneticAlgorithm:
    def __init__(self, degree, num_points_sampled,  iterations=1000, population_size=11175, mutation_rate=0.1, crossover_rate=0.1, given_curve=None):
        self.population_size = population_size
        # (n^2 - n)/2 = self.population_size <-- n must be manually set
        self.n = 150

        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_points_sampled = num_points_sampled
        self.degree = degree

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

        # variables to store the final parameters and fitness values for plotting
        self.fitness_values_population_avg = []
        # stores the best parameters of the population for each iteration
        self.fitness_values_population_best = []
        self.par_end = None
        self.fitness_end = None
    # function that calculates the squared distance between the original and reconstructed Bezier control points
    # this function can only be used after the simulated annealing algorithm has been executed
    def calc_real_error(self):
        return np.sum(np.abs(self.bezier_curve.get_control_points() - self.par_end))

    # takes control points which are the parameters in the simulated annealing algorithm -
    # compares the current parameters with the actual control points of the Bezier curve based on the squared l2-norm
    def calc_fitness(self, parameters):
        points_same_coordinates_penality = 0

        threshold = 1  # threshold below which points that are considered too close to each other

        amount_too_close = 0  # amount of points that are too close to each other -> will be penalized in the fitness function

        for i in range(parameters.shape[0] - 1):
            amount_too_close += parameters[np.sum(np.abs(parameters - parameters[i])) < threshold].shape[0]

        # if there are points that are too close to each other, add a penalty to the fitness function, the weight is currently set to 1000000 without much thought
        points_same_coordinates_penality += amount_too_close * 10

        sample_from_par = self.bezier_curve.sample_curve(self.num_points_sampled, curve_to_sample=parameters)

        assert (
                    sample_from_par.shape == self.sampled_points.shape), "Sampled points from the Bezier curve and sampled points from the parameters must have the same shape"

        # try to keep fitness function as low as possible
        fitness = np.sum(np.square(self.sampled_points - sample_from_par)) + points_same_coordinates_penality

        return fitness

    def calc_fitness_population(self, population):
        # calculate the fitness for each individual in the population
        fitness_values = np.array([np.array([idx, self.calc_fitness(individual)]) for idx, individual in enumerate(population)])
        assert(fitness_values.shape == (self.population_size, 2)), "Fitness values must have shape (population_size, 2)"
        fitness_values = fitness_values[fitness_values[:, 1].argsort()]
        return fitness_values

    # start_point = first_sampled_point
    # end_point = last_sampled_point
    # inbetween points are evenly spaced points on the Bezier curve
    def start_points_on_curve(self):
        par_initial = self.sampled_points[::self.num_points_sampled // (self.degree + 1)]
        assert (par_initial.shape[0] == self.degree + 1), "Number of initial parameters must match degree + 1"
        assert np.all(par_initial[0] == self.sampled_points[0]), "First point must be the first sampled point"
        par_initial[-1] = self.sampled_points[-1]  # ensure last point is the last sampled point

        return par_initial

    def start_points_random(self):
        par_initial = np.random.rand(self.population_size, self.degree + 1, 2) * 100
        assert (par_initial.shape[1] == self.degree + 1), "Number of initial parameters must match degree + 1"

        return par_initial

    def curve_sex(self):
        pass

    def gen_alg(self):
        par_cur = self.start_points_random()

        # calculate the fitness for the initial population
        fitness_values = self.calc_fitness_population(par_cur)

        fit_best = fitness_values[0, 1]
        par_best = par_cur[fitness_values[0, 0].astype(int)]

        self.fitness_values_population_best.append(fit_best)
        self.fitness_values_population_avg.append(np.mean(fitness_values[:, 1]))


        for i in range(self.iterations):
            # print update
            if i % (self.iterations / 10) == 0:
                print(f"Iteration {i}, Fit best: {fit_best}, Fit new: {fitness_values[0, 1]}")
            new_population = np.zeros_like(par_cur)

            pairs = [(x, y) for x in range(self.n) for y in range(x + 1, self.n)]

            for idx, (i, j) in enumerate(pairs):
                new_population[idx] = par_cur[fitness_values[i, 0].astype(int)] + par_cur[fitness_values[j, 0].astype(int)]


            par_cur = new_population/2
            # calculate the fitness for the new population
            fitness_values = self.calc_fitness_population(par_cur)

            if fitness_values[0,1] < fit_best:
                fit_best = fitness_values[0, 1]
                par_best = par_cur[fitness_values[0, 0].astype(int)]

            self.fitness_values_population_best.append(fitness_values[0,1])
            self.fitness_values_population_avg.append(np.mean(fitness_values[:, 1]))



        self.par_end = par_best
        self.fitness_end = fit_best

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.fitness_values_population_best, label="Best Fitness")
        plt.plot(self.fitness_values_population_avg, label="Average Fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Value")
        plt.title("Genetic Algorithm Fitness Progression")
        plt.legend()
        plt.grid(True)
        plt.show()

    # this function plots the reconstructed Bezier curve and the original Bezier curve (based on sampled points)
    def plot_bezier_curve(self):
        t_values = np.linspace(0, 1, 10000)
        reconstructed_bezier_curve = np.array(
            [self.bezier_curve.calc_bezier_value(t, self.par_end) for t in t_values])
        original_curve = np.array(
            [self.bezier_curve.calc_bezier_value(t, self.bezier_curve.control_points) for t in t_values])

        plt.scatter(reconstructed_bezier_curve[:, 0], reconstructed_bezier_curve[:, 1],
                    label='Reconstructed Bezier Curve')
        # plt.scatter(self.sampled_points[:, 0], self.sampled_points[:, 1], color='orange', label='Sampled Points')
        plt.scatter(original_curve[:, 0], original_curve[:, 1], label='Original Bezier Curve')

        plt.title("Reconstructed Bezier Curve")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
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

    def plot_all(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))

        # Plot fitness values
        axs[0].plot(self.fitness_values_population_best[20:])
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

if __name__ == '__main__':
    genetic = GeneticAlgorithm(degree=6, num_points_sampled=100, iterations=10, mutation_rate=0.1, crossover_rate=0.1)
    genetic.gen_alg()
    genetic.plot_all()
