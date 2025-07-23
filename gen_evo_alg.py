from bezier_curve_creation import BezierCurve, BezierCurveMultiProcess
import scipy as sp
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

class GeneticAlgorithm:
    def __init__(self, degree, num_points_sampled,  iterations=1000, population_size=4950, mutation_rate=0.1, crossover_rate=0.1, given_curve=None, mut_all=False, given_population=None, preround=False):
        self.population_size = population_size
        # (n^2 - n)/2 = self.population_size <-- n must be manually set
        self.n = 100

        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_points_sampled = num_points_sampled
        self.degree = degree
        self.mut_all = mut_all
        self.given_population = given_population
        self.preround = preround # <--- used to determine the fitness of a population of one thread, to then use this information when merging the populations of multiple threads

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

        # stors the best final parameters and fitness values after the algorithm has been executed
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

    # calculates the fitness for the whole population, calls calc_fitness for each individual in the population and returns the fitness values sorted by fitness (best=lowest first)
    def calc_fitness_population(self, population):
        # calculate the fitness for each individual in the population
        fitness_values = np.array([np.array([idx, self.calc_fitness(individual)]) for idx, individual in enumerate(population)])
        assert(fitness_values.shape == (self.population_size, 2)), f"Fitness values must have shape (population_size, 2), but got {fitness_values.shape}"
        fitness_values = fitness_values[fitness_values[:, 1].argsort()]
        # print(f"Fitness values shape: {fitness_values.shape}, first 5 values: {fitness_values[:5]}")
        # returns array of shape (population_size, 2) where the first column is the index of the individual in the population array and the second column is the fitness value
        return fitness_values

    # as the names suggest: this function generates the initial parameters for the population at random
    def start_points_random(self):
        par_initial = np.random.rand(self.population_size, self.degree + 1, 2) * 100
        par_initial[:,0,:] = self.sampled_points[0]  # ensure first point is the first sampled point
        par_initial[:,-1,:] = self.sampled_points[-1]
        assert (par_initial.shape[1] == self.degree + 1), "Number of initial parameters must match degree + 1"
        return par_initial

    # evenly spaces the control points line by line
    # THIS DOES NOT WORK WELL
    def start_points_evenly_spaced(self):
        x_distribution = np.linspace(0,100, self.degree + 1)
        y_distribution = np.linspace(0,100, self.population_size)
        par_initial = np.array([[x, y]  for y in y_distribution for x in x_distribution]).reshape(self.population_size, self.degree + 1, 2)
        print(par_initial)
        assert (par_initial.shape[1] == self.degree + 1), "Number of initial parameters must match degree + 1"
        return par_initial

    # ------------------------- BELOW: Algorithm implementations -------------------------

    # this is the random algorithm, which always randomly generates new parameters and calculates the fitness for each iteration
    # used as a sanity check to see if the genetic algorithm is actually better than just randomly generating parameters
    def rand_alg(self):
        par_cur = self.start_points_random()
        # par_cur = self.start_points_evenly_spaced()

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

            new_population = self.start_points_random()
            par_cur = new_population

            # calculate the fitness for the new population
            fitness_values = self.calc_fitness_population(par_cur)

            if fitness_values[0, 1] < fit_best:
                fit_best = fitness_values[0, 1]
                par_best = par_cur[fitness_values[0, 0].astype(int)]

            self.fitness_values_population_best.append(fitness_values[0, 1])
            self.fitness_values_population_avg.append(np.mean(fitness_values[:, 1]))

        self.par_end = par_best
        self.fitness_end = fit_best

    def gen_alg(self):
        par_cur = self.start_points_random()
        # par_cur = self.start_points_evenly_spaced()

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
                # weighted version
                l1_norm = np.linalg.norm([i+1,j+1], ord = 1)
                weights = [(1+i)/l1_norm, (1+j)/l1_norm]
                assert(np.sum(weights) == 1.0)
                # note that the fitness values are sorted, so the first value is the best one -> smaller  value -> switch weights such that the better one has a higher weight
                new_population[idx] = par_cur[fitness_values[i, 0].astype(int)] *weights[1]  + par_cur[fitness_values[j, 0].astype(int)]*weights[0]

                # unweighted version
                # new_population[idx] = par_cur[fitness_values[i, 0].astype(int)] *0.5 + par_cur[fitness_values[j, 0].astype(int)] *0.5

            par_cur = new_population

            # calculate the fitness for the new population
            fitness_values = self.calc_fitness_population(par_cur)

            # check if the best fitness value has improved
            if fitness_values[0,1] < fit_best:
                fit_best = fitness_values[0, 1]
                par_best = par_cur[fitness_values[0, 0].astype(int)]

            self.fitness_values_population_best.append(fitness_values[0,1])
            self.fitness_values_population_avg.append(np.mean(fitness_values[:, 1]))

        # save the best parameters and fitness values
        self.par_end = par_best
        self.fitness_end = fit_best

    def exponential_cooldown(self, i):
        return math.e ** (-(0.001) * (i+13))

    def gen_alg_with_SA(self):
        par_cur = None

        if self.given_population is None:
            par_cur = self.start_points_random()
            # par_cur = self.start_points_evenly_spaced()
        else:
            par_cur = self.given_population
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

            # if i == 2 or i == 70 or i == 120:
            #
            #     assert(self.population_size%5 == 0), "Population size must be divisible by 5 for this implementation"
            #     pop_fifth = int(self.population_size / 5)
            #     random_matrix = np.random.rand(pop_fifth, self.degree + 1, 2)
            #
            #     change = 0
            #
            #     if i == 2:
            #         change = 5
            #     elif i == 70:
            #         change = 1
            #     elif i == 120:
            #         change = 0.5
            #
            #     print("Parents reproduce themselves, with shifts")
            #     best_parents = fitness_values[:pop_fifth, 0]
            #     best_parents_indices = best_parents.astype(int)
            #     best_parents_curves = par_cur[best_parents_indices]
            #     new_population[:pop_fifth] = best_parents_curves + random_matrix * change  - (change / 2)
            #     random_matrix = np.random.rand(pop_fifth, self.degree + 1, 2)
            #     new_population[pop_fifth:2*pop_fifth] = best_parents_curves + random_matrix* (change/2) - (change/4)
            #
            #     new_population[2*pop_fifth:3*pop_fifth] = best_parents_curves
            #
            #     random_matrix = np.random.rand(pop_fifth, self.degree + 1, 2)
            #     new_population[3*pop_fifth:4*pop_fifth] = best_parents_curves + random_matrix * change - (change / 2)
            #
            #     random_matrix = np.random.rand(pop_fifth, self.degree + 1, 2)
            #     new_population[4*pop_fifth:] = best_parents_curves +  random_matrix * (change / 2) - (change / 4)
            #
            pairs = [(x, y) for x in range(self.n) for y in range(x + 1, self.n)]

            for idx, (i, j) in enumerate(pairs):
                random_number = np.random.random()


                if random_number >= self.exponential_cooldown(i):
                    # weighted version
                    l1_norm = np.linalg.norm([i + 1, j + 1], ord=1)
                    weights = [(1 + i) / l1_norm, (1 + j) / l1_norm]
                    assert (np.sum(weights) == 1.0)
                    # note that the fitness values are sorted, so the first value is the best one -> smaller  value -> switch weights such that the better one has a higher weight
                    new_population[idx] = par_cur[fitness_values[i, 0].astype(int)] * weights[1] + par_cur[fitness_values[j, 0].astype(int)] * weights[0]

                    # unweighted version
                    # new_population[idx] = par_cur[fitness_values[i, 0].astype(int)] *0.5 + par_cur[fitness_values[j, 0].astype(int)] *0.5

                else:
                    if self.mut_all:
                        # THIS CASE IS OUTDATED
                        # mutate all control points of the current individual
                        new_population[idx] = par_cur[fitness_values[i, 0].astype(int)]  + (np.random.rand(self.degree+1,2)*10 -5)
                    else:
                        # mutate only one control point of the current individual
                        new_population[idx] = par_cur[fitness_values[i, 0].astype(int)].copy()
                        random_control_point = np.random.randint(1, self.degree)

                        # checks the fitness value of the individual and mutates the control point accordingly
                        if fitness_values[i, 1] < 10:
                            new_population[idx][random_control_point] += (np.random.rand(2) * 10 - 5)
                        elif fitness_values[i, 1] < 50:
                            new_population[idx][random_control_point] += (np.random.rand(2) * 20 - 10)
                        else:
                            new_population[idx][random_control_point] += (np.random.rand(2) * 40 - 20)

            par_cur = new_population

            # calculate the fitness for the new population
            # i'm doing this twice which is not efficient -- change in future
            fitness_values = self.calc_fitness_population(par_cur)

            # check if the best fitness value has improved
            if fitness_values[0, 1] < fit_best:
                fit_best = fitness_values[0, 1]
                par_best = par_cur[fitness_values[0, 0].astype(int)]

            self.fitness_values_population_best.append(fitness_values[0, 1])
            self.fitness_values_population_avg.append(np.mean(fitness_values[:, 1]))

        # save the best parameters and fitness values
        self.par_end = par_best
        self.fitness_end = fit_best

        return fit_best, par_cur, fitness_values



    # ------------------------- BELOW: Plotting function -------------------------
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


    def plot_all(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))

        plot_from_iteration = 0
        # Plot fitness values
        axs[0].plot(self.fitness_values_population_best[plot_from_iteration:])
        axs[0].set_title("Fitness Values Over Iterations")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Fitness Value")
        axs[0].text(0.5, 0.95, f"Plotting from iteration: {plot_from_iteration}", transform=axs[0].transAxes, ha="center", va="top", fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":3})

        # Plot control points
        original_control_points = self.bezier_curve.get_control_points()
        reconstructed_control_points = self.par_end
        for idx, (x, y) in enumerate(original_control_points):
            axs[1].scatter(x, y, color='blue', label='Original Control Points' if idx == 0 else "")
            axs[1].text(x, y, str(idx), color='blue', fontsize=8, ha='right', va='bottom')
        for idx, (x, y) in enumerate(reconstructed_control_points):
            axs[1].scatter(x, y, color='red', label='Reconstructed Control Points' if idx == 0 else "")
            axs[1].text(x, y, str(idx), color='red', fontsize=8, ha='left', va='top')
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

def run_ge(instance):
    fit_best, par_best, fitness_stats = instance.gen_alg_with_SA()
    return fit_best, par_best, fitness_stats

def run_ge_plot(instance):
    fit_best, par_best, fitness_stats = instance.gen_alg_with_SA()
    instance.plot_all()
    return fit_best, par_best, fitness_stats

if __name__ == '__main__':
    # Note that mutation and crossover rates are not used in the current implementation of the genetic algorithm
    # genetic = GeneticAlgorithm(degree=6, num_points_sampled=100, iterations=20, mutation_rate=0.1, crossover_rate=0.1)
    # genetic.rand_alg()
    # genetic.gen_alg()
    # genetic.gen_alg_with_SA()
    # genetic.plot_all()

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


    # currently only num_points_sampled % (degree + 1) == 0 is supported, because of the way the initial parameters are set on the curve
    ge_instances_mut_one_point_pre_run = [
        GeneticAlgorithm(degree=degree, num_points_sampled=100, iterations=5, mutation_rate=0.1, crossover_rate=0.1, given_curve=curve_to_reconstruct)
        for _ in
        range(runs_on_same_curve)]

    results = []
    # Use multiprocessing to run the simulated annealing algorithm on multiple instances in parallel
    with mp.Pool(processes=num_processes) as pool:
        results += pool.map(run_ge_plot, ge_instances_mut_one_point_pre_run)


    results_sorted = sorted(
            results,
            key=lambda x: np.sum(x[2][:, 1])
        )





    par_best_5= np.zeros((4950, degree + 1, 2))
    index = 0


    # VERSION: Take the first 5 best results and use their parameters to create a new population
    for cur_fit, cur_par, fitness_stats in results_sorted[:5]:
        par_best_5[index*2*495:(index+1)*2*495] = cur_par[fitness_stats[:2*495, 0].astype(int)]
        index += 1

    par_best = np.zeros_like(par_best_5)
    index = 0
    # VERSION: Take the first 10 best results and use their parameters to create a new population
    for cur_fit, cur_par, fitness_stats in results:
        par_best[index * 495:(index + 1) * 495] = cur_par[fitness_stats[:495, 0].astype(int)]
        index += 1


    # Run with pop from 5 best
    ge_instances_mut_one_point_merged_run = [
        GeneticAlgorithm(degree=degree, num_points_sampled=100, iterations=5, mutation_rate=0.1, crossover_rate=0.1,
                         given_curve=curve_to_reconstruct, given_population=par_best)
        for _ in
        range(runs_on_same_curve)]

    with mp.Pool(processes=num_processes) as pool:
        pool.map(run_ge_plot, ge_instances_mut_one_point_merged_run)


    # run with pop from 10 best
    ge_instances_mut_one_point_merged_run = [
        GeneticAlgorithm(degree=degree, num_points_sampled=100, iterations=6, mutation_rate=0.1, crossover_rate=0.1,
                         given_curve=curve_to_reconstruct, given_population=par_best)
        for _ in
        range(runs_on_same_curve)]

    with mp.Pool(processes=num_processes) as pool:
        pool.map(run_ge_plot, ge_instances_mut_one_point_merged_run)




    # Comparison
    # ge_instances_mut_one_point = [
    #     GeneticAlgorithm(degree=degree, num_points_sampled=100, iterations=8, mutation_rate=0.1, crossover_rate=0.1,
    #                      given_curve=curve_to_reconstruct)
    #     for _ in
    #     range(runs_on_same_curve)]
    #
    # results = []
    # # Use multiprocessing to run the simulated annealing algorithm on multiple instances in parallel
    # with mp.Pool(processes=num_processes) as pool:
    #     results += pool.map(run_ge_plot, ge_instances_mut_one_point)


    # ge_instances_mut_all_points = [
    #     GeneticAlgorithm(degree=degree, num_points_sampled=200, iterations=100, mutation_rate=0.1, crossover_rate=0.1,
    #                      given_curve=curve_to_reconstruct, mut_all = True)
    #     for _ in
    #     range(runs_on_same_curve)]

    # Use multiprocessing to run the simulated annealing algorithm on multiple instances in parallel
    # with mp.Pool(processes=num_processes) as pool:
    #     pool.map(run_ge_plot, ge_instances_mut_all_points)