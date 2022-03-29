/*

HillVallEA

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA


*/

#include "population.hpp"
#include "mathfunctions.hpp"
#include "fitness.h"

namespace hillvallea {

    // Constructor
    population_t::population_t() {

    }

    // Destructor
    population_t::~population_t() {}

    /**
     * Make a deep copy of this population
     * @return A deep copy of this population
     */
    population_pt population_t::deepCopyPopulation()
    {
        population_pt newPopulation = std::make_shared<population_t>();

        newPopulation->sols.resize(this->sols.size());
        for (size_t solution_index = 0; solution_index < this->sols.size(); ++solution_index) {
            newPopulation->sols[solution_index] = std::make_shared<solution_t>(*this->sols[solution_index]);
        }

        return newPopulation;
    }

    // dimensions
    //------------------------------------------
    size_t population_t::size() const { return sols.size(); }

    size_t population_t::problem_size() const { return sols[0]->param.size(); }

    // Population mean
    void population_t::mean(vec_t &mean) const {
        // Compute the sample mean
        //-------------------------------------------
        // set the mean to zero.
        mean.resize(problem_size());
        mean.fill(0);

        for (auto sol = sols.begin(); sol != sols.end(); ++sol) {
            mean += (*sol)->param;
        }

        mean /= (double) sols.size();

    }

    // weighted Population mean
    void population_t::weighted_mean(vec_t &mean, const vec_t &weights) const {
        // Compute the sample mean
        //-------------------------------------------
        // set the mean to zero.
        assert(sols.size() == weights.size());
        mean.resize(problem_size());
        mean.fill(0);

        for (size_t i = 0; i < sols.size(); ++i) {
            mean += weights[i] * sols[i]->param;
        }

        // mean /= (double)sols.size();

    }

    void population_t::weighted_transformed_mean(vec_t &mean, const vec_t &weights) const {

        assert(weights.size() >= sols.size());
        mean.resize(problem_size());
        mean.fill(0.0);

        // Compute the weighted sample mean
        for (size_t i = 0; i < sols.size(); ++i) {
            mean += sols[i]->param_transformed * weights[i];
        }

    }

    // Population mean
    void
    population_t::weighted_mean_of_selection(vec_t &mean, const vec_t &weights, const size_t selection_size) const {

        size_t range = (size_t) (std::min(selection_size, sols.size()));

        assert(weights.size() >= range);
        mean.resize(problem_size());
        mean.fill(0.0);


        // Compute the weighted sample mean
        for (size_t i = 0; i < range; ++i) {
            mean += sols[i]->param * weights[i];
        }

    }

    // population covariance
    void population_t::covariance(const vec_t &mean, matrix_t &covariance) const {
        // Compute the sample covariance
        // use the maximum likelihood estimate (see e.g. wikipedia)
        //-------------------------------------------
        covariance.reset(problem_size(), problem_size(), 0.0);

        // First do the maximum-likelihood estimate from data
        for (size_t i = 0; i < problem_size(); i++) {
            for (size_t j = i; j < problem_size(); j++) {
                for (size_t k = 0; k < sols.size(); k++) {
                    covariance[i][j] += (sols[k]->param[i] - mean[i]) * (sols[k]->param[j] - mean[j]);
                }

                covariance[i][j] /= (double) sols.size();
            }
        }

        for (size_t i = 0; i < problem_size(); i++)
            for (size_t j = 0; j < i; j++)
                covariance[i][j] = covariance[j][i];
    }

    // population covariance
    void population_t::covariance_univariate(const vec_t &mean, matrix_t &covariance) const {
        // Compute the sample covariance
        // use the maximum likelihood estimate (see e.g. wikipedia)
        //-------------------------------------------
        covariance.reset(problem_size(), problem_size(), 0.0);

        /* First do the maximum-likelihood estimate from data */
        for (size_t i = 0; i < problem_size(); i++) {
            for (size_t k = 0; k < sols.size(); k++) {
                covariance[i][i] += (sols[k]->param[i] - mean[i]) * (sols[k]->param[i] - mean[i]);
            }

            covariance[i][i] /= (double) sols.size();
        }
    }

    // population covariance
    void population_t::covariance_univariate(const vec_t &mean, vec_t &covariance) const {
        // Compute the sample covariance
        // use the maximum likelihood estimate (see e.g. wikipedia)
        //-------------------------------------------
        covariance.resize(problem_size());
        covariance.fill(0.0);

        /* First do the maximum-likelihood estimate from data */
        for (size_t i = 0; i < problem_size(); i++) {
            for (size_t k = 0; k < sols.size(); k++) {
                covariance[i] += (sols[k]->param[i] - mean[i]) * (sols[k]->param[i] - mean[i]);
            }

            covariance[i] /= (double) sols.size();
        }
    }

    // evalute the population
    //-------------------------------------------------------------------------------------
    int population_t::evaluate(const fitness_pt fitness_function, const size_t skip_number_of_elites) {
        size_t i_start = skip_number_of_elites;

        if (fitness_function->dynamic_objective) {
            i_start = 0;
        }

        int number_of_evaluations = 0;
        for (size_t i = i_start; i < sols.size(); ++i) {
            fitness_function->evaluate(*sols[i]);
            number_of_evaluations++;
            // assert(isfinite(sols[i]->f));
        }

        return number_of_evaluations;
    }

    int population_t::evaluate_with_gradients(const fitness_pt fitness_function, const size_t skip_number_of_elites) {
        size_t i_start = skip_number_of_elites;

        if (fitness_function->dynamic_objective) {
            i_start = 0;
        }

        for (size_t i = i_start; i < sols.size(); ++i) {
            fitness_function->evaluate_with_gradients(*sols[i]);
            // assert(isfinite(sols[i]->f));
        }

        return ((int) (sols.size() - skip_number_of_elites));
    }


    // Fill the given population by uniform initialization in the range [min,max),
    // for all dimensions equal
    //----------------------------------------------------------------------------------------
    void population_t::fill_uniform(const size_t sample_size, const size_t problem_size, const vec_t &lower_param_range,
                                    const vec_t &upper_param_range, rng_pt rng) {

        // resize the solutions vector.
        sols.resize(sample_size);

        // sample  solutions and evaluate them
        for (size_t i = 0; i < sols.size(); ++i) {

            // if the solution is not yet initialized, do it now.
            if (sols[i] == nullptr) {
                solution_pt sol = std::make_shared<solution_t>(problem_size);
                sols[i] = sol;
            }

            // sample a new solution ...
            sample_uniform(sols[i]->param, problem_size, lower_param_range, upper_param_range, rng);

        }
    }

    void population_t::fill_greedy_uniform(const size_t sample_size, const size_t problem_size, double sample_ratio,
                                           const vec_t &lower_param_range, const vec_t &upper_param_range, rng_pt rng) {

        // resize the solutions vector.
        sols.resize((size_t) (sample_ratio * sample_size));

        // sample  solutions and evaluate them
        for (size_t i = 0; i < sols.size(); ++i) {

            // if the solution is not yet initialized, do it now.
            if (sols[i] == nullptr) {
                sols[i] = std::make_shared<solution_t>(problem_size);
            }

            // sample a new solution ...
            sample_uniform(sols[i]->param, problem_size, lower_param_range, upper_param_range, rng);

        }

        if (sample_ratio > 1) {
            std::vector<solution_pt> new_sols;
            selectSolutionsBasedOnParameterDiversity(sols, sample_size, new_sols, rng);
            sols = new_sols;
        }
    }

    // reject samples of which the nearest d+1 solutions
    void population_t::fill_with_rejection(const size_t sample_size, const size_t problem_size, double sample_ratio,
                                           const std::vector<solution_pt> &previous_sols,
                                           const vec_t &lower_param_range, const vec_t &upper_param_range, rng_pt rng) {

        // resize the solutions vector.
        sols.resize((size_t) (sample_ratio * sample_size));

        vec_t dist(previous_sols.size(), 0.0);

        size_t number_of_nearest_neighbours = problem_size + 1;


        std::uniform_real_distribution<double> unif(0, 1);

        // sample solutions and evaluate them
        for (size_t i = 0; i < sols.size(); ++i) {

            // if the solution is not yet initialized, do it now.
            //if (sols[i] == nullptr) {
            sols[i] = std::make_shared<solution_t>(problem_size);
            //}

            // sample a new solution ...
            sample_uniform(sols[i]->param, problem_size, lower_param_range, upper_param_range, rng);

            if (previous_sols.size() > 0) {
                // for each solution, find the nearest solutions from the previous pop.
                //-----------------------------------------------------------------------
                size_t nearest_index = 0, furthest_index = 0;
                for (size_t j = 0; j < previous_sols.size(); ++j) {
                    dist[j] = sols[i]->param_distance(*previous_sols[j]);

                    if (dist[j] < dist[nearest_index]) {
                        nearest_index = j;
                    }

                    if (dist[j] > dist[furthest_index]) {
                        furthest_index = j;
                    }
                }

                bool accept_sample = false;
                int cluster_index = previous_sols[nearest_index]->cluster_number; // note, cluster_number == -1 if not in selection, but also count those!

                if (cluster_index == -1) {
                    continue;
                }

                for (size_t j = 1; j < number_of_nearest_neighbours; ++j) {
                    size_t old_nearest_index = nearest_index;
                    nearest_index = furthest_index;

                    for (size_t k = 0; k < previous_sols.size(); k++) {

                        if (dist[k] > dist[old_nearest_index] && dist[k] < dist[nearest_index]) {
                            nearest_index = k;
                        }
                    }
                    if (cluster_index != previous_sols[nearest_index]->cluster_number) {
                        accept_sample = true;
                        break;
                    }
                }

                // do not accept sample if all neighbours are from the same cluster
                if (!accept_sample) {
                    // reject sample
                    if (unif(*rng) > 0.1) {
                        i--;
                    }
                }
            }
        }

        if (sample_ratio > 1) {
            std::vector<solution_pt> new_sols;
            selectSolutionsBasedOnParameterDiversity(sols, sample_size, new_sols, rng);
            sols = new_sols;
        }
    }

    // Fill the given population by normal sampling
    //-------------------------------------------------------------------------------------------------------------------------------
    int population_t::fill_normal(const size_t sample_size, const size_t problem_size, const vec_t &mean,
                                  const matrix_t &MatrixRoot, const vec_t &lower_param_range,
                                  const vec_t &upper_param_range, const size_t number_of_elites, rng_pt rng) {

        // Resize the population vector
        //--------------------------------------------
        sols.resize(sample_size);

        int number_of_samples = 0;

        // for each sol in the pop, sample.
        for (size_t i = 0; i < sols.size(); ++i) {

            // save the elite (if it is defined)
            if (i < number_of_elites && sols[i] != nullptr)
                continue;

            // if the solution is not yet initialized, do it now.
            if (sols[i] == nullptr) {
                solution_pt sol = std::make_shared<solution_t>(problem_size);
                sols[i] = sol;
            }

            number_of_samples += sample_normal(sols[i]->param, sols[i]->param_transformed, problem_size, mean,
                                               MatrixRoot, lower_param_range, upper_param_range, rng);

        }

        return number_of_samples;
    }


    int population_t::fill_normal_univariate(const size_t sample_size, const size_t problem_size, const vec_t &mean,
                                             const vec_t &cholesky, const vec_t &lower_param_range,
                                             const vec_t &upper_param_range, const size_t number_of_elites,
                                             rng_pt rng) {

        // Resize the population vector
        //--------------------------------------------
        sols.resize(sample_size);

        int number_of_samples = 0;

        // for each sol in the pop, sample.
        for (size_t i = 0; i < sols.size(); ++i) {

            // save the elite (if it is defined)
            if (i < number_of_elites && sols[i] != nullptr)
                continue;

            // if the solution is not yet initialized, do it now.
            if (sols[i] == nullptr) {
                solution_pt sol = std::make_shared<solution_t>(problem_size);
                sols[i] = sol;
            }


            number_of_samples += sample_normal_univariate(sols[i]->param, sols[i]->param_transformed, problem_size,
                                                          mean, cholesky, lower_param_range, upper_param_range, rng);

        }

        return number_of_samples;
    }

    // Truncation selection (selection percentage)
    // select the selection_percentage*population_size best individuals in the population
    //-------------------------------------------------------------------------------------
    void population_t::truncation_percentage(population_t &selection, double selection_percentage) const {
        // Determine selection size
        size_t number_selected = (size_t) selection_percentage * sols.size();

        // Set minimum of 1 solution
        if (number_selected == 0)
            number_selected = 1;

        // Get the parent population size to compute the selection fraction.
        // Then, call truncation by number.
        truncation_size(selection, number_selected);

    }


    // Truncation selection (selection size)
    // select the selection_size best individuals in the population
    // the population is already sorted.
    //-------------------------------------------------------------------------------------
    void population_t::truncation_size(population_t &selection, size_t selection_size) const {

        selection.sols.resize(selection_size);

        // copy the pointers from the parents to the selection
        std::copy(sols.begin(), sols.begin() + selection_size, selection.sols.begin());

    }

    // Add the solutions of another population to this one
    //---------------------------------------------------------------------
    void population_t::addSolutions(const population_t &pop) {
        this->sols.insert(this->sols.end(), pop.sols.begin(), pop.sols.end());
    }

    // do we have improvement over the given objective value
    // assumes a sorted population
    //--------------------------------------------
    bool population_t::improvement_over(const double objective) const {
        if (sols[0]->f < objective) {
            return true;
        }

        return false;
    }

    // Sort the population such that best = first
    //-------------------------------------------------------------------------------------
    void population_t::sort_on_fitness() {
        std::sort(sols.begin(), sols.end(), solution_t::better_solution_via_pointers);
    }

    std::vector<size_t> population_t::sort_solution_index_on_fitness() {
        // Create indices
        std::vector<size_t> indices_solutions(sols.size());
        std::iota(indices_solutions.begin(), indices_solutions.end(), 0);

        // Retrieve fitness
        std::vector<double> fitness_value_solutions;
        for (size_t solution_index = 0; solution_index < sols.size(); ++solution_index) {
            fitness_value_solutions.push_back(sols[solution_index]->f);
        }

        // Sort the indices on fitness
        std::sort(indices_solutions.begin(),
                  indices_solutions.end(),
                  [&](double i, double j) { return fitness_value_solutions[i] < fitness_value_solutions[j]; });

        return indices_solutions;
    }

    // Population Statistics
    //--------------------------------------------------------------------
    solution_pt population_t::first() const {

        if (sols.size() < 1)
            return nullptr;
        else
            return sols[0];
    }

    solution_pt population_t::last() const {
        if (sols.size() < 1)
            return nullptr;
        else
            return sols[sols.size() - 1];
    }

    /**
     * Retrieves the index of the best solution of the population
     * @return The index of the best solution
     */
    size_t population_t::bestSolutionIndex() const {
        size_t bestSolutionIndex = 0;
        for (size_t solutionIndex = 1; solutionIndex < this->sols.size(); ++solutionIndex) {
            if (solution_t::better_solution(*this->sols[solutionIndex], *this->sols[bestSolutionIndex])) {
                bestSolutionIndex = solutionIndex;
            }
        }

        return bestSolutionIndex;
    }

    /**
     * Returns a pointer to a copy of the best solution of the population
     * @return A pointer to a copy of the best solution in the population
     */
    solution_pt population_t::bestSolution() const {
        size_t bestSolutionIndex = this->bestSolutionIndex();
        return std::make_shared<solution_t>(*this->sols[bestSolutionIndex]);
    }

    /**
     * Returns the best fitness of a population
     * @return the best fitness of a population
     */
    double population_t::best_fitness() const {
        size_t bestSolutionIndex = this->bestSolutionIndex();
        return this->sols[bestSolutionIndex]->f;
    }

    /**
     * Returns the sum of fitness of a population
     * @return the sum of fitness of a population
     */
    double population_t::sum_fitness() const {
        double sum_fitness = 0.0;

        for (auto sol = sols.begin(); sol != sols.end(); ++sol) {
            sum_fitness += (*sol)->f;
        }

        return sum_fitness;
    }

    /**
     * Returns the average fitness of a population
     * @return The average fitness of a population
     */
    double population_t::average_fitness() const {
        double sum_fitness = this->sum_fitness();

        double average_fitness = sum_fitness / this->size();

        return average_fitness;
    }

    double population_t::fitness_variance() const {

        double mean = 0;
        double variance = 0;

        for (auto sol = sols.begin(); sol != sols.end(); ++sol)
            mean += (*sol)->f;

        mean /= (sols.size());

        for (auto sol = sols.begin(); sol != sols.end(); ++sol)
            variance += ((*sol)->f - mean) * ((*sol)->f - mean);

        variance /= (sols.size());

        return variance;

    }

    double population_t::relative_fitness_std() const {


        double mean = 0;
        double variance = 0;

        for (auto sol = sols.begin(); sol != sols.end(); ++sol)
            mean += (*sol)->f;

        mean /= (sols.size());

        for (auto sol = sols.begin(); sol != sols.end(); ++sol)
            variance += ((*sol)->f - mean) * ((*sol)->f - mean);

        variance /= (sols.size());

        if (fabs(mean) <= 0)
            return 0.0;
        else
            return sqrt(variance) / fabs(mean);

    }

    // Average penalty of the population
    //-------------------------------------------
    double population_t::average_constraint() const {
        double average_fitness = 0.0;

        for (auto sol = sols.begin(); sol != sols.end(); ++sol) {
            average_fitness += (*sol)->constraint;
        }

        average_fitness /= size();

        return average_fitness;
    }

    double population_t::relative_constraint_std() const {

        double mean = 0;
        double variance = 0;

        for (auto sol = sols.begin(); sol != sols.end(); ++sol)
            mean += (*sol)->constraint;

        mean /= (sols.size());

        for (auto sol = sols.begin(); sol != sols.end(); ++sol)
            variance += ((*sol)->constraint - mean) * ((*sol)->constraint - mean);

        variance /= (sols.size());

        if (fabs(mean) <= 0)
            return 0.0;
        else
            return sqrt(variance) / fabs(mean);

    }


    // Sort the population such that highest probability = first
    //-------------------------------------------------------------------------------------
    void population_t::sort_on_probability() {
        std::sort(sols.begin(), sols.end(), solution_t::higher_probability);
    }

    // set the fitness ranks
    void population_t::set_fitness_rank() {
        // sort the population
        sort_on_fitness();

        // set the fitness ranks
        for (size_t i = 0; i < sols.size(); ++i) {
            sols[i]->fitness_rank = (int) i;
        }
    }


    // set probability rank
    void population_t::set_probability_rank() {

        // sort the population on probability
        sort_on_probability();

        // set the fitness rank
        for (size_t i = 0; i < sols.size(); ++i) {
            sols[i]->probability_rank = (int) i;
        }

    }

    double population_t::compute_DFC() {

        set_probability_rank();
        set_fitness_rank();

        double fitness_correlation = 0.0;
        double N = (double) size();
        double rankdifference;

        if (N <= 1)
            return fitness_correlation;

        for (auto sol : sols) {
            rankdifference = (double) (sol->fitness_rank - sol->probability_rank);
            fitness_correlation += rankdifference * rankdifference;
        }

        // compute the predictive value
        // https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
        fitness_correlation = 1 - fitness_correlation / (N * (N * N - 1) / 6.0);

        return fitness_correlation;


    }

}
