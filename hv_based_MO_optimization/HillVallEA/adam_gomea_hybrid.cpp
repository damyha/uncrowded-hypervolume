/*
 * ADAM_GOMEA_hybrid.cpp
 *
 * Implementation by D. Ha
 */

#include "adam_gomea_hybrid.h"
#include "fitness.h"
#include "population.hpp"
#include "optimizer.hpp"

/**
 * Constructor
 * @param number_of_parameters The number of parameters
 * @param lower_param_bounds The lower bounds vector
 * @param upper_param_bounds The upper bounds vector
 * @param init_univariate_bandwidth Todo: Find out what this is
 * @param version_GOMEA The local optimizer index of GOMEA to use
 * @param version_gradient_optimizer The local optimizer index of ADAM to use
 * @param method_ADAM The method of
 * @param fitness_function The fitness function to examine
 * @param rng The random number generator
 */
hillvallea::adam_gomea_hybrid_t::adam_gomea_hybrid_t(
        const size_t number_of_parameters,
        const vec_t & lower_param_bounds,
        const vec_t & upper_param_bounds,
        double init_univariate_bandwidth,
        const int version_GOMEA,
        const int version_gradient_optimizer,
        const int method_gradient_optimizer,
        const fitness_pt fitness_function,
        rng_pt rng) : optimizer_t(
                number_of_parameters,
                lower_param_bounds,
                upper_param_bounds,
                init_univariate_bandwidth,
                fitness_function,
                rng){

            // Set (remaining) settings
            this->version_EA = version_GOMEA;
            this->version_gradient_optimizer = version_gradient_optimizer;
            this->method_gradient_optimizer = method_gradient_optimizer;

            this->population_initialized = false;
            this->selection_fraction = 0.35;    // Taken from gomea.cpp

            // Initialize Evolutionary Algorithm
            this->optimizer_EA = init_optimizer(version_GOMEA, number_of_parameters,
                                                lower_param_bounds, upper_param_bounds,
                                                init_univariate_bandwidth, fitness_function, rng);
            this->optimizer_EA->active = false;

            // Initialize Gradient algorithm
            int version_gradient_and_method = version_gradient_optimizer * 10 + method_gradient_optimizer;
            this->optimizer_gradient = init_optimizer(version_gradient_and_method , number_of_parameters,
                                                      lower_param_bounds, upper_param_bounds,
                                                      init_univariate_bandwidth, fitness_function, rng);
            this->optimizer_gradient->active = false;

        }

/**
 * Deconstruct
 */
hillvallea::adam_gomea_hybrid_t::~adam_gomea_hybrid_t() {

}

/**
 * The name of the algorithm
 * @return The name of the algorithm
 */
std::string hillvallea::adam_gomea_hybrid_t::name() const { return "Hybrid_GOMEA_ADAM"; }

/**
 * Setter for population
 * @param pop The population received
 * @param target_popsize The target size of the population
 */
void hillvallea::adam_gomea_hybrid_t::initialize_from_population(population_pt pop, size_t target_popsize)
{
    // Set parameters
    this->pop = pop;
    this->population_size = (int) target_popsize;
    this->initial_population_size = std::min(population_size, (int) pop->size());
    this->number_of_generations = 0;

    // Copy the population
    this->pop = pop;
    pop->sort_on_fitness();

    // Set the best solution
    this->best = solution_t(*pop->sols[0]);

    // Set population as initialized
    this->population_initialized = true;
}

void hillvallea::adam_gomea_hybrid_t::generation(size_t sample_size, int & number_of_evaluations)
{
    try {
        // Remove this later
        if ( !this->optimizer_gradient->active && this->number_of_generations == 0)
        {
            this->optimizer_gradient->initialize_from_population(pop, population_size);
            this->optimizer_gradient->active = true;
            this->optimizer_EA->active = false;
        }
        else if ( !this->optimizer_EA->active && number_of_evaluations >= 2000/8)
        {
            this->optimizer_EA->initialize_from_population(pop, population_size);
            this->optimizer_EA->active = true;
            this->optimizer_gradient->active = false;
        }

        if (optimizer_gradient->active)
        {
            this->optimizer_gradient->generation(sample_size, number_of_evaluations);
        } else
        {
            this->optimizer_EA->generation(sample_size, number_of_evaluations);
        }


        this->pop = this->optimizer_gradient->pop;
        average_fitness_history.push_back(this->optimizer_gradient->average_fitness_history.back());

        // Check if solution has improved
        for (size_t i = 0; i < population_size; ++i) {
            if (solution_t::better_solution(*pop->sols[i], best)) {
                best = solution_t(*pop->sols[i]);
            }
        }



        this->number_of_generations++;

    }catch (std::exception& exception){
        std::cerr << exception.what() << std::endl;
    }



}


bool hillvallea::adam_gomea_hybrid_t::checkTerminationCondition()
{
    return this->optimizer_gradient->checkTerminationCondition();
}

