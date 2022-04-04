#pragma once

/*
 * ADAM_GOMEA_hybrid.h
 * Todo: write something here about the final algorithm
 *
 * Implementation by D. Ha
 */

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"

namespace hillvallea {
    class adam_gomea_hybrid_t : public optimizer_t {
    public:

        // Constructor and destructor
        adam_gomea_hybrid_t(
                const size_t number_of_parameters,
                const vec_t & lower_param_bounds,
                const vec_t & upper_param_bounds,
                double init_univariate_bandwidth,
                int version_GOMEA,
                int version_gradient_optimizer,
                int method_gradient_optimizer,
                fitness_pt fitness_function,
                rng_pt rng);

        ~adam_gomea_hybrid_t();

        // Optimizer inherited methods
        std::string name() const;
        void initialize_from_population(population_pt pop, size_t target_popsize);
        void generation(size_t sample_size, int & number_of_evaluations);
        bool checkTerminationCondition();


    private:
        // Optimizers
        optimizer_pt optimizer_EA, optimizer_gradient;  // Pointers to the EA and gradient algorithm

        // Parameters optimizers algorithm
        int version_EA;                     // The optimizer index of the EA
        int version_gradient_optimizer;     // The optimizer index of the gradient algorithm
        int method_gradient_optimizer;      // The application method of the gradient algorithm
        vec_t lower_init_ranges;            // The lower initialization range (used for Gradient Algorithm)
        vec_t upper_init_ranges;            // The upper initialization range (used for Gradient Algorithm)

        // Parameters population
        bool population_initialized;        // Boolean if population is initialized
        size_t initial_population_size;     // The initial population size
        int population_size;                // The current population size


    };



}
