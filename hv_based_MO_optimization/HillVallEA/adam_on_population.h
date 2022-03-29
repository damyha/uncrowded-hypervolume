/*
 * ADAM applied on a population stead of a single solution
 *
 * By D. Ha
 *
 * Each solution can be selected to have ADAM/GAMO to be performed on.
 * Each solution keeps its own algorithm parameters
*/

#include "hillvallea_internal.hpp"
#include "optimizer.hpp"
#include "fitness.h"
#include "hillvallea_internal.hpp"
#include "optimizer.hpp"

namespace hillvallea
{
    class adam_on_population_t : public optimizer_t {

    public:
        // Constructor and Destructor
        adam_on_population_t(size_t number_of_parameters,
                             const vec_t & lower_param_bounds,
                             const vec_t & upper_param_bounds,
                             double init_univariate_bandwidth,
                             int version,
                             int version_method,
                             fitness_pt fitness_function,
                             rng_pt rng);
        ~adam_on_population_t();

        // HillVallEA methods
        optimizer_pt clone() const;
        adam_on_population_pt clone(population_pt newPopulation);
        std::string name() const override;
        void initialize_from_population(population_pt pop, size_t target_popsize) override;
        void generation(size_t sample_size, int &external_number_of_evaluations) override;
        bool checkTerminationCondition();

        // Gradient Algorithm specific parameters
        int version;                            // The local optimizer index of the gradient algorithm
        int version_method;                     // The method of how to apply a local optimizer to the population
        size_t population_size;                 // The current population size
        size_t initial_population_size;         // The initial population size

        bool population_initialized;            // Boolean if population has been initialized
        bool use_momentum_with_nag;             // Todo: Find out what this is
        bool accept_only_improvements;          // Boolean if improvements should only  be accepted

        int no_improvement_stretch;             // Number of generations without any improvement: Not used here because this optimizers forces the execution of local optimizers
        double weighted_number_of_evaluations;  // The number of times a function evaluation is executed

        // Hybrid methods
        void generation(size_t sample_size, int &external_number_of_evaluations, size_t number_of_calls_to_execute);
        void executeGenerationOnSpecificSolution(size_t solutionIndex);
        std::vector<size_t> sortPopulationIndexOnBestFitness(); // Sort a population of fitness and returns the indices of the solutions

        // Algorithm methods
        std::vector<size_t> determineSolutionIndicesToApplyAlgorithmOn(size_t number_of_calls_to_execute);

        // Getters and setters
        void setGradientOptimizers(const std::vector<adam_pt> &gradientOptimizers);
        void setTouchedParameterIdx(const std::vector<std::vector<size_t>> &touchedParameterIdx);
        void setLowerInitRanges(const vec_t &lowerInitRanges);
        void setUpperInitRanges(const vec_t &upperInitRanges);
        void setGammaWeight(double gammaWeight);
        void setStepSizeDecayFactor(double stepSizeDecayFactor);
        void setFiniteDifferencesMultiplier(double finiteDifferencesMultiplier);

        std::string convertUHV_ADAMModelNameToString();

    private:
        // Gradient Algorithm parameters
        std::vector<adam_pt> gradient_optimizers;               // The gradient optimizer (per solution)
        std::vector<std::vector<size_t>> touched_parameter_idx; // The indices that should be altered
        vec_t lower_init_ranges;                                // The lower initialization range (used for gamma)/sol
        vec_t upper_init_ranges;                                // The upper initialization range (used for gamma)/sol

        // Gradient Algorithm specific parameters
        double gamma_weight;                    // The gamma_weight value (default taken from ADAM.cpp)
        double stepSizeDecayFactor;             // The value used to decrease the step size if it can't find an improvement
        double finite_differences_multiplier;   // The finite differences multiplier

        // Algorithm parameters initialization methods
        void determineInitRanges();
        void initializeGradientAlgorithmSpecificParameters();
        void initializeGammaVector();
        void initializeLocalOptimizers();

        // Other functions
        std::vector <size_t> selectTopNSolutions(size_t topNSolutions, size_t number_of_calls_to_execute);


    };

}

