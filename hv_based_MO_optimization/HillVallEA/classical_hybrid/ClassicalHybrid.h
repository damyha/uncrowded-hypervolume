/*
 * ClassicalHybrid.h
 * This optimizer executes the EA and gradient algorithm based on a fixed pattern.
 *
 *
 * Implementation by D. Ha
 */

#include "../adam_on_population.h"
#include "../gomea.hpp"
#include "../hillvallea_internal.hpp"
#include "../optimizer.hpp"
#include "../population.hpp"
#include "../../UHV.hpp"

namespace hillvallea {
    class ClassicalHybrid : public optimizer_t {
    public:
        // Constructor and destructor
        ClassicalHybrid(
                std::shared_ptr<UHV_t> uhvFitnessFunction,
                int localOptimizerIndexUHV_GOMEA,
                int localOptimizerIndexUHV_ADAM,
                int indexApplicationMethodUHV_ADAM,
                const size_t numberOfSOParameters,
                const vec_t & soLowerParameterBounds,
                const vec_t & soUpperParameterBounds,
                double initUniVariateBandwidth,
                rng_pt rng);

        ~ClassicalHybrid();

        // Optimizer inherited methods
        std::string name() const;
        void initialize_from_population(population_pt pop, size_t target_popsize);
        void generation(size_t sample_size, int & number_of_evaluations);

        // UHV Fitness function
        std::shared_ptr<UHV_t> uhvFitnessFunction;  // The UHV fitness function

        // Parameters population and statistics
        bool populationInitialized;    // Boolean if population is initialized
        size_t populationSize;         // The current population size (can be bigger than the initial population size)

        // Intra generational statistics
        // These variables are flushed every generation
        std::vector<std::string> vecOptimizerNameOfPopulation;      // The optimizer that was applied to create the intermediate population
        std::vector<population_pt> vecPopulationCreatedByOptimizer; // The intermediate populations created by the optimizers (first entry is first optimizer)
        std::vector<solution_t> vecBestSolutionsAfterOptimizer;     // The best solution ever found after executing the optimizer
        std::vector<int> vecNumberOfSOEvaluationsByOptimizer;       // The total number of so-evaluations when an optimizer finished executing
        std::vector<size_t> vecNumberOfMOEvaluationsByOptimizer;    // The total number of mo-evaluations when an optimizer finished executing
        std::vector<size_t> vecNumberOfCallsByOptimizer;            // The number of calls done in the current generation per optimizer

        // Reset functions
        void resetIntraGenerationalOptimizerStatistics();   // Resets the intra generational statistical data

        // Initialization methods
        void initializeOptimizers();
        void initializeUHV_GOMEA();
        void initializeUHV_ADAM_POPULATION();

        // Optimizers
        gomea_pt optimizerUHV_GOMEA;                            // UHV-GOMEA optimizer
        adam_on_population_pt optimizerUHV_ADAM_POPULATION;     // UHV-ADAM best solution optimizer

    private:
        // Optimizer to apply
        bool applyUHV_GOMEA;            // Should UHV-GOMEA be applied on a population branch
        bool applyUHV_ADAM_POPULATION;  // Should UHV-Adam on population be applied on a population branch

        // Optimizer indices
        int localOptimizerIndexUHV_GOMEA;   // Optimizer index of UHV-GOMEA
        int localOptimizerIndexUHV_ADAM;    // Optimizer index of UHV-ADAM
        int indexApplicationMethodUHV_ADAM; // Method of applying UHV-ADAM on a population

        size_t numberOfUHV_GOMEACalls;              // The number of UHV-GOMEA calls to execute per generation
        size_t numberOfUHV_ADAM_POPULATIONCalls;    // The number of UHV-ADAM calls to execute per generation

        // Optimizer execution
        void executeUHV_GOMEA(
                size_t sample_size,
                int & totalNumberOfSOEvaluations);
        void executeUHV_ADAM_BEST(
                size_t sample_size,
                size_t callsToExecuteThisGeneration,
                int & totalNumberOfSOEvaluations);

        // Statistical writing
        void storeOptimizerStatistics(std::string optimizerName, int currentNumberOfSOEvaluations, size_t currentNumberOfMOEvaluations, size_t currentNumberOfCalls);

        // Helper methods
        std::string convertUHV_GOMEALinkageModelNameToString(int linkageModelUHV_GOMEA);
        std::string convertUHV_ADAMModelNameToString(int versionUHV_ADAM);
    };
}

