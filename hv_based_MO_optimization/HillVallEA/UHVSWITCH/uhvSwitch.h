/*
 * uhvSwitch.h
 * This algorithm applies UHV-GOMEA until a certain UHV threshold is met, after which a gradient algorithm is applied.
 *
 * Implementation by D. Ha
 */

#include "../../UHV.hpp"
#include "../adam_on_population.h"
#include "../gomea.hpp"
#include "../hillvallea_internal.hpp"
#include "../optimizer.hpp"
#include "../population.hpp"

namespace hillvallea {
    class UHVSwitch : public optimizer_t {
    public:
        // Constructor and destructor
        UHVSwitch(
                std::shared_ptr<UHV_t> fitnessFunction,
                int versionUHVGOMEA,
                int versionUHVADAM,
                int indexApplicationMethodUHV_ADAM,
                double uhvSwitchValue,
                const size_t number_of_parameters,
                const vec_t & lower_param_bounds,
                const vec_t & upper_param_bounds,
                double init_univariate_bandwidth,
                rng_pt rng);

        ~UHVSwitch();

        // Optimizer inherited methods
        std::string name() const;
        void initialize_from_population(population_pt pop, size_t target_popsize);
        void generation(size_t sample_size, int & number_of_evaluations);
//        bool checkTerminationCondition();

        // UHV Fitness function
        std::shared_ptr<UHV_t> uhvFitnessFunction;  // The UHV fitness function

        // Parameters population and statistics
        bool population_initialized;    // Boolean if population is initialized
        size_t population_size;         // The current population size (can be bigger than the initial population size)

        // Intra generational statistics
        // These variables are flushed every generation
        std::vector<std::string> vecOptimizerNameOfPopulation;      // The optimizer that was applied to create the intermediate population
        std::vector<population_pt> vecPopulationCreatedByOptimizer; // The intermediate populations created by the optimizers (first entry is first optimizer)
        std::vector<int> vecNumberOfSOEvaluationsByOptimizer;       // The total number of so-evaluations when an optimizer finished executing
        std::vector<size_t> vecNumberOfMOEvaluationsByOptimizer;    // The total number of mo-evaluations when an optimizer finished executing

        // Initialize optimizers
        void initializeUHV_GOMEA();
        void initializeUHV_ADAM_BEST();

        // Reset functions
        void resetIntraGenerationalOptimizerStatistics();   // Resets the intra generational statistical data

        // Optimizers
        gomea_pt optimizerUHV_GOMEA;                    // UHV-GOMEA optimizer
        adam_on_population_pt optimizerUHV_ADAM_BEST;   // UHV-ADAM best solution optimizer

    private:
        // Optimizer to apply
        bool applyUHVGradient;          // Should UHV-Adam on best solution be applied on a population branch
        int optimizerIndexUHV_GOMEA;    // Optimizer index of UHV-GOMEA
        int optimizerIndexUHVGradientT; // Optimizer index of the gradient
        int uhvGradientVersion;         // The method of how to apply the gradient

        double uhvSwitchValue;  // The value to switch from UHV-GOMEA to the gradient algorithm
        bool hasSwitched;       // Boolean if switching point was reached

        // Optimizer execution
        void executeUHV_GOMEA(
                size_t sample_size,
                int & totalNumberOfSOEvaluations,
                size_t & MOEvaluationsUsedByUHV_GOMEA,
                double & improvementByUHV_GOMEA);
        void executeUHV_ADAM_BEST(
                size_t sample_size,
                size_t callsToExecuteThisGeneration,
                int & totalNumberOfSOEvaluations,
                size_t & MOEvaluationsUsedByUHV_ADAM_BEST,
                double & improvementByUHV_ADAM_BEST);

        // Helper methods
        void initializeOptimizers();

        // Optimizer statistics methods
        void storeOptimizerStatistics(std::string optimizerName, int currentNumberOfSOEvaluations, size_t currentNumberOfMOEvaluations);
        std::string convertUHV_GOMEALinkageModelNameToString(int linkageModelUHV_GOMEA);
        std::string convertUHV_ADAMModelNameToString(int versionUHV_ADAM);
        population_pt copyPopulation();


    };
}
