/*
 * naive_resource_allocation_scheme_t.h
 * This optimizer uses a resource allocation scheme inspired by:
 * "Combining Gradient Techniques for Numerical Multi-Objective Evolutionary Optimization" by
 * P A.N. Bosman and E D. de Jong
 *
 * The resource allocation scheme is naive in the sense that it does not realize that the algorithms use history.
 *
 * 'vec_populations_created_by_optimizers', 'vec_optimizer_name_of_population' and 'vec_optimizer_number_of_so_evaluations'
 * are used to create a single statistics file that contains the statistics of every optimizer that was applied. Specifically
 * the variables keep track of the order in which the optimizers were applied and are flushed after a generation.
 *
 * Implementation by D. Ha
 */

#include "GenerationalArchive.h"

#include "../adam_on_population.h"
#include "../gomea.hpp"
#include "../hillvallea_internal.hpp"
#include "../optimizer.hpp"
#include "../population.hpp"
#include "../../UHV.hpp"

namespace hillvallea {
    class naive_resource_allocation_scheme_t : public optimizer_t {
    public:
        // Constructor and destructor
        naive_resource_allocation_scheme_t(
                std::shared_ptr<UHV_t> uhvFitnessFunction,
                int localOptimizerIndexUHV_GOMEA,
                int localOptimizerIndexUHV_ADAM,
                int indexApplicationMethodUHV_ADAM,
                int improvementMetricIndex,
                const size_t numberOfSOParameters,
                const vec_t & soLowerParameterBounds,
                const vec_t & soUpperParameterBounds,
                double initUniVariateBandwidth,
                double memoryDecayFactor,
                double initialStepSizeFactor,
                double stepSizeDecayFactor,
                rng_pt rng);

        ~naive_resource_allocation_scheme_t();

        // Optimizer inherited methods
        std::string name() const;
        void initialize_from_population(population_pt pop, size_t target_popsize);
        void generation(size_t sample_size, int & number_of_evaluations);
//        bool checkTerminationCondition();

        // UHV Fitness function
        std::shared_ptr<UHV_t> uhvFitnessFunction;  // The UHV fitness function

        // Resource allocation parameters
        double memoryDecayFactor;       // Eta of memory decay

        // Gradient algorithm parameters
        double initialStepSizeFactor;   // The factor used to estimate the initial step size
        double stepSizeDecayFactor;     // The factor used to decrease the step size

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
        std::vector<double> vecImprovementsByOptimizer;             // The number of improvements found this generation
        std::vector<double> vecRewardsByOptimizer;                  // The reward value of this generation
        std::vector<size_t> vecNumberOfCallsByOptimizer;            // The number of calls done in the current generation per optimizer

        // Optimizers
        gomea_pt optimizerUHV_GOMEA;                    // UHV-GOMEA optimizer
        adam_on_population_pt optimizerUHV_ADAM_BEST;   // UHV-ADAM best solution optimizer

        // Reset functions
        void resetIntraGenerationalOptimizerStatistics();   // Resets the intra generational statistical data
        void resetResourceAllocationSchemeStatistics();     // Resets the resource allocation variables

        void initializeUHV_GOMEA();
        void initializeUHV_ADAM_BEST();

        void mergePopulations(population_pt populationUHV_GOMEA,
                              population_pt populationUHV_ADAM_BEST);



    private:
        // Optimizer to apply
        bool applyUHV_GOMEA;            // Should UHV-GOMEA be applied on a population branch
        bool applyUHV_ADAM_BEST;        // Should UHV-Adam by applied on a population

        // Hybrid analysis: Used to analyze this algorithm
        bool doHybridAnalysis;          // Overrides the resource allocation scheme and switches to a fixed schedule
        bool writeEveryStep;            // Write statistics for every call
        int generationToSwitch;         // Generation to switch to fixed schedule
        bool forceUHV_GOMEA;            // Should UHV-GOMEA be executed when doing the hybrid analysis
        bool forceUHV_ADAM;             // Should UHV-ADAM be executed when doing the hybrid analysis

        bool separateExecution;         // Execute the EA and gradient algorithms on separate populations

        // Optimizer indices
        int localOptimizerIndexUHV_GOMEA;   // Optimizer index of UHV-GOMEA
        int localOptimizerIndexUHV_ADAM;    // Optimizer index of UHV-ADAM
        int indexApplicationMethodUHV_ADAM; // Method of applying UHV-ADAM on a population

        // Resource allocation specific parameters
        int improvementMetricIndex; // Improvement metric index
        bool calculatePreAndPostImprovement;    // Calculate the improvement pre and post

        size_t numberOfUHV_GOMEACalls;              // The number of UHV-GOMEA calls to execute per generation
        double minimumNumberOfUHV_ADAM_BestCalls;   // The minimum number of UHV-Adam-Best calls to execute if the threshold is met

        // Optimizer resource allocation variables
        size_t generationThresholdUHV_ADAM_BEST;                  // Lower bound of generation when UHV-ADAM-BEST should be applied
        std::vector<size_t> vecNumberOfMOEvalsOfUHV_GOMEA;      // Eps-GOMEA:       Vector that keeps track of the number of evaluations used by UHV-GOMEA
        std::vector<size_t> vecNumberOfMOEvalsOfUHV_ADAM_Best;  // Eps-ADAM-BEST:   Vector that keeps track of the number of evaluations used by UHV-ADAM-BEST
        std::vector<double> vecImprovementsOfUHV_GOMEA;         // I-GOMEA:         Number of improvements obtained by UHV-GOMEA
        std::vector<double> vecImprovementsOfUHV_ADAM_Best;     // I-ADAM-Best:     Number of improvements obtained by UHV-ADAM-BEST
        double callsToExecuteNextGenerationUHV_ADAM_BEST;         // Number of calls to execute UHV-ADAM-BEST in this generation
        double storedCallCountUHV_ADAM_BEST;                      // Stores the number of UHV-ADAM-Best calls to resume after the wait threshold

        // Generational Archive
        bool useGenerationalArchive;    // Boolean if the generational archive should be used

        GenerationalArchive generationalMOSolutionsArchive;     // The MO solutions archive use for non dominated improvement metrics

        // Initialization methods
        void initializeOptimizers();

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
                double & improvementByUHV_ADAM_BEST,
                size_t & numberOfCallsThisGenerationByUHV_ADAM_BEST);
//        void execute_uhv_adam_all_solution(size_t sample_size);

        // Resource allocation methods
        void calculateResourcesNextGeneration(
                size_t MOEvaluationsUsedThisGenerationByUHV_GOMEA,
                size_t MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST,
                double improvementThisGenerationUHV_GOMEA,
                double improvementThisGenerationByUHV_ADAM_BEST,
                size_t numberOfCallsThisGenerationByUHV_ADAM_BEST);
        void storeResourceAllocationStatisticsUHV_GOMEA(size_t MOEvaluationsUsedUHV_GOMEA, double improvementUHV_GOMEA);
        void storeResourceAllocationStatisticsUHV_ADAM_BEST(size_t MOEvaluationsUsedUHV_ADAM_BEST, double improvementUHV_ADAM_BEST);
        void calculateResourcesUHV_GOMEA(size_t minimumRequiredMOEvaluations, size_t & sumEvaluationsRequiredByUHV_GOMEA, double & sumImprovementsUHV_GOMEA);
        void calculateResourcesUHV_ADAM_BEST(size_t & sumEvaluationsRequiredByUHV_ADAMBEST, double & sumImprovementsUHV_ADAM_BEST);

        bool optimizerHasContributed(bool optimizerActive, double improvementUHVThisGeneration);
        double applyMemoryDecay(double nextValue, double previousValue);

        // Improvement metric methods
        void determineImprovementMetricVariables();
        double calculateImprovements(population_pt populationBefore, population_pt populationAfter, int indexSolutionChanged);

        // Optimizer statistics methods
        void storeOptimizerStatistics(std::string optimizerName,
                                      int currentNumberOfSOEvaluations,
                                      size_t currentNumberOfMOEvaluations,
                                      double currentImprovementsFound,
                                      size_t currentNumberOfCalls);

        // Helper methods
        std::string convertUHV_GOMEALinkageModelNameToString(int linkageModelUHV_GOMEA);
        std::string convertUHV_ADAMModelNameToString(int versionUHV_ADAM);



    };
}
