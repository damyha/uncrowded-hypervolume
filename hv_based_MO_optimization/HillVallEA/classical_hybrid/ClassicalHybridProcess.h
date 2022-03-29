/*
 * ClassicalHybridProcess.h
 *
 * This class takes care of the optimization process.
 * Think of initialization of the process, termination of the hybrid algorithm and writing of statistics
 *
 * Implementation by D. Ha 2021
 */

#include "../hillvallea_internal.hpp"
#include "../optimizer.hpp"
#include "../../UHV.hpp"


namespace hillvallea{
    class ClassicalHybridProcess {
    public:
        // Constructor and destructor
        ClassicalHybridProcess(
                std::shared_ptr<UHV_t> fitnessFunction,
                int localOptimizerIndexUHV_GOMEA,
                int localOptimizerIndexUHV_ADAM,
                int indexApplicationMethodUHV_ADAM,
                int numberOfSOParameters,
                size_t solutionSetSize,
                const vec_t &soLowerInitializationRange,
                const vec_t &soUpperInitializationRange,
                int maximumNumberOfMOEvaluations,
                int maximumTimeInSeconds,
                double valueVTR,
                bool useVTR,
                int randomSeedNumber,
                bool writeSolutionsOptimizers,
                bool writeStatisticsOptimizers,
                std::string writeDirectory,
                std::string fileAppendix
                );

        ~ClassicalHybridProcess();

        void run_optimization(size_t populationSize);  // Executes UHV-CLASSICAL-HYBRID

        // The optimizer
        classicalHybrid_pt uhvClassicalHybrid;

        // Getters and setters
        int get_number_of_evaluations();
        clock_t get_starting_time();

        // Public variables
        std::shared_ptr<UHV_t> uhvFitnessFunction;  // The uhv fitness function

    private:
        // General variables
        int numberOfSOParameters;               // The number of (SO) parameters
        size_t solutionSetSize;                 // The solution set size
        vec_t lowerInitializationRanges;        // The lower initialization range per parameter
        vec_t upperInitializationRanges;        // The upper initialization range per parameter
        vec_t lowerParameterBounds;             // The minimum value a solution can take per parameter
        vec_t upperParameterBounds;             // The maximum value a solution can take per parameter
        int maximumNumberOfMOEvaluations;       // The maximum number of (SO) evaluations before termination
        int maximumTimeInSeconds;               // The time limit in seconds the optimization is allowed to use
        bool useVTR;                            // Should the process terminate when a certain UHV is reached
        double valueVTR;                        // The UHV termination value
        int randomSeedNumber;                   // The random seed number
        bool writeSolutionsOptimizers;          // Write the population after applying the optimizer
        bool writeStatisticsOptimizers;         // Write the statistics after applying the optimizer
        std::string writeDirectory;             // The directory to write files to
        std::string file_appendix;              // The file appendix

        // Derived variables
        std::shared_ptr<std::mt19937> rng;  // The rng object

        // Optimizer variables
        int localOptimizerIndexUHV_GOMEA;       // Optimizer index of UHV-GOMEA
        int localOptimizerIndexUHV_ADAM;        // Optimizer index of UHV-ADAM
        int indexApplicationMethodUHV_ADAM;     // Application method of UHV-ADAM

        // Statistical variables
        solution_t bestSolution;    // The best solution found
        bool terminated;            // Has the process terminated
        bool success;               // Has the process successfully terminated
        int numberOfSOEvaluations;  // The number of SO evaluations
        int numberOfMOEvaluations;  // The number of MO evaluations
        int numberOfGenerations;    // The number of generations
        clock_t startingTime;       // The starting time

        double scaledSearchVolume;  // Reduces round-off errors and +inf errors for large number_of_parameters;

        // Initialization methods
        void init_default_params();     // Initialize the default parameters of this optimization process
        void reset_log_variables();     // Resets log variables to the default values

        // Termination methods
        bool terminate_on_runtime();                                // Terminate on runtime
        bool terminate_on_vtr(optimizer_pt optimizer_selected);     // Terminate on UHV value reached

        // Logging methods
        // Creates a new statistics file that contains the column headers
        void create_initial_statistics_file(std::string &statisticsPath,
                                            std::shared_ptr<std::ofstream> statisticsStream);
        void writeOptimizersStatistics(classicalHybrid_pt uhvClassicalHybrid,
                                       size_t currentGenerationIndex,
                                       std::shared_ptr <std::ofstream> statistics_file);

    };
}
