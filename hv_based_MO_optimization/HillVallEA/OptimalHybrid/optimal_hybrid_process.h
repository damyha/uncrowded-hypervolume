/*
 * optimal_hybrid_process.h
 *
 * This class takes care of the optimization process.
 * Think of termination conditions, writing statistics to files, etc.
 *
 * Implementation by D. Ha 2021
 */

#include "../hillvallea_internal.hpp"
#include "../optimizer.hpp"

#include "../../UHV.hpp"

namespace hillvallea{

    class optimal_hybrid_process_t {
    public:

        // Constructor and destructor
        optimal_hybrid_process_t(
                std::shared_ptr<UHV_t> fitness_function,
                int localOptimizerIndexUHV_GOMEA,
                int localOptimizerIndexUHV_ADAM,
                int versionUHV_ADAM,
                int reinitializeOptimizers,
                int number_of_parameters,
                const vec_t &lower_init_ranges,
                const vec_t &upper_init_ranges,
                int maximumNumberOfMOEvaluations,
                int maximum_number_of_seconds,
                bool use_vtr,
                double vtr,
                size_t numberOFUHVGOMEACallsPerStep,
                size_t numberOfUHVADAMCallsPerStep,
                size_t maxMOEvaluationsPerGeneration,
                int random_seed,
                bool write_solutions_per_branch,
                bool write_statistics_per_branch,
                std::string write_directory,
                std::string file_appendix
        );

        ~optimal_hybrid_process_t();

        // Statistical variables
        solution_t bestSolution;                    // The best solution found
        optimal_hybrid_optimizer_pt uhv_optimizer;  // The optimal hybrid optimizer

        void run_optimization(size_t population_size);  // Executes UHV-OPTIMAL-HYBRID

    private:
        // General variables
        std::shared_ptr<UHV_t> uhvFitnessFunction;  // The (SO) fitness function
        int reinitializeOptimizers;                 // Keep the state of the optimizers every branch
        int number_of_parameters;                   // The number of (SO) parameters
        size_t solution_set_size;                   // The solution set size
        vec_t lower_init_ranges;                    // The lower initialization range per parameter
        vec_t upper_init_ranges;                    // The upper initialization range per parameter
        vec_t lower_param_bounds;                   // The minimum value a solution can take per parameter
        vec_t upper_param_bounds;                   // The maximum value a solution can take per parameter
        int maximumNumberOfMOEvaluations;           // The maximum number of (MO) evaluations before termination
        int maximum_number_of_seconds;              // The time limit in seconds the optimization is allowed to use
        bool use_vtr;                               // Should the process terminate when a certain UHV is reached
        double vtr;                                 // The UHV termination value
        int random_seed;                            // The random seed number
        bool write_solutions_per_branch;            // Write the solutions per branch (NOT IMPLEMENTED)
        bool write_statistics_per_branch;           // Write statistics files per branch
        std::string write_directory;                // The directory to write files to
        std::string file_appendix;                  // The file appendix

        // Derived variables
        std::shared_ptr<std::mt19937> rng;  // The rng object

        // Optimizer variables
        int localOptimizerIndexUHV_GOMEA;   // The optimizer index of UHV-GOMEA
        int localOptimizerIndexUHV_ADAM;    // The optimizer index of UHV-ADAM
        int versionUHV_ADAM;                // The application method of UHV-ADAM

        size_t numberOFUHVGOMEACallsPerStep;    // The number of UHV-GOMEA calls per step
        size_t numberOfUHVADAMCallsPerStep;     // The number of UHV-ADAM calls per step
        size_t maxMOEvaluationsPerGeneration;   // The maximum number

        // Statistical variables
        bool terminated;                // Has the process terminated
        bool success;                   // Has the process sucessfully terminated
        int number_of_evaluations;      // The number of evaluations
        int number_of_generations;      // The number of generations
        clock_t starting_time;          // The starting time

        double scaled_search_volume;    // Reduces round-off errors and +inf errors for large number_of_parameters;

        // Initialization methods
        void init_default_params();     // Initialize the default parameters of this optimization process
        void reset_log_variables();     // Resets log variables to the default values

        // Termination methods
        bool terminate_on_runtime();    // Terminate on runtime
        bool terminate_on_vtr(optimal_hybrid_optimizer_pt optimizer);

        // Logging methods
        // Creates a new statistics file that contains the column headers
        void createInitialStatisticsFile(std::string & initialStatisticsFilePath,
                                         std::shared_ptr<std::ofstream> initialStatisticsFile);
        // Write the population branches to their respective statistics files
        void write_population_branch_statistics(
                std::vector<optimization_branch_pt> bestOptimizationBranches,
                std::shared_ptr<UHV_t> uhvFitnessFunction);




    };


}



