/*
 * uhvSwitchProcess.cpp
 *
 * This class takes care of the optimization process.
 * Think of initialization of the process, termination of the algorithm and writing of statistics
 *
 * Implementation by D. Ha 2021
 */

#include "../hillvallea_internal.hpp"
#include "../optimizer.hpp"
#include "../../UHV.hpp"

namespace hillvallea{

    class UHVSWITCHProcess {
    public:

        // Constructor and destructor
        UHVSWITCHProcess(
                std::shared_ptr<UHV_t> fitness_function,
                int local_optimizer_index_gomea,
                int local_optimizer_index_gradient,
                int indexApplicationMethodUHV_ADAM,
                int number_of_parameters,
                size_t solution_set_size,
                const vec_t &lower_init_ranges,
                const vec_t &upper_init_ranges,
                int maximum_number_of_evaluations,
                int maximum_number_of_seconds,
                double vtr,
                double uhvSwitchValue,
                bool use_vtr,
                int random_seed,
                bool write_solutions_optimizers,
                bool write_statistics_optimizers,
                std::string write_directory,
                std::string file_appendix
                );

        ~UHVSWITCHProcess();

        void run_optimization(size_t population_size);  // Executes UHV-SWITCH

        // The optimizer
        UHVSwitch_pt uhv_switch;

        // Getters and setters
        int get_number_of_evaluations();
        clock_t get_starting_time();

        // Public variables
        std::shared_ptr<UHV_t> fitness_function;// The (SO) fitness function

    private:
        // General variables
        int number_of_parameters;               // The number of (SO) parameters
        size_t solution_set_size;               // The solution set size
        vec_t lower_init_ranges;                // The lower initialization range per parameter
        vec_t upper_init_ranges;                // The upper initialization range per parameter
        vec_t lower_param_bounds;               // The minimum value a solution can take per parameter
        vec_t upper_param_bounds;               // The maximum value a solution can take per parameter
        int maximum_number_of_evaluations;      // The maximum number of (SO) evaluations before termination
        int maximum_number_of_seconds;          // The time limit in seconds the optimization is allowed to use
        bool use_vtr;                           // Should the process terminate when a certain UHV is reached
        double vtr;                             // The UHV termination value
        double uhvSwitchValue;                  // The UHV value to switch from UHV-GOMEA to the UHV-Gradient algorithm
        int random_seed;                        // The random seed number
        bool write_solutions_optimizers;        // Write the population after applying the optimizer
        bool write_statistics_optimizers;       // Write the statistics after applying the optimizer
        std::string write_directory;            // The directory to write files to
        std::string file_appendix;              // The file appendix

        // Derived variables
        std::shared_ptr<std::mt19937> rng;  // The rng object

        // Optimizer variables
        int local_optimizer_index_gomea;    // The optimizer index of UHV-GOMEA
        int local_optimizer_index_gradient; // The optimizer index of the gradient algorithm
        int indexApplicationMethodUHV_ADAM; // The application method to apply on the gradient algorithm on a population

        // Statistical variables
        solution_t best_solution;       // The best solution found
        bool terminated;                // Has the process terminated
        bool success;                   // Has the process successfully terminated
        int number_of_evaluations;      // The number of evaluations
        int number_of_generations;      // The number of generations
        clock_t starting_time;          // The starting time

        double scaled_search_volume;    // Reduces round-off errors and +inf errors for large number_of_parameters;

        // Initialization methods
        void init_default_params();     // Initialize the default parameters of this optimization process
        void reset_log_variables();     // Resets log variables to the default values

        // Termination methods
        bool terminate_on_runtime();                                // Terminate on runtime
        bool terminate_on_vtr(optimizer_pt optimizer_selected);     // Terminate on UHV value reached

        // Logging methods
        // Creates a new statistics file that contains the column headers
        void create_initial_statistics_file(
                std::string &initial_population_branch_path,
                std::shared_ptr<std::ofstream> initial_population_branch_file);
        void writeOptimizersStatistics(
                std::vector<std::string> vecNamesOptimizers,
                std::vector<population_pt> vecPopulationsCreatedByOptimizers,
                std::vector<int> vecNumberOfSOEvaluationsPerOptimizer,
                std::vector<size_t> vecNumberOfMOEvaluationsPerOptimizer,
                size_t currentGenerationIndex,
                std::shared_ptr<std::ofstream> statistics_file);

    };


}



