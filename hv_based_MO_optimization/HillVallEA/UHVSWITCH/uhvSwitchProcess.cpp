/*
 * uhvSwitchProcess.cpp
 *
 * This class takes care of the optimization process.
 * Think of initialization of the process, termination of the algorithm and writing of statistics
 *
 * Implementation by D. Ha 2021
 */


#include "uhvSwitchProcess.h"
#include "uhvSwitch.h"

#include "../fitness.h"
#include "../hillvallea_internal.hpp"
#include "../population.hpp"
#include "../solution.hpp"

namespace hillvallea{

    /**
     * Constructor
     * @param fitness_function The fitness function
     * @param local_optimizer_index_gomea The local optimizer index of UHV-GOMEA
     * @param number_of_parameters The number of (SO) parameters
     * @param solution_set_size The solution set size
     * @param lower_init_ranges The lower initialization range per parameter
     * @param upper_init_ranges The upper initialization range per parameter
     * @param maximum_number_of_evaluations The maximum number of single objective evaluations
     * @param maximum_number_of_seconds The maximum time for the algorithm to run
     * @param vtr The UHV value to reach
     * @param use_vtr Termination on UHV value allowed
     * @param memory_decay_factor The memory decay factor
     * @param random_seed The random seed number
     * @param write_solutions_optimizers Should solutions be written after applying an optimizer
     * @param write_statistics_optimizers Should statistics be written after applying an optimizer
     * @param write_directory The write directory
     * @param file_appendix The appendix of the files (probably not used due to max file length)
     */
    UHVSWITCHProcess::UHVSWITCHProcess(
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
    )
    {
        // Copy all parameters
        this->fitness_function = fitness_function;
        this->local_optimizer_index_gomea = local_optimizer_index_gomea;
        this->local_optimizer_index_gradient = local_optimizer_index_gradient;
        this->indexApplicationMethodUHV_ADAM = indexApplicationMethodUHV_ADAM;
        this->number_of_parameters = number_of_parameters;
        this->solution_set_size = solution_set_size;
        this->lower_init_ranges = lower_init_ranges;
        this->upper_init_ranges = upper_init_ranges;
        this->maximum_number_of_evaluations = maximum_number_of_evaluations;
        this->maximum_number_of_seconds = maximum_number_of_seconds;
        this->use_vtr = use_vtr;
        this->vtr = vtr;
        this->uhvSwitchValue = uhvSwitchValue;
        this->random_seed = random_seed;
        this->write_solutions_optimizers = write_solutions_optimizers;
        this->write_statistics_optimizers = write_statistics_optimizers;
        this->write_directory = write_directory;
        this->file_appendix = file_appendix;

        // Derive parameters
        this->rng = std::make_shared<std::mt19937>((unsigned long) (random_seed));
        std::uniform_real_distribution<double> unif(0, 1);

        // Initialize default parameters
        init_default_params();
    }

    /**
     * Destructor
     */
    UHVSWITCHProcess::~UHVSWITCHProcess() {}


    /**
     * Initializes the default parameters of the process
     */
    void UHVSWITCHProcess::init_default_params()
    {
        // Determine the scaled search volume
        double scaled_search_volume_result = 1.0;
        for (size_t i = 0; i < this->upper_init_ranges.size(); ++i) {
            double upper_min_lower_init = this->upper_init_ranges[i] - this->lower_init_ranges[i];
            scaled_search_volume_result *= pow(upper_min_lower_init, 1.0 / this->number_of_parameters);
        }
        this->scaled_search_volume = scaled_search_volume_result;

        // Set single objective init bounds to parameter bounds if it's beyond the param bounds
        fitness_function->get_param_bounds(lower_param_bounds, upper_param_bounds);
        for (size_t i = 0; i < lower_param_bounds.size(); ++i) {
            if (lower_init_ranges[i] < lower_param_bounds[i]) { lower_init_ranges[i] = lower_param_bounds[i]; }
            if (upper_init_ranges[i] > upper_param_bounds[i]) { upper_init_ranges[i] = upper_param_bounds[i]; }
        }
    }

    /**
     * Resets log variables to the default values
     */
    void UHVSWITCHProcess::reset_log_variables()
    {
        // Reset all log variables
        this->starting_time = clock();
        this->success = false;
        this->terminated = false;
        this->number_of_evaluations = 0;
        this->number_of_generations = 0;
    }

    /**
     * Terminate when max runtime is exceeded
     * @return True if max runtime is exceeded
     */
    bool UHVSWITCHProcess::terminate_on_runtime() {
        // Stop if time has ran out
        if (this->maximum_number_of_seconds > 0) {
            clock_t current_time = clock();
            double runtime = double(current_time - this->starting_time) / CLOCKS_PER_SEC;

            return runtime > maximum_number_of_seconds;
        }

        return false;
    }

    /**
     * Terminate when UHV value has been reached
     * Todo This termination is not optimal since ADAM does not reshuffle the best solution to index 0
     * @return Boolean if UHV-value has been reached
     */
    bool UHVSWITCHProcess::terminate_on_vtr(optimizer_pt optimizer_selected)
    {
        // Initialize variables
        bool vtrHit, shallowVTRHit = false;

        // Determine variables
        size_t best_solution_index = optimizer_selected->best_solution_index;
        solution_pt best_solution = optimizer_selected->pop->sols[best_solution_index];

        // Shallow check if VTR has been hit and which VTR type is hit
        if (!fitness_function->redefine_vtr) {
            shallowVTRHit = (best_solution->constraint == 0) && (best_solution->f <= vtr);
        } else {
            shallowVTRHit = fitness_function->vtr_reached(*best_solution, vtr);
        }

        // Find out if VTR has truly been reached
        if (shallowVTRHit) {

            // Check for round off errors
            if (fitness_function->partial_evaluations_available &&
                fitness_function->has_round_off_errors_in_partial_evaluations) {

                // re-evaluate solution when vtr is assumed to be hit.
                fitness_function->evaluate_with_gradients(best_solution);
                this->number_of_evaluations++;

                if (!fitness_function->redefine_vtr) {
                    vtrHit = (best_solution->constraint == 0) && (best_solution->f <= vtr);
                } else {
                    vtrHit = fitness_function->vtr_reached(*best_solution, vtr);
                }
            }
            else {
                vtrHit = true;
            }
        }
        else {
            vtrHit = false;
        }

        return vtrHit;
    }

    /**
     * Do the optimization (Process copied from hillvallea_t::runSerial3())
     * Why not reuse hillvallea_t::runSerial3()? Because statistics might need to be written after each optimizer is done.
     */
    void UHVSWITCHProcess::run_optimization(size_t population_size) {
        // Reset log variables
        this->reset_log_variables();
        this->number_of_evaluations = 0;

        // Create initial population
        population_pt initial_pop = std::make_shared<population_t>();

        // Initialize the population
        if (fitness_function->redefine_random_initialization) {
            fitness_function->init_solutions_randomly(*initial_pop,
                                                      population_size,
                                                      this->lower_init_ranges,
                                                      this->upper_init_ranges,
                                                      0,
                                                      this->rng);
        } else {
            initial_pop->fill_uniform((size_t) population_size,
                                      this->number_of_parameters,
                                      this->lower_init_ranges,
                                      this->upper_init_ranges,
                                      this->rng);
        }


        // Evaluate the initial population (do not count evals) and sort on fitness
        this->number_of_evaluations += initial_pop->evaluate_with_gradients(this->fitness_function, 0);
        initial_pop->sort_on_fitness();

        // Create initial statistics file
        std::string initial_population_statistics_path;
        std::shared_ptr<std::ofstream> initial_population_statistics_file(new std::ofstream);
        if (write_statistics_optimizers) {
            this->create_initial_statistics_file(initial_population_statistics_path, initial_population_statistics_file);
        }

        // Init the UHV-OPTIMAL-HYBRID optimizer
        double init_univariate_bandwidth = scaled_search_volume * pow(initial_pop->size(), -1.0 / number_of_parameters);
        this->uhv_switch = std::make_shared<UHVSwitch>(
                this->fitness_function,
                this->local_optimizer_index_gomea,
                this->local_optimizer_index_gradient,
                this->indexApplicationMethodUHV_ADAM,
                this->uhvSwitchValue,
                this->number_of_parameters,
                this->lower_param_bounds,
                this->upper_param_bounds,
                init_univariate_bandwidth,
                this->rng);

        this->uhv_switch->initialize_from_population(initial_pop, population_size);

        // Write initial population's generational statistics
        if (write_statistics_optimizers) {
            writeOptimizersStatistics(
                    uhv_switch->vecOptimizerNameOfPopulation,
                    uhv_switch->vecPopulationCreatedByOptimizer,
                    uhv_switch->vecNumberOfSOEvaluationsByOptimizer,
                    uhv_switch->vecNumberOfMOEvaluationsByOptimizer,
                    0,
                    initial_population_statistics_file);
        }

        // Optimization loop
        bool statisticsRecentlyWritten = false; // Prevents the process from writing the same data twice at the end
        while (true)
        {
            // Terminate if max number of evaluations has been reached
            if (maximum_number_of_evaluations > 0 && this->number_of_evaluations >= maximum_number_of_evaluations)
            {
                printf("Max evals reached %d/%d (SO-evals)\n", this->number_of_evaluations, maximum_number_of_evaluations);
                break;
            }

            // Terminate if time limit has been reached
            if (this->terminate_on_runtime())
            {
                printf("Time limit reached\n");
                break;
            }

            // Terminate if VTR has been hit
            if (this->use_vtr && this->terminate_on_vtr(uhv_switch))
            {
                printf("VTR reached %lf/%lf \n", uhv_switch->best.f, vtr);
                success = true;
                break;
            }

            statisticsRecentlyWritten = false;

            uhv_switch->generation(population_size, this->number_of_evaluations);

            // Todo: Write generational solutions

            // Write statistics
            if (write_statistics_optimizers) {
//                if (this->number_of_generations < 100 || this->number_of_generations % 100 == 0)
                {
                    statisticsRecentlyWritten = true;
                    writeOptimizersStatistics(
                            uhv_switch->vecOptimizerNameOfPopulation,
                            uhv_switch->vecPopulationCreatedByOptimizer,
                            uhv_switch->vecNumberOfSOEvaluationsByOptimizer,
                            uhv_switch->vecNumberOfMOEvaluationsByOptimizer,
                            number_of_generations,
                            initial_population_statistics_file);
                }
            }

            // Update statistics
            this->number_of_generations++;
        }

        // Write final statistics
        if (write_statistics_optimizers && !statisticsRecentlyWritten)
        {
            writeOptimizersStatistics(
                    uhv_switch->vecOptimizerNameOfPopulation,
                    uhv_switch->vecPopulationCreatedByOptimizer,
                    uhv_switch->vecNumberOfSOEvaluationsByOptimizer,
                    uhv_switch->vecNumberOfMOEvaluationsByOptimizer,
                    number_of_generations,
                    initial_population_statistics_file);
        }

        // Close all statistics files
        initial_population_statistics_file->close();


    }

    /**
     * Creates a new statistics file that contains the column headers
     * @param initial_population_statistics_path A pointer to the string that contains the path of the statistics file
     * @param initial_population_statistics_file A pointer to the ofstream of the statistics file
     */
    void UHVSWITCHProcess::create_initial_statistics_file(
            std::string & initial_population_statistics_path,
            std::shared_ptr<std::ofstream> initial_population_statistics_file)
    {
        // Determine statistics file name
        initial_population_statistics_path = this->write_directory + "statistics_UHV-SWITCH.dat";

        // Create and open statistics file
        initial_population_statistics_file->open(initial_population_statistics_path, std::ofstream::out | std::ofstream::trunc);

        // Add header to statistics file
        *initial_population_statistics_file
                <<  "Algorithm        Gen    Evals          Time                    Best-f   "
                << "Best-constr   Average-obj       Std-obj    Avg-constr    Std-constr "
                << fitness_function->write_solution_info_header(false) << std::endl;
    }


    /**
     * Writes all the intermediate statistics of each optimizer that was executed.
     * Assumes that the file is already open. Todo: might be better to write a statistics object.
     * @param vecNamesOptimizers A vector of the names of the optimizer that were used
     * @param vecPopulationsCreatedByOptimizers A vector of populations created by an optimizer
     * @param vecNumberOfSOEvaluationsPerOptimizer A vector of total SO evaluations that have passed after executing the optimizer
     * @param vecNumberOfMOEvaluationsPerOptimizer A vector of total MO evaluations that have passed after executing the optimizer
     * @param currentGenerationIndex The current generation
     * @param statistics_file The statistics file to write the data to.
     */
    void UHVSWITCHProcess::writeOptimizersStatistics(
            std::vector<std::string> vecNamesOptimizers,
            std::vector<population_pt> vecPopulationsCreatedByOptimizers,
            std::vector<int> vecNumberOfSOEvaluationsPerOptimizer,
            std::vector<size_t> vecNumberOfMOEvaluationsPerOptimizer,
            size_t currentGenerationIndex,
            std::shared_ptr<std::ofstream> statistics_file)
    {
        // Sanity check on data vectors
        if (vecPopulationsCreatedByOptimizers.size() != vecNamesOptimizers.size() ||
                vecPopulationsCreatedByOptimizers.size() != vecNumberOfSOEvaluationsPerOptimizer.size() ||
                vecPopulationsCreatedByOptimizers.size() != vecNumberOfMOEvaluationsPerOptimizer.size())
        {
            throw std::runtime_error("Statistics vectors are not of equal size:Pop: "
                + std::to_string(vecPopulationsCreatedByOptimizers.size()) + ", " +
                "Opt: " + std::to_string(vecNamesOptimizers.size()) + ", " +
                "SO Evals: " + std::to_string(vecNumberOfSOEvaluationsPerOptimizer.size()) +
                "MO Evals: " + std::to_string(vecNumberOfMOEvaluationsPerOptimizer.size()));
        }

        // Check not empty
        if (vecPopulationsCreatedByOptimizers.size() == 0 ||
                vecNamesOptimizers.size() == 0 ||
                vecNumberOfSOEvaluationsPerOptimizer.size() == 0 ||
                vecNumberOfMOEvaluationsPerOptimizer.size() == 0)
        {
            throw std::runtime_error("Statistics vectors are empty:Pop: "
                                     + std::to_string(vecPopulationsCreatedByOptimizers.size()) + ", " +
                                     "Opt: " + std::to_string(vecNamesOptimizers.size()) + ", " +
                                     "SO Evals: " + std::to_string(vecNumberOfSOEvaluationsPerOptimizer.size()) +
                                     "MO Evals: " + std::to_string(vecNumberOfMOEvaluationsPerOptimizer.size()));
        }

        // Determine time that has passed
        clock_t current_time = clock();
        double runtime = double(current_time - this->starting_time) / CLOCKS_PER_SEC;

        // Prepare empty variables
        std::vector<solution_pt> empty_archive;
        empty_archive.clear();

        // Go over optimizers
        for (size_t optimizerIndex = 0; optimizerIndex < vecPopulationsCreatedByOptimizers.size(); ++optimizerIndex )
        {
            // Retrieve data
            std::string optimizerName = vecNamesOptimizers[optimizerIndex];
            population_pt optimizerPopulation = vecPopulationsCreatedByOptimizers[optimizerIndex];
            int numberOfSOEvaluations = vecNumberOfSOEvaluationsPerOptimizer[optimizerIndex];
            size_t numberOfMOEvaluations = vecNumberOfMOEvaluationsPerOptimizer[optimizerIndex];

            solution_pt best_solution_pt = optimizerPopulation->bestSolution();

            // Check if ofstream is already open
            if (statistics_file->is_open())
            {
                // Adjust fitness function parameters Todo: This is a quick implementation, but might be prone to errors
                fitness_function->number_of_mo_evaluations = numberOfMOEvaluations;

                // Write to statistics file
                *statistics_file
                        << std::setw(14) << optimizerName
                        << " " << std::setw(3) << currentGenerationIndex
                        << " " << std::setw(8) << numberOfSOEvaluations
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << runtime
                        << " " << std::setw(25) << std::scientific << std::setprecision(16) << best_solution_pt->f
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << best_solution_pt->constraint
                        << " " << std::setw(13) << std::scientific << std::setprecision(16) << optimizerPopulation->average_fitness()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << optimizerPopulation->relative_fitness_std()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << optimizerPopulation->average_constraint()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << optimizerPopulation->relative_constraint_std()
                        << " " << fitness_function->write_additional_solution_info(*best_solution_pt, empty_archive, false)
                        << std::endl;
            }
            else {
                throw std::runtime_error("Statistics file was not open.");
            }
        }
    }

    /**
     * Getter of 'number_of_evaluations'
     * @return 'number_of_evaluations'
     */
    int UHVSWITCHProcess::get_number_of_evaluations()
    {
        return this->number_of_evaluations;
    }

    /**
     * Getter of 'starting_time'
     * @return 'starting_time'
     */
    clock_t UHVSWITCHProcess::get_starting_time()
    {
        return this->starting_time;
    }



}