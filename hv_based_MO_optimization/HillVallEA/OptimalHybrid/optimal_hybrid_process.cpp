/*
 * optimal_hybrid_process.cpp
 *
 * This class takes care of the optimization process.
 * Think of termination conditions, writing statistics to files, etc.
 *
 * Implementation by D. Ha 2021
 *
 */

#include "optimal_hybrid_process.h"
#include "optimal_hybrid_optimizer.h"

#include "../fitness.h"
#include "../hillvallea_internal.hpp"


namespace hillvallea{

    /**
     * Constructor
     * @param fitness_function  The fitness function
     * @param local_optimizer_index_gomea The local optimizer index of UHV-GOMEA
     * @param localOptimizerIndexUHV_ADAM The local optimizer index of UHV-ADAM
     * @param reinitializeOptimizers Should the optimizers reinitialize every step
     * @param number_of_parameters The number of (SO) parameters
     * @param lower_init_ranges The lower initialization range per parameter
     * @param upper_init_ranges The upper initialization range per parameter
     * @param maximum_number_of_evaluations The maximum number of single objective evaluations
     * @param maximum_number_of_seconds The maximum time for the algorithm to run
     * @param vtr The UHV value to reach
     * @param use_vtr Termination on UHV value allowed
     * @param random_seed The random seed number
     * @param write_solutions_per_branch Should solutions of every branch be written
     * @param write_statistics_per_branch Should statistics be written for every branch
     * @param write_directory The write directory
     * @param file_appendix The appendix of the files
     */
    optimal_hybrid_process_t::optimal_hybrid_process_t(
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
        )
    {
        // Copy all parameters
        this->uhvFitnessFunction = fitness_function;
        this->localOptimizerIndexUHV_GOMEA = localOptimizerIndexUHV_GOMEA;
        this->localOptimizerIndexUHV_ADAM = localOptimizerIndexUHV_ADAM;
        this->versionUHV_ADAM = versionUHV_ADAM;
        this->reinitializeOptimizers = reinitializeOptimizers;
        this->number_of_parameters = number_of_parameters;
        this->lower_init_ranges = lower_init_ranges;
        this->upper_init_ranges = upper_init_ranges;
        this->maximumNumberOfMOEvaluations = maximumNumberOfMOEvaluations;
        this->maximum_number_of_seconds = maximum_number_of_seconds;
        this->use_vtr = use_vtr;
        this->vtr = vtr;
        this->numberOFUHVGOMEACallsPerStep = numberOFUHVGOMEACallsPerStep;
        this->numberOfUHVADAMCallsPerStep = numberOfUHVADAMCallsPerStep;
        this->maxMOEvaluationsPerGeneration = maxMOEvaluationsPerGeneration;
        this->random_seed = random_seed;
        this->write_solutions_per_branch = false;   // Todo: not implemented
        this->write_statistics_per_branch = true;   // Todo: make this an option
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
    optimal_hybrid_process_t::~optimal_hybrid_process_t() {}


    /**
     * Initializes the default parameters of the process
     */
    void optimal_hybrid_process_t::init_default_params()
    {
        // Determine the scaled search volume
        double scaled_search_volume_result = 1.0;
        for (size_t i = 0; i < this->upper_init_ranges.size(); ++i) {
            double upper_min_lower_init = this->upper_init_ranges[i] - this->lower_init_ranges[i];
            scaled_search_volume_result *= pow(upper_min_lower_init, 1.0 / this->number_of_parameters);
        }
        this->scaled_search_volume = scaled_search_volume_result;

        // Set single objective init bounds to parameter bounds if it's beyond the param bounds
        uhvFitnessFunction->get_param_bounds(lower_param_bounds, upper_param_bounds);
        for (size_t i = 0; i < lower_param_bounds.size(); ++i) {
            if (lower_init_ranges[i] < lower_param_bounds[i]) { lower_init_ranges[i] = lower_param_bounds[i]; }
            if (upper_init_ranges[i] > upper_param_bounds[i]) { upper_init_ranges[i] = upper_param_bounds[i]; }
        }
    }

    /**
     * Resets log variables to the default values
     */
    void optimal_hybrid_process_t::reset_log_variables()
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
    bool optimal_hybrid_process_t::terminate_on_runtime() {
        // Stop if we run out of time.
        if (maximum_number_of_seconds > 0) {
            clock_t current_time = clock();
            double runtime = double(current_time - starting_time) / CLOCKS_PER_SEC;

            return runtime > maximum_number_of_seconds;
        }

        return false;
    }

    /**
     * Terminate when UHV value has been reached
     * @return Boolean if UHV-value has been reached
     */
    bool optimal_hybrid_process_t::terminate_on_vtr(optimal_hybrid_optimizer_pt optimizer)
    {
        // Initialize variables
        bool vtrHit, shallowVTRHit = false;

        // Go over all active branches
        std::vector<optimization_branch_pt> activeBranches = optimizer->activeOptimizationBranches;

        for(size_t branchIndex = 0; branchIndex < activeBranches.size(); ++branchIndex)
        {
            // Retrieve current fitness
            solution_pt bestSolutionOfBranch = activeBranches[branchIndex]->getCurrentPopulation()->bestSolution();

            // Shallow check if VTR has been hit and which VTR type is hit
            if (!uhvFitnessFunction->redefine_vtr) {
                shallowVTRHit = (bestSolutionOfBranch->constraint == 0) && (bestSolutionOfBranch->f <= vtr);
            } else {
                shallowVTRHit = uhvFitnessFunction->vtr_reached(*bestSolutionOfBranch, vtr);
            }

            // Find out if VTR has truly been reached
            if (shallowVTRHit) {

                // Check for round off errors
                if (uhvFitnessFunction->partial_evaluations_available &&
                        uhvFitnessFunction->has_round_off_errors_in_partial_evaluations) {

                    // re-evaluate solution when vtr is assumed to be hit.
                    uhvFitnessFunction->evaluate(bestSolutionOfBranch);

                    if (!uhvFitnessFunction->redefine_vtr) {
                        vtrHit = (bestSolutionOfBranch->constraint == 0) && (bestSolutionOfBranch->f <= vtr);
                    } else {
                        vtrHit = uhvFitnessFunction->vtr_reached(*bestSolutionOfBranch, vtr);
                    }
                }
                else {
                    vtrHit = true;
                }
            }
            else {
                vtrHit = false;
            }
        }

        return vtrHit;
    }

    /**
     * Do the optimization (Process copied from hillvallea_t::runSerial3())
     * Why not reuse hillvallea_t::runSerial3()? Because I need to write multiple statistics files.
     */
    void optimal_hybrid_process_t::run_optimization(size_t population_size) {
        // Reset log variables
        this->reset_log_variables();

        // Create the initial population
        population_pt  initialPopulation = std::make_shared<population_t>();

        // Initialize the population
        if (uhvFitnessFunction->redefine_random_initialization) {
            uhvFitnessFunction->init_solutions_randomly(*initialPopulation,
                                                      (size_t) population_size,
                                                      this->lower_init_ranges,
                                                      this->upper_init_ranges,
                                                      0,
                                                      this->rng);
        } else {
            initialPopulation->fill_uniform((size_t) population_size,
                                      this->number_of_parameters,
                                      this->lower_init_ranges,
                                      this->upper_init_ranges,
                                      this->rng);
        }

        // Evaluate the initial population (do not count evals) and sort on fitness
        initialPopulation->evaluate_with_gradients(this->uhvFitnessFunction, 0);
        initialPopulation->sort_on_fitness();

        // Init the UHV-OPTIMAL-HYBRID optimizer
        double init_univariate_bandwidth = scaled_search_volume * pow(initialPopulation->size(), -1.0 / number_of_parameters);
        this->uhv_optimizer = std::make_shared<optimal_hybrid_optimizer_t>(
                this->uhvFitnessFunction,
                this->reinitializeOptimizers,
                this->number_of_parameters,
                this->lower_param_bounds,
                this->upper_param_bounds,
                init_univariate_bandwidth,
                this->localOptimizerIndexUHV_GOMEA,
                this->localOptimizerIndexUHV_ADAM,
                this->versionUHV_ADAM,
                this->numberOFUHVGOMEACallsPerStep,
                this->numberOfUHVADAMCallsPerStep,
                this->maxMOEvaluationsPerGeneration,
                this->write_statistics_per_branch,
                this->rng,
                this->write_directory);
        uhv_optimizer->initialize_from_population(initialPopulation);

        // Create initial statistics file
        std::string initialStatisticsFilePath;
        std::shared_ptr<std::ofstream> initialStatisticsFile(new std::ofstream);
        if (write_statistics_per_branch) {
            this->createInitialStatisticsFile(initialStatisticsFilePath, initialStatisticsFile);
            uhv_optimizer->activeOptimizationBranches[0]->setStatisticsFileParameters(initialStatisticsFilePath,
                                                                                      initialStatisticsFile);
        }

        while (true)
        {
            // Terminate on set number of generations: Used for debugging
//            if (number_of_generations > 10)
//            {
//                break;
//            }

            // Terminate if VTR has been hit
            if (this->use_vtr && this->terminate_on_vtr(uhv_optimizer))
            {
                printf("VTR reached %lf/%lf \n", uhv_optimizer->best.f, vtr);
                success = true;
                break;
            }

            // Terminate if max number of evaluations has been reached Todo: select MO-evals from all branches
            auto currentNumberOfEvaluations = (size_t) uhv_optimizer->activeOptimizationBranches[0]->getCurrentNumberOfMOEvaluations();
            if (maximumNumberOfMOEvaluations > 0 &&
                currentNumberOfEvaluations >= maximumNumberOfMOEvaluations)
            {
                printf("Max evals reached %ld/%d (SO-evals)\n", currentNumberOfEvaluations, maximumNumberOfMOEvaluations);
                break;
            }

            // Terminate if time limit has been reached
            if (this->terminate_on_runtime())
            {
                printf("Time limit reached\n");
                break;
            }

//            printf("Executing generation.\n");
            uhv_optimizer->generation(population_size);
//            printf("Done with generation. \n");

            // Update best solution Todo: make the optimizer keep trck of this/ select from all branches available
            this->bestSolution = solution_t(*uhv_optimizer->activeOptimizationBranches[0]->getCurrentPopulation()->bestSolution());

            // Write to statistics file
            if (write_statistics_per_branch) {
//                printf("Writing Statistics.\n");
                write_population_branch_statistics(
                        uhv_optimizer->activeOptimizationBranches,
                        this->uhvFitnessFunction);
            }

            // Update statistics
            this->number_of_generations++;

        }

        // Close all statistics files
        uhv_optimizer->closeAllStatisticsFiles();
//        printf("Closed all statistics files\n");


    }

    /**
     * Creates a new statistics file that contains the column headers
     */
    void optimal_hybrid_process_t::createInitialStatisticsFile(std::string & initialStatisticsFilePath,
                                                               std::shared_ptr<std::ofstream> initialStatisticsFile)
    {
        // Determine statistics file name
        initialStatisticsFilePath = this->write_directory + "statistics_UHV-OPT_b0.dat";

        // Create and open statistics file
        initialStatisticsFile->open(initialStatisticsFilePath, std::ofstream::out | std::ofstream::trunc);

        // Add header to statistics file
        *initialStatisticsFile
                << "Algorithm        Gen    Evals  Calls          Time                    Best-f   Best-constr   Average-obj       "
                << "Std-obj    Avg-constr    Std-constr                 Best-HV               Best-IGD"
                << "                Best-GD size             Archive-HV            Archive-IGD             Archive-GD size   MO-evals"
                << std::endl;
    }

    /**
     * Writes all the population branches in population_branches to their respective statistics files
     * @param population_branches The vector of population branches
     * @param statistics_files_population_branches The vector of statistics files associated to a population branch
     */
    void optimal_hybrid_process_t::write_population_branch_statistics(
            std::vector<optimization_branch_pt> bestOptimizationBranches,
            std::shared_ptr<UHV_t> uhvFitnessFunction)
    {
        assert(bestOptimizationBranches.size() == 1);

        // Determine time that has passed
        clock_t current_time = clock();
        double runtime = double(current_time - this->starting_time) / CLOCKS_PER_SEC;

        // Prepare empty variables
        std::vector<solution_pt> empty_archive;
        empty_archive.clear();

        // Go over all population branches
        for (size_t branchIndex = 0; branchIndex < bestOptimizationBranches.size(); ++branchIndex )
        {
            optimization_branch_pt selectedBranch = bestOptimizationBranches[branchIndex];

            // Check if ofstream is already open
            if (selectedBranch->getStatisticsFile()->is_open())
            {
                // Retrieve history parameters
                std::vector<population_pt> historyPopulations;
                std::vector<std::string> historyOptimizerNames;
                std::vector<size_t> historyGenerations, historyCalls, historySOEvaluations;
                std::vector<double> historyMOEvaluations;
                selectedBranch->getHistoryParameters(historyPopulations,
                                                     historyOptimizerNames,
                                                     historyGenerations,
                                                     historyCalls,
                                                     historySOEvaluations,
                                                     historyMOEvaluations);

                // Write the history
                for(size_t stepIndex = 0; stepIndex < historyPopulations.size(); ++stepIndex)
                {
                    // Retrieve data
                    population_pt populationSelected = historyPopulations[stepIndex];
                    std::string optimizerName =  historyOptimizerNames[stepIndex];
                    size_t generations = historyGenerations[stepIndex];
                    size_t numberOfCalls = historyCalls[stepIndex];
                    size_t evaluationsSO = historySOEvaluations[stepIndex];
                    double evaluationsMO = historyMOEvaluations[stepIndex];

                    solution_pt bestSolution = populationSelected->bestSolution();

                    // Prepare the fitness function
                    uhvFitnessFunction->number_of_mo_evaluations = evaluationsMO;

                    *selectedBranch->getStatisticsFile()
                            << std::setw(14) << optimizerName
                            << " " << std::setw(3) << generations
                            << " " << std::setw(8) << evaluationsSO
                            << " " << std::setw(8) << std::fixed << std::setprecision(3) << numberOfCalls
                            << " " << std::setw(13) << std::scientific << std::setprecision(3) << runtime
                            << " " << std::setw(25) << std::scientific << std::setprecision(16) << bestSolution->f
                            << " " << std::setw(13) << std::scientific << std::setprecision(3) << bestSolution->constraint
                            << " " << std::setw(13) << std::scientific << std::setprecision(16) << populationSelected->average_fitness()
                            << " " << std::setw(13) << std::scientific << std::setprecision(3) << populationSelected->relative_fitness_std()
                            << " " << std::setw(13) << std::scientific << std::setprecision(3) << populationSelected->average_constraint()
                            << " " << std::setw(13) << std::scientific << std::setprecision(3) << populationSelected->relative_constraint_std()
                            << " " << uhvFitnessFunction->write_additional_solution_info(*bestSolution, empty_archive, false)
                            << std::endl;

                }
            }
            else {
                throw std::runtime_error("Population branch " + std::to_string(0) + "'s statistics file was not open.");    // Todo: fix this
            }
        }
    }

    /*
     * Todo: Copy pasted function for quick fix
     */


}