/*
 * uhvSwitch.h
 * This algorithm applies UHV-GOMEA until a certain UHV threshold is met, after which a gradient algorithm is applied.
 *
 * Some warnings:
 * - Do not sort UHV-GOMEA's population
 * - The population pointer is shared between the algorithms
 *
 * Implementation by D. Ha
 */

#include "uhvSwitch.h"

#include "../fitness.h"



namespace hillvallea{

    UHVSwitch::UHVSwitch(
            std::shared_ptr<UHV_t> fitnessFunction,
            int versionUHVGOMEA,
            int versionUHVADAM,
            int indexApplicationMethodUHV_ADAM,
            double uhvSwitchValue,
            const size_t number_of_parameters,
            const vec_t & lower_param_bounds,
            const vec_t & upper_param_bounds,
            double init_univariate_bandwidth,
            rng_pt rng) : optimizer_t(
                number_of_parameters,
                lower_param_bounds,
                upper_param_bounds,
                init_univariate_bandwidth,
                fitnessFunction,
                rng)
    {
        // Copy settings
        this->optimizerIndexUHV_GOMEA = versionUHVGOMEA;
        this->optimizerIndexUHVGradientT = versionUHVADAM;
        this->uhvGradientVersion = indexApplicationMethodUHV_ADAM;
        this->uhvFitnessFunction = fitnessFunction;
        this->uhvSwitchValue = uhvSwitchValue;

        // Initialize variables
        this->population_initialized = false;
        this->population_size = 0;
        this->hasSwitched = false;

        this->selection_fraction = 0.35;    // Taken from gomea.cpp
        this->use_boundary_repair = false;

        // Reset statistical data
        this->resetIntraGenerationalOptimizerStatistics();

        // Initialize optimizers
        this->applyUHVGradient = true;


        this->initializeOptimizers();

        // Asserts that optimizers are running (in case multiple gradient algorithms are used)
        assert(applyUHVGradient);
    }

    /**
     * Destructor
     */
    UHVSwitch::~UHVSwitch() { };

    /**
     * Initializes the optimizer objects
     */
    void UHVSwitch::initializeOptimizers()
    {
        initializeUHV_GOMEA();

        // Create UHV-ADAM on best solution optimizer
        if (this->applyUHVGradient)
        {
            initializeUHV_ADAM_BEST();
        }
    }

    /**
     * Initializes UHV-GOMEA
     */
    void UHVSwitch::initializeUHV_GOMEA()
    {
        // Create UHV-GOMEA optimizer
        this->optimizerUHV_GOMEA = std::make_shared<gomea_t>(
                this->number_of_parameters,
                this->lower_param_bounds,
                this->upper_param_bounds,
                this->init_univariate_bandwidth,
                this->optimizerIndexUHV_GOMEA,
                this->fitness_function,
                this->rng);
    }

    /**
     * Initializes UHV-ADAM-BEST
     */
    void UHVSwitch::initializeUHV_ADAM_BEST()
    {
        this->optimizerUHV_ADAM_BEST = std::make_shared<adam_on_population_t>(
                this->number_of_parameters,
                this->lower_param_bounds,
                this->upper_param_bounds,
                this->init_univariate_bandwidth,
                this->optimizerIndexUHVGradientT,
                this->uhvGradientVersion,
                this->fitness_function,
                this->rng);
    }


    /**
     * The name of the algorithm
     * @return The name of the algorithm
     */
    std::string UHVSwitch::name() const { return "UHV-SWITCH"; }


    /**
     * A setter of the initial population.
     * Todo: This method currently does not support 'target_popsize'
     * @param pop The initial population (that's sorted on fitness)
     * @param target_popsize The size the population can be shaped to (usually set equal to initial population size)
     */
    void UHVSwitch::initialize_from_population(population_pt pop, size_t target_popsize)
    {
        // Set parameters population
        this->pop = pop;
        this->population_size = pop->size();

        // Set default parameters
        this->active = true;

        // Set the best solution
        this->best = *this->pop->bestSolution();

        // Update initial optimizer statistics
        this->storeOptimizerStatistics(
                "NONE",
                (int) this->uhvFitnessFunction->number_of_evaluations,
                this->uhvFitnessFunction->number_of_mo_evaluations);

        // Prepare UHV-GOMEA
        this->optimizerUHV_GOMEA->initialize_from_population(this->pop, this->pop->size());
        this->optimizerUHV_GOMEA->average_fitness_history.push_back(this->pop->average_fitness());

        // Prepare UHV-ADAM-Best
        if (this->applyUHVGradient)
        {
            // Pass population to UHV-ADAM (best)
            this->optimizerUHV_ADAM_BEST->initialize_from_population(this->pop, this->pop->size());
            this->optimizerUHV_ADAM_BEST->average_fitness_history.push_back(this->pop->average_fitness());
        }

        // Set population as initialized
        this->population_initialized = true;
    }

    /**
     * Execute a generation, where a generation means that every eligible optimizer is executed
     * @param sample_size The sample size of the population
     * @param number_of_evaluations The total number of evaluations
     */
    void UHVSwitch::generation(size_t sample_size, int & number_of_evaluations)
    {
        // Prepare Resource allocation variables
        double improvementThisGenerationUHV_GOMEA, improvementThisGenerationByUHV_ADAM_BEST = 0;
        size_t MOEvaluationsUsedThisGenerationByUHV_GOMEA, MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST = 0;

        // Reset the intra generational statistics variables
        this->resetIntraGenerationalOptimizerStatistics();


        // Apply UHV-GOMEA when threshold not met yet
        if (this->pop->best_fitness() >= uhvSwitchValue)
        {
            this->executeUHV_GOMEA(sample_size, number_of_evaluations, MOEvaluationsUsedThisGenerationByUHV_GOMEA, improvementThisGenerationUHV_GOMEA);

            // Store optimizer statistics
            this->storeOptimizerStatistics(
                    this->convertUHV_GOMEALinkageModelNameToString(this->optimizerIndexUHV_GOMEA),
                    number_of_evaluations,
                    this->uhvFitnessFunction->number_of_mo_evaluations);
        } else
        {
            // Apply UHV-ADAM-Best
            if (this->applyUHVGradient)
            {
                // Determine number of calls to execute if generation threshold is met
                size_t NumberOfUHV_ADAM_BESTCallsToExecuteThisGeneration = 10;

                // Reinitialize the optimizer
                if (!hasSwitched)
                {
                    // Reinitialize the optimizer
                    initializeUHV_ADAM_BEST();
                    this->optimizerUHV_ADAM_BEST->initialize_from_population(this->pop, this->pop->size());

                    // Determine number of MO evaluations
                    double numberOfMOEvaluationsBefore = uhvFitnessFunction->number_of_mo_evaluations;

                    // Compute initial gradient of population
                    for (size_t i = 0; i < this->pop->size(); ++i) {
                        uhvFitnessFunction->evaluate_with_gradients(optimizerUHV_ADAM_BEST->pop->sols[i]);
                    }

                    // Change back the number of MO evaluations
                    this->uhvFitnessFunction->number_of_mo_evaluations = numberOfMOEvaluationsBefore;
                }

                this->executeUHV_ADAM_BEST(
                        sample_size,
                        NumberOfUHV_ADAM_BESTCallsToExecuteThisGeneration,
                        number_of_evaluations,
                        MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST,
                        improvementThisGenerationByUHV_ADAM_BEST);

                // Store optimizer statistics
                this->storeOptimizerStatistics(
                        this->convertUHV_ADAMModelNameToString(this->uhvGradientVersion),   // Todo: Fix this later
                        number_of_evaluations,
                        this->uhvFitnessFunction->number_of_mo_evaluations);
            }


            hasSwitched = true;
        }



        // Update statistics
        this->number_of_generations++;
    }

    /**
     * Execute UHV-GOMEA and pass on important results.
     * (Internal note: Do not sort on population as this breaks GOMEA: 'pop->sort_on_fitness()')
     * @param sample_size The sample size of the population
     * @param totalNumberOfSOEvaluations The current (total) number of evaluations which will be increased with new evaluations
     * @param MOEvaluationsUsedByUHV_GOMEA The number of MO evaluations used by UHV-GOMEA
     * @param improvementByUHV_GOMEA The improvement found by UHV-GOMEA
     */
    void UHVSwitch::executeUHV_GOMEA(
            size_t sample_size,
            int & totalNumberOfSOEvaluations,
            size_t & MOEvaluationsUsedByUHV_GOMEA,
            double & improvementByUHV_GOMEA)
    {
        // Initialize variables
        size_t numberOfMOEvaluationsBefore = this->uhvFitnessFunction->number_of_mo_evaluations;
        double bestFitnessBefore = this->pop->best_fitness();

        // Prepare optimizer
        this->optimizerUHV_GOMEA->best = solution_t(this->best);

        // Do a generation and pass on the number of evaluations used
        this->optimizerUHV_GOMEA->generation(sample_size, totalNumberOfSOEvaluations);

        // Calculate and pass on the number of MO evaluations used
        size_t currentNumberOfMOEvaluations = this->uhvFitnessFunction->number_of_mo_evaluations;
        MOEvaluationsUsedByUHV_GOMEA = currentNumberOfMOEvaluations - numberOfMOEvaluationsBefore;

        // Calculate and pass on the hyper volume improvement
        double currentBestFitness = this->optimizerUHV_GOMEA->pop->best_fitness();
        improvementByUHV_GOMEA = currentBestFitness - bestFitnessBefore;
        improvementByUHV_GOMEA = improvementByUHV_GOMEA < 0 ? improvementByUHV_GOMEA : 0;


        // Update global best statistics
        if (solution_t::better_solution(this->optimizerUHV_GOMEA->best, this->best)) {
            this->best = solution_t(this->optimizerUHV_GOMEA->best);
            this->best_solution_index = 0;
        }
    }

    /**
     * Execute UHV-ADAM-BEST-SOLUTION
     * @param sample_size The sample size of the population
     * @param callsToExecuteThisGeneration The number of UHV-ADAM-Best calls to execute this generation
     * @param totalNumberOfSOEvaluations The current (total) number of SO evaluations which will be increased with new evaluations
     * @param MOEvaluationsUsedByUHV_ADAM_BEST The number of MO evaluations used by UHV-ADAM-Best
     * @param improvementByUHV_ADAM_BEST The improvement found by UHV-Adam-Best
     */
    void UHVSwitch::executeUHV_ADAM_BEST(
            size_t sample_size,
            size_t callsToExecuteThisGeneration,
            int & totalNumberOfSOEvaluations,
            size_t & MOEvaluationsUsedByUHV_ADAM_BEST,
            double & improvementByUHV_ADAM_BEST)
    {
        // Initialize variables
        size_t numberOfMOEvaluationsBefore = this->uhvFitnessFunction->number_of_mo_evaluations;
        double bestFitnessBefore = this->pop->best_fitness();

        // Prepare optimizer
        this->optimizerUHV_ADAM_BEST->best = solution_t(this->best);

        // Execute UHV-ADAM-BEST
        this->optimizerUHV_ADAM_BEST->generation(sample_size, totalNumberOfSOEvaluations, callsToExecuteThisGeneration);

        // Calculate the resources spent and the rewards
        size_t currentNumberOfMOEvaluations = this->uhvFitnessFunction->number_of_mo_evaluations;
        MOEvaluationsUsedByUHV_ADAM_BEST = currentNumberOfMOEvaluations - numberOfMOEvaluationsBefore;

        // Todo Make the average more robust?
        double currentBestFitness = this->optimizerUHV_ADAM_BEST->pop->best_fitness();
        double resultImprovementByUHV_ADAM_BEST = (currentBestFitness - bestFitnessBefore) / 1;
        improvementByUHV_ADAM_BEST = resultImprovementByUHV_ADAM_BEST < 0 ? resultImprovementByUHV_ADAM_BEST : 0;

        // Update global best statistics
        if (solution_t::better_solution(this->optimizerUHV_ADAM_BEST->best, this->best)) {
            this->best = solution_t(this->optimizerUHV_ADAM_BEST->best);
            this->best_solution_index = this->optimizerUHV_ADAM_BEST->best_solution_index;
        }
    }

    /**
     * Resets the intra generational statistics variables.
     * These statistics should be refreshed every generation.
     */
    void UHVSwitch::resetIntraGenerationalOptimizerStatistics()
    {
        this->vecOptimizerNameOfPopulation.resize(0);
        this->vecPopulationCreatedByOptimizer.resize(0);
        this->vecNumberOfSOEvaluationsByOptimizer.resize(0);
        this->vecNumberOfMOEvaluationsByOptimizer.resize(0);
    }

    /**
     * This method stores the statistics data of an optimizer.
     * @param optimizerName The name of the optimizer that was executed
     * @param currentNumberOfSOEvaluations The current number (total) of Single Objective evaluations executed
     * @param currentNumberOfMOEvaluations The current number (total) of Multi Objective evaluations executed
     * @param currentNumberOfCalls The number of calls executed in this generation
     */
    void UHVSwitch::storeOptimizerStatistics(
            std::string optimizerName,
            int currentNumberOfSOEvaluations,
            size_t currentNumberOfMOEvaluations)
    {
        // Copy current population
        population_pt copied_population = this->copyPopulation();

        // Write intra generational statistics
        this->vecOptimizerNameOfPopulation.push_back(optimizerName);
        this->vecPopulationCreatedByOptimizer.push_back(copied_population);
        this->vecNumberOfSOEvaluationsByOptimizer.push_back(currentNumberOfSOEvaluations);
        this->vecNumberOfMOEvaluationsByOptimizer.push_back(currentNumberOfMOEvaluations);
    }

    /**
     * Returns the optimizer name of UHV-GOMEA and its linkage model as a string
     * @param uhv_gomea_linkage_model_index The linkage model index of UHV-GOMEA
     * @return The optimizer name and linkage model as a string
     */
    std::string UHVSwitch::convertUHV_GOMEALinkageModelNameToString(int linkageModelUHV_GOMEA)
    {
        switch (linkageModelUHV_GOMEA) {
            case 64: return "UHV-GOMEA-Lm";
            case 66: return "UHV-GOMEA-Lt";
            case 50: return "UHV-GOMEA-Lf";
            default: return "UHV-GOMEA-Unknown";
        }
    }

    /**
     * Returns the optimizer name of UHV-ADAM and its version
     * @param uhv_adam_model_version The version of UHV-ADAM [0:best, 1:All, 2:Top3]
     * @return The optimizer name and version as a string
     */
    std::string UHVSwitch::convertUHV_ADAMModelNameToString(int versionUHV_ADAM)
    {
        switch (versionUHV_ADAM) {
            case 0: return "UHV-ADAM-Best";
            case 1: return "UHV-ADAM-All";
            case 2: return "UHV-ADAM-TOP3";
            case 3: return "UHV-ADAM-Random";
            default: return "UHV-ADAM-Unknown";
        }
    }

    /**
     * Deep copies this->pop
     * @return this->pop's copy
     */
    population_pt UHVSwitch::copyPopulation()
    {
        population_pt new_population = std::make_shared<population_t>();

        // Deep copy solutions
        new_population->sols.resize(this->pop->sols.size());
        for (size_t solution_index = 0; solution_index < this->pop->sols.size(); ++solution_index)
        {
            new_population->sols[solution_index] = std::make_shared<solution_t>(* this->pop->sols[solution_index]);
        }

        return new_population;
    }



}

