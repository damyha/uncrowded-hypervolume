/*
 * ClassicalHybrid.cpp
 * This optimizer executes the EA and gradient algorithm based on a fixed pattern.
 *
 *
 * Implementation by D. Ha
 */

#include "ClassicalHybrid.h"

namespace hillvallea{
    /**
     * Constructor
     * @param uhvFitnessFunction The UHV fitness function
     * @param localOptimizerIndexUHV_GOMEA The local optimizer index of UHV-GOMEA
     * @param localOptimizerIndexUHV_ADAM  The local optimizer index of UHV-ADAM-POPULATION
     * @param indexApplicationMethodUHV_ADAM The application method index of UHV-ADAM-POPULATION
     * @param numberOfSOParameters The number of single objective parameters
     * @param soLowerParameterBounds The single objective lower initialization bound
     * @param soUpperParameterBounds The single objective upper initialization bound
     * @param initUniVariateBandwidth The uni variate bandwidth
     * @param rng The RNG object
     */
    ClassicalHybrid::ClassicalHybrid(std::shared_ptr<UHV_t> uhvFitnessFunction,
                                     int localOptimizerIndexUHV_GOMEA,
                                     int localOptimizerIndexUHV_ADAM,
                                     int indexApplicationMethodUHV_ADAM,
                                     const size_t numberOfSOParameters,
                                     const vec_t &soLowerParameterBounds,
                                     const vec_t &soUpperParameterBounds,
                                     double initUniVariateBandwidth, rng_pt rng) : optimizer_t(
                                             numberOfSOParameters,
                                             soLowerParameterBounds,
                                             soUpperParameterBounds,
                                             initUniVariateBandwidth,
                                             uhvFitnessFunction,
                                             rng) {
        // Default variables
        this->numberOfUHV_GOMEACalls = 1;
        this->numberOfUHV_ADAM_POPULATIONCalls = 1; // Needs to be set during population initialization
        this->selection_fraction = 0.35;    // Taken from gomea.cpp

        // Copy settings
        this->uhvFitnessFunction = uhvFitnessFunction;

        this->localOptimizerIndexUHV_GOMEA = localOptimizerIndexUHV_GOMEA;
        this->localOptimizerIndexUHV_ADAM = localOptimizerIndexUHV_ADAM;
        this->indexApplicationMethodUHV_ADAM = indexApplicationMethodUHV_ADAM;

        // Initialize variables
        this->populationInitialized = false;
        this->populationSize = 0;
        this->use_boundary_repair = false;

        this->resetIntraGenerationalOptimizerStatistics();

        // Determine which optimizers to apply
        this->applyUHV_GOMEA = true;
        this->applyUHV_ADAM_POPULATION = true;

        // Initialize optimizers
        this->initializeOptimizers();
    }

    /**
     * Destructor
     */
    ClassicalHybrid::~ClassicalHybrid() { };

    /**
     * Initializes the optimizers objects
     */
    void ClassicalHybrid::initializeOptimizers()
    {
        // Create UHV-GOMEA optimizer
        if (this->applyUHV_GOMEA)
            initializeUHV_GOMEA();

        // Create UHV-ADAM on best solution optimizer
        if (this->applyUHV_ADAM_POPULATION)
            initializeUHV_ADAM_POPULATION();
    }
    /**
     * Initializes UHV-GOMEA
     */
    void ClassicalHybrid::initializeUHV_GOMEA()
    {
        // Create UHV-GOMEA optimizer
        this->optimizerUHV_GOMEA = std::make_shared<gomea_t>(
                this->number_of_parameters,
                this->lower_param_bounds,
                this->upper_param_bounds,
                this->init_univariate_bandwidth,
                this->localOptimizerIndexUHV_GOMEA,
                this->fitness_function,
                this->rng);
    }

    /**
     * Initializes UHV-ADAM-BEST
     */
    void ClassicalHybrid::initializeUHV_ADAM_POPULATION()
    {
        this->optimizerUHV_ADAM_POPULATION = std::make_shared<adam_on_population_t>(
                this->number_of_parameters,
                this->lower_param_bounds,
                this->upper_param_bounds,
                this->init_univariate_bandwidth,
                this->localOptimizerIndexUHV_ADAM,
                this->indexApplicationMethodUHV_ADAM,
                this->fitness_function,
                this->rng);
    }

    /**
     * The name of the algorithm
     * @return The name of the algorithm
     */
    std::string ClassicalHybrid::name() const { return "UHV-CLASSICAL_HYBRID"; }

    /**
     * A setter of the initial population.
     * @param pop The initial population (that's sorted on fitness)
     * @param target_popsize The size the population can be shaped to (usually set equal to initial population size)
     */
    void ClassicalHybrid::initialize_from_population(population_pt pop, size_t target_popsize)
    {
        // Set parameters population
        this->pop = pop;
        this->populationSize = pop->size();
        this->numberOfUHV_ADAM_POPULATIONCalls = populationSize;

        // Retrieve parameters
        this->active = true;
        this->best = *this->pop->bestSolution();

        // Update initial optimizer statistics
        this->storeOptimizerStatistics(
                "NONE",
                (int) this->populationSize,
                (size_t) this->uhvFitnessFunction->number_of_mo_evaluations,
                0);

        // Initialize population of UHV-GOMEA
        if (this->applyUHV_GOMEA)
        {
            // Pass population to UHV-GOMEA
            this->optimizerUHV_GOMEA->initialize_from_population(this->pop, this->populationSize);
            this->optimizerUHV_GOMEA->average_fitness_history.push_back(this->pop->average_fitness());
        }

        // Initialize population of UHV-ADAM-Best
        if (this->applyUHV_ADAM_POPULATION)
        {
            // Pass population to UHV-ADAM (best)
            this->optimizerUHV_ADAM_POPULATION->initialize_from_population(this->pop, this->populationSize);
            this->optimizerUHV_ADAM_POPULATION->average_fitness_history.push_back(this->pop->average_fitness());
        }

        // Set population as initialized
        this->populationInitialized = true;
    }

    /**
     * Execute a generation, where a generation means that every eligible optimizer is executed
     * @param sample_size The sample size of the population
     * @param number_of_evaluations The total number of evaluations
     */
    void ClassicalHybrid::generation(size_t sample_size, int & number_of_evaluations)
    {
        // Reset the intra generational statistics variables
        this->resetIntraGenerationalOptimizerStatistics();

        // Apply UHV-GOMEA
        if (this->applyUHV_GOMEA)
        {
            // Execute UHV-GOMEA
            this->executeUHV_GOMEA(sample_size, number_of_evaluations);

            // Store optimizer statistics
            this->storeOptimizerStatistics(optimizerUHV_GOMEA->convertUHV_GOMEALinkageModelNameToString(),
                                           number_of_evaluations,
                                           uhvFitnessFunction->number_of_mo_evaluations,
                                           numberOfUHV_GOMEACalls);
        }

        // Apply UHV-ADAM-POPULATION
        if (this->applyUHV_ADAM_POPULATION)
        {
            // Execute UHV-ADAM-POPULATION
            this->executeUHV_ADAM_BEST(sample_size,
                                       numberOfUHV_ADAM_POPULATIONCalls,
                                       number_of_evaluations);

            // Store optimizer statistics
            this->storeOptimizerStatistics(optimizerUHV_ADAM_POPULATION->convertUHV_ADAMModelNameToString(),
                                           number_of_evaluations,
                                           this->uhvFitnessFunction->number_of_mo_evaluations,
                                           numberOfUHV_ADAM_POPULATIONCalls);
        }

        // Update statistics
        this->number_of_generations++;

    }

    /**
     * Execute UHV-GOMEA and pass on important results.
     * (Internal note: Do not sort on population as this breaks GOMEA: 'pop->sort_on_fitness()')
     * @param sample_size The sample size of the population
     * @param totalNumberOfSOEvaluations The current (total) number of evaluations which will be increased with new evaluations
     */
    void ClassicalHybrid::executeUHV_GOMEA(size_t sample_size,
                                           int & totalNumberOfSOEvaluations)
    {
        // Prepare optimizer
        this->optimizerUHV_GOMEA->best = this->best;

        // Execute UHV-GOMEA
        for(size_t generationIndex = 0; generationIndex < numberOfUHV_GOMEACalls; ++generationIndex) {
            this->optimizerUHV_GOMEA->generation(sample_size, totalNumberOfSOEvaluations);
        }

        // Update global best statistics
        solution_pt currentBestSolution = pop->bestSolution();
        if (solution_t::better_solution(*currentBestSolution, best)) {
            this->best = *currentBestSolution;
        }

    }

    /**
     * Execute UHV-ADAM-BEST-SOLUTION
     * @param sample_size The sample size of the population
     * @param callsToExecuteThisGeneration The number of UHV-ADAM-Best calls to execute this generation
     * @param totalNumberOfSOEvaluations The current (total) number of SO evaluations which will be increased with new evaluations
     */
    void ClassicalHybrid::executeUHV_ADAM_BEST(size_t sample_size,
                              size_t callsToExecuteThisGeneration,
                              int & totalNumberOfSOEvaluations)
    {
        // Reinitialize UHV-ADAM-BEST
        this->initializeUHV_ADAM_POPULATION();
        this->optimizerUHV_ADAM_POPULATION->initialize_from_population(this->pop, this->populationSize);
        this->optimizerUHV_ADAM_POPULATION->best = *this->pop->bestSolution();

        // Keep the current number of MO evaluations
        size_t numberOfMOEvaluationsBefore = (size_t) this->uhvFitnessFunction->number_of_mo_evaluations;

        // Prepare population by computing gradients and resetting history parameters of relevant solutions
        std::vector<size_t> solutionIndicesToInspect =
                optimizerUHV_ADAM_POPULATION->determineSolutionIndicesToApplyAlgorithmOn(callsToExecuteThisGeneration);
        for (auto const &solIndex : solutionIndicesToInspect) {
            uhvFitnessFunction->evaluate_with_gradients(pop->sols[solIndex]);

            pop->sols[solIndex]->adam_mt.assign(pop->sols[solIndex]->adam_mt.size(), 0);
            pop->sols[solIndex]->adam_vt.assign(pop->sols[solIndex]->adam_vt.size(), 0);
        }

        // Determine if gradient came free or should be counted
        if(!this->uhvFitnessFunction->use_finite_differences)
            uhvFitnessFunction->number_of_mo_evaluations = (double) numberOfMOEvaluationsBefore;

        // Execute UHV-ADAM-BEST
        for (auto const &solutionIndex : solutionIndicesToInspect) {
            // Execute on a single solution
            optimizerUHV_ADAM_POPULATION->executeGenerationOnSpecificSolution(solutionIndex);

            // Update statistics
            totalNumberOfSOEvaluations++;
        }

        // Update global best statistics
        solution_pt currentBestSolution = pop->bestSolution();
        if (solution_t::better_solution(*currentBestSolution, best)) {
            this->best = *currentBestSolution;
        }
    }

    /**
     * This method stores the statistics data of an optimizer.
     * @param optimizerName The name of the optimizer that was executed
     * @param currentNumberOfSOEvaluations The current number (total) of Single Objective evaluations executed
     * @param currentNumberOfMOEvaluations The current number (total) of Multi Objective evaluations executed
     * @param currentNumberOfCalls The number of calls executed in this generation
     */
    void ClassicalHybrid::storeOptimizerStatistics(
            std::string optimizerName,
            int currentNumberOfSOEvaluations,
            size_t currentNumberOfMOEvaluations,
            size_t currentNumberOfCalls) {
        // Copy current population
        population_pt copied_population = this->pop->deepCopyPopulation();

        // Write intra generational statistics
        this->vecOptimizerNameOfPopulation.push_back(optimizerName);
        this->vecPopulationCreatedByOptimizer.push_back(copied_population);
        this->vecBestSolutionsAfterOptimizer.push_back(best);
        this->vecNumberOfSOEvaluationsByOptimizer.push_back(currentNumberOfSOEvaluations);
        this->vecNumberOfMOEvaluationsByOptimizer.push_back(currentNumberOfMOEvaluations);
        this->vecNumberOfCallsByOptimizer.push_back(currentNumberOfCalls);
    }


    /**
     * Resets the intra generational statistics variables.
     * These statistics should be refreshed every generation.
     */
    void ClassicalHybrid::resetIntraGenerationalOptimizerStatistics()
    {
        this->vecOptimizerNameOfPopulation.resize(0);
        this->vecPopulationCreatedByOptimizer.resize(0);
        this->vecBestSolutionsAfterOptimizer.resize(0);
        this->vecNumberOfSOEvaluationsByOptimizer.resize(0);
        this->vecNumberOfMOEvaluationsByOptimizer.resize(0);
        this->vecNumberOfCallsByOptimizer.resize(0);
    }
}