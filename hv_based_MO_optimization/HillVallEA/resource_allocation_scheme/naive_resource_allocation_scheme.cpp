/*
 * naive_resource_allocation_scheme.cpp
 * This optimizer optimizes a problem via a resource allocation scheme inspired by:
 *  "Combining Gradient Techniques for Numerical Multi-Objective Evolutionary Optimization" by
 *  P A.N. Bosman and E D. de Jong.
 *
 * Some warnings:
 * - Do not sort UHV-GOMEA's population
 * - The population pointer is shared between the algorithms
 *
 * Implementation by D. Ha
 */

#include "naive_resource_allocation_scheme.h"

#include "../fitness.h"
#include "../adam.hpp"

// Todo:
// Currently experimenting with separating application of algorithms
// In the curren state I mimic the behavior by applying the algorithms on separate populations and to merge those back
// together.
// One of the issues is that UHV-GOMEA is doing too many evaluations
// Another issue is that I need to merge the populations. Currently I pick the best solution per solution index.

namespace hillvallea {
    /**
     * Constructor
     * @param fitness_function The fitness function
     * @param localOptimizerIndexUHV_GOMEA The local optimizer index of UHV-GOMEA
     * @param localOptimizerIndexUHV_ADAM The local optimizer index of UHV-ADAM
     * @param indexApplicationMethodUHV_ADAM The application method of applying UHV-ADAM on a population
     * @param number_of_parameters The number of (SO) parameters
     * @param lower_param_bounds The minimum value a solution can take per parameter
     * @param upper_param_bounds The maximum value a solution can take per parameter
     * @param init_univariate_bandwidth
     * @param rng The random number generator object
     */
    naive_resource_allocation_scheme_t::naive_resource_allocation_scheme_t(
            std::shared_ptr<UHV_t> uhvFitnessFunction,
            int localOptimizerIndexUHV_GOMEA,
            int localOptimizerIndexUHV_ADAM,
            int indexApplicationMethodUHV_ADAM,
            int improvementMetricIndex,
            const size_t numberOfSOParameters,
            const vec_t &soLowerParameterBounds,
            const vec_t &soUpperParameterBounds,
            double initUniVariateBandwidth,
            double memoryDecayFactor,
            double initialStepSizeFactor,
            double stepSizeDecayFactor,
            rng_pt rng) : optimizer_t(
            numberOfSOParameters,
            soLowerParameterBounds,
            soUpperParameterBounds,
            initUniVariateBandwidth,
            uhvFitnessFunction,
            rng) {
        // Default variables
        this->numberOfUHV_GOMEACalls = 1;
        this->minimumNumberOfUHV_ADAM_BestCalls = 10;
        this->selection_fraction = 0.35;    // Taken from gomea.cpp

        // Copy settings
        this->uhvFitnessFunction = uhvFitnessFunction;

        this->localOptimizerIndexUHV_GOMEA = localOptimizerIndexUHV_GOMEA;
        this->localOptimizerIndexUHV_ADAM = localOptimizerIndexUHV_ADAM;
        this->indexApplicationMethodUHV_ADAM = indexApplicationMethodUHV_ADAM;

        this->improvementMetricIndex = improvementMetricIndex;

        // Initialize variables
        this->populationInitialized = false;
        this->populationSize = 0;
        this->use_boundary_repair = false;

        // Set default values
        this->memoryDecayFactor = memoryDecayFactor;
        this->initialStepSizeFactor = initialStepSizeFactor;
        this->stepSizeDecayFactor = stepSizeDecayFactor;

        // Determine the improvement metric variables
        this->determineImprovementMetricVariables();

        // Initialize the generational MO solutions archive
        if (useGenerationalArchive)
            generationalMOSolutionsArchive = GenerationalArchive();

        // Reset statistical data
        this->resetIntraGenerationalOptimizerStatistics();

        // Determine which optimizers to apply
        this->applyUHV_GOMEA = true;
        this->applyUHV_ADAM_BEST = true;

        this->doHybridAnalysis = false;
        this->writeEveryStep = false;
        this->generationToSwitch = 20;
        this->forceUHV_GOMEA = false;
        this->forceUHV_ADAM = true;

        this->separateExecution = false;

        // Initialize optimizers
        this->initializeOptimizers();

        // Reset resource allocation scheme variables
        this->resetResourceAllocationSchemeStatistics();
        this->callsToExecuteNextGenerationUHV_ADAM_BEST = this->minimumNumberOfUHV_ADAM_BestCalls;
    }

    /**
     * Destructor
     */
    naive_resource_allocation_scheme_t::~naive_resource_allocation_scheme_t() {};


    /**
     * Initializes the optimizers objects
     */
    void naive_resource_allocation_scheme_t::initializeOptimizers() {
        // Create UHV-GOMEA optimizer
        if (this->applyUHV_GOMEA) {
            initializeUHV_GOMEA();
        }

        // Create UHV-ADAM on best solution optimizer
        if (this->applyUHV_ADAM_BEST) {
            initializeUHV_ADAM_BEST();
        }
    }

    /**
     * Initializes UHV-GOMEA
     */
    void naive_resource_allocation_scheme_t::initializeUHV_GOMEA() {
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
    void naive_resource_allocation_scheme_t::initializeUHV_ADAM_BEST() {
        this->optimizerUHV_ADAM_BEST = std::make_shared<adam_on_population_t>(
                this->number_of_parameters,
                this->lower_param_bounds,
                this->upper_param_bounds,
                this->init_univariate_bandwidth,
                this->localOptimizerIndexUHV_ADAM,
                this->indexApplicationMethodUHV_ADAM,
                this->fitness_function,
                this->rng);

        this->optimizerUHV_ADAM_BEST->setGammaWeight(initialStepSizeFactor);
        this->optimizerUHV_ADAM_BEST->setStepSizeDecayFactor(stepSizeDecayFactor);

    }

    /**
     * Determine when to calculate the improvements.
     * If true, the improvements are calculated pre and post executing all calls
     * If false, the improvements are calculated after each call
     */
    void naive_resource_allocation_scheme_t::determineImprovementMetricVariables() {
        if (improvementMetricIndex == 0) {
            // Best fitness between populations
            this->calculatePreAndPostImprovement = writeEveryStep;  // Depends on if every step needs to be written
            this->useGenerationalArchive = false;
        } else if (improvementMetricIndex == 1) {
            // Average fitness between populations
            this->calculatePreAndPostImprovement = writeEveryStep;  // Depends on if every step needs to be written
            this->useGenerationalArchive = false;
        } else if (improvementMetricIndex == 2) {
            // Number of UHV-Improvements w.r.t. original solution
            this->calculatePreAndPostImprovement = false;
            this->useGenerationalArchive = false;
        } else if (improvementMetricIndex == 3) {
            // Number of UHV-Improvements w.r.t. previous population
            this->calculatePreAndPostImprovement = false;
            this->useGenerationalArchive = false;
        } else if (improvementMetricIndex == 4) {
            // Number of MO-solutions that are not dominated w.r.t new solution
            this->calculatePreAndPostImprovement = false;
            this->useGenerationalArchive = false;
        } else if (improvementMetricIndex == 5) {
            // Number of MO-solutions that are not dominated w.r.t. the old population
            this->calculatePreAndPostImprovement = false;
            this->useGenerationalArchive = true;
        } else {
            std::runtime_error("Unknown improvement metric index detected.");
        }
    }


    /**
     * The name of the algorithm
     * @return The name of the algorithm
     */
    std::string naive_resource_allocation_scheme_t::name() const { return "UHV-HYBRID_PROTOTYPE"; }


    /**
     * A setter of the initial population.
     * Todo: This method currently does not support 'target_popsize'
     * @param pop The initial population (that's sorted on fitness)
     * @param target_popsize The size the population can be shaped to (usually set equal to initial population size)
     */
    void naive_resource_allocation_scheme_t::initialize_from_population(population_pt pop, size_t target_popsize) {
        // Set parameters population
        this->pop = pop;
        this->populationSize = pop->size();

        // Initialize parameters
        this->active = true;
        this->best = *this->pop->bestSolution();

        // Update initial optimizer statistics
        this->storeOptimizerStatistics(
                "NONE",
                (int) this->pop->size(),
                (size_t) this->uhvFitnessFunction->number_of_mo_evaluations,
                0,
                0);

        // Initialize population of UHV-GOMEA
        if (this->applyUHV_GOMEA) {
            // Pass population to UHV-GOMEA
            this->optimizerUHV_GOMEA->initialize_from_population(this->pop, this->pop->size());
            this->optimizerUHV_GOMEA->average_fitness_history.push_back(this->pop->average_fitness());
        }

        // Initialize population of UHV-ADAM-Best
        if (this->applyUHV_ADAM_BEST) {
            // Pass population to UHV-ADAM (best)
            this->optimizerUHV_ADAM_BEST->initialize_from_population(this->pop, this->pop->size());
            this->optimizerUHV_ADAM_BEST->average_fitness_history.push_back(this->pop->average_fitness());
        }

        // Set population as initialized
        this->populationInitialized = true;
    }

    /**
     * Execute a generation, where a generation means that every eligible optimizer is executed
     * @param sample_size The sample size of the population
     * @param number_of_evaluations The total number of evaluations
     */
    void naive_resource_allocation_scheme_t::generation(size_t sample_size, int &number_of_evaluations) {
        // Prepare Resource allocation variables
        double improvementThisGenerationUHV_GOMEA = 0, improvementThisGenerationByUHV_ADAM_BEST = 0;
        size_t MOEvaluationsUsedThisGenerationByUHV_GOMEA = 0, MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST = 0;
        size_t numberOfCallsThisGenerationByUHV_GOMEA = 0, numberOfCallsThisGenerationByUHV_ADAM_BEST = 0;

        // Reset the intra generational statistics variables
        this->resetIntraGenerationalOptimizerStatistics();

        // Force execute optimizer
        bool useFixedSchedule = false;
        if (doHybridAnalysis) {
            if (number_of_generations >= generationToSwitch) {
                useFixedSchedule = true;
            }
        }

        // Reset the generational archive
        if (useGenerationalArchive) {
            generationalMOSolutionsArchive.initializeGenerationalMOSolutionsArchive(pop);
//            printf("Size archive start: %ld\n", generationalMOSolutionsArchive.getArchiveMoSolutions().size());
        }

        // Do separate execution of algorithms
        population_pt populationUHV_GOMEA, populationUHV_ADAM_BEST;
        if (separateExecution) {
            populationUHV_GOMEA = this->pop->deepCopyPopulation();
            populationUHV_ADAM_BEST = this->pop->deepCopyPopulation();
        }

//        printf("Current Generation: %d\n", number_of_generations);

        // Apply UHV-GOMEA
        if (this->applyUHV_GOMEA) {
            // Set copy of population as current population
            if (separateExecution) {
                this->pop = populationUHV_GOMEA;
                this->optimizerUHV_GOMEA->pop = populationUHV_GOMEA;
            }

            // Execute UHV-GOMEA
//            printf("Executing UHV-GOMEA\n");
//            printf("Before applying UHV-GOMEA: %.4e (best), %.4e (avg)  \n", pop->best_fitness(), pop->average_fitness());
            if(!useFixedSchedule || forceUHV_GOMEA) {
                this->executeUHV_GOMEA(sample_size,
                                       number_of_evaluations,
                                       MOEvaluationsUsedThisGenerationByUHV_GOMEA,
                                       improvementThisGenerationUHV_GOMEA);
                numberOfCallsThisGenerationByUHV_GOMEA = numberOfUHV_GOMEACalls;
            }


            // Store Resource allocation variables of UHV-GOMEA
            this->storeResourceAllocationStatisticsUHV_GOMEA(MOEvaluationsUsedThisGenerationByUHV_GOMEA,
                                                             improvementThisGenerationUHV_GOMEA);

            // Store optimizer statistics
            this->storeOptimizerStatistics(
                    optimizerUHV_GOMEA->convertUHV_GOMEALinkageModelNameToString(),
                    number_of_evaluations,
                    uhvFitnessFunction->number_of_mo_evaluations,
                    improvementThisGenerationUHV_GOMEA,
                    numberOfCallsThisGenerationByUHV_GOMEA);

//            printf("After applying UHV-GOMEA: %.4e (best), %.4e (avg)  \n", pop->best_fitness(), pop->average_fitness());
        }

        // Apply UHV-ADAM-Best
        if (this->applyUHV_ADAM_BEST) {
            // Set copy of population as current population
            if (separateExecution) {
                this->pop = populationUHV_ADAM_BEST;
                this->optimizerUHV_ADAM_BEST->pop = populationUHV_ADAM_BEST;
            }

//            printf("Executing UHV-ADAM\n");
//            printf("Before applying UHV-ADAM: %.4e (best), %.4e (avg)  \n", pop->best_fitness(), pop->average_fitness());

            if(!useFixedSchedule || forceUHV_ADAM) {
                // Determine number of calls to execute if generation threshold is met
                size_t numberOfUHV_ADAM_BESTCallsToExecuteThisGeneration;
                if (useFixedSchedule) {
                    numberOfUHV_ADAM_BESTCallsToExecuteThisGeneration = populationSize;
                } else {
                    numberOfUHV_ADAM_BESTCallsToExecuteThisGeneration =
                            (size_t) std::max(this->callsToExecuteNextGenerationUHV_ADAM_BEST,
                                              this->minimumNumberOfUHV_ADAM_BestCalls);
                }

//                printf("Executing %ld calls of UHV-ADAM-Best\n", numberOfUHV_ADAM_BESTCallsToExecuteThisGeneration);
                this->executeUHV_ADAM_BEST(
                        sample_size,
                        numberOfUHV_ADAM_BESTCallsToExecuteThisGeneration,
                        number_of_evaluations,
                        MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST,
                        improvementThisGenerationByUHV_ADAM_BEST,
                        numberOfCallsThisGenerationByUHV_ADAM_BEST);

            }

            // Store Resource allocation statistics
            this->storeResourceAllocationStatisticsUHV_ADAM_BEST(MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST,
                                                                 improvementThisGenerationByUHV_ADAM_BEST);

            // Store optimizer statistics
            if (!writeEveryStep) {
                this->storeOptimizerStatistics(
                        optimizerUHV_ADAM_BEST->convertUHV_ADAMModelNameToString(),
                        number_of_evaluations,
                        this->uhvFitnessFunction->number_of_mo_evaluations,
                        improvementThisGenerationByUHV_ADAM_BEST,
                        numberOfCallsThisGenerationByUHV_ADAM_BEST);
            }

//            printf("After applying UHV-ADAM: %.4e (best), %.4e (avg)  \n", pop->best_fitness(), pop->average_fitness());
        }

        if (separateExecution) {
            this->mergePopulations(populationUHV_GOMEA, populationUHV_ADAM_BEST);
        }


        // Calculate resources required in next generation
        this->calculateResourcesNextGeneration(MOEvaluationsUsedThisGenerationByUHV_GOMEA,
                                               MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST,
                                               improvementThisGenerationUHV_GOMEA,
                                               improvementThisGenerationByUHV_ADAM_BEST,
                                               numberOfCallsThisGenerationByUHV_ADAM_BEST);


        // Update statistics
        this->number_of_generations++;

//        printf("Size archive start: %ld\n", generationalMOSolutionsArchive.getArchiveMoSolutions().size());
    }

    /**
     * Merges the populations when the algorithms are executed separately.
     * Takes the best solutions per population index
     * @param populationUHV_GOMEA The population obtained by executing UHV-GOMEA
     * @param populationUHV_ADAM_BEST The population obtained by executing UHV-ADAM-POP
     */
    void naive_resource_allocation_scheme_t::mergePopulations(population_pt populationUHV_GOMEA,
                                                              population_pt populationUHV_ADAM_BEST) {
        for (size_t solutionIndex = 0; solutionIndex < populationSize; ++solutionIndex) {
            // Retrieve the solutions
            solution_pt solutionUHV_GOMEA = populationUHV_GOMEA->sols[solutionIndex];
            solution_pt solutionUHV_ADAM = populationUHV_ADAM_BEST->sols[solutionIndex];

            if (solutionUHV_ADAM->f < solutionUHV_GOMEA->f) {
                populationUHV_GOMEA->sols[solutionIndex] = solutionUHV_ADAM;
            }

        }

        this->pop = populationUHV_GOMEA;

    }

    /**
     * Execute UHV-GOMEA and pass on important results.
     * (Internal note: Do not sort on population as this breaks GOMEA: 'pop->sort_on_fitness()')
     * @param sample_size The sample size of the population
     * @param totalNumberOfSOEvaluations The current (total) number of evaluations which will be increased with new evaluations
     * @param MOEvaluationsUsedByUHV_GOMEA The number of MO evaluations used by UHV-GOMEA
     * @param improvementByUHV_GOMEA The improvement found by UHV-GOMEA
     */
    void naive_resource_allocation_scheme_t::executeUHV_GOMEA(
            size_t sample_size,
            int &totalNumberOfSOEvaluations,
            size_t &MOEvaluationsUsedByUHV_GOMEA,
            double &improvementByUHV_GOMEA) {
        // Initialize variables
        size_t numberOfMOEvaluationsBefore = this->uhvFitnessFunction->number_of_mo_evaluations;
        double improvementCount = 0;

        // Prepare optimizer
        this->optimizerUHV_GOMEA->best = solution_t(best);

        // Do improvement metric specific executions of generations + logging of improvements
        if (calculatePreAndPostImprovement) {
            // Track the current population
            population_pt copyCurrentPopulation = this->pop->deepCopyPopulation();

            // Do a generation and pass on the number of evaluations used
            for (size_t generationIndex = 0; generationIndex < numberOfUHV_GOMEACalls; ++generationIndex) {
                // Execute a generation
                this->optimizerUHV_GOMEA->generation(sample_size, totalNumberOfSOEvaluations);

                // Update global best statistics
                solution_pt currentBestSolution = pop->bestSolution();
                if (solution_t::better_solution(*currentBestSolution, best)) {
                    this->best = solution_t(*currentBestSolution);
                }
            }

            // Determine the improvement obtained
            improvementCount += calculateImprovements(copyCurrentPopulation,
                                                      optimizerUHV_GOMEA->pop,
                                                      -1);
        } else {
            // Do a generation and pass on the number of evaluations used
            for (size_t generationIndex = 0; generationIndex < numberOfUHV_GOMEACalls; ++generationIndex) {
                // Track the populations created
                population_pt copyCurrentPopulation = this->pop->deepCopyPopulation();

                // Execute a generation
                this->optimizerUHV_GOMEA->generation(sample_size, totalNumberOfSOEvaluations);

                // Update global best statistics
                solution_pt currentBestSolution = pop->bestSolution();
                if (solution_t::better_solution(*currentBestSolution, best)) {
                    this->best = solution_t(*currentBestSolution);
                }

                // Determine the improvement obtained
                improvementCount += calculateImprovements(copyCurrentPopulation,
                                                          optimizerUHV_GOMEA->pop,
                                                          -1);
            }
        }

        // Calculate and pass on the number of MO evaluations used and improvements
        size_t currentNumberOfMOEvaluations = this->uhvFitnessFunction->number_of_mo_evaluations;

        MOEvaluationsUsedByUHV_GOMEA = currentNumberOfMOEvaluations - numberOfMOEvaluationsBefore;
        improvementByUHV_GOMEA = improvementCount;
    }

    /**
     * Execute UHV-ADAM-BEST-SOLUTION
     * @param sample_size The sample size of the population
     * @param callsToExecuteThisGeneration The number of UHV-ADAM-Best calls to execute this generation
     * @param totalNumberOfSOEvaluations The current (total) number of SO evaluations which will be increased with new evaluations
     * @param MOEvaluationsUsedByUHV_ADAM_BEST The number of MO evaluations used by UHV-ADAM-Best
     * @param improvementByUHV_ADAM_BEST The improvement found by UHV-Adam-Best
     * @param numberOfCallsThisGenerationByUHV_ADAM_BEST The number of UHV-Adam-Best calls executed in this generation
     */
    void naive_resource_allocation_scheme_t::executeUHV_ADAM_BEST(
            size_t sample_size,
            size_t callsToExecuteThisGeneration,
            int &totalNumberOfSOEvaluations,
            size_t &MOEvaluationsUsedByUHV_ADAM_BEST,
            double &improvementByUHV_ADAM_BEST,
            size_t &numberOfCallsThisGenerationByUHV_ADAM_BEST) {
        // Initialize variables
        size_t resultMOEvaluationsUsedByUHV_ADAM_BEST = 0;
        double resultImprovementByUHV_ADAM_BEST = 0;
        size_t resultNumberOfCallsThisGenerationByUHV_ADAM_BEST = 0;

        // Check if generation threshold has been met
        if (this->number_of_generations >= this->generationThresholdUHV_ADAM_BEST) {
            // Reinitialize UHV-ADAM-BEST (Resets variable: number_of_generations)
            initializeUHV_ADAM_BEST();
            this->optimizerUHV_ADAM_BEST->initialize_from_population(this->pop, this->populationSize);
            this->optimizerUHV_ADAM_BEST->best = solution_t(best);

            // Initialize number of MO evaluations and improvements found
            size_t numberOfMOEvaluationsBefore = this->uhvFitnessFunction->number_of_mo_evaluations;
            double improvementCount = 0;

            // Prepare population by computing gradients and resetting history parameters of relevant solutions
            std::vector<size_t> solutionIndicesToInspect =
                    optimizerUHV_ADAM_BEST->determineSolutionIndicesToApplyAlgorithmOn(callsToExecuteThisGeneration);

            // Determine unique solution indices. This is used to reset the gradient algorithms
            std::vector<size_t> uniqueSolutionIndices = solutionIndicesToInspect;
            std::sort(uniqueSolutionIndices.begin(), uniqueSolutionIndices.end());
            uniqueSolutionIndices.erase(std::unique(uniqueSolutionIndices.begin(), uniqueSolutionIndices.end()), uniqueSolutionIndices.end());

            // Calculate the gradients and reset the gradient algorithm parameters
            // UHV-GOMEA does not calculate any gradients which are needed by UHV-ADAM. Letting UHV-GOMEA take care of
            // the gradients is expensive. It is more efficient to calculate the gradients of those solutions that need
            // it, especially when finite difference (FD) approximations are used. FD "increases [the MO-evaluations]
            // from 'p' to '(1+n)*p'". See paper on UHV-ADAM for quote. This means that you only need to add 'n*p'
            // evaluations for every soluton set that needs gradients.
            for (auto const &solIndex : uniqueSolutionIndices) {
                // Determine the MO-evaluations before calculating the gradient
                size_t temp_MO_Evaluations = uhvFitnessFunction->number_of_mo_evaluations;

                // Calculate gradients
                uhvFitnessFunction->evaluate_with_gradients(pop->sols[solIndex]);

                // Determine the cost of the gradient
                if (this->uhvFitnessFunction->use_finite_differences) {
                    // Add n*p MO-evaluations
                    uhvFitnessFunction->number_of_mo_evaluations = temp_MO_Evaluations;
                    uhvFitnessFunction->number_of_mo_evaluations += number_of_parameters;
                } else {
                    // Free gradients
                    uhvFitnessFunction->number_of_mo_evaluations = temp_MO_Evaluations;
                }

                // Reset parameters (Gamma is taken care of during reinitialization of the gradient algorithm)
                pop->sols[solIndex]->adam_mt.assign(pop->sols[solIndex]->adam_mt.size(), 0);
                pop->sols[solIndex]->adam_vt.assign(pop->sols[solIndex]->adam_vt.size(), 0);
            }



            // Do improvement metric specific executions of generations + logging of improvements
            if (calculatePreAndPostImprovement) {
                // Copy the pre execution population
                population_pt copyCurrentPopulation = this->pop->deepCopyPopulation();

                // Execute UHV-ADAM-BEST
                for (auto const &solutionIndex : solutionIndicesToInspect) {
                    // Execute on a single solution
                    this->optimizerUHV_ADAM_BEST->executeGenerationOnSpecificSolution(solutionIndex);

                    // Update statistics
                    totalNumberOfSOEvaluations++;
                    resultNumberOfCallsThisGenerationByUHV_ADAM_BEST++;

                    // Update global best statistics
                    solution_pt currentSolution = pop->sols[solutionIndex];
                    if (solution_t::better_solution(*currentSolution, best)) {
                        this->best = solution_t(*currentSolution);
                    }
                }

                // Update the number of improvements found
                improvementCount += calculateImprovements(copyCurrentPopulation,
                                                          optimizerUHV_ADAM_BEST->pop,
                                                          -1);
            } else {
                // Execute UHV-ADAM-BEST
                for (auto const &solutionIndex : solutionIndicesToInspect) {
                    // Track the populations created
                    population_pt copyCurrentPopulation = this->pop->deepCopyPopulation();

                    // Execute on a single solution
                    this->optimizerUHV_ADAM_BEST->executeGenerationOnSpecificSolution(solutionIndex);

                    // Update the number of improvements found
                    improvementCount += calculateImprovements(copyCurrentPopulation,
                                                              optimizerUHV_ADAM_BEST->pop,
                                                              solutionIndex);

                    // Update statistics
                    totalNumberOfSOEvaluations++;
                    resultNumberOfCallsThisGenerationByUHV_ADAM_BEST++;

                    // Update global best statistics
                    solution_pt currentSolution = pop->sols[solutionIndex];
                    if (solution_t::better_solution(*currentSolution, best)) {
                        this->best = solution_t(*currentSolution);
                    }

                    // Write statistics
                    if (writeEveryStep) {
                        // Write statistics
                        this->storeOptimizerStatistics(
                                optimizerUHV_ADAM_BEST->convertUHV_ADAMModelNameToString(),
                                totalNumberOfSOEvaluations,
                                uhvFitnessFunction->number_of_mo_evaluations,
                                improvementCount,
                                1); // alternative: resultNumberOfCallsThisGenerationByUHV_ADAM_BEST
                    }
                }
            }


            // Calculate the resources spent and improvements found
            size_t currentNumberOfMOEvaluations = this->uhvFitnessFunction->number_of_mo_evaluations;

            resultMOEvaluationsUsedByUHV_ADAM_BEST = currentNumberOfMOEvaluations - numberOfMOEvaluationsBefore;
            resultImprovementByUHV_ADAM_BEST = improvementCount;
        }

        // Assign the resource allocation data
        MOEvaluationsUsedByUHV_ADAM_BEST = resultMOEvaluationsUsedByUHV_ADAM_BEST;
        improvementByUHV_ADAM_BEST = resultImprovementByUHV_ADAM_BEST;
        numberOfCallsThisGenerationByUHV_ADAM_BEST = resultNumberOfCallsThisGenerationByUHV_ADAM_BEST;
    }

    /**
     * Calculate the improvements obtained by UHV-GOMEA based on the improvement metric index
     * @param populationBefore The population before apply UHV-GOMEA
     * @param populationAfter The population after applying UHV-GOMEA
     * @param indexSolutionChanged index of specific solution that was changed otherwise "-1"
     * @return The improvements obtained by UHV-GOMEA based on the improvement metric index
     */
    double naive_resource_allocation_scheme_t::calculateImprovements(
            population_pt populationBefore,
            population_pt populationAfter,
            int indexSolutionChanged) {
        // Prepare variable
        double result = 0;

        // Determine which solution indices of the population must be inspected
        std::vector<size_t> solutionIndicesToInspect;
        if (improvementMetricIndex != 0 && improvementMetricIndex != 1) {
            if (indexSolutionChanged >= 0) {
                solutionIndicesToInspect.resize(1);
                solutionIndicesToInspect[0] = indexSolutionChanged;
            } else {
                solutionIndicesToInspect.resize(populationSize);
                std::iota(std::begin(solutionIndicesToInspect), std::end(solutionIndicesToInspect), 0);
            }
        }

        // Calculate the improvement
        if (this->improvementMetricIndex == 0) {
            // Difference best solution's fitness
            result = populationAfter->best_fitness() - populationBefore->best_fitness();
        } else if (this->improvementMetricIndex == 1) {
            // Difference average fitness of population
            result = populationAfter->average_fitness() - populationBefore->average_fitness();
        } else if (this->improvementMetricIndex == 2) {
            // Number of times the UHV improved with respect to the original solution
            for (const auto solutionIndex : solutionIndicesToInspect) {
                solution_pt solutionBefore = populationBefore->sols[solutionIndex];
                solution_pt solutionAfter = populationAfter->sols[solutionIndex];

                if (solutionAfter->f < solutionBefore->f) --result;
            }
        } else if (this->improvementMetricIndex == 3) {
            // Number of times the UHV improved with respect to the population
            double previousBestFitness = populationBefore->best_fitness();

            for (const auto solutionIndex : solutionIndicesToInspect) {
                solution_pt solutionAfter = populationAfter->sols[solutionIndex];

                if (solutionAfter->f < previousBestFitness) --result;
            }
        } else if (this->improvementMetricIndex == 4) {
            // Number of non dominated MO-solution created w.r.t the original solutions
            // - Count the number of solutions that is not dominated by any solution in the pop and

            for (const auto &solutionIndex: solutionIndicesToInspect) {
                // Retrieve the solutions before and after
                solution_pt solutionBefore = populationBefore->sols[solutionIndex];
                solution_pt solutionAfter = populationAfter->sols[solutionIndex];

                // Retrieve the MO solutions before and after
                std::vector<hicam::solution_pt> moSolutionsBefore = solutionBefore->mo_test_sols;
                std::vector<hicam::solution_pt> moSolutionsAfter = solutionAfter->mo_test_sols;

                // Go over all newly created MO solutions
                for (size_t moSolutionIndex = 0; moSolutionIndex < moSolutionsAfter.size(); ++moSolutionIndex) {
                    // Retrieve the MO solution to investigate
                    hicam::solution_pt moSolutionAfter = moSolutionsAfter[moSolutionIndex];

                    bool dominatedByOtherSolutions = false;

                    // Check if the MO solution is not dominated by any of the mo solutions of the previous solution set
                    for (const auto &otherSolution: moSolutionsBefore) {
                        if (GenerationalArchive::solutionDominates(otherSolution, moSolutionAfter)) {
                            dominatedByOtherSolutions = true;
                            break;
                        }
                    }

                    // Check if any of the current MO solutions dominates this solution
                    if (!dominatedByOtherSolutions) {
                        for (size_t moInspectedSolutionIndex = 0;
                             moInspectedSolutionIndex < moSolutionIndex; ++moInspectedSolutionIndex) {
                            hicam::solution_pt otherSolution = moSolutionsAfter[moInspectedSolutionIndex];

                            if (GenerationalArchive::solutionDominates(otherSolution, moSolutionAfter)) {
                                dominatedByOtherSolutions = true;
                                break;
                            }
                        }
                    }

                    // Check if solution is eligible for reward
                    if (!dominatedByOtherSolutions) result--;
                }
            }
        } else if (this->improvementMetricIndex == 5) {
            // Number of non dominated MO-solution created w.r.t the population
            // - Count the number of solutions that is not dominated by any solution in the pop and

            // Add the new population's MO solution to the generational MO solutions archive
            for (const auto &solutionIndex: solutionIndicesToInspect) {
                solution_pt currentSolution = populationAfter->sols[solutionIndex];

                for (const auto &currentMOSolution : currentSolution->mo_test_sols) {
                    bool addedToArchive = generationalMOSolutionsArchive.addMOSolutionToGenerationalArchive(
                            currentMOSolution);

                    if (addedToArchive) result--;
                }
            }
        }

        return result;
    }


    /**
     * Calculate the resources that are required next generation
     * @param MOEvaluationsUsedThisGenerationByUHV_GOMEA  The number of MO evaluations used by UHV-GOMEA this generation
     * @param MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST The number of MO evaluations used by UHV-ADAm-Best this generation
     * @param improvementThisGenerationUHV_GOMEA The improvements found this generation by UHV-GOMEA
     * @param improvementThisGenerationByUHV_ADAM_BEST The improvements found this generation by UHV-ADAM-BEST
     * @param numberOfCallsThisGenerationByUHV_ADAM_BEST The number of UHV-ADAm-BEST calls executed this generation
     */
    void naive_resource_allocation_scheme_t::calculateResourcesNextGeneration(
            size_t MOEvaluationsUsedThisGenerationByUHV_GOMEA,
            size_t MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST,
            double improvementThisGenerationUHV_GOMEA,
            double improvementThisGenerationByUHV_ADAM_BEST,
            size_t numberOfCallsThisGenerationByUHV_ADAM_BEST
    ) {
        // Calculate the resource of UHV-ADAM-BEST
        size_t sumEvaluationsRequiredByUHV_ADAM_BEST = 0;
        double sumImprovementsUHV_ADAM_BEST = 0;
        if (this->applyUHV_ADAM_BEST)
            this->calculateResourcesUHV_ADAM_BEST(sumEvaluationsRequiredByUHV_ADAM_BEST, sumImprovementsUHV_ADAM_BEST);

        // Determine minimum required MO evaluations
        size_t minimumMOEvaluations = sumEvaluationsRequiredByUHV_ADAM_BEST;   // Todo: Needs to be a max of all local optimizers used.

        // Calculate the resources of UHV-GOMEA
        size_t sumEvaluationsRequiredByUHV_GOMEA = 0;
        double sumImprovementsUHV_GOMEA = 0;
        if (this->applyUHV_GOMEA)
            this->calculateResourcesUHV_GOMEA(minimumMOEvaluations, sumEvaluationsRequiredByUHV_GOMEA,
                                              sumImprovementsUHV_GOMEA);


        // Determine which optimizer have contributed
        bool uhvGOMEAHasContributed = optimizerHasContributed(this->applyUHV_GOMEA, sumImprovementsUHV_GOMEA);
        bool uhvAdamBestHasContributed = optimizerHasContributed(this->applyUHV_ADAM_BEST,
                                                                 sumImprovementsUHV_ADAM_BEST);

        // Calculate reward Todo: Rework this part
        double rewardUHVGOMEA = uhvGOMEAHasContributed ? -sumImprovementsUHV_GOMEA /
                                                         (double) sumEvaluationsRequiredByUHV_GOMEA : 0;
        double rewardUHVAdamBest = uhvAdamBestHasContributed ? -sumImprovementsUHV_ADAM_BEST /
                                                               (double) sumEvaluationsRequiredByUHV_ADAM_BEST : 0;
        double rewardTotal = rewardUHVGOMEA + rewardUHVAdamBest;

        // When numerical precision is reached no more rewards can be obtained.
        double scaledRewardUHVGOMEA = rewardTotal > 0 ? rewardUHVGOMEA / rewardTotal : 0;
        double scaledRewardUHVAdamBest = rewardTotal > 0 ? rewardUHVAdamBest / rewardTotal : 0;

        // Calculate the budget for next generation according to this generation
        size_t totalNumberOfEvaluationsToRedistributeForThisGeneration =
                sumEvaluationsRequiredByUHV_GOMEA + sumEvaluationsRequiredByUHV_ADAM_BEST;
        double numberOfEvalsForUHVAdamBestNextGenerationAccordingToThisGeneration =
                scaledRewardUHVAdamBest * (double) totalNumberOfEvaluationsToRedistributeForThisGeneration;

        // Determine the average evaluations per call
        double averageEvaluationsPerCallUHVAdamBest = numberOfCallsThisGenerationByUHV_ADAM_BEST > 0 ?
                                                      (double) MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST /
                                                      (double) numberOfCallsThisGenerationByUHV_ADAM_BEST : 0;

        // Determine the number of calls according to this generation
        double numberOfCallsOfUHVAdamBestAccordingToThisGeneration = averageEvaluationsPerCallUHVAdamBest > 0 ?
                                                                     (numberOfEvalsForUHVAdamBestNextGenerationAccordingToThisGeneration /
                                                                      averageEvaluationsPerCallUHVAdamBest) : 0;

        // Check if UHV-GOMEA has done improvements and if not force the other optimizer to do something
        if (!uhvGOMEAHasContributed) {
            if (numberOfCallsOfUHVAdamBestAccordingToThisGeneration <= this->minimumNumberOfUHV_ADAM_BestCalls)
                numberOfCallsOfUHVAdamBestAccordingToThisGeneration = this->minimumNumberOfUHV_ADAM_BestCalls;
        }

        // Apply memory decay: The planned number of calls should be used instead of actual number of calls ('numberOfCallsThisGenerationByUHV_ADAM_BEST')
        double uhvAdamBestPlannedCallsToExecuteNextGeneration =
                applyMemoryDecay(numberOfCallsOfUHVAdamBestAccordingToThisGeneration,
                                 (double) this->callsToExecuteNextGenerationUHV_ADAM_BEST);

        // Apply waiting scheme
        if (number_of_generations >= generationThresholdUHV_ADAM_BEST) {
            // Set lower bound on number of planned calls
            if (uhvAdamBestPlannedCallsToExecuteNextGeneration < 0.01 * this->minimumNumberOfUHV_ADAM_BestCalls)
                uhvAdamBestPlannedCallsToExecuteNextGeneration = 0.01 * this->minimumNumberOfUHV_ADAM_BestCalls;

            // Set next threshold
            if (uhvAdamBestPlannedCallsToExecuteNextGeneration < 1 * this->minimumNumberOfUHV_ADAM_BestCalls) {
                this->generationThresholdUHV_ADAM_BEST = number_of_generations + 1 +
                                                         ((size_t) this->minimumNumberOfUHV_ADAM_BestCalls /
                                                          uhvAdamBestPlannedCallsToExecuteNextGeneration);
            }

            // Store the current planned number of calls if waiting is triggered such that future calls know what the previous memory value is
            this->storedCallCountUHV_ADAM_BEST = uhvAdamBestPlannedCallsToExecuteNextGeneration;

        } else {
            // Threshold is still active, pass on the number of calls to the future
            uhvAdamBestPlannedCallsToExecuteNextGeneration = this->storedCallCountUHV_ADAM_BEST;
        }

        // Cap the number of calls
        if (uhvAdamBestPlannedCallsToExecuteNextGeneration > populationSize) {
            uhvAdamBestPlannedCallsToExecuteNextGeneration = populationSize;
        }

        this->callsToExecuteNextGenerationUHV_ADAM_BEST = uhvAdamBestPlannedCallsToExecuteNextGeneration;


//        printf("Resource allocations of generation %d\n", number_of_generations);
//        printf("Raw resources UHV-GOMEA: REWARD:%.4e / RESOURCES:%ld\n", improvementThisGenerationUHV_GOMEA, MOEvaluationsUsedThisGenerationByUHV_GOMEA);
//        printf("Sum resources UHV-GOMEA: REWARD:%.4e / RESOURCES:%ld\n", sumImprovementsUHV_GOMEA, sumEvaluationsRequiredByUHV_GOMEA);
//        printf("Raw resources UHV-ADAM: REWARD:%.4e / RESOURCES:%ld\n", improvementThisGenerationByUHV_ADAM_BEST, MOEvaluationsUsedThisGenerationByUHV_ADAM_BEST);
//        printf("Sum resources UHV-ADAM: REWARD:%.4e / RESOURCES:%ld\n", sumImprovementsUHV_ADAM_BEST, sumEvaluationsRequiredByUHV_ADAM_BEST);
//        printf("Contribution of algorithms: UHV-GOMEA: %d, UHV-ADAM:%d\n", uhvGOMEAHasContributed, uhvAdamBestHasContributed);
//        printf("Resources up for grabs: %ld\n", totalNumberOfEvaluationsToRedistributeForThisGeneration);
//        printf("Rewards: UHV-GOMEA: %d(%.4e), UHV-ADAM:%d(%.4e) out of 100 percent(%.4e)\n", (int) (100*scaledRewardUHVGOMEA), rewardUHVGOMEA, (int) (100*scaledRewardUHVAdamBest), rewardUHVAdamBest, rewardTotal);
//        printf("Budget of next generation: UHV-ADAM evals: %lf, Calls: %lf, Memory Decay calls:%lf\n", numberOfEvalsForUHVAdamBestNextGenerationAccordingToThisGeneration, numberOfCallsOfUHVAdamBestAccordingToThisGeneration, uhvAdamBestPlannedCallsToExecuteNextGeneration);
//        printf("Next generation to execute UHV-ADAM: %ld\n", this->generationThresholdUHV_ADAM_BEST);
//        printf("\n");

//        if(number_of_generations == 100) exit(0);

    }

    /**
     * Calculates the number of evaluations taken into consideration and UHV-improvement of UHV-GOMEA.
     * See "Combining Gradient Techniques for Numerical Multi-Objective Evolutionary Optimization" for more details.
     * @param minimumRequiredMOEvaluations The minimum number of MO evaluations that needs to be counted.
     * @param sumEvaluationsRequiredByUHV_GOMEA The resulting sum of evaluations required by UHV-GOMEA
     * @param sumImprovementsUHV_GOMEA The resulting sum of improvements of UHV-GOMEA
     */
    void naive_resource_allocation_scheme_t::calculateResourcesUHV_GOMEA(
            size_t minimumRequiredMOEvaluations,
            size_t &sumEvaluationsRequiredByUHV_GOMEA,
            double &sumImprovementsUHV_GOMEA) {
        // Prepare variables
        size_t resultSumEvaluationsUHV_GOMEA = 0;
        double resultSumImprovementsUHV_GOMEA = 0;

        // Retrieve considered number of evaluations and resources of UHV-GOMEA
        int generation_index = this->number_of_generations;
        while (generation_index >= 0) {
            // Add the UHV and evaluations
            resultSumEvaluationsUHV_GOMEA += this->vecNumberOfMOEvalsOfUHV_GOMEA[generation_index];
            resultSumImprovementsUHV_GOMEA += this->vecImprovementsOfUHV_GOMEA[generation_index];

            // Check if the evaluations meet the 'fairness of comparison' requirement
            if (resultSumEvaluationsUHV_GOMEA >= minimumRequiredMOEvaluations) break;

            generation_index--;
        }

        sumEvaluationsRequiredByUHV_GOMEA = resultSumEvaluationsUHV_GOMEA;
        sumImprovementsUHV_GOMEA = resultSumImprovementsUHV_GOMEA;

    }

    void naive_resource_allocation_scheme_t::calculateResourcesUHV_ADAM_BEST(
            size_t &sumEvaluationsRequiredByUHV_ADAMBEST,
            double &sumImprovementsUHV_ADAM_BEST
    ) {

        // Prepare variables
        size_t resultSumEvaluationsUHV_ADAM_BEST = 0;
        double resultSumImprovementsUHV_ADAM_BEST = 0;

        // Retrieve considered number of evaluations and resources of UHV-ADAM-Best
        int generationIndex = this->number_of_generations;
        int lbGenerationIndex = std::max(generationIndex, 0);
        while (generationIndex >= lbGenerationIndex) {
            // Add the UHV and evaluations
            resultSumEvaluationsUHV_ADAM_BEST += this->vecNumberOfMOEvalsOfUHV_ADAM_Best[generationIndex];
            resultSumImprovementsUHV_ADAM_BEST += this->vecImprovementsOfUHV_ADAM_Best[generationIndex];

            generationIndex--;
        }

        sumEvaluationsRequiredByUHV_ADAMBEST = resultSumEvaluationsUHV_ADAM_BEST;
        sumImprovementsUHV_ADAM_BEST = resultSumImprovementsUHV_ADAM_BEST;

    }

    /**
     * Determines if an optimizer has contributed
     * @param optimizerActive Boolean if optimizer is active (Set in constructor)
     * @param improvementUHVThisGeneration The UHV obtained by the optimizer
     * @return
     */
    bool naive_resource_allocation_scheme_t::optimizerHasContributed(
            bool optimizerActive,
            double improvementUHVThisGeneration) {
        return (optimizerActive && improvementUHVThisGeneration < 0);
    }

    /**
     * Apply memory decay / moving average
     * @param nextValue The number of calls of the next generation according to this generation
     * @param previousValue The number of calls that was executed this generation
     * @return
     */
    double naive_resource_allocation_scheme_t::applyMemoryDecay(double nextValue, double previousValue) {
        double result;

        if (nextValue > previousValue)
            result = nextValue;
        else
            result = this->memoryDecayFactor * previousValue + (1 - this->memoryDecayFactor) * nextValue;

        return result;
    }

    /**
     * Reset all variables of the resource allocation scheme
     */
    void naive_resource_allocation_scheme_t::resetResourceAllocationSchemeStatistics() {
        this->vecNumberOfMOEvalsOfUHV_GOMEA.resize(0);
        this->vecNumberOfMOEvalsOfUHV_ADAM_Best.resize(0);
        this->vecImprovementsOfUHV_GOMEA.resize(0);
        this->vecImprovementsOfUHV_ADAM_Best.resize(0);

        this->generationThresholdUHV_ADAM_BEST = this->number_of_generations;
        this->callsToExecuteNextGenerationUHV_ADAM_BEST = 0;
    }

    /**
     * Resets the intra generational statistics variables.
     * These statistics should be refreshed every generation.
     */
    void naive_resource_allocation_scheme_t::resetIntraGenerationalOptimizerStatistics() {
        this->vecOptimizerNameOfPopulation.resize(0);
        this->vecPopulationCreatedByOptimizer.resize(0);
        this->vecBestSolutionsAfterOptimizer.resize(0);
        this->vecNumberOfSOEvaluationsByOptimizer.resize(0);
        this->vecNumberOfMOEvaluationsByOptimizer.resize(0);
        this->vecImprovementsByOptimizer.resize(0);
        this->vecRewardsByOptimizer.resize(0);
        this->vecNumberOfCallsByOptimizer.resize(0);
    }


    /**
     * Stores the resource allocation statistics of UHV-GOMEA
     * @param MOEvaluationsUsedUHV_GOMEA The number of MO evaluations used by UHV-GOMEA
     * @param improvementUHV_GOMEA The improvement metric obtained by UHV-GOMEA
     */
    void naive_resource_allocation_scheme_t::storeResourceAllocationStatisticsUHV_GOMEA(
            size_t MOEvaluationsUsedUHV_GOMEA,
            double improvementUHV_GOMEA) {
        // Store the resource allocation data
        this->vecNumberOfMOEvalsOfUHV_GOMEA.push_back(MOEvaluationsUsedUHV_GOMEA);
        this->vecImprovementsOfUHV_GOMEA.push_back(improvementUHV_GOMEA);
    }

    void naive_resource_allocation_scheme_t::storeResourceAllocationStatisticsUHV_ADAM_BEST(
            size_t MOEvaluationsUsedUHV_ADAM_BEST,
            double improvementUHV_ADAM_BEST) {
        // Store the resource allocation data
        this->vecNumberOfMOEvalsOfUHV_ADAM_Best.push_back(MOEvaluationsUsedUHV_ADAM_BEST);
        this->vecImprovementsOfUHV_ADAM_Best.push_back(improvementUHV_ADAM_BEST);
    }

    /**
     * This method stores the statistics data of an optimizer.
     * @param optimizerName The name of the optimizer that was executed
     * @param currentNumberOfSOEvaluations The current number (total) of Single Objective evaluations executed
     * @param currentNumberOfMOEvaluations The current number (total) of Multi Objective evaluations executed
     * @param currentNumberOfCalls The number of calls executed in this generation
     */
    void naive_resource_allocation_scheme_t::storeOptimizerStatistics(
            std::string optimizerName,
            int currentNumberOfSOEvaluations,
            size_t currentNumberOfMOEvaluations,
            double currentImprovementsFound,
            size_t currentNumberOfCalls) {
        // Copy current population
        population_pt copied_population = this->pop->deepCopyPopulation();

        // Calculate reward Todo: It might be more fair to use the actual reward
        bool contributed = optimizerHasContributed(true, currentImprovementsFound);
        double reward = contributed ? -currentImprovementsFound / (double) currentNumberOfMOEvaluations : 0;

        // Write intra generational statistics
        this->vecOptimizerNameOfPopulation.push_back(optimizerName);
        this->vecPopulationCreatedByOptimizer.push_back(copied_population);
        this->vecBestSolutionsAfterOptimizer.push_back(solution_t(best));
        this->vecNumberOfSOEvaluationsByOptimizer.push_back(currentNumberOfSOEvaluations);
        this->vecNumberOfMOEvaluationsByOptimizer.push_back(currentNumberOfMOEvaluations);
        this->vecImprovementsByOptimizer.push_back(currentImprovementsFound);
        this->vecRewardsByOptimizer.push_back(reward);
        this->vecNumberOfCallsByOptimizer.push_back(currentNumberOfCalls);
    }
}



