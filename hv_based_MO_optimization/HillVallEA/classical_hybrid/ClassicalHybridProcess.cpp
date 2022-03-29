/*
 * ClassicalHybridProcess.cpp
 *
 * This class takes care of the optimization process.
 * Think of initialization of the process, termination of the hybrid algorithm and writing of statistics
 *
 * Implementation by D. Ha 2021
 */


#include "ClassicalHybridProcess.h"
#include "ClassicalHybrid.h"

#include "../fitness.h"
#include "../hillvallea_internal.hpp"
#include "../population.hpp"
#include "../solution.hpp"

namespace hillvallea {
    /**
     * Constructor
     * @param fitnessFunction The UHV fitness function
     * @param localOptimizerIndexUHV_GOMEA The local optimizer index of UHV-GOMEA
     * @param localOptimizerIndexUHV_ADAM The local optimizer index of UHV-ADAM-POPULATION
     * @param indexApplicationMethodUHV_ADAM The index of the application method of UHV-ADAM-POPULATION
     * @param numberOfSOParameters The number of single objective parameters
     * @param solutionSetSize The solution set size
     * @param soLowerInitializationRange The single objective lower initialization range
     * @param soUpperInitializationRange The single objective upper initialization range
     * @param maximumNumberOfMOEvaluations The maximum number of MO evaluations to spend
     * @param maximumTimeInSeconds The maximum number of seconds to spend
     * @param valueVTR The value to reach
     * @param useVTR Boolean if VTR termination should be enabled
     * @param randomSeedNumber The number for the random number generator
     * @param writeSolutionsOptimizers Write the population to a file (not implemented yet)
     * @param writeStatisticsOptimizers Write statistics
     * @param writeDirectory Directory to write files to
     * @param fileAppendix File appendix (not used)
     */
    ClassicalHybridProcess::ClassicalHybridProcess(
            std::shared_ptr <UHV_t> fitnessFunction,
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
            std::string fileAppendix) {
        // Copy all parameters
        this->uhvFitnessFunction = fitnessFunction;
        this->localOptimizerIndexUHV_GOMEA = localOptimizerIndexUHV_GOMEA;
        this->localOptimizerIndexUHV_ADAM = localOptimizerIndexUHV_ADAM;
        this->indexApplicationMethodUHV_ADAM = indexApplicationMethodUHV_ADAM;
        this->numberOfSOParameters = numberOfSOParameters;
        this->solutionSetSize = solutionSetSize;
        this->lowerInitializationRanges = soLowerInitializationRange;
        this->upperInitializationRanges = soUpperInitializationRange;
        this->maximumNumberOfMOEvaluations = maximumNumberOfMOEvaluations;
        this->maximumTimeInSeconds = maximumTimeInSeconds;
        this->useVTR = useVTR;
        this->valueVTR = valueVTR;
        this->randomSeedNumber = randomSeedNumber;
        this->writeSolutionsOptimizers = writeSolutionsOptimizers;
        this->writeStatisticsOptimizers = writeStatisticsOptimizers;
        this->writeDirectory = writeDirectory;
        this->file_appendix = fileAppendix;

        // Derive parameters
        this->rng = std::make_shared<std::mt19937>((unsigned long) (randomSeedNumber));
        std::uniform_real_distribution<double> unif(0, 1);

        // Initialize default parameters
        init_default_params();
    }

    /**
     * Destructor
     */
    ClassicalHybridProcess::~ClassicalHybridProcess() {}

    /**
     * Initializes the default parameters of the process
     */
    void ClassicalHybridProcess::init_default_params() {
        // Determine the scaled search volume
        double scaledSearchVolumeResult = 1.0;
        for (size_t i = 0; i < this->upperInitializationRanges.size(); ++i) {
            double upperMinLowerInit = this->upperInitializationRanges[i] - this->lowerInitializationRanges[i];
            scaledSearchVolumeResult *= pow(upperMinLowerInit, 1.0 / this->numberOfSOParameters);
        }
        this->scaledSearchVolume = scaledSearchVolumeResult;

        // Set SO init bounds to parameter bounds if it's beyond the param bounds
        uhvFitnessFunction->get_param_bounds(lowerParameterBounds, upperParameterBounds);
        for (size_t i = 0; i < upperParameterBounds.size(); ++i) {
            if (lowerInitializationRanges[i] < lowerParameterBounds[i])
                lowerInitializationRanges[i] = lowerParameterBounds[i];

            if (upperInitializationRanges[i] > upperParameterBounds[i])
                upperInitializationRanges[i] = upperParameterBounds[i];
        }
    }

    /**
     * Resets log variables to the default values
     */
    void ClassicalHybridProcess::reset_log_variables() {
        // Reset all log variables
        this->startingTime = clock();
        this->success = false;
        this->terminated = false;
        this->numberOfSOEvaluations = 0;
        this->numberOfMOEvaluations = 0;
        this->numberOfGenerations = 0;
    }

    /**
     * Terminate when max runtime is exceeded
     * @return True if max runtime is exceeded
     */
    bool ClassicalHybridProcess::terminate_on_runtime() {
        // Stop if time has run out
        if (this->maximumTimeInSeconds > 0) {
            clock_t currentTime = clock();
            double runtime = double(currentTime - this->startingTime) / CLOCKS_PER_SEC;

            return runtime > this->maximumTimeInSeconds;
        }

        return false;
    }

    /**
     * Terminate when UHV value has been reached
     * @return Boolean if UHV-value has been reached
     */
    bool ClassicalHybridProcess::terminate_on_vtr(optimizer_pt optimizer_selected) {
        // Initialize variables
        bool vtrHit, shallowVTRHit = false;

        // Determine variables
        solution_pt bestSolutionFound = optimizer_selected->pop->bestSolution();

        // Shallow check if VTR has been hit and which VTR type is hit
        if (!uhvFitnessFunction->redefine_vtr) {
            shallowVTRHit = (bestSolutionFound->constraint == 0) && (bestSolutionFound->f <= valueVTR);
        } else {
            shallowVTRHit = uhvFitnessFunction->vtr_reached(*bestSolutionFound, valueVTR);
        }

        // Find out if VTR has truly been reached
        if (shallowVTRHit) {

            // Check for round off errors
            if (uhvFitnessFunction->partial_evaluations_available &&
                uhvFitnessFunction->has_round_off_errors_in_partial_evaluations) {

                // re-evaluate solution when vtr is assumed to be hit.
                double numberOFMOEvaluationsBefore = uhvFitnessFunction->number_of_mo_evaluations;
                uhvFitnessFunction->evaluate_with_gradients(bestSolutionFound);
                double numberOFMOEvaluationsAfter = uhvFitnessFunction->number_of_mo_evaluations;

                this->numberOfSOEvaluations++;
                this->numberOfMOEvaluations += (int) (numberOFMOEvaluationsAfter -
                                                      numberOFMOEvaluationsBefore);  // Todo: This probably does not get logged because we never reach this

                if (!uhvFitnessFunction->redefine_vtr) {
                    vtrHit = (bestSolutionFound->constraint == 0) && (bestSolutionFound->f <= valueVTR);
                } else {
                    vtrHit = uhvFitnessFunction->vtr_reached(*bestSolutionFound, valueVTR);
                }
            } else {
                vtrHit = true;
            }
        } else {
            vtrHit = false;
        }

        return vtrHit;
    }

    /**
     * Do the optimization (Process copied from hillvallea_t::runSerial3())
     * Why not reuse hillvallea_t::runSerial3()? Because statistics might need to be written after each optimizer is done.
     */
    void ClassicalHybridProcess::run_optimization(size_t populationSize) {
        // Reset log variables
        this->reset_log_variables();

        // Create initial population
        population_pt initialPopulation = std::make_shared<population_t>();

        // Initialize the population
        if (uhvFitnessFunction->redefine_random_initialization) {
            uhvFitnessFunction->init_solutions_randomly(*initialPopulation,
                                                        populationSize,
                                                        this->lowerInitializationRanges,
                                                        this->upperInitializationRanges,
                                                        0,
                                                        this->rng);
        } else {
            initialPopulation->fill_uniform(populationSize,
                                            this->numberOfSOParameters,
                                            this->lowerInitializationRanges,
                                            this->upperInitializationRanges,
                                            this->rng);
        }


        // Evaluate the initial population and sort on fitness
        double numberOFMOEvaluationsBefore = uhvFitnessFunction->number_of_mo_evaluations;
        this->numberOfSOEvaluations += initialPopulation->evaluate_with_gradients(this->uhvFitnessFunction, 0);
        double numberOFMOEvaluationsAfter = uhvFitnessFunction->number_of_mo_evaluations;
        this->numberOfMOEvaluations += (int) (numberOFMOEvaluationsAfter - numberOFMOEvaluationsBefore);
        initialPopulation->sort_on_fitness();

        // Create statistics file
        std::string statisticsFilepath;
        std::shared_ptr <std::ofstream> statisticsStream(new std::ofstream);
        if (writeStatisticsOptimizers) {
            this->create_initial_statistics_file(statisticsFilepath,
                                                 statisticsStream);
        }

        // Init the UHV-HYBRID optimizer
        double initUniVariateBandwidth =
                this->scaledSearchVolume * pow(initialPopulation->size(), -1.0 / numberOfSOParameters);
        this->uhvClassicalHybrid = std::make_shared<ClassicalHybrid>(
                this->uhvFitnessFunction,
                this->localOptimizerIndexUHV_GOMEA,
                this->localOptimizerIndexUHV_ADAM,
                this->indexApplicationMethodUHV_ADAM,
                this->numberOfSOParameters,
                this->lowerParameterBounds,
                this->upperParameterBounds,
                initUniVariateBandwidth,
                this->rng);
        uhvClassicalHybrid->initialize_from_population(initialPopulation, populationSize);

        // Write initial population's generational statistics
        if (writeStatisticsOptimizers) {
            writeOptimizersStatistics(uhvClassicalHybrid, 0, statisticsStream);
        }

        // Optimization loop
        bool statisticsRecentlyWritten = false; // Prevents the process from writing the same data twice at the end
        while (true) {
            // Terminate if max number of evaluations has been reached
            if (maximumNumberOfMOEvaluations > 0 && this->numberOfMOEvaluations >= maximumNumberOfMOEvaluations) {
                printf("Max evaluations reached %d/%d (SO-evals)\n",
                       this->numberOfMOEvaluations, maximumNumberOfMOEvaluations);
                break;
            }

            // Terminate if time limit has been reached
            if (terminate_on_runtime()) {
                printf("Time limit reached\n");
                break;
            }

            // Terminate if VTR has been hit
            if (this->useVTR && this->terminate_on_vtr(uhvClassicalHybrid)) {
                printf("VTR reached %lf/%lf \n", uhvClassicalHybrid->best.f, valueVTR);
                success = true;
                break;
            }

            // Prepare to execute a generation
            statisticsRecentlyWritten = false;

            // Execute a generation
            double numberOFMOEvaluationsBefore = this->uhvFitnessFunction->number_of_mo_evaluations;
            uhvClassicalHybrid->generation(populationSize, this->numberOfSOEvaluations);
            double numberOFMOEvaluationsAfter = this->uhvFitnessFunction->number_of_mo_evaluations;
            this->numberOfMOEvaluations += (int) (numberOFMOEvaluationsAfter - numberOFMOEvaluationsBefore);

            // Todo: Write generational solutions

            // Write statistics
            if (writeStatisticsOptimizers) {
//                            if (this->number_of_generations < 50 || this->number_of_generations % 100 == 0)
                {
                    statisticsRecentlyWritten = true;
                    writeOptimizersStatistics(uhvClassicalHybrid, numberOfGenerations, statisticsStream);
                }
            }

            // Update statistics
            this->numberOfGenerations++;
        }

        // Write final statistics
        if (writeStatisticsOptimizers && !statisticsRecentlyWritten) {
            writeOptimizersStatistics(uhvClassicalHybrid, numberOfGenerations, statisticsStream);
        }

        // Close all statistics files
        statisticsStream->close();
    }

    /**
     * Creates a new statistics file that contains the column headers
     * @param initial_population_statistics_path A pointer to the string that contains the path of the statistics file
     * @param initial_population_statistics_file A pointer to the ofstream of the statistics file
     */
    void ClassicalHybridProcess::create_initial_statistics_file(
            std::string &statisticsPath,
            std::shared_ptr <std::ofstream> statisticsStream) {
        // Determine statistics file name
        statisticsPath = this->writeDirectory + "statistics_UHV-CLASSICAL_HYBRID.dat";

        // Create and open statistics file
        statisticsStream->open(statisticsPath, std::ofstream::out | std::ofstream::trunc);

        // Add header to statistics file
        *statisticsStream
                << "Algorithm        Gen    Evals  Calls          Time                    Best-f   "
                << "Best-constr   Average-obj       Std-obj    Avg-constr    Std-constr "
                << uhvFitnessFunction->write_solution_info_header(false) << std::endl;
    }

    /**
     * Writes all the intermediate statistics of each optimizer that was executed.
     * Assumes that the file is already open.
     * @param uhvClassicalHybrid The classical hybrid containing the data
     * @param currentGenerationIndex The current generation
     * @param statistics_file The statistics file to write the data to.
     */
    void ClassicalHybridProcess::writeOptimizersStatistics(classicalHybrid_pt uhvClassicalHybrid,
                                                           size_t currentGenerationIndex,
                                                           std::shared_ptr <std::ofstream> statistics_file) {
        // Retrieve data
        std::vector <std::string> vecNamesOptimizers = uhvClassicalHybrid->vecOptimizerNameOfPopulation;
        std::vector <population_pt> vecPopulationsCreatedByOptimizers = uhvClassicalHybrid->vecPopulationCreatedByOptimizer;
        std::vector <solution_t> vecBestSolutionsAfterOptimizer = uhvClassicalHybrid->vecBestSolutionsAfterOptimizer;
        std::vector<int> vecNumberOfSOEvaluationsPerOptimizer = uhvClassicalHybrid->vecNumberOfSOEvaluationsByOptimizer;
        std::vector <size_t> vecNumberOfMOEvaluationsPerOptimizer = uhvClassicalHybrid->vecNumberOfMOEvaluationsByOptimizer;
        std::vector <size_t> vecCurrentNumberOfCallsPerOptimizer = uhvClassicalHybrid->vecNumberOfCallsByOptimizer;


        // Sanity check on data vectors
        if (vecPopulationsCreatedByOptimizers.size() != vecNamesOptimizers.size() ||
            vecPopulationsCreatedByOptimizers.size() != vecBestSolutionsAfterOptimizer.size() ||
            vecPopulationsCreatedByOptimizers.size() != vecNumberOfSOEvaluationsPerOptimizer.size() ||
            vecPopulationsCreatedByOptimizers.size() != vecNumberOfMOEvaluationsPerOptimizer.size() ||
            vecPopulationsCreatedByOptimizers.size() != vecCurrentNumberOfCallsPerOptimizer.size()) {
            throw std::runtime_error("Statistics vectors are not of equal size:Pop: "
                                     + std::to_string(vecPopulationsCreatedByOptimizers.size()) + ", " +
                                     "Opt: " + std::to_string(vecNamesOptimizers.size()) + ", " +
                                     "SO Evals: " + std::to_string(vecNumberOfSOEvaluationsPerOptimizer.size()) +
                                     "MO Evals: " + std::to_string(vecNumberOfMOEvaluationsPerOptimizer.size()) +
                                     "Calls: " + std::to_string(vecCurrentNumberOfCallsPerOptimizer.size()));
        }

        // Check not empty
        if (vecPopulationsCreatedByOptimizers.empty() ||
            vecNamesOptimizers.empty() ||
            vecBestSolutionsAfterOptimizer.empty() ||
            vecNumberOfSOEvaluationsPerOptimizer.empty() ||
            vecNumberOfMOEvaluationsPerOptimizer.empty() ||
            vecCurrentNumberOfCallsPerOptimizer.empty()) {
            throw std::runtime_error("Statistics vectors are empty:Pop: "
                                     + std::to_string(vecPopulationsCreatedByOptimizers.size()) + ", " +
                                     "Opt: " + std::to_string(vecNamesOptimizers.size()) + ", " +
                                     "SO Evals: " + std::to_string(vecNumberOfSOEvaluationsPerOptimizer.size()) +
                                     "MO Evals: " + std::to_string(vecNumberOfMOEvaluationsPerOptimizer.size()) +
                                     "Calls: " + std::to_string(vecCurrentNumberOfCallsPerOptimizer.size()));
        }

        // Determine time that has passed
        clock_t currentTime = clock();
        double runtime = double(currentTime - this->startingTime) / CLOCKS_PER_SEC;

        // Prepare empty variables
        std::vector <solution_pt> emptyArchive;
        emptyArchive.clear();

        // Go over optimizers
        for (size_t optimizerIndex = 0; optimizerIndex < vecPopulationsCreatedByOptimizers.size(); ++optimizerIndex) {
            // Retrieve data
            std::string optimizerName = vecNamesOptimizers[optimizerIndex];
            population_pt optimizerPopulation = vecPopulationsCreatedByOptimizers[optimizerIndex];
            solution_t bestSolution = vecBestSolutionsAfterOptimizer[optimizerIndex];
            int numberOfSOEvaluations = vecNumberOfSOEvaluationsPerOptimizer[optimizerIndex];
            size_t numberOfMOEvaluations = vecNumberOfMOEvaluationsPerOptimizer[optimizerIndex];
            size_t numberOfCallsExecuted = vecCurrentNumberOfCallsPerOptimizer[optimizerIndex];


            // Check if ofstream is already open
            if (statistics_file->is_open()) {
                // Adjust fitness function parameters Todo: This is a quick implementation, but might be prone to errors
                uhvFitnessFunction->number_of_mo_evaluations = numberOfMOEvaluations;

                // Write to statistics file
                *statistics_file
                        << std::setw(14) << optimizerName
                        << " " << std::setw(3) << currentGenerationIndex
                        << " " << std::setw(8) << numberOfSOEvaluations
                        << " " << std::setw(8) << std::fixed << std::setprecision(3) << numberOfCallsExecuted
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << runtime
                        << " " << std::setw(25) << std::scientific << std::setprecision(16) << bestSolution.f
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << bestSolution.constraint
                        << " " << std::setw(13) << std::scientific << std::setprecision(16)
                        << optimizerPopulation->average_fitness()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3)
                        << optimizerPopulation->relative_fitness_std()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3)
                        << optimizerPopulation->average_constraint()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3)
                        << optimizerPopulation->relative_constraint_std()
                        << " " << uhvFitnessFunction->write_additional_solution_info(bestSolution, emptyArchive, false)
                        << std::endl;
            } else {
                throw std::runtime_error("Statistics file was not open.");
            }
        }
    }

    /**
     * Getter of 'number_of_evaluations'
     * @return 'number_of_evaluations'
     */
    int ClassicalHybridProcess::get_number_of_evaluations() {
        return this->numberOfMOEvaluations;
    }

    /**
     * Getter of 'starting_time'
     * @return 'starting_time'
     */
    clock_t ClassicalHybridProcess::get_starting_time() {
        return this->startingTime;
    }
}