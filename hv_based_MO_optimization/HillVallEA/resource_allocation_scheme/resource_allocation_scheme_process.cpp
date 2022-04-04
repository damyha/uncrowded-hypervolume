/*
 * resource_allocation_scheme_process.cpp
 *
 * This class takes care of the optimization process.
 * Think of initialization of the process, termination of the hybrid algorithm and writing of statistics
 *
 * Implementation by D. Ha 2021
 */


#include "resource_allocation_scheme_process.h"
#include "naive_resource_allocation_scheme.h"

#include "../fitness.h"
#include "../hillvallea_internal.hpp"
#include "../population.hpp"
#include "../solution.hpp"

namespace hillvallea {

    /**
     * Constructor
     * @param fitnessFunction The uhv fitness function
     * @param localOptimizerIndexUHV_GOMEA The local optimizer index of UHV-GOMEA
     * @param localOptimizerIndexUHV_ADAM The local optimizer index of UHV-ADAM
     * @param indexApplicationMethodUHV_ADAM The method index of how a gradient algorithm should be applied on a pop.
     * @param numberOfSOParameters The number of SO parameters
     * @param solutionSetSize The solution set size
     * @param soLowerInitializationRange The SO lower initialization range per parameter
     * @param soUpperInitializationRange The SO upper initialization range per parameter
     * @param maximumNumberOfMOEvaluations The maximum number of MO evaluations
     * @param maximumTimeInSeconds The maximum time for the algorithm to run in seconds
     * @param valueVTR The UHV value to reach
     * @param useVTR Termination on UHV value allowed
     * @param memoryDecayFactor The memory decay factor
     * @param randomSeedNumber The random seed number
     * @param writeSolutionsOptimizers Should solutions be written after applying an optimizer
     * @param writeStatisticsOptimizers Should statistics be written after applying an optimizer
     * @param writeDirectory The write directory
     * @param fileAppendix The appendix of the files (probably not used due to max file length)
     */
    resource_allocation_scheme_process_t::resource_allocation_scheme_process_t(
            std::shared_ptr<UHV_t> fitnessFunction,
            int localOptimizerIndexUHV_GOMEA,
            int localOptimizerIndexUHV_ADAM,
            int indexApplicationMethodUHV_ADAM,
            int improvementMetricIndex,
            int numberOfSOParameters,
            size_t solutionSetSize,
            const vec_t &soLowerInitializationRange,
            const vec_t &soUpperInitializationRange,
            int maximumNumberOfMOEvaluations,
            int maximumTimeInSeconds,
            double valueVTR,
            bool useVTR,
            double memoryDecayFactor,
            double initialStepSizeFactor,
            double stepSizeDecayFactor,
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
        this->improvementMetricIndex = improvementMetricIndex;
        this->numberOfSOParameters = numberOfSOParameters;
        this->solutionSetSize = solutionSetSize;
        this->lowerInitializationRanges = soLowerInitializationRange;
        this->upperInitializationRanges = soUpperInitializationRange;
        this->maximumNumberOfMOEvaluations = maximumNumberOfMOEvaluations;
        this->maximumTimeInSeconds = maximumTimeInSeconds;
        this->useVTR = useVTR;
        this->valueVTR = valueVTR;
        this->memoryDecayFactor = memoryDecayFactor;
        this->initialStepSizeFactor = initialStepSizeFactor;
        this->stepSizeDecayFactor = stepSizeDecayFactor;
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

        // Settings
        this->writeBestSolutionToFile = false;
    }

    /**
     * Destructor
     */
    resource_allocation_scheme_process_t::~resource_allocation_scheme_process_t() {}


    /**
     * Initializes the default parameters of the process
     */
    void resource_allocation_scheme_process_t::init_default_params() {
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
    void resource_allocation_scheme_process_t::reset_log_variables() {
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
    bool resource_allocation_scheme_process_t::terminate_on_runtime() {
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
    bool resource_allocation_scheme_process_t::terminate_on_vtr(optimizer_pt optimizer_selected) {
        // Initialize variables
        bool vtrHit, shallowVTRHit = false;

        // Determine variables
        solution_t bestSolutionFound = optimizer_selected->best;

        // Shallow check if VTR has been hit and which VTR type is hit
        if (!uhvFitnessFunction->redefine_vtr) {
            shallowVTRHit = (bestSolutionFound.constraint == 0) && (bestSolutionFound.f <= valueVTR);
        } else {
            shallowVTRHit = uhvFitnessFunction->vtr_reached(bestSolutionFound, valueVTR);
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
                this->numberOfMOEvaluations += (int) (numberOFMOEvaluationsAfter - numberOFMOEvaluationsBefore);  // Todo: This probably does not get logged because we never reach this

                if (!uhvFitnessFunction->redefine_vtr) {
                    vtrHit = (bestSolutionFound.constraint == 0) && (bestSolutionFound.f <= valueVTR);
                } else {
                    vtrHit = uhvFitnessFunction->vtr_reached(bestSolutionFound, valueVTR);
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
    void resource_allocation_scheme_process_t::run_optimization(size_t populationSize) {
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
        std::shared_ptr<std::ofstream> statisticsStream(new std::ofstream);
        if (writeStatisticsOptimizers) {
            this->create_initial_statistics_file(statisticsFilepath,
                                                 statisticsStream);
        }

        // Init the UHV-HYBRID optimizer
        double initUniVariateBandwidth = this->scaledSearchVolume * pow(initialPopulation->size(), -1.0 / numberOfSOParameters);
        this->uhvHybrid = std::make_shared<naive_resource_allocation_scheme_t>(
                this->uhvFitnessFunction,
                this->localOptimizerIndexUHV_GOMEA,
                this->localOptimizerIndexUHV_ADAM,
                this->indexApplicationMethodUHV_ADAM,
                this->improvementMetricIndex,
                this->numberOfSOParameters,
                this->lowerParameterBounds,
                this->upperParameterBounds,
                initUniVariateBandwidth,
                this->memoryDecayFactor,
                this->initialStepSizeFactor,
                this->stepSizeDecayFactor,
                this->rng);

        uhvHybrid->initialize_from_population(initialPopulation, populationSize);

        // Write initial population's generational statistics
        if (writeStatisticsOptimizers) {
            writeOptimizersStatistics(uhvHybrid, 0, statisticsStream);

            // Write initial best solution (together with the statistics)
            if (writeBestSolutionToFile) {
                writeBestSolution(uhvHybrid, 0);
            }
        }

        // Write initial population's solution to a file
        if (writeSolutionsOptimizers) {
            writePopulation(uhvHybrid, 0);
        }

        // Optimization loop
        bool statisticsRecentlyWritten = false, populationRecentlyWritten = false; // Prevents the process from writing the same data twice at the end
        while (true) {
            // Terminate if max number of evaluations has been reached
            if (maximumNumberOfMOEvaluations > 0 && this->numberOfMOEvaluations >= maximumNumberOfMOEvaluations) {
                printf("Max evaluations reached %d/%d (SO-evals)\n", this->numberOfMOEvaluations,
                       maximumNumberOfMOEvaluations);
                break;
            }

            // Terminate if time limit has been reached
            if (terminate_on_runtime()) {
                printf("Time limit reached\n");
                break;
            }

            // Terminate if VTR has been hit
            if (this->useVTR && this->terminate_on_vtr(uhvHybrid)) {
                printf("VTR reached %lf/%lf \n", uhvHybrid->best.f, valueVTR);
                success = true;
                break;
            }

            // Prepare to execute a generation
            statisticsRecentlyWritten = false;

            // Execute a generation
            double numberOFMOEvaluationsBefore = this->uhvFitnessFunction->number_of_mo_evaluations;
            uhvHybrid->generation(populationSize, this->numberOfSOEvaluations);
            double numberOFMOEvaluationsAfter = this->uhvFitnessFunction->number_of_mo_evaluations;
            this->numberOfMOEvaluations += (int) (numberOFMOEvaluationsAfter - numberOFMOEvaluationsBefore);

            // Write statistics
            if (writeStatisticsOptimizers) {
//                if (this->numberOfGenerations < 50 || this->numberOfGenerations % 100 == 0)
                {
                    statisticsRecentlyWritten = true;
                    writeOptimizersStatistics(uhvHybrid, numberOfGenerations, statisticsStream);
                }

                // Write the best solution (together with the statistics)
                if (writeBestSolutionToFile) {
                    writeBestSolution(uhvHybrid, numberOfGenerations);
                }
            }

            // Write population
            if (writeSolutionsOptimizers) {
                //                if (this->numberOfGenerations < 50 || this->numberOfGenerations % 100 == 0)
                {
                    populationRecentlyWritten = true;
                    writePopulation(uhvHybrid, numberOfGenerations);
                }
            }

            // Update statistics
            this->numberOfGenerations++;
        }

        // Write final statistics
        if (writeStatisticsOptimizers && !statisticsRecentlyWritten) {
            writeOptimizersStatistics(uhvHybrid, numberOfGenerations, statisticsStream);

            // Write the best solution (together with the statistics)
            if (writeBestSolutionToFile) {
                writeBestSolution(uhvHybrid, numberOfGenerations);
            }
        }

        // Write population
        if (writeSolutionsOptimizers && !populationRecentlyWritten) {
            writePopulation(uhvHybrid, numberOfGenerations);
        }

        // Close all statistics files
        statisticsStream->close();
    }

    /**
     * Creates a new statistics file that contains the column headers
     * @param initial_population_statistics_path A pointer to the string that contains the path of the statistics file
     * @param initial_population_statistics_file A pointer to the ofstream of the statistics file
     */
    void resource_allocation_scheme_process_t::create_initial_statistics_file(
            std::string &statisticsPath,
            std::shared_ptr<std::ofstream> statisticsStream) {
        // Determine statistics file name
        statisticsPath = this->writeDirectory + "statistics_UHV-HYBRID-NAIVE.dat";

        // Create and open statistics file
        statisticsStream->open(statisticsPath, std::ofstream::out | std::ofstream::trunc);

        // Add header to statistics file
        *statisticsStream
                << "     Algorithm  Gen    Evals            Improvements                 Rewards Calls          Time"
                << "                  Best-f   Best-constr             Average-obj       Std-obj    Avg-constr    Std-constr "
                << uhvFitnessFunction->write_solution_info_header(false) << std::endl;
    }


    /**
     * Writes all the intermediate statistics of each optimizer that was executed.
     * Assumes that the file is already open.
     * @param hybridOptimizer The hybrid optimizer that contains the intermediate statistics
     * @param currentGenerationIndex The current generation
     * @param statistics_file The statistics file to write the data to.
     */
    void resource_allocation_scheme_process_t::writeOptimizersStatistics(
            naive_resource_allocation_scheme_pt hybridOptimizer,
            size_t currentGenerationIndex,
            std::shared_ptr<std::ofstream> statistics_file) {
        // Retrieve data
        std::vector<std::string> vecNamesOptimizers = hybridOptimizer->vecOptimizerNameOfPopulation;
        std::vector<population_pt> vecPopulationsCreatedByOptimizers = hybridOptimizer->vecPopulationCreatedByOptimizer;
        std::vector<solution_t> vecBestSolutionsAfterOptimizer = hybridOptimizer->vecBestSolutionsAfterOptimizer;
        std::vector<int> vecNumberOfSOEvaluationsPerOptimizer = hybridOptimizer->vecNumberOfSOEvaluationsByOptimizer;
        std::vector<size_t> vecNumberOfMOEvaluationsPerOptimizer = hybridOptimizer->vecNumberOfMOEvaluationsByOptimizer;
        std::vector<double> vecImprovementsByOptimizer = hybridOptimizer->vecImprovementsByOptimizer;
        std::vector<double> vecRewardsByOptimizer = hybridOptimizer->vecRewardsByOptimizer;
        std::vector<size_t> vecCurrentNumberOfCallsPerOptimizer = hybridOptimizer->vecNumberOfCallsByOptimizer;


        // Sanity check on data vectors
        size_t correctLength = vecNamesOptimizers.size();
        if (vecPopulationsCreatedByOptimizers.size() != correctLength ||
                vecBestSolutionsAfterOptimizer.size() != correctLength ||
                vecNumberOfSOEvaluationsPerOptimizer.size() != correctLength ||
                vecNumberOfMOEvaluationsPerOptimizer.size() != correctLength ||
                vecImprovementsByOptimizer.size() != correctLength ||
                vecRewardsByOptimizer.size() != correctLength ||
                vecCurrentNumberOfCallsPerOptimizer.size() != correctLength
                ) {
            throw std::runtime_error("Statistics vectors are not of equal size:Pop: "
                                     + std::to_string(vecPopulationsCreatedByOptimizers.size()) + ", " +
                                     "Opt: " + std::to_string(vecNamesOptimizers.size()) + ", " +
                                     "SO Evals: " + std::to_string(vecNumberOfSOEvaluationsPerOptimizer.size()) + ", " +
                                     "MO Evals: " + std::to_string(vecNumberOfMOEvaluationsPerOptimizer.size()) + ", " +
                                     "Improvements : " + std::to_string(vecImprovementsByOptimizer.size()) + ", " +
                                     "Rewards : " + std::to_string(vecRewardsByOptimizer.size()) + ", " +
                                     "Calls: " + std::to_string(vecCurrentNumberOfCallsPerOptimizer.size()));
        }

        // Check not empty
        if (vecPopulationsCreatedByOptimizers.empty() ||
            vecNamesOptimizers.empty() ||
            vecNumberOfSOEvaluationsPerOptimizer.empty() ||
            vecNumberOfMOEvaluationsPerOptimizer.empty() ||
            vecImprovementsByOptimizer.empty() ||
            vecRewardsByOptimizer.empty() ||
            vecCurrentNumberOfCallsPerOptimizer.empty()) {
            throw std::runtime_error("Statistics vectors are empty:Pop: "
            + std::to_string(vecPopulationsCreatedByOptimizers.size()) + ", " +
            "Opt: " + std::to_string(vecNamesOptimizers.size()) + ", " +
            "SO Evals: " + std::to_string(vecNumberOfSOEvaluationsPerOptimizer.size()) + ", " +
            "MO Evals: " + std::to_string(vecNumberOfMOEvaluationsPerOptimizer.size()) + ", " +
            "Improvements: " + std::to_string(vecImprovementsByOptimizer.size()) + ", " +
            "Rewards: " + std::to_string(vecRewardsByOptimizer.size()) + ", " +
            "Calls: " + std::to_string(vecCurrentNumberOfCallsPerOptimizer.size()));
        }

        // Determine time that has passed
        clock_t currentTime = clock();
        double runtime = double(currentTime - this->startingTime) / CLOCKS_PER_SEC;

        // Prepare empty variables
        std::vector<solution_pt> emptyArchive;
        emptyArchive.clear();

        // Go over optimizers
        for (size_t optimizerIndex = 0; optimizerIndex < vecPopulationsCreatedByOptimizers.size(); ++optimizerIndex) {
            // Retrieve data
            std::string optimizerName = vecNamesOptimizers[optimizerIndex];
            population_pt optimizerPopulation = vecPopulationsCreatedByOptimizers[optimizerIndex];
            solution_t bestSolution = vecBestSolutionsAfterOptimizer[optimizerIndex];
            int numberOfSOEvaluations = vecNumberOfSOEvaluationsPerOptimizer[optimizerIndex];
            size_t numberOfMOEvaluations = vecNumberOfMOEvaluationsPerOptimizer[optimizerIndex];
            double improvements = vecImprovementsByOptimizer[optimizerIndex];
            double reward = vecRewardsByOptimizer[optimizerIndex];
            size_t numberOfCallsExecuted = vecCurrentNumberOfCallsPerOptimizer[optimizerIndex];

            // Check if ofstream is already open
            if (statistics_file->is_open()) {
                // Adjust fitness function parameters Todo: This is a quick implementation, but might be prone to errors
                uhvFitnessFunction->number_of_mo_evaluations = numberOfMOEvaluations;

                // Write to statistics file
                *statistics_file
                        << std::setw(14) << optimizerName
                        << " " << std::setw(4) << currentGenerationIndex
                        << " " << std::setw(8) << numberOfSOEvaluations
                        << " " << std::setw(23) << std::scientific << std::setprecision(16) << improvements
                        << " " << std::setw(23) << std::scientific << std::setprecision(16) << reward
                        << " " << std::setw(5) << std::fixed << std::setprecision(5) << numberOfCallsExecuted
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << runtime
                        << " " << std::setw(23) << std::scientific << std::setprecision(16) << bestSolution.f
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << bestSolution.constraint
                        << " " << std::setw(23) << std::scientific << std::setprecision(16) << optimizerPopulation->average_fitness()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << optimizerPopulation->relative_fitness_std()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << optimizerPopulation->average_constraint()
                        << " " << std::setw(13) << std::scientific << std::setprecision(3) << optimizerPopulation->relative_constraint_std()
                        << " " << uhvFitnessFunction->write_additional_solution_info(bestSolution, emptyArchive, false)
                        << std::endl;
            } else {
                throw std::runtime_error("Statistics file was not open.");
            }
        }
    }

    /**
     * Writes the populations to a file
     * @param hybridOptimizer
     * @param currentGenerationIndex
     */
    void resource_allocation_scheme_process_t::writePopulation(
            naive_resource_allocation_scheme_pt hybridOptimizer,
            size_t currentGenerationIndex) {

        // Retrieve the populations that were created
        std::vector<population_pt> vecPopulationCreatedByOptimizer = hybridOptimizer->vecPopulationCreatedByOptimizer;

        // Write every population
        for (size_t optimizerIndex = 0; optimizerIndex < vecPopulationCreatedByOptimizer.size(); ++optimizerIndex) {
            // Create file name
            std::ostringstream fileNameStream;
            fileNameStream << "populationHybrid_generation" << currentGenerationIndex
                << "_subgeneration" << optimizerIndex
                << ".dat";
            std::string fileName = fileNameStream.str();
            std::string filePath = this->writeDirectory + fileName;

            // Create and open statistics file.
            std::shared_ptr<std::ofstream> populationStream(new std::ofstream);
            populationStream->open(filePath, std::ofstream::out | std::ofstream::trunc);

            // Retrieve the population
            population_pt currentPopulation = vecPopulationCreatedByOptimizer[optimizerIndex];

            // Write the population
            for (const auto currentSolution : currentPopulation->sols) {
                // Write the parameters
                for(const auto parameter : currentSolution->param) {
                    *populationStream << std::setw(23) << std::scientific << std::setprecision(16) << parameter << " ";
                }

                // Write the fitness
                *populationStream << "   " << std::setw(23) << std::scientific << std::setprecision(16) << currentSolution->f << std::endl;

            }
        }
    }

    /**
     * Write the best solution to a file
     * @param hybridOptimizer
     * @param currentGenerationIndex
     */
    void resource_allocation_scheme_process_t::writeBestSolution(
            naive_resource_allocation_scheme_pt hybridOptimizer,
            size_t currentGenerationIndex) {
        // Retrieve the data
        std::vector<solution_t> vecBestSolutionsAfterOptimizer = hybridOptimizer->vecBestSolutionsAfterOptimizer;
        std::vector<size_t> vecNumberOfMOEvaluationsByOptimizer = hybridOptimizer->vecNumberOfMOEvaluationsByOptimizer;

        // Write every best solution
        for (size_t optimizerIndex = 0; optimizerIndex < vecBestSolutionsAfterOptimizer.size(); ++optimizerIndex) {
            // Create file name
            std::ostringstream fileNameStream;
            fileNameStream << "bestSolutionHybridAlgorithm_generation" << currentGenerationIndex
                           << "_subgeneration" << optimizerIndex
                           << ".dat";
            std::string fileName = fileNameStream.str();
            std::string filePath = this->writeDirectory + fileName;

            // Create and open statistics file.
            std::shared_ptr<std::ofstream> populationStream(new std::ofstream);
            populationStream->open(filePath, std::ofstream::out | std::ofstream::trunc);

            // Retrieve data of optimizer
            solution_t currentBestSolution = vecBestSolutionsAfterOptimizer[optimizerIndex];
            size_t currentMOEvals = vecNumberOfMOEvaluationsByOptimizer[optimizerIndex];

            // Write the parameters
            for(const auto parameter : currentBestSolution.param) {
                *populationStream << std::setw(23) << std::scientific << std::setprecision(16) << parameter << " ";
            }

            // Write break
            *populationStream << "   ";

            // Write the fitness
            *populationStream << std::setw(23) << std::scientific << std::setprecision(16) << currentBestSolution.f;

            // Write MO-evals
            *populationStream << " " << std::setw(8) << std::fixed << currentMOEvals;

            // Write end
            *populationStream << std::endl;
        }

    }

    /**
     * Getter of 'number_of_evaluations'
     * @return 'number_of_evaluations'
     */
    int resource_allocation_scheme_process_t::get_number_of_evaluations() {
        return this->numberOfMOEvaluations;
    }

    /**
     * Getter of 'starting_time'
     * @return 'starting_time'
     */
    clock_t resource_allocation_scheme_process_t::get_starting_time() {
        return this->startingTime;
    }


}