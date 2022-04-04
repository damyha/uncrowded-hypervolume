/**
 * UHV-CLASSICAL HYBRID
 * This algorithm uses a fixed schedule that alternates between the EA and gradient algorithm.
 *
 * Implementation by Damy Ha 2021
 * Most code copied from S.C. Maree
 */

// Includes
#include "UHV.hpp"
#include "bezier.hpp"

#include "HillVallEA/hillvallea.hpp"
#include "HillVallEA/hillvallea_internal.hpp"
#include "HillVallEA/sofomore.hpp"
#include "HillVallEA/solution.hpp"
#include "HillVallEA/classical_hybrid/ClassicalHybridProcess.h"
#include "HillVallEA/classical_hybrid/ClassicalHybrid.h"



// for MO problems
//#include "../domination_based_MO_optimization/mohillvallea/hicam_external.h"
#include "../domination_based_MO_optimization/gomea/MO-RV-GOMEA.h"
#include "../benchmark_functions/mo_benchmarks.h"

// Todos left to implement:
// - Check if VTR has been reached currently does not allow partial evaluations.
//   If you want to implement this you also need to take care of the re-evaluation that is done when checking for VTR hits.
//   This extra check is not logged in the statistics file.
// - This executable is currently not able to write populations to population files.

/*************/
/* Variables */
/*************/

// Print settings
bool printVerboseOverview;              // Print the start parameters

// Write settings (user defined)
bool writeStatisticsOptimizers;         // Write statistics for every generation of each optimizer
bool writeSolutionsOptimizers;          // Write solutions for every generation of each optimizer (not implemented yet)

// General algorithm settings (user defined)
bool collect_all_mo_sols_in_archive;    // Use an archive
int problemIndex;                       // Problem index number
size_t numberOfMOParameters;            // Dimension: n
size_t solutionSetSize;                 // Solution set size
double lowerInitializationBound;        // Lower initialization bound
double upperInitializationBound;        // Upper initialization bound
size_t populationSize;                  // The population size
//size_t approximation_set_size;          // Approximation set size
size_t elitistArchiveTargetSize;        // NOT USED: The size of the elitist archive
int maximumNumberOfMOEvaluations;       // Maximum number of Multi objective evaluations
int maximumTimeInSeconds;               // Time limit in seconds
int useVTR;                             // Enable termination when UHV is reached
double valueVTR;                        // UHV value to reach (positive number tuned negative)
int randomSeedNumber;                   // Seed to create random numbers
std::string writeDirectory;             // Output directory that contains the output files

// Specific algorithm settings
bool useFiniteDifferenceApproximation;  // Use finite differences as gradient
bool enableNiching;                     // NOT USED: Enable niching
bool useBezierInterpolation;            // NOT USED: Use bezier interpolation
size_t numberOfReferencePoints;         // NOT USED: Number of reference points for bezier interpolation

int linkageModelUHV_GOMEA;              // The linkage model of UHV-GOMEA
int indexVersionUHV_ADAM;               // The gradient optimizer to use (0:ADAM - 1:GAMO)
int indexApplicationMethodUHV_ADAM;     // Method of applying gradient optimizer on population

// Retrieved variables
int localOptimizerIndexUHV_GOMEA;       // The local optimizer index of UHV-GOMEA
int localOptimizerIndexUHV_ADAM;        // The local optimizer index of UHV-ADAM
hicam::fitness_pt moFitnessFunction;    // The multi objective fitness function

// Settings interleaved multi-start scheme
unsigned int number_of_subgenerations_per_population_factor;   // NOT USED: Multiplication factor of interleaved multi-start scheme
unsigned int maximum_number_of_populations;                     // NOT USED: Maximum number of populations

// Algorithm specific variables
bool scaledSearchVolume;                // Variable that indicates the initial search range

// Problem parameters defined by user
double factorAlpha;     // Alpha value of problem
double factorBeta;      // Beta value of problem
double factorLambda;    // Lambda of problem
double factorFrequency; // Frequency of problem


// Derived Variables
hicam::vec_t moLowerInitRanges, moUpperInitRanges;  // The lower and upper initialization ranges for MO optimization
std::string file_appendix;                                  // Part of the output file name

/**
 * Prints help to the console.
 */
void printHelp() {
    printf("Usage: uhv_classical_hybrid [-?] [-P] [-s] [-w] [-v] [-e] [-r] lmod vgrad mgrad pro dim ssize low upp pop eva sec vtr ndec alpha beta lmbda freq rnd wrp\n"); // [-n] [-f]
    printf(" -?: Prints out this usage information.\n");
    printf(" -P: Prints out a list of all installed optimization problems.\n");
    printf(" -s: Enables computing and writing of statistics for every population branch.\n");
    printf(" -w: Enable writing of solutions and their fitness every generation.\n");
    printf(" -v: Verbose mode. Prints the settings before starting the run + output of generational statistics to terminal.\n");
    // printf(" -n: Enable niching \n");
    printf(" -f: Enforce use of finite differences for gradient-based methods\n");
    printf(" -r: Enables use of vtr (value-to-reach) termination condition based on the hyper volume.\n");
    printf("\n");
    printf(" lmod: Linkage model of UHV-GOMEA (0 = marginal linkage (Lm), 1 = linkage tree (Lt), 2 = full linkage (Lf).\n");
    printf("vgrad: Linkage model of UHV-ADAM (0 = ADAM, 1 = GAMO).\n");
    printf("mgrad: Method of applying UHV-ADAM on a population (0 = Best, 1 = ALL, 2:Best-3).\n");
    printf("  pro: Multi-objective optimization problem index (minimization).\n");
    printf("  dim: Number of parameters (if the problem is configurable).\n");
    printf("ssize: Solution set size (number of solutions on the front).\n");
    printf("  low: Overall initialization lower bound.\n");
    printf("  upp: Overall initialization upper bound.\n");
    printf("  pop: Population size.\n");
    printf("  eva: Maximum number of evaluations of the multi-objective problem allowed.\n");
    printf("  sec: Time limit in seconds.\n");
    printf("  vtr: The value to reach. If the hyper volume of the best feasible solution reaches this value, termination is enforced (if -r is specified).\n");
    printf("alpha: Alpha of problem.\n");
    printf(" beta: Beta of problem.\n");
    printf("lmbda: Lambda of problem.\n");
    printf(" freq: Frequency of problem.\n");
    printf("  rnd: Random seed.\n");
    printf("  wrp: write path.\n");
}

/**
 * Returns the problems installed.
 */
void printAllInstalledProblems() {
    int i = 0;
    hicam::fitness_pt objective = getObjectivePointer(i);

    std::cout << "Installed optimization problems:\n";

    // Print problems
    while (objective != nullptr) {
        std::cout << std::setw(3) << i << ": " << objective->name() << std::endl;

        i++;
        objective = getObjectivePointer(i);
    }
}


/**
 * Print parameters of run
 * @param singleObjectiveFitnessFunction The MO-objective function as SO-objective function
 */
void printComputationSettings(const hillvallea::fitness_pt &singleObjectiveFitnessFunction) {
    std::cout << "Problem settings:" << std::endl;
    std::cout << "\tfunction_name = " << moFitnessFunction->name() << std::endl;
    std::cout << "\tproblem_index = " << problemIndex << std::endl;
    std::cout << "\tmo_number_of_parameters = " << numberOfMOParameters << std::endl;
    std::cout << "\tinit_range = [" << lowerInitializationBound << ", " << upperInitializationBound << "]" << std::endl;
    std::cout << "\tHV reference point = " << moFitnessFunction->hypervolume_max_f0 << ", "
              << moFitnessFunction->hypervolume_max_f1 << std::endl;

    std::cout << "Run settings:" << std::endl;
    std::cout << "\tmax_number_of_MO_evaluations = " << maximumNumberOfMOEvaluations << std::endl;
    std::cout << "\tmaximum_number_of_seconds = " << maximumTimeInSeconds << std::endl;
    std::cout << "\tuse_vtr = " << useVTR << std::endl;
    std::cout << "\tvtr = " << valueVTR << std::endl;

    std::cout << "Optimizer settings:" << std::endl;
    std::cout << "\tUHV-CLASSICAL_HYBRID" << std::endl;
    std::cout << "\tUHV-GOMEA-local_optimizer_index = " << localOptimizerIndexUHV_GOMEA << std::endl;
    std::cout << "\tUHV-ADAM-local_optimizer_index = " << localOptimizerIndexUHV_ADAM << std::endl;
    std::cout << "\tUHV-ADAM-application_method = " << indexApplicationMethodUHV_ADAM << std::endl;
    std::cout << "\tsolution_set_size = " << solutionSetSize << std::endl;
    std::cout << "\tso_number_of_parameters = " << singleObjectiveFitnessFunction->number_of_parameters << std::endl;
    std::cout << "\tuse_finite_differences = " << (useFiniteDifferenceApproximation ? "Yes" : "No") << std::endl;
    std::cout << "\tpopulation_size = " << populationSize << std::endl;
    std::cout << "\tenable_niching = " << (enableNiching ? "Yes" : "No") << std::endl;
    std::cout << "\trandom_seed = " << randomSeedNumber << std::endl;
}

void printComputationResults(const hillvallea::classicalHybridProcess_pt process) {
    std::cout << "Best:" << std::endl;
    std::cout << "\tFitness = " << std::fixed << std::setprecision(14) << -process->uhvClassicalHybrid->best.f
              << std::endl;
    std::cout << "\tMO-fevals = " << process->uhvFitnessFunction->number_of_mo_evaluations << std::endl;
    std::cout << "\truntime = " << double(clock() - process->get_starting_time()) / CLOCKS_PER_SEC << " sec"
              << std::endl;

    std::cout << "pareto_front" << (useBezierInterpolation ? numberOfReferencePoints : 0) << " = [";
    for (auto &mo_test_sol : process->uhvClassicalHybrid->best.mo_test_sols) {
        std::cout << "\n\t" << std::fixed << std::setw(10) << std::setprecision(4) << mo_test_sol->obj;
    }
    std::cout << " ];\n";

    std::cout << "pareto_set" << (useBezierInterpolation ? numberOfReferencePoints : 0) << " = [";
    for (auto &mo_test_sol : process->uhvClassicalHybrid->best.mo_test_sols) {
        std::cout << "\n\t";
        for (size_t i = 0; i < mo_test_sol->param.size(); ++i) {
            std::cout << std::fixed << std::setw(10) << std::setprecision(4) << mo_test_sol->param[i] << " ";
        }
    }
    std::cout << " ];\n";
}

/**
 * Writes the parameters of a run to a file
 */
void writeParametersFile() {
    // Determine parameters file path
    std::string filepath = writeDirectory + "parameters.txt";

    // Create parameters file
    std::ofstream parameters_file;
    parameters_file.open(filepath, std::ofstream::out | std::ofstream::trunc);

    // Write parameters to file
    std::string text = "";
    text += "Algorithm:UHV-Classical_Hybrid, ";
    text += "EAModel:" + std::to_string(linkageModelUHV_GOMEA) + ", ";
    text += "GradientModel:" + std::to_string(indexVersionUHV_ADAM) + ", ";
    text += "GradientApplication:" + std::to_string(indexApplicationMethodUHV_ADAM) + ", ";
    text += "FiniteDifferenceApproximation:" + std::to_string(useFiniteDifferenceApproximation) + ", ";
    text += "ProblemIndex:" + std::to_string(problemIndex) + ", ";
    text += "ProblemDimension:" + std::to_string(numberOfMOParameters) + ", ";
    text += "ssize:" + std::to_string(solutionSetSize) + ", ";
    text += "lb:" + std::to_string(lowerInitializationBound) + ", ";
    text += "ub:" + std::to_string(upperInitializationBound) + ", ";
    text += "popsize:" + std::to_string(populationSize) + ", ";
    text += "maxevals:" + std::to_string(maximumNumberOfMOEvaluations) + ", ";
    text += "seed:" + std::to_string(randomSeedNumber) + ", ";
    text += "alpha:" + std::to_string(factorAlpha) + ", ";
    text += "beta:" + std::to_string(factorBeta) + ", ";
    text += "lambda:" + std::to_string(factorLambda) + ", ";
    text += "frequency:" + std::to_string(factorFrequency);
    parameters_file << text << std::endl;

    // Close file
    parameters_file.close();
}

/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError(char **argv, int index) {
    printf("Illegal option: %s\n\n", argv[index]);
    printHelp();
}


/**
 * Initializes the options variables to the default value
 */
void initializeOptions() {
    // Print settings
    printVerboseOverview = false;

    // Write settings
    writeStatisticsOptimizers = false;
    writeSolutionsOptimizers = false;

    // General algorithm settings
    useVTR = 0;
    collect_all_mo_sols_in_archive = false;

    // Algorithm specific settings
    enableNiching = false;
    useFiniteDifferenceApproximation = false;
}

/**
 * Depending on the problem the user has selected, some problem parameters must be disabled.
 */
void cleanProblemParameters() {
    if (problemIndex == 35) {
        // Check problem 35
        factorLambda = NAN;
    } else {
        // Other problems
        factorAlpha = NAN;
        factorBeta = NAN;
        factorLambda = NAN;
        factorFrequency = NAN;
    }
}

/**
 * Given the command line arguments, the option values are retrieved.
 * @param argc The number of arguments (int)
 * @param argv The argument vector (char **)
 * @param index A pointer to the current input to investigate. This pointer is later reused to read parameters (it skips options).
 * @return Boolean if the program should stop after printing help
 */
bool parseOptions(int argc, char **argv, int *index) {
    // Initialize variables
    bool quitProgram = false;   // Quit program after printing help?
    double dummy;

    // Initialize options to default value
    initializeOptions();

    // Loop over input
    for (; (*index) < argc; (*index)++) {
        // Check for option flag
        if (argv[*index][0] == '-') {
            // Check if negative number (which means it's not an option)
            if (sscanf(argv[*index], "%lf", &dummy) && argv[*index][1] != '\0')
                break;

            if (argv[*index][1] == '\0') {
                // Check if flag not empty
                optionError(argv, *index);
                throw std::runtime_error("'-' detected, but no character found. ");
            } else if (argv[*index][2] != '\0') {
                // Check if flag only single character
                optionError(argv, *index);
                throw std::runtime_error("The flag is too long. Detected: " + std::to_string(argv[*index][2]) + ".");
            } else {
                // Go over options
                switch (argv[*index][1]) {
                    case '?':
                        printHelp();
                        quitProgram = true;
                        break;
                    case 'P':
                        printAllInstalledProblems();
                        quitProgram = true;
                        break;
                    case 's':
                        writeStatisticsOptimizers = true;
                        break;
                    case 'w':
                        writeSolutionsOptimizers = true;
                        printf("-w Not implemented yet.\n");
                        break;
                    case 'v':
                        printVerboseOverview = true;
                        break;
//                    case 'n': enable_niching = true; break;
//                    case 'e': collect_all_mo_sols_in_archive = true; break;
                    case 'f':
                        useFiniteDifferenceApproximation = true;
                        break;
                    case 'r':
                        useVTR = 1;
                        break; // HV-based vtr (note, use_vtr = 2 is an IGD-based VTR)
                    default:
                        optionError(argv, *index);
                        printHelp();
                        throw std::runtime_error("Unknown flag detected: " + std::to_string(argv[*index][1]) + ".");
                }
            }
        } else
            // No more options
            break;
    }

    return quitProgram;
}

/**
 * Given the command line arguments, the parameters values are retrieved.
 * @param argc The number of arguments (int)
 * @param argv The argument vector (char **)
 * @param index A pointer to the current input to investigate. It assumes that options were already read.
 */
void parseParameters(int argc, char **argv, const int *index) {
    // Initialize Constants
    int totalNumberOfParameters = 18;

    // Check for correct number of parameters
    if ((argc - *index) != totalNumberOfParameters) {
        printHelp();
        throw std::runtime_error("Number of parameters is incorrect, require " +
                                 std::to_string(totalNumberOfParameters) + " parameters (you provided " +
                                 std::to_string(argc - *index) + ").\n\n");
    }

    // Todo: Output specific errors?
    int noError = 1;
    noError = noError && sscanf(argv[*index + 0], "%d", &linkageModelUHV_GOMEA);
    noError = noError && sscanf(argv[*index + 1], "%d", &indexVersionUHV_ADAM);
    noError = noError && sscanf(argv[*index + 2], "%d", &indexApplicationMethodUHV_ADAM);
    noError = noError && sscanf(argv[*index + 3], "%d", &problemIndex);
    noError = noError && sscanf(argv[*index + 4], "%zd", &numberOfMOParameters);
    noError = noError && sscanf(argv[*index + 5], "%zd", &solutionSetSize);
    noError = noError && sscanf(argv[*index + 6], "%lf", &lowerInitializationBound);
    noError = noError && sscanf(argv[*index + 7], "%lf", &upperInitializationBound);
    noError = noError && sscanf(argv[*index + 8], "%zd", &populationSize);
    noError = noError && sscanf(argv[*index + 9], "%d", &maximumNumberOfMOEvaluations);
    noError = noError && sscanf(argv[*index + 10], "%d", &maximumTimeInSeconds);
    noError = noError && sscanf(argv[*index + 11], "%lf", &valueVTR);
    noError = noError && sscanf(argv[*index + 12], "%lf", &factorAlpha);        // Problem setting
    noError = noError && sscanf(argv[*index + 13], "%lf", &factorBeta);         // Problem setting
    noError = noError && sscanf(argv[*index + 14], "%lf", &factorLambda);       // Problem setting
    noError = noError && sscanf(argv[*index + 15], "%lf", &factorFrequency);    // Problem setting
    noError = noError && sscanf(argv[*index + 16], "%d", &randomSeedNumber);
    writeDirectory = argv[*index + 17];

    // Clean up the problem parameters
    cleanProblemParameters();

    if (!noError) {
        printHelp();
        throw std::runtime_error("An error occurred when parameters were parsed.");
    }
}

/**
 * Assigns the problem parameters to the problems
 */
void assignParametersToProblems() {
    if (hicam::biExpCosineDecay *c = dynamic_cast<hicam::biExpCosineDecay *>(moFitnessFunction.get())) {
        c->set_alpha(factorAlpha);
        c->set_beta(factorBeta);
        c->set_lambda(factorLambda);
        c->set_frequency(factorFrequency);
    } else if (hicam::biConvexSphereMinSphereCosine *c = dynamic_cast<hicam::biConvexSphereMinSphereCosine *>(moFitnessFunction.get())) {
        c->set_alpha(factorAlpha);
        c->set_beta(factorBeta);
        c->set_lambda(factorLambda);
        c->set_frequency(factorFrequency);
    }
}

/**
 * Checks the values of the options
 */
void checkOptions() {
    // Determine UHV-GOMEA local optimizer index
    switch (linkageModelUHV_GOMEA) {
        case 0:
            // Marginal linkage
            localOptimizerIndexUHV_GOMEA = 64;
            break;
        case 1:
            // Linkage tree
            localOptimizerIndexUHV_GOMEA = 66;
            break;
        case 2:
            // Full linkage
            localOptimizerIndexUHV_GOMEA = 50;
            break;
        default: {
            throw std::runtime_error(std::string("Error: GOMEA's linkage model is invalid (read: ") +
                                     std::to_string(linkageModelUHV_GOMEA) + std::string(").\n"));
        }
    }

    // Determine Gradient Algorithm local optimizer index
    switch (indexVersionUHV_ADAM) {
        case 0:
            // ADAM
            localOptimizerIndexUHV_ADAM = 50;
            break;
        case 1:
            // GAMO
            localOptimizerIndexUHV_ADAM = 64;
            break;
        case 2:
            // Plain gradient
            localOptimizerIndexUHV_ADAM = 80;
            break;
        case 3:
            // Plain gradient
            localOptimizerIndexUHV_ADAM = 81;
            break;
        default: {
            throw std::runtime_error(std::string("The gradient algorithm's version is invalid (read: ") +
                                     std::to_string(indexVersionUHV_ADAM) + ").\n");
        }
    }

    // Check Gradient Algorithm application method
    if (indexApplicationMethodUHV_ADAM != 0 &&
        indexApplicationMethodUHV_ADAM != 1 &&
        indexApplicationMethodUHV_ADAM != 2 &&
        indexApplicationMethodUHV_ADAM != 3) {
        throw std::runtime_error("Unknown gradient algorithm application method detected: " +
                                 std::to_string(indexApplicationMethodUHV_ADAM) + ".\n");
    }

    // Retrieve fitness function and check if problem exists
    moFitnessFunction = getObjectivePointer(problemIndex);
    if (moFitnessFunction == nullptr) {
        throw std::runtime_error(std::string("Error: Could not find a fitness function associated with index: "
                                             + std::to_string(problemIndex) + ").\n"));
    }

    // Set dimension 'n' of fitness function
    if (numberOfMOParameters <= 0) {
        // Only occurs when numberOfMOParameters == 0 since mo_number_of_parameters is size_t
        throw std::runtime_error("Error: number of MO parameters <= 0 (read: " +
                                 std::to_string(numberOfMOParameters) +
                                 "). Require MO number of parameters >= 1.\n");
    } else {
        moFitnessFunction->set_number_of_parameters(numberOfMOParameters);
    }

    // Set parameters of problem
    assignParametersToProblems();

    // Check solution set size is valid
    if (solutionSetSize <= 0) {
        // Only occurs when solution_set_size == 0 since solution_set_size is size_t
        throw std::runtime_error("Error: solution set size <= 0 (read: " +
                                 std::to_string(solutionSetSize) + "). Require >= 1.");
    }

    // Deactivate unused parameters set size and change elitist archive settings
    collect_all_mo_sols_in_archive = false;
    elitistArchiveTargetSize = 0;
//    approximation_set_size = 0;

    // Set the SO initialization ranges the initital initialization ranges per dimension
    moLowerInitRanges.resize(moFitnessFunction->number_of_parameters, lowerInitializationBound);
    moUpperInitRanges.resize(moFitnessFunction->number_of_parameters, upperInitializationBound);

    // Check if ranges are logical if non random initialization is used
    if (!moFitnessFunction->redefine_random_initialization) {
        if (lowerInitializationBound >= upperInitializationBound) {
            throw std::runtime_error("Error: init range invalid (read lower " +
                                     std::to_string(lowerInitializationBound) +
                                     ", upper, " + std::to_string(upperInitializationBound) + ")");
        }
    }

    // Set random seed
    if (randomSeedNumber < 0) {
        randomSeedNumber = (int) clock();
    }

    // Determine output file name root
    std::stringstream ss;
    ss << "_UHV-CLASSICAL_HYBRID" << linkageModelUHV_GOMEA
       << "_problem" << problemIndex
       << "_p" << solutionSetSize
       << "_run" << std::setw(3) << std::setfill('0') << randomSeedNumber;
    file_appendix = ss.str();
}

/**
 * Initialize the algorithm specific variables
 */
void initializeAlgorithmSpecificVariables() {
    // Enable bezier representation of solutions sets (see publication S.C. Maree et al., PPSN2020, BezEA)
    useBezierInterpolation = false;
    numberOfReferencePoints = solutionSetSize;

    // Retrieve pareto set
    moFitnessFunction->get_pareto_set();

    // Set interleaved multi-start scheme settings
    number_of_subgenerations_per_population_factor = 2;
    maximum_number_of_populations = 1;
}

/**
 * Initialize the settings of the algorithm
 * @param argc The number of arguments (int)
 * @param argv The argument vector (char **)
 * @return bool if process was successfully completed
 */
bool initializeProcess(int argc, char **argv) {

    bool completedSuccessfully = false;

    try {
        int index = 1;

        // Parse options
        bool quitProgram = parseOptions(argc, argv, &index);

        // Check if options causes the program to stop
        if (!quitProgram) {
            // Parse parameters
            parseParameters(argc, argv, &index);

            // Check the options
            checkOptions();

            // Initialize remaining variables
            initializeAlgorithmSpecificVariables();

            completedSuccessfully = true;
        }
    } catch (std::exception &exception) {
        std::cerr << exception.what() << std::endl;
    }

    return completedSuccessfully;
}


int main(int argc, char **argv) {
    // Initialize process variables
    bool varsInitialized = initializeProcess(argc, argv);

    // Check if process is successfully initialized
    if (varsInitialized) {
        // Initialize as single objective problem
        std::shared_ptr <hillvallea::UHV_t> uhvFitnessFunction;
        uhvFitnessFunction = std::make_shared<hillvallea::UHV_t>(moFitnessFunction, solutionSetSize,
                                                                 collect_all_mo_sols_in_archive,
                                                                 elitistArchiveTargetSize, nullptr,
                                                                 useFiniteDifferenceApproximation);

        // Create lower and upper bounds for SO fitness function
        hillvallea::vec_t soLowerInitRanges, soUpperInitRanges;
        soLowerInitRanges.resize(uhvFitnessFunction->number_of_parameters, lowerInitializationBound);
        soUpperInitRanges.resize(uhvFitnessFunction->number_of_parameters, upperInitializationBound);

        // Check if IGD-based VTR should be used
        if (useVTR == 2) uhvFitnessFunction->redefine_vtr = true;

        // Change sign of VTR due to minimizing HV(X)
        valueVTR *= -1;

        // Print Computation Settings
        if (printVerboseOverview) printComputationSettings(uhvFitnessFunction);

        // Initialize the optimization process
        hillvallea::classicalHybridProcess_pt process;
        process = std::make_shared<hillvallea::ClassicalHybridProcess>(
                uhvFitnessFunction,
                localOptimizerIndexUHV_GOMEA,
                localOptimizerIndexUHV_ADAM,
                indexApplicationMethodUHV_ADAM,
                uhvFitnessFunction->number_of_parameters,
                solutionSetSize,
                soLowerInitRanges,
                soUpperInitRanges,
                maximumNumberOfMOEvaluations,
                maximumTimeInSeconds,
                valueVTR,
                useVTR,
                randomSeedNumber,
                writeSolutionsOptimizers,
                writeStatisticsOptimizers,
                writeDirectory,
                file_appendix);


        // Execute optimization
        process->run_optimization(populationSize);

        // Write parameters file
        writeParametersFile();

        // Print Computation Results
        if (printVerboseOverview)
            printComputationResults(process);

    }


    printf("Finished executing\n");   // Todo: Remove this line.
    std::exit(0);

}