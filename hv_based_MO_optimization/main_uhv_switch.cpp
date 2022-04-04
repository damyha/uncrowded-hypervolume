/**
 * UHV-SWITCH
 * This algorithm applies UHV-GOMEA until a certain UHV threshold is met, after which a gradient algorithm is applied.
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
#include "HillVallEA/UHVSWITCH//uhvSwitch.h"
#include "HillVallEA/UHVSWITCH//uhvSwitchProcess.h"



// for MO problems
//#include "../domination_based_MO_optimization/mohillvallea/hicam_external.h"
#include "../domination_based_MO_optimization/gomea/MO-RV-GOMEA.h"
#include "../benchmark_functions/mo_benchmarks.h"


/*************/
/* Variables */
/*************/

// Print settings
bool print_verbose_overview;

// Write settings (user defined)
bool write_statistics_optimizers;       // Write statistics for every generation of each optimizer
bool write_solutions_optimizers;        // Write solutions for every generation of each optimizer

// General algorithm settings (user defined)
bool collect_all_mo_sols_in_archive;    // NOT USED: Use an archive
int problem_index;                      // Problem index number
size_t mo_number_of_parameters;         // Dimension: n
size_t solution_set_size;               // Solution set size
double lower_init;                      // Lower initialization bound
double upper_init;                      // Upper initialization bound
size_t population_size;                 // The population size
size_t approximation_set_size;          // Approximation set size
size_t elitist_archive_size_target;     // NOT USED: The size of the elitist archive
int maximum_number_of_mo_evaluations;   // Maximum number of Multi objective evaluations
int maximum_number_of_so_evaluations;   // Maximum number of Single objective evaluations derived from Max MO-evaluations
int maximum_number_of_seconds;          // Time limit in seconds
int use_vtr;                            // Enable termination when UHV is reached
double value_to_reach;                  // UHV value to reach
int random_seed;                        // Seed to create random numbers
std::string write_directory;            // Output directory that contains the output files

// Specific algorithm settings
bool enable_niching;                    // NOT USED: Enable niching
int linkage_model_index_UHVGOMEA;       // The linkage model of UHV-GOMEA
int versionUHVADAM;                     // The version of the gradient algorithm
int indexApplicationMethodUHV_ADAM;     // The application method of UHV-ADAM
bool use_finite_differences;            // Use finite differences as gradient
bool use_bezier_interpolation;          // NOT USED: Use bezier interpolation
size_t number_of_reference_points;      // NOT USED: Number of reference points for bezier interpolation
double uhvSwitchValue;                  // UHV value to switch from UHV-GOMEA to the gradient algorithm

// Retrieved variables
int local_optimizer_index_gomea;        // The local optimizer index of UHV-GOMEA
int localOptimizerIndexUHVADAM;         // The local optimizer index of UHV-ADAM
hicam::fitness_pt mo_fitness_function;  // The multi objective fitness function

// Settings interleaved multi-start scheme
unsigned int number_of_subgenerations_per_population_factor ;   // NOT USED: Multiplication factor of interleaved multi-start scheme
unsigned int maximum_number_of_populations;                     // NOT USED: Maximum number of populations

// Optimalization parameters
bool scaled_search_volume;  // Todo: Find out what this does

// Problem parameters defined by user
double f_alpha;         // Alpha value of problem
double f_beta;          // Beta value of problem
double f_lambda;        // Lambda of problem
double f_frequency;     // Frequency of problem


// Derived Variables
hicam::vec_t mo_lower_init_ranges, mo_upper_init_ranges;    // The lower and upper ranges per dimension
std::string file_appendix;                                  // Part of the output file name

/**
 * Prints help to the console.
 */
void printHelp()
{
    printf("Usage: uhv_switch [-?] [-P] [-s] [-w] [-v] [-e] [-r] lmod1 vgrad pro dim ssize low upp pop eva sec vtr swtch alpha beta lmbda freq rnd wrp\n"); // [-n] [-f]
    printf(" -?: Prints out this usage information.\n");
    printf(" -P: Prints out a list of all installed optimization problems.\n");
    printf(" -s: Enables computing and writing of statistics for every population branch.\n");
    printf(" -w: Enable writing of solutions and their fitness every generation.\n");
    printf(" -v: Verbose mode. Prints the settings before starting the run + output of generational statistics to terminal.\n");
    // printf(" -n: Enable niching \n");
    printf(" -f: Enforce use of finite differences for gradient-based methods\n");
    printf(" -r: Enables use of vtr (value-to-reach) termination condition based on the hyper volume.\n");
    printf("\n");
    printf("lmod1: Linkage model of UHV-GOMEA (0 = marginal linkage (Lm), 1 = linkage tree (Lt), 2 = full linkage (Lf).\n");
    printf("vgrad: Linkage model of UHV-ADAM (0 = ADAM, 1 = GAMO.\n");
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
    printf("swtch: UHV value to switch from UHV-GOMEA to the gradient.\n");
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
void printAllInstalledProblems()
{
    int i = 0;
    hicam::fitness_pt objective = getObjectivePointer(i);

    std::cout << "Installed optimization problems:\n";

    // Print problems
    while(objective != nullptr)
    {
        std::cout << std::setw(3) << i << ": " << objective->name() << std::endl;

        i++;
        objective = getObjectivePointer(i);
    }
}


/**
 * Print parameters
 * @param single_objective_fitness_function The MO-objective function as SO-objective function
 */
void printComputationSettings(const hillvallea::fitness_pt& single_objective_fitness_function)
{
    std::cout << "Problem settings:" << std::endl;
    std::cout << "\tfunction_name = " << mo_fitness_function->name() << std::endl;
    std::cout << "\tproblem_index = " << problem_index << std::endl;
    std::cout << "\tmo_number_of_parameters = " << mo_number_of_parameters << std::endl;
    std::cout << "\tinit_range = [" << lower_init << ", " << upper_init << "]" << std::endl;
    std::cout << "\tHV reference point = " << mo_fitness_function->hypervolume_max_f0 << ", " << mo_fitness_function->hypervolume_max_f1 << std::endl;

    std::cout << "Run settings:" << std::endl;
    std::cout << "\tmax_number_of_MO_evaluations = " << maximum_number_of_mo_evaluations << std::endl;
    std::cout << "\tmaximum_number_of_seconds = " << maximum_number_of_seconds << std::endl;
    std::cout << "\tuse_vtr = " << use_vtr << std::endl;
    std::cout << "\tvtr = " << value_to_reach << std::endl;

    std::cout << "Optimizer settings:" << std::endl;
    std::cout << "\tUHV-SWITCH" << std::endl;
    std::cout << "\tUHV-GOMEA-local_optimizer_index = " << local_optimizer_index_gomea << std::endl;
    std::cout << "\tUHV-ADAM-version = " << versionUHVADAM << std::endl;
    std::cout << "\tUHV-ADAM-application = " << indexApplicationMethodUHV_ADAM << std::endl;
    std::cout << "\tsolution_set_size = " << solution_set_size << std::endl;
    std::cout << "\tso_number_of_parameters = " << single_objective_fitness_function->number_of_parameters << std::endl;
    std::cout << "\tuse_finite_differences = " << (use_finite_differences ? "yes" : "no") << std::endl;
    std::cout << "\tpopsize = " << population_size << std::endl;
    std::cout << "\tswitching value = " << uhvSwitchValue << std::endl;
    std::cout << "\tenable_niching = " << (enable_niching ? "yes" : "no") << std::endl;
    std::cout << "\trandom_seed = " << random_seed << std::endl;
}

void printComputationResults(const hillvallea::UHVSWITCHProcess_pt process)
{
    std::cout << "Best:" << std::endl;
    std::cout << "\tFitness = " << std::fixed << std::setprecision(14) << -process->uhv_switch->best.f << std::endl;
    std::cout << "\tMO-fevals = " << process->fitness_function->number_of_mo_evaluations << std::endl;
    std::cout << "\truntime = " << double(clock() - process->get_starting_time()) / CLOCKS_PER_SEC << " sec" << std::endl;

    std::cout << "pareto_front" << (use_bezier_interpolation ? number_of_reference_points : 0) << " = [";
    for(auto & mo_test_sol : process->uhv_switch->best.mo_test_sols) {
        std::cout << "\n\t" << std::fixed << std::setw(10) << std::setprecision(4) << mo_test_sol->obj;
    }
    std::cout << " ];\n";

    std::cout << "pareto_set" << (use_bezier_interpolation ? number_of_reference_points : 0) << " = [";
    for(auto & mo_test_sol : process->uhv_switch->best.mo_test_sols)
    {
        std::cout << "\n\t";
        for(size_t i = 0; i < mo_test_sol->param.size(); ++i) {
            std::cout << std::fixed << std::setw(10) << std::setprecision(4) << mo_test_sol->param[i] << " ";
        }
    }
    std::cout << " ];\n";
}

/**
 * Writes the parameters of a run to a file
 */
void writeParametersFile()
{
    // Determine parameters file path
    std::string filepath = write_directory + "parameters.txt";

    // Create parameters file
    std::ofstream parameters_file;
    parameters_file.open(filepath, std::ofstream::out | std::ofstream::trunc);

    // Write parameters to file
    std::string text = "";
    text += "Algorithm:UHV-SWITCH, ";
    text += "EAModel:" + std::to_string(linkage_model_index_UHVGOMEA) + ", ";
    text += "GradientModel:" + std::to_string(versionUHVADAM) + ", ";
    text += "GradientApplication:" + std::to_string(indexApplicationMethodUHV_ADAM) + ", ";
    text += "FiniteDifferenceApproximation:" + std::to_string(use_finite_differences) + ", ";
    text += "ProblemIndex:" + std::to_string(problem_index) + ", ";
    text += "ProblemDimension:" + std::to_string(mo_number_of_parameters) + ", ";
    text += "ssize:" + std::to_string(solution_set_size) + ", ";
    text += "lb:" + std::to_string(lower_init) + ", ";
    text += "ub:" + std::to_string(upper_init) + ", ";
    text += "popsize:" + std::to_string(population_size) + ", ";
    text += "maxevals:" + std::to_string(maximum_number_of_mo_evaluations) + ", ";
    text += "SwitchValue:" + std::to_string(uhvSwitchValue) + ", ";
    text += "seed:" + std::to_string(random_seed) + ", ";
    text += "alpha:" + std::to_string(f_alpha) + ", ";
    text += "beta:" + std::to_string(f_beta) + ", ";
    text += "lambda:" + std::to_string(f_lambda) + ", ";
    text += "frequency:" + std::to_string(f_frequency);
    parameters_file << text << std::endl;

    // Close file
    parameters_file.close();
}

/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError(char **argv, int index)
{
    printf("Illegal option: %s\n\n", argv[index]);
    printHelp();
}


/**
 * Initializes the options variables to the default value
 */
void initializeOptions(){
    // Print settings
    print_verbose_overview = false;

    // Write settings
    write_statistics_optimizers = false;
    write_solutions_optimizers = false;

    // General algorithm settings
    use_vtr = 0;
    collect_all_mo_sols_in_archive = false;

    // Algorithm specific settings
    enable_niching = false;
    use_finite_differences = false;
}

/**
 * Depending on the problem the user has selected, some problem parameters must be disabled.
 */
void cleanProblemParameters()
{
    if (problem_index == 35)
    {
        // Check problem 35
        f_lambda = NAN;
    } else {
        // Other problems
        f_alpha = NAN;
        f_beta = NAN;
        f_lambda = NAN;
        f_frequency = NAN;
    }
}

/**
 * Given the command line arguments, the option values are retrieved.
 * @param argc The number of arguments (int)
 * @param argv The argument vector (char **)
 * @param index A pointer to the current input to investigate. This pointer is later reused to read parameters (it skips options).
 * @return Boolean if the program should stop after printing help
 */
bool parseOptions(int argc, char **argv, int *index)
{
    bool quit_program = false;   // Quit program after printing help?
    double dummy;

    // Initialize options to default value
    initializeOptions();

    // Loop over input
    for (; (*index) < argc; (*index)++)
    {
        // Check for option flag
        if (argv[*index][0] == '-')
        {
            // Check if negative number (which means it's not an option)
            if (sscanf(argv[*index], "%lf", &dummy) && argv[*index][1] != '\0')
                break;

            // Check if flag not empty
            if (argv[*index][1] == '\0'){
                optionError(argv, *index);
                throw std::runtime_error("'-' detected, but no character found. ");
            }
            // Check if flag only single character
            else if (argv[*index][2] != '\0')
            {
                optionError(argv, *index);
                throw std::runtime_error("The flag is too long. Detected: " + std::to_string(argv[*index][2]) + ".");
            }
            else
            {
                // Go over options
                switch (argv[*index][1])
                {
                    case '?': printHelp(); quit_program=true; break;
                    case 'P': printAllInstalledProblems(); quit_program=true; break;
                    case 's': write_statistics_optimizers = true; break;
                    case 'w': write_solutions_optimizers = true; printf("-w Not implemented yet.\n"); break;
                    case 'v': print_verbose_overview = true; break;
//                    case 'n': enable_niching = true; break;
//                    case 'e': collect_all_mo_sols_in_archive = true; break;
                    case 'f': use_finite_differences = true; break;
                    case 'r': use_vtr = 1; break; // HV-based vtr (note, use_vtr = 2 is an IGD-based VTR)
                    default: optionError(argv, *index);
                        printHelp();
                        throw std::runtime_error("Unknown flag detected: " + std::to_string(argv[*index][1]) + ".");
                }
            }
        }
        else
            // No more options
            break;
    }

    return quit_program;
}

/**
 * Given the command line arguments, the parameters values are retrieved.
 * @param argc The number of arguments (int)
 * @param argv The argument vector (char **)
 * @param index A pointer to the current input to investigate. It assumes that options were already read.
 */
void parseParameters(int argc, char **argv, const int *index)
{
    int n_params = 19;

    // Check for correct number of parameters
    if ((argc - *index) != n_params)
    {
        printHelp();
        throw std::runtime_error("Number of parameters is incorrect, require " +
            std::to_string(n_params)+ " parameters (you provided " + std::to_string(argc - *index) + ").\n\n");
    }

    // Todo: Output specific errors?
    int noError = 1;
    noError = noError && sscanf(argv[*index + 0], "%d", &linkage_model_index_UHVGOMEA);
    noError = noError && sscanf(argv[*index + 1], "%d", &versionUHVADAM);
    noError = noError && sscanf(argv[*index + 2], "%d", &indexApplicationMethodUHV_ADAM);
    noError = noError && sscanf(argv[*index + 3], "%d", &problem_index);
    noError = noError && sscanf(argv[*index + 4], "%zd", &mo_number_of_parameters);
    noError = noError && sscanf(argv[*index + 5], "%zd", &solution_set_size);
    noError = noError && sscanf(argv[*index + 6], "%lf", &lower_init);
    noError = noError && sscanf(argv[*index + 7], "%lf", &upper_init);
    noError = noError && sscanf(argv[*index + 8], "%zd", &population_size);
    noError = noError && sscanf(argv[*index + 9], "%d", &maximum_number_of_mo_evaluations);
    noError = noError && sscanf(argv[*index + 10], "%d", &maximum_number_of_seconds);
    noError = noError && sscanf(argv[*index + 11], "%lf", &value_to_reach);
    noError = noError && sscanf(argv[*index + 12], "%lf", &uhvSwitchValue);
    noError = noError && sscanf(argv[*index + 13], "%lf", &f_alpha);        // Problem setting
    noError = noError && sscanf(argv[*index + 14], "%lf", &f_beta);         // Problem setting
    noError = noError && sscanf(argv[*index + 15], "%lf", &f_lambda);       // Problem setting
    noError = noError && sscanf(argv[*index + 16], "%lf", &f_frequency);    // Problem setting
    noError = noError && sscanf(argv[*index + 17], "%d", &random_seed);
    write_directory = argv[*index + 18];

    // Clean up the problem parameters
    cleanProblemParameters();

    if (!noError)
    {
        printHelp();
        throw std::runtime_error("An error occurred when parameters were parsed.");
    }
}

/**
 * Assigns the problem parameters to the problems
 */
void assignParametersToProblems()
{
    if (hicam::biExpCosineDecay* c = dynamic_cast<hicam::biExpCosineDecay*>(mo_fitness_function.get()))
    {
        c->set_alpha(f_alpha);
        c->set_beta(f_beta);
        c->set_lambda(f_lambda);
        c->set_frequency(f_frequency);
    }
    else if (hicam::biConvexSphereMinSphereCosine* c = dynamic_cast<hicam::biConvexSphereMinSphereCosine*>(mo_fitness_function.get()))
    {
        c->set_alpha(f_alpha);
        c->set_beta(f_beta);
        c->set_lambda(f_lambda);
        c->set_frequency(f_frequency);
    }
}

/**
 * Checks the values of the options
 */
void checkOptions()
{
    // Determine UHV-GOMEA local optimizer index
    switch (linkage_model_index_UHVGOMEA) {
            case 0: local_optimizer_index_gomea = 64; break; // marginal linkage
            case 1: local_optimizer_index_gomea = 66; break; // linkage tree
            case 2: local_optimizer_index_gomea = 50; break; // full linkage
        default:{
            throw std::runtime_error(std::string("Error: UHV-GOMEA's linkage model is invalid (read: ") +
                std::to_string(linkage_model_index_UHVGOMEA) + std::string(").\n"));
        }
    }

    // Determine Gradient Algorithm local optimizer index
    switch (versionUHVADAM) {
        case 0: localOptimizerIndexUHVADAM = 50; break; // ADAM
        case 1: localOptimizerIndexUHVADAM = 64; break; // GAMO
        case 2: localOptimizerIndexUHVADAM = 80; break; // PLAIN GRADIENT
        case 3: localOptimizerIndexUHVADAM = 81; break; // LINE SEARCH APPROXIMATION
        default:{
            throw std::runtime_error(std::string("Error: The gradient algorithm's version is invalid (read: ") +
                                     std::to_string(versionUHVADAM) + std::string(").\n"));
        }
    }

    // Check Gradient Algorithm application method
    if (indexApplicationMethodUHV_ADAM != 0 &&
        indexApplicationMethodUHV_ADAM != 1 &&
        indexApplicationMethodUHV_ADAM != 2 &&
        indexApplicationMethodUHV_ADAM != 3)
    {
        throw std::runtime_error("Unknown gradient algorithm application method detected: " +
                                 std::to_string(indexApplicationMethodUHV_ADAM) + ".\n");
    }

    // Retrieve fitness function and check if problem exists
    mo_fitness_function = getObjectivePointer(problem_index);
    if (mo_fitness_function == nullptr)
    {
        throw std::runtime_error(std::string("Error: unknown problem index (read index "
                                             + std::to_string(problem_index)+ ").\n"));
    }

    // Set dimension 'n' of fitness function
    if (mo_number_of_parameters <= 0)
    {
        // Error should not happen since mo_number_of_parameters is size_t
        throw std::runtime_error("Error: number of MO parameters <= 0 (read: " +
            std::to_string(mo_number_of_parameters)+"). Require MO number of parameters >= 1.\n");
    }
    else{
        mo_fitness_function->set_number_of_parameters(mo_number_of_parameters);
    }

    // Set parameters of problem
    assignParametersToProblems();

    // Check solution set size of test points
    if (solution_set_size <= 0)
    {
        // Error should not happen since solution_set_size is size_t
        throw std::runtime_error("Error: solution set size <= 0 (read: " +
                                 std::to_string(solution_set_size) + "). Require >= 1.");
    }

    // Deactivate unused parameters set size and change elitist archive settings
    collect_all_mo_sols_in_archive = false;
    elitist_archive_size_target = 0;
    approximation_set_size = 0;

    // Initialize the initital initialization ranges per dimension
    mo_lower_init_ranges.resize(mo_fitness_function->number_of_parameters, lower_init);
    mo_upper_init_ranges.resize(mo_fitness_function->number_of_parameters, upper_init);

    // Check if ranges are logical
    if(!mo_fitness_function->redefine_random_initialization)
    {
        if(lower_init >= upper_init) {
            throw std::runtime_error("Error: init range invalid (read lower " +
                std::to_string(lower_init)+ ", upper, " + std::to_string(upper_init)+ ")");
        }
    }

    // Set random seed
    if(random_seed < 0) {
        random_seed = (int) clock();
    }

    // Determine output file name root: Probably not used any more
    std::stringstream ss;
    ss << "_UHV-HYBRID" << linkage_model_index_UHVGOMEA << "_problem" << problem_index << "_p" << solution_set_size << "_run" << std::setw(3) << std::setfill('0') << random_seed;
    file_appendix = ss.str();
}

/**
 * Initialize the algorithm specific variables
 */
void initializeAlgorithmSpecificVariables(){
    // Enable bezier representation of solutions sets (see publication S.C. Maree et al., PPSN2020, BezEA)
    use_bezier_interpolation = false;
    number_of_reference_points = solution_set_size;

    // Retrieve pareto set
    mo_fitness_function->get_pareto_set();  // Todo: Find out what exactly is happening here

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
bool initializeProcess(int argc, char **argv){

    bool completed_successfully = false;

    try {
        int index = 1;

        // Parse options
        bool quit_program = parseOptions(argc, argv, &index);

        // Check if options causes the program to stop
        if (!quit_program)
        {
            // Parse parameters
            parseParameters(argc, argv, &index);

            // Check the options
            checkOptions();

            // Initialize remaining variables
            initializeAlgorithmSpecificVariables();

            completed_successfully = true;
        }
    } catch (std::exception& exception){
        std::cerr << exception.what() << std::endl;
    }

    return completed_successfully;
}

int main(int argc, char **argv)
{
    // Initialize process variables
    bool vars_initialized = initializeProcess(argc, argv);

    // CHeck if process is successfully initialized
    if (vars_initialized) {
        // Initialize as single objective problem
        std::shared_ptr<hillvallea::UHV_t> fitness_function;
        fitness_function = std::make_shared<hillvallea::UHV_t>(mo_fitness_function, solution_set_size, collect_all_mo_sols_in_archive, elitist_archive_size_target, nullptr, use_finite_differences);

        // Get lower and upper bounds for single objective fitness function
        hillvallea::vec_t lower_init_ranges, upper_init_ranges;
        lower_init_ranges.resize(fitness_function->number_of_parameters, lower_init);
        upper_init_ranges.resize(fitness_function->number_of_parameters, upper_init);

        // Check if IGD-based VTR should be used
        if(use_vtr == 2)
            fitness_function->redefine_vtr = true;

        // Change sign of VTR due to minimizing HV(X)
        value_to_reach *= -1;

        // Calculate maximum number of SO evaluations
        maximum_number_of_so_evaluations = (int) ( maximum_number_of_mo_evaluations / (double) solution_set_size );

        // Print Computation Settings
        if(print_verbose_overview)
            printComputationSettings(fitness_function);

        // Initialize the optimization process
        hillvallea::UHVSWITCHProcess_pt process;
        process = std::make_shared<hillvallea::UHVSWITCHProcess>(
                fitness_function,
                local_optimizer_index_gomea,
                localOptimizerIndexUHVADAM,
                indexApplicationMethodUHV_ADAM,
                fitness_function->number_of_parameters,
                solution_set_size,
                lower_init_ranges,
                upper_init_ranges,
                maximum_number_of_so_evaluations,
                maximum_number_of_seconds,
                value_to_reach,
                uhvSwitchValue,
                use_vtr,
                random_seed,
                write_solutions_optimizers,
                write_statistics_optimizers,
                write_directory,
                file_appendix);

        // Execute optimization
        process->run_optimization(population_size);

        // Write parameters file
        writeParametersFile();

        // Print Computation Results
        if (print_verbose_overview)
            printComputationResults(process);

    }

    std::exit(0);

}