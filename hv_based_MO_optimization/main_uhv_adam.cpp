/*

UHV-Gradient

Implementation by S.C. Maree, 2020

Corresponding publication:

Gadient-based Multi-objective optimization with the uncrowded hypervolume,
T.M. Deist, S.C. Maree, T. Alderliesten, P.A.N. Bosman, PPSN-2020


*/

#include "HillVallEA/sofomore.hpp"
#include "HillVallEA/adam.hpp"
#include "HillVallEA/hillvallea.hpp"
#include "HillVallEA/fitness.h"
#include "HillVallEA/mathfunctions.hpp"
#include "UHV.hpp"
#include "bezier.hpp"

// for MO problems
#include "../domination_based_MO_optimization/mohillvallea/hicam_external.h"
#include "../domination_based_MO_optimization/gomea/MO-RV-GOMEA.h"
#include "../benchmark_functions/mo_benchmarks.h"


// all setting variables
int optimizer_type;
int local_optimizer_index;
int problem_index;
size_t mo_number_of_parameters;
size_t number_of_test_points;
double lower_init;
double upper_init;
size_t popsize;
int random_seed;
bool collect_all_mo_sols_in_archive;
size_t elitist_archive_size_target;
size_t approximation_set_size;
int maximum_number_of_mo_evaluations;
int maximum_number_of_so_evaluations;
int maximum_number_of_seconds;
int use_vtr;
double value_to_reach;
bool write_generational_statistics;
bool write_generational_solutions;
std::string write_directory;
bool use_bezier_interpolation;
size_t number_of_reference_points;
bool use_finite_differences;
double finite_differences_multiplier;
double gamma_weight;
bool enable_niching;
std::string file_appendix;
hicam::fitness_pt mo_fitness_function;
hicam::vec_t mo_lower_init_ranges, mo_upper_init_ranges;
double HL_tol; // no longer used

// IMS population sizing scheme
unsigned int number_of_subgenerations_per_population_factor;
unsigned int maximum_number_of_populations;
bool print_verbose_overview;
bool print_generational_statistics;

// Problem parameters defined by user
double f_alpha;
double f_beta;
double f_lambda;
double f_frequency;


void printUsage(void) {
    printf("Usage: uhv_grad [-?] [-P] [-s] [-w] [-v] [-f] [-e] [-r] opt pro dim ssize low upp ela eva sec vtr rnd wrp\n"); // [-n] pop
    printf(" -?: Prints out this usage information.\n");
    printf(" -P: Prints out a list of all installed optimization problems.\n");
    printf(" -s: Enables computing and writing of statistics every generation.\n");
    printf(" -w: Enable writing of solutions and their fitnesses every generation.\n");
    printf(" -v: Verbose mode. Prints the settings before starting the run + ouput of generational statistics to terminal.\n");
    // printf(" -n: Enable niching \n");
    printf(" -e: Enable collecting all solutions in an elitist archive\n");
    printf(" -f: Force use of finite differences instead of analytical gradient.\n");
    printf(" -r: Enables use of vtr (value-to-reach) termination condition based on the hypervolume.\n");
    printf("\n");
    printf("  opt: Gradient method (0 = ADAM, 1 = GAMO, 2 = PLAIN, 3 = LINEAPPROXIMATION).\n");
    printf("  pro: Multi-objective optimization problem index (minimization).\n");
    printf("  dim: Number of parameters (if the problem is configurable).\n");
    printf("ssize: Solution set size (number of solutions on the front).\n");
    printf("  low: Overall initialization lower bound.\n");
    printf("  upp: Overall initialization upper bound.\n");
    // printf("  pop: Population size.\n");
    printf("  ela: Max Elitist archive size target (if -e is specified).\n");
    printf("  eva: Maximum number of evaluations of the multi-objective problem allowed.\n");
    printf("  sec: Time limit in seconds.\n");
    printf("  vtr: The value to reach. If the hypervolume of the best feasible solution reaches this value, termination is enforced (if -r is specified).\n");
    printf("  rnd: Random seed.\n");
    printf("  wrp: write path.\n");
    printf("alpha: Alpha of problem.\n");
    printf(" beta: Beta of problem.\n");
    printf("lmbda: Lambda of problem.\n");
    printf(" freq: Frequency of problem.\n");
    exit(0);

}

/**
 * Returns the problems installed.
 */
void printAllInstalledProblems(void) {
    int i = 0;

    hicam::fitness_pt objective = getObjectivePointer(i);

    std::cout << "Installed optimization problems:\n";

    while (objective != nullptr) {
        std::cout << std::setw(3) << i << ": " << objective->name() << std::endl;

        i++;
        objective = getObjectivePointer(i);
    }

    exit(0);
}

/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError(char **argv, int index) {
    printf("Illegal option: %s\n\n", argv[index]);

    printUsage();
}

void parseOptions(int argc, char **argv, int *index) {
    double dummy;

    write_generational_statistics = 0;
    write_generational_solutions = 0;
    print_verbose_overview = 0;
    use_vtr = 0;
    enable_niching = 0;
    collect_all_mo_sols_in_archive = 0;
    use_finite_differences = 0;

    for (; (*index) < argc; (*index)++) {
        if (argv[*index][0] == '-') {
            /* If it is a negative number, the option part is over */
            if (sscanf(argv[*index], "%lf", &dummy) && argv[*index][1] != '\0')
                break;

            if (argv[*index][1] == '\0')
                optionError(argv, *index);
            else if (argv[*index][2] != '\0')
                optionError(argv, *index);
            else {
                switch (argv[*index][1]) {
                    case '?':
                        printUsage();
                        break;
                    case 'P':
                        printAllInstalledProblems();
                        break;
                    case 's':
                        write_generational_statistics = 1;
                        break;
                    case 'w':
                        write_generational_solutions = 1;
                        break;
                    case 'v':
                        print_verbose_overview = 1;
                        break;
                        // case 'n': enable_niching = true; break;
                    case 'e':
                        collect_all_mo_sols_in_archive = true;
                        break;
                    case 'f':
                        use_finite_differences = true;
                        break;
                    case 'r':
                        use_vtr = 1;
                        break; // HV-based vtr (note, use_vtr = 2 is an IGD-based VTR)
                    default:
                        optionError(argv, *index);
                }
            }
        } else /* Argument is not an option, so option part is over */
            break;
    }

}

/**
 * Depending on the problem the user has selected, some problem parameters must be disabled.
 */
void cleanProblemParameters() {
    if (problem_index == 35) {
        // Check problem 35
        f_lambda = NAN;
    }
    else {
        // Other problems
        f_alpha = NAN;
        f_beta = NAN;
        f_lambda = NAN;
        f_frequency = NAN;
    }
}

void parseParameters(int argc, char **argv, int *index) {
    int noError;

    int n_params = 16;
    if ((argc - *index) != n_params) {
        printf("Number of parameters is incorrect, require %d parameters (you provided %d).\n\n", n_params,
               (argc - *index));

        printUsage();
    }

    noError = 1;

    noError = noError && sscanf(argv[*index + 0], "%u", &optimizer_type);
    noError = noError && sscanf(argv[*index + 1], "%d", &problem_index);
    noError = noError && sscanf(argv[*index + 2], "%zd", &mo_number_of_parameters);
    noError = noError && sscanf(argv[*index + 3], "%zd", &number_of_test_points);
    noError = noError && sscanf(argv[*index + 4], "%lf", &lower_init);
    noError = noError && sscanf(argv[*index + 5], "%lf", &upper_init);
    // noError = noError && sscanf(argv[*index + 6], "%zd", &popsize);
    noError = noError && sscanf(argv[*index + 6], "%zu", &elitist_archive_size_target);
    noError = noError && sscanf(argv[*index + 7], "%d", &maximum_number_of_mo_evaluations);
    noError = noError && sscanf(argv[*index + 8], "%d", &maximum_number_of_seconds);
    noError = noError && sscanf(argv[*index + 9], "%lf", &value_to_reach);
    noError = noError && sscanf(argv[*index + 10], "%d", &random_seed);
    write_directory = argv[*index + 11];
    noError = noError && sscanf(argv[*index + 12], "%lf", &f_alpha);
    noError = noError && sscanf(argv[*index + 13], "%lf", &f_beta);
    noError = noError && sscanf(argv[*index + 14], "%lf", &f_lambda);
    noError = noError && sscanf(argv[*index + 15], "%lf", &f_frequency);

    // Clean up the problem parameters
    cleanProblemParameters();

    if (!noError) {
        printf("Error parsing parameters.\n\n");
        printUsage();
    }

}

void assignParametersToProblems() {
    if (hicam::biExpCosineDecay *c = dynamic_cast<hicam::biExpCosineDecay *>(mo_fitness_function.get())) {
        c->set_alpha(f_alpha);
        c->set_beta(f_beta);
        c->set_lambda(f_lambda);
        c->set_frequency(f_frequency);
    } else if (hicam::biConvexSphereMinSphereCosine *c = dynamic_cast<hicam::biConvexSphereMinSphereCosine *>(mo_fitness_function.get())) {
        c->set_alpha(f_alpha);
        c->set_beta(f_beta);
        c->set_lambda(f_lambda);
        c->set_frequency(f_frequency);
    }

}

void checkOptions(void) {
    if (optimizer_type == 0) {
        local_optimizer_index = 50; // ADAM
    } else if (optimizer_type == 1) {
        local_optimizer_index = 64; // GAMO
    } else if (optimizer_type == 2) {
        local_optimizer_index = 80; // Plain
    } else if (optimizer_type == 3) {
        local_optimizer_index = 81; // Line approximation
    } else {
        printf("\n");
        printf("Error: optimizer type invalid (read: %d).", optimizer_type);
        printf("\n\n");

        exit(0);
    }

    mo_fitness_function = getObjectivePointer(problem_index);
    if (mo_fitness_function == nullptr) {
        printf("\n");
        printf("Error: unknown index for problem (read index %d).", problem_index);
        printf("\n\n");

        exit(0);
    }

    mo_fitness_function->set_number_of_parameters(mo_number_of_parameters);

    if (mo_number_of_parameters <= 0) {
        printf("\n");
        printf("Error: number of MO parameters <= 0 (read: %d). Require MO number of parameters >= 1.",
               (int) mo_number_of_parameters);
        printf("\n\n");

        exit(0);
    }

    // Set parameters of problem
    assignParametersToProblems();

    if (number_of_test_points <= 0) {
        printf("\n");
        printf("Error: solution set size <= 0 (read: %d). Require >= 1.", (int) number_of_test_points);
        printf("\n\n");

        exit(0);
    }

    approximation_set_size = elitist_archive_size_target;
    if (elitist_archive_size_target <= 0) {
        collect_all_mo_sols_in_archive = false;
        elitist_archive_size_target = 0;
        approximation_set_size = 0;
    }

    // initialize the init ranges
    mo_lower_init_ranges.resize(mo_fitness_function->number_of_parameters, lower_init);
    mo_upper_init_ranges.resize(mo_fitness_function->number_of_parameters, upper_init);

    if (!mo_fitness_function->redefine_random_initialization) {
        if (lower_init >= upper_init) {
            printf("\n");
            printf("Error: init range empty (read lower %f, upper, %f)", lower_init, upper_init);
            printf("\n\n");

            exit(0);
        }
    }

    // Enable bezier representation of solutions sets (see publication S.C. Maree et al., PPSN2020, BezEA)
    // disabled in this interface
    use_bezier_interpolation = false;
    number_of_reference_points = number_of_test_points; // for BezEA, this number can be different, and is generally lower.

    // Gradient & Finite difference settings
    finite_differences_multiplier = 1e-6;
    gamma_weight = 0.01;


    // prepares the fitness function
    mo_fitness_function->get_pareto_set();

    HL_tol = 0; // this is no longer used.

    // Interleaved multi-start scheme
    // set number of pops larger than 0 to increase popsize over time
    number_of_subgenerations_per_population_factor = 2;
    maximum_number_of_populations = 1;

    if (random_seed < 0) {
        random_seed = (int) clock();
    }

    // File appendix for writing
    std::stringstream ss;
    ss << "_opt" << optimizer_type << "_problem" << problem_index << "_p" << number_of_reference_points << "_run"
       << std::setw(3) << std::setfill('0') << random_seed;
    file_appendix = ss.str();

}


void parseCommandLine(int argc, char **argv) {
    int index;

    index = 1;

    parseOptions(argc, argv, &index);

    parseParameters(argc, argv, &index);
}

void interpretCommandLine(int argc, char **argv) {

    parseCommandLine(argc, argv);
    checkOptions();
}

/**
 * Writes the parameters of a run to a file
 */
void writeParametersFile() {
    // Determine parameters file path
    std::string filepath = write_directory + "parameters.txt";

    // Create parameters file
    std::ofstream parameters_file;
    parameters_file.open(filepath, std::ofstream::out | std::ofstream::trunc);

    // Write parameters to file
    std::string text = "";
    text += "Algorithm:UHV-ADAM, ";
    text += "GradientModel:" + std::to_string(optimizer_type) + ", ";
    text += "ProblemIndex:" + std::to_string(problem_index) + ", ";
    text += "FiniteDifferenceApproximation:" + std::to_string(use_finite_differences) + ", ";
    text += "ProblemDimension:" + std::to_string(mo_number_of_parameters) + ", ";
    text += "ssize:" + std::to_string(number_of_test_points) + ", ";
    text += "lb:" + std::to_string(lower_init) + ", ";
    text += "ub:" + std::to_string(upper_init) + ", ";
    text += "popsize:1, ";
    text += "maxevals:" + std::to_string(maximum_number_of_mo_evaluations) + ", ";
    text += "seed:" + std::to_string(random_seed) + ", ";
    text += "alpha:" + std::to_string(f_alpha) + ", ";
    text += "beta:" + std::to_string(f_beta) + ", ";
    text += "lambda:" + std::to_string(f_lambda) + ", ";
    text += "frequency:" + std::to_string(f_frequency);
    parameters_file << text << std::endl;

    // Close file
    parameters_file.close();
}

// Main: Run the CEC2013 niching benchmark
//--------------------------------------------------------
int main(int argc, char **argv) {

    interpretCommandLine(argc, argv);


    // create optimizer
    if (print_verbose_overview) {
        std::cout << "Problem settings:\n\tfunction_name = " << mo_fitness_function->name() << "\n\tproblem_index = "
                  << problem_index << "\n\tmo_number_of_parameters = " << mo_number_of_parameters
                  << "\n\tinit_range = [" << lower_init << ", " << upper_init << "]\n\tHV reference point = "
                  << mo_fitness_function->hypervolume_max_f0 << ", " << mo_fitness_function->hypervolume_max_f1 << "\n";
        std::cout << "Run settings:\n\tmax_number_of_MO_evaluations = " << maximum_number_of_mo_evaluations
                  << "\n\tmaximum_number_of_seconds = " << maximum_number_of_seconds << "\n\tuse_vtr = " << use_vtr
                  << "\n\tvtr = " << value_to_reach << "\n";
        std::cout << "Archive settings:\n\tCollect all MO-sol in archive = "
                  << (collect_all_mo_sols_in_archive ? "yes" : "no") << "\n\tElitist_archive_target_size = "
                  << elitist_archive_size_target << "\n\tApproximation_set_size = " << approximation_set_size << "\n";
    }

    // set SO problem
    hillvallea::fitness_pt fitness_function = std::make_shared<hillvallea::UHV_t>(mo_fitness_function,
                                                                                  number_of_reference_points,
                                                                                  collect_all_mo_sols_in_archive,
                                                                                  elitist_archive_size_target, nullptr,
                                                                                  use_finite_differences);

    hillvallea::vec_t lower_init_ranges, upper_init_ranges;
    lower_init_ranges.resize(fitness_function->number_of_parameters, lower_init);
    upper_init_ranges.resize(fitness_function->number_of_parameters, upper_init);

    std::stringstream ss;
    ss << "_UHVGRAD" << file_appendix;
    file_appendix = ss.str();

    if (print_verbose_overview) {
        std::cout << "Optimizer settings: \n\t" << (optimizer_type == 0 ? "ADAM" : "GAMO")
                  << " \n\tlocal_optimizer_index = " << local_optimizer_index << "\n\ttest_points = "
                  << number_of_test_points << "\n\tnumber_of_reference_points = " << number_of_reference_points
                  << "\n\tso_number_of_parameters = " << fitness_function->number_of_parameters
                  << "\n\tuse_finite_differences = " << (use_finite_differences ? "yes" : "no") << "\n\tpopsize = "
                  << popsize << "\n\tenable_niching = " << (enable_niching ? "yes" : "no") << "\n\trandom_seed = "
                  << random_seed << "\n";
    }

    if (use_vtr == 2) // IGD-based VTR
    {
        fitness_function->redefine_vtr = true;
    }

    // Internally, minimization is performed of -HV(X)
    // Therefore, the HV-based VTR is negated
    value_to_reach *= -1;
    maximum_number_of_so_evaluations = (int) (maximum_number_of_mo_evaluations / (double) number_of_reference_points);


    hillvallea::adam_t opt(
            fitness_function,
            local_optimizer_index,
            lower_init_ranges,
            upper_init_ranges,
            maximum_number_of_so_evaluations,
            maximum_number_of_seconds,
            value_to_reach,
            use_vtr,
            random_seed,
            write_generational_solutions,
            write_generational_statistics,
            write_directory,
            file_appendix,
            gamma_weight,
            finite_differences_multiplier
    );

    opt.run();

// Write parameters file
    writeParametersFile();

    if (print_verbose_overview) {
        std::cout << "Best: \n\tHV = " << std::fixed << std::setprecision(14) << -opt.best.f << "\n\tMO-fevals = "
                  << mo_fitness_function->number_of_evaluations << "\n\truntime = "
                  << double(clock() - opt.starting_time) / CLOCKS_PER_SEC << " sec" << std::endl;


        std::cout << "pareto_front" << (use_bezier_interpolation ? number_of_reference_points : 0) << " = [";
        for (size_t k = 0; k < opt.best.mo_test_sols.size(); ++k) {
            std::cout << "\n\t" << std::fixed << std::setw(10) << std::setprecision(4) << opt.best.mo_test_sols[k]->obj;
        }
        std::cout << " ];\n";


        std::cout << "pareto_set" << (use_bezier_interpolation ? number_of_reference_points : 0) << " = [";
        for (size_t k = 0; k < opt.best.mo_test_sols.size(); ++k) {
            std::cout << "\n\t";
            for (size_t i = 0; i < opt.best.mo_test_sols[k]->param.size(); ++i) {
                std::cout << std::fixed << std::setw(10) << std::setprecision(4) << opt.best.mo_test_sols[k]->param[i]
                          << " ";
            }
        }
        std::cout << " ];\n";
    }
}
