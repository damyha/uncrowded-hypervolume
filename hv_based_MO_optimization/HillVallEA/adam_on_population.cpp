/*
 * ADAM applied on a population stead of a single solution
 *
 * Warnings:
 * - Do not manually change the population. Please use 'initialize_from_population' or 'changePopulation'
 *      such that the best solution is always calculated.
 *
 * By D. Ha
 *
*/

#include "adam_on_population.h"
#include "adam.hpp"
#include "gomea.hpp"
#include "mathfunctions.hpp"


/**
 * Constructor of ADAM_on_population
 * @param numberOfParameters
 * @param lowerParamBounds
 * @param upperParamBounds
 * @param initUnivariateBandwidth
 * @param fitnessFunction
 * @param rng1
 * @param number_of_parameters
 * @param lower_param_bounds
 * @param upper_param_bounds
 * @param init_univariate_bandwidth
 * @param version
 * @param fitness_function
 * @param rng
 */
hillvallea::adam_on_population_t::adam_on_population_t(
        const size_t number_of_parameters,
        const vec_t &lower_param_bounds,
        const vec_t &upper_param_bounds,
        double init_univariate_bandwidth,
        int version,
        int version_method,
        fitness_pt fitness_function,
        rng_pt rng) : optimizer_t(number_of_parameters,
                                  lower_param_bounds,
                                  upper_param_bounds,
                                  init_univariate_bandwidth,
                                  fitness_function, rng) {

    // Initialize settings
    this->version = version;
    this->version_method = version_method;
    this->initial_population_size = 0;
    this->population_size = 0;

    this->population_initialized = false;
    this->use_momentum_with_nag = false;
    this->accept_only_improvements = false;

    this->no_improvement_stretch = 0;
    this->weighted_number_of_evaluations = 0;

    // Private variables
    this->gamma_weight = 0.01;          // Should be overwritten from outside (to save me from changing it everywhere)
    this->stepSizeDecayFactor = 0.99;   // Should be overwritten from outside (to save me from changing it everywhere)
    this->finite_differences_multiplier = 1e-6; // Default value taken from Adam.cpp

}

/**
 * Destructor of ADAM_on_population
 */
hillvallea::adam_on_population_t::~adam_on_population_t() {

};

/**
 * Clone the current optimizer to a new object.
 * This method is not implemented yet
 * @return The cloned optimizer object.
 */
hillvallea::optimizer_pt hillvallea::adam_on_population_t::clone() const {

    throw std::runtime_error("Clone method of ADAM_ON_POPULATION not implemented yet.");
    // Create new optimizer
    /*
     * Variables taken care of during intialization:
     *  version, version_method
     */
    adam_on_population_pt opt = std::make_shared<adam_on_population_t>(
            number_of_parameters, lower_param_bounds, upper_param_bounds,
            init_univariate_bandwidth, version, version_method, fitness_function, rng);

    // Copy data
    opt->active = active;
    opt->number_of_parameters = number_of_parameters;
    opt->lower_param_bounds = lower_param_bounds;
    opt->upper_param_bounds = upper_param_bounds;
    opt->fitness_function = fitness_function;
    opt->number_of_generations = number_of_generations;
    opt->rng = rng;
    opt->best = best;
    opt->average_fitness_history = average_fitness_history;
    opt->selection_fraction = selection_fraction;
    opt->init_univariate_bandwidth = init_univariate_bandwidth;

    // Copy population to new object
    opt->pop = std::make_shared<population_t>();
    opt->pop->addSolutions(*pop);

    // Copy the stopping criterea
    opt->param_std_tolerance = param_std_tolerance;
    opt->fitness_std_tolerance = fitness_std_tolerance;

    return opt;
}

hillvallea::adam_on_population_pt hillvallea::adam_on_population_t::clone(population_pt newPopulation) {
    // Create new optimizer
    // Variables taken care of: gamma_weight, finite_differences_multiplier
    adam_on_population_pt newOptimizer = std::make_shared<adam_on_population_t>(
            number_of_parameters, lower_param_bounds, upper_param_bounds,
            init_univariate_bandwidth, version, version_method, fitness_function, rng);

    // Insert population into optimizer
    newOptimizer->pop = newPopulation;
    newOptimizer->population_size = newPopulation->size();
    newOptimizer->initial_population_size = initial_population_size;
    newOptimizer->population_initialized = true;

    // Copy other settings
    newOptimizer->use_momentum_with_nag = use_momentum_with_nag;
    newOptimizer->accept_only_improvements = accept_only_improvements;
    newOptimizer->no_improvement_stretch = no_improvement_stretch;
    newOptimizer->weighted_number_of_evaluations = weighted_number_of_evaluations;

    // Copy touched_parameter_idx
    newOptimizer->setTouchedParameterIdx(touched_parameter_idx);

    // Initialize lower_init_ranges and upper_init_ranges
    newOptimizer->setLowerInitRanges(lower_init_ranges);
    newOptimizer->setUpperInitRanges(upper_init_ranges);

    // Copy the optimizer
    std::vector <adam_pt> copyOptimizers;
    copyOptimizers.resize(gradient_optimizers.size());
    for (size_t i = 0; i < gradient_optimizers.size(); ++i) {
        copyOptimizers[i] = gradient_optimizers[i]->clone();
    }
    newOptimizer->setGradientOptimizers(copyOptimizers);

    return newOptimizer;
}

/**
 * The name of the optimizer
 * @return The name of the optimizer
 */
std::string hillvallea::adam_on_population_t::name() const { return "ADAM_P_" + std::to_string(this->version); }


/**
 * Initializing the population from an existing population
 * Assumes that the fitness (and gradient) of the population already has been calculated
 * @param pop The population that replaces the old/empty population
 * @param target_popsize The population size
 */
void hillvallea::adam_on_population_t::initialize_from_population(population_pt pop, size_t target_popsize) {
    // Copy the population
    this->pop = pop;

    // Set Initialization flag
    this->population_initialized = true;

    // Finds and sets the best solution and its index
    this->best = *this->pop->bestSolution();

    // Set population sizes
    this->population_size = target_popsize;
    this->initial_population_size = std::min((int) population_size, (int) pop->size());

    // Reset settings
    this->number_of_generations = 0;
    this->no_improvement_stretch = 0;
    this->weighted_number_of_evaluations = 0;

    // Reset the No improvement Stretch (NIS) of the population
//    for (size_t j = 0; j < pop->size(); ++j) {
//        pop->sols[j]->NIS = 0;
//    }

//    // Fill the remaining population slots with copies (if the new population size is larger than the initial)
//    pop->sols.resize(target_popsize);
//    for (size_t j = initial_population_size; j < population_size; j++) {
//        pop->sols[j] = copySolution(pop->sols[j % initial_population_size]);
//    }

    // Determine initialization ranges
    this->determineInitRanges();

    // Reset touched_parameter_idx
    this->initializeGradientAlgorithmSpecificParameters();

    // Reinitialize all optimizers
    this->initializeLocalOptimizers();

    // Initialize gamma vector
    this->initializeGammaVector();

}

/**
 * Execute one generation of ADAM on the population.
 * Assumes that 'initialize_from_population' was ran first
 * @param sample_size The sample size
 * @param number_of_evaluations A pointer to the number of evaluations
 */
void hillvallea::adam_on_population_t::generation(size_t sample_size, int &external_number_of_evaluations) {
    throw std::runtime_error("Deprecated generation function is called.");
}


/**
 * Applies the gradient algorithm on a single solution of the population
 * @param solutionIndex The solution index to apply the gradient algorithm on
 */
void hillvallea::adam_on_population_t::executeGenerationOnSpecificSolution(size_t solutionIndex) {
    // Determine which gradient algorithm should be applied
    if (version == 50) {
        // Apply ADAM
        weighted_number_of_evaluations += gradient_optimizers[solutionIndex]->gradientOffspring(
                pop->sols[solutionIndex],
                touched_parameter_idx,
                pop->sols[solutionIndex]->gamma_vector);
    } else if (version == 64) {
        // Apply GAMO
        weighted_number_of_evaluations += gradient_optimizers[solutionIndex]->HIGAMOgradientOffspring(
                pop->sols[solutionIndex],
                touched_parameter_idx,
                pop->sols[solutionIndex]->gamma_vector);
    } else if (version == 80) {
        // Apply plain gradient
        weighted_number_of_evaluations += gradient_optimizers[solutionIndex]->plainGradientOffspring(
                pop->sols[solutionIndex],
                touched_parameter_idx,
                pop->sols[solutionIndex]->gamma_vector);
    } else if (version == 81) {
        // Apply Line search
        weighted_number_of_evaluations += gradient_optimizers[solutionIndex]->lineSearchOffspring(
                pop->sols[solutionIndex],
                touched_parameter_idx,
                pop->sols[solutionIndex]->gamma_vector);
    }


    // Check if a better solution was found
    if (solution_t::better_solution(*pop->sols[solutionIndex], this->best)) {
        this->best = solution_t(*pop->sols[solutionIndex]);
        this->best_solution_index = solutionIndex;
    }

    // Update optimizer statistics
    gradient_optimizers[solutionIndex]->number_of_generations++;
}

/**
 * Execute multiple calls of ADAM on the population.
 * Assumes that 'initialize_from_population' was ran first
 * @param sample_size The sample size
 * @param external_number_of_evaluations A pointer to the number of evaluations
 * @param number_of_calls_to_execute The number of calls to execute
 */
void hillvallea::adam_on_population_t::generation(size_t sample_size,
                                                  int &external_number_of_evaluations,
                                                  size_t number_of_calls_to_execute) {
    // Check if population is initialized
    if (population_initialized) {

        // Initialize variable
        double current_number_of_evaluations = this->weighted_number_of_evaluations;

        // Determine which solution indices to apply the algorithm on
        std::vector <size_t> solutions_to_apply = this->determineSolutionIndicesToApplyAlgorithmOn(
                number_of_calls_to_execute);

        // Execute gradient algorithm on solutions that were picked
        for (auto &solution_index : solutions_to_apply) {
            this->executeGenerationOnSpecificSolution(solution_index);
        }

        // Update statistics
        number_of_generations++;
        external_number_of_evaluations += (int) round(weighted_number_of_evaluations - current_number_of_evaluations);
        average_fitness_history.push_back(pop->average_fitness());

    } else {
        throw std::runtime_error("Population was not initialized, but a generation has been requested\n");
    }

}


/**
 * Todo Fix this later
 * @return
 */
bool hillvallea::adam_on_population_t::checkTerminationCondition() {
    return false;
}


/////////////////////////////
///  Algorithms Variables ///
/////////////////////////////

/*
 * Goes over every solution variable and determine the initialization ranges of the population
 */
void hillvallea::adam_on_population_t::determineInitRanges() {
    // Initialize result variables
    vec_t lower_range(number_of_parameters, 1e300);
    vec_t upper_range(number_of_parameters, -1e300);

    // Go over solutions
    for (size_t population_index; population_index < pop->sols.size(); ++population_index) {
        // Go over parameters
        for (size_t parameter_index = 0; parameter_index < number_of_parameters; ++parameter_index) {
            // Set solution's parameter as lowest range/ highest range
            if (pop->sols[population_index]->param[parameter_index] < lower_range[parameter_index])
                lower_range[parameter_index] = pop->sols[population_index]->param[parameter_index];

            if (pop->sols[population_index]->param[parameter_index] > upper_range[parameter_index])
                upper_range[parameter_index] = pop->sols[population_index]->param[parameter_index];
        }
    }

    // Assign result
    this->lower_init_ranges = lower_range;
    this->upper_init_ranges = upper_range;

}

/**
 * Initializes touched_parameter_idx.
 */
void hillvallea::adam_on_population_t::initializeGradientAlgorithmSpecificParameters() {
    if (version == 50) {
        // ADAM
        // Set the parameter indices that are optimized
        touched_parameter_idx.resize(1);
        touched_parameter_idx[0].resize(number_of_parameters);
        for (size_t i = 0; i < number_of_parameters; ++i) {
            touched_parameter_idx[0][i] = i;
        }
    } else if (version == 64) {
        // GAMO
        // Determine variables
        size_t p = pop->sols[0]->mo_test_sols.size();
        size_t n = pop->sols[0]->mo_test_sols[0]->number_of_parameters();

        // Resize the touched parameter
        touched_parameter_idx.resize(p);

        size_t ki = 0;
        for (size_t FOS_idx = 0; FOS_idx < p; ++FOS_idx) {
            touched_parameter_idx[FOS_idx].resize(n);
            for (size_t i = 0; i < n; ++i) {
                touched_parameter_idx[FOS_idx][i] = ki;
                ki++;
            }
        }
        assert(ki == number_of_parameters);

//        maximum_no_improvement_stretch *= n; This variable is not initialized and not used
    } else if (version == 80) {
        // Plain gradient
        // Set the parameter indices that are optimized
        touched_parameter_idx.resize(1);
        touched_parameter_idx[0].resize(number_of_parameters);
        for (size_t i = 0; i < number_of_parameters; ++i) {
            touched_parameter_idx[0][i] = i;
        }
    } else if (version == 81) {
        // Line search
        // Set the parameter indices that are optimized
        touched_parameter_idx.resize(1);
        touched_parameter_idx[0].resize(number_of_parameters);
        for (size_t i = 0; i < number_of_parameters; ++i) {
            touched_parameter_idx[0][i] = i;
        }

    } else {
        throw std::runtime_error(std::to_string(version) + " is an unknown version to ADAM_ON_POPULATION\n");
    }
}

/**
 * Initializes the gamma vector.
 * Assumes that the local optimizer has been initialized first, since it takes the initial gamma from the optimizer
 */
void hillvallea::adam_on_population_t::initializeGammaVector() {
    // Assign version specific gamma per solution
    for (size_t solution_index = 0; solution_index < pop->size(); ++solution_index) {
        // Retrieve initial gamma
        double initial_gamma = gradient_optimizers[solution_index]->gamma;

        // Restructure the gamma vector according to the local optimizer version
        if (version == 50) {
            // ADAM
            pop->sols[solution_index]->gamma_vector.assign(1, initial_gamma);
        } else if (version == 64) {
            // GAMO
            size_t p = pop->sols[0]->mo_test_sols.size();
            pop->sols[solution_index]->gamma_vector.assign(p, initial_gamma);
        } else if (version == 80) {
            // Plain gradient
            pop->sols[solution_index]->gamma_vector.assign(1, initial_gamma);
        } else if (version == 81) {
            // Line search
            pop->sols[solution_index]->gamma_vector.assign(1, initial_gamma);
        } else {
            throw std::runtime_error(std::to_string(version) + " is an unknown version to ADAM_ON_POPULATION\n");
        }
    }

}

///////////////////////
/// Local optimizer ///
///////////////////////

/**
 * Initializes the gradient optimizers
 */
void hillvallea::adam_on_population_t::initializeLocalOptimizers() {
    // Prepare gradient optimizers vector
    gradient_optimizers.resize(population_size);

    // Initialize gradient optimizers
    for (size_t optimizer_index = 0; optimizer_index < population_size; ++optimizer_index) {
        gradient_optimizers[optimizer_index] = std::make_shared<adam_t>(
                fitness_function,
                version,
                lower_init_ranges,
                upper_init_ranges,
                -1,
                -1,
                0.0,
                false,
                0,
                false,
                false,
                "",
                "",
                gamma_weight,
                finite_differences_multiplier);
        // Override settings
        gradient_optimizers[optimizer_index]->stepSizeShrinkFactor = stepSizeDecayFactor;


        // Copy over the settings
        gradient_optimizers[optimizer_index]->accept_only_improvements = this->accept_only_improvements;
        gradient_optimizers[optimizer_index]->use_momentum_with_nag = this->use_momentum_with_nag;
        gradient_optimizers[optimizer_index]->rng = this->rng;
    }
}

/**
 * Determines the indices of the solutions where ADAM/GAMO should be applied on based on the application method
 * (read: version_method)
 * @param number_of_calls_to_execute The number of calls to execute
 * @return A vector of solution indices to apply the gradient algorithm on
 */
std::vector <size_t> hillvallea::adam_on_population_t::determineSolutionIndicesToApplyAlgorithmOn(
        size_t number_of_calls_to_execute) {
    std::vector <size_t> result;

    if (this->version_method == 0) {
        // Apply all calls on best solution
        size_t bestSolutionIndex = this->pop->bestSolutionIndex();
        for (size_t i = 0; i < number_of_calls_to_execute; ++i) {
            result.push_back(bestSolutionIndex);
        }
    } else if (this->version_method == 1) {
        // Apply calls on as many solutions as possible but only once
        size_t max_population_index = min(number_of_calls_to_execute, this->population_size);

        for (size_t solution_index = 0; solution_index < max_population_index; ++solution_index) {
            result.push_back(solution_index);
        }
    } else if (this->version_method == 2) {
        // Apply calls on top 3 best solutions. Assumes that population > 3, hw
        size_t numberOfBestSolutions = 3;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 3) {
        // Random solutions
        for (size_t i = 0; i < number_of_calls_to_execute; ++i) {
            int randomSolutionIndex = randomInt(this->pop->size(), *this->rng);
            result.push_back(randomSolutionIndex);
        }

    } else if (this->version_method == 4) {
        // 2.5 % of best solutions
        size_t numberOfBestSolutions = (size_t) (0.025 * (double) pop->size());
        if (numberOfBestSolutions <= 0) numberOfBestSolutions = 1;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 5) {
        // 5 % of best solutions
        size_t numberOfBestSolutions = (size_t) (0.05 * (double) pop->size());
        if (numberOfBestSolutions <= 0) numberOfBestSolutions = 1;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 6) {
        // 10 % of best solutions
        size_t numberOfBestSolutions = (size_t) (0.1 * (double) pop->size());
        if (numberOfBestSolutions <= 0) numberOfBestSolutions = 1;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 7) {
        // 20 % of best solutions
        size_t numberOfBestSolutions = (size_t) (0.2 * (double) pop->size());
        if (numberOfBestSolutions <= 0) numberOfBestSolutions = 1;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 8) {
        // 35 % of best solutions
        size_t numberOfBestSolutions = (size_t) (0.35 * (double) pop->size());
        if (numberOfBestSolutions <= 0) numberOfBestSolutions = 1;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 9) {
        // 50 % of best solutions
        size_t numberOfBestSolutions = (size_t) (0.5 * (double) pop->size());
        if (numberOfBestSolutions <= 0) numberOfBestSolutions = 1;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 10) {
        // Apply calls on top 5 best solutions. Assumes that population > 5, hw
        size_t numberOfBestSolutions = 5;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 11) {
        // Apply calls on top 10 best solutions. Assumes that population > 10, hw
        size_t numberOfBestSolutions = 10;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else if (this->version_method == 12) {
        // Apply calls on top 20 best solutions. Assumes that population > 20, hw
        size_t numberOfBestSolutions = 20;
        result = selectTopNSolutions(numberOfBestSolutions, number_of_calls_to_execute);
    } else {
        // Unknown version
        throw std::runtime_error("Gradient algorithm has detected an unknown application method detected: " +
                                 std::to_string(this->version_method) + ".\n");
    }

    return result;
}

/**
 * Selects the best N solutions of a population and assigns budget to these solutions until all
 * number_of_calls_to_execute have been scheduled.
 * @param topNSolutions The best N number of solutions
 * @param number_of_calls_to_execute The number of calls available to spend
 * @return A list of indices that represent the solution index of a population
 */
std::vector <size_t> hillvallea::adam_on_population_t::selectTopNSolutions(size_t topNSolutions,
                                                                           size_t number_of_calls_to_execute) {
    // Initialize result
    std::vector <size_t> result;

    // Sort population indices on fitness
    std::vector <size_t> indices_solutions_sorted = this->sortPopulationIndexOnBestFitness();

    // Schedule calls
    size_t solution_selected = 0;
    size_t calls_scheduled = 0;
    while (calls_scheduled < number_of_calls_to_execute) {
        // Add solution index to planned calls
        result.push_back(indices_solutions_sorted[solution_selected]);

        // Update the number of calls scheduled
        ++calls_scheduled;

        // Determine the next solution index to schedule
        ++solution_selected;
        if (solution_selected >= topNSolutions)
            solution_selected = 0;
    }

    return result;
}

/**
 * Sorts the population index on fitness and sets 'best' and 'best_solution_index' given the population
 */
std::vector <size_t> hillvallea::adam_on_population_t::sortPopulationIndexOnBestFitness() {
    // Sort a vector of solution indices sorted on fitness
    std::vector <size_t> indices_solutions_sorted = this->pop->sort_solution_index_on_fitness();

    // Assign results
    this->best_solution_index = indices_solutions_sorted[0];
    this->best = solution_t(*this->pop->sols[this->best_solution_index]);

    return indices_solutions_sorted;


}

void hillvallea::adam_on_population_t::setGradientOptimizers(const std::vector <adam_pt> &gradientOptimizers) {
    gradient_optimizers = gradientOptimizers;
}

void
hillvallea::adam_on_population_t::setTouchedParameterIdx(const std::vector <std::vector<size_t>> &touchedParameterIdx) {
    touched_parameter_idx = touchedParameterIdx;
}

void hillvallea::adam_on_population_t::setLowerInitRanges(const hillvallea::vec_t &lowerInitRanges) {
    lower_init_ranges = lowerInitRanges;
}

void hillvallea::adam_on_population_t::setUpperInitRanges(const hillvallea::vec_t &upperInitRanges) {
    upper_init_ranges = upperInitRanges;
}

void hillvallea::adam_on_population_t::setGammaWeight(double gammaWeight) {
    gamma_weight = gammaWeight;
}

void hillvallea::adam_on_population_t::setStepSizeDecayFactor(double stepSizeDecayFactor) {
    this->stepSizeDecayFactor = stepSizeDecayFactor;
}

void hillvallea::adam_on_population_t::setFiniteDifferencesMultiplier(double finiteDifferencesMultiplier) {
    finite_differences_multiplier = finiteDifferencesMultiplier;
}

std::string hillvallea::adam_on_population_t::convertUHV_ADAMModelNameToString() {
    std::string result = "UHV-";

    // Add algorithm name
    switch (version) {
        case 50:
            result += "ADAM";
            break;
        case 64:
            result += "GAMO";
            break;
        case 80:
            result += "GRAD";
            break;
        case 81:
            result += "LINEAPP";
            break;
        default:
            result += "UNKNOWN";
            break;
    }

    // Add application method
    switch (version_method) {
        case 0:
            result += "-Best";
            break;
        case 1:
            result += "-All";
            break;
        case 2:
            result += "-TOP3";
            break;
        case 3:
            result += "-Random";
            break;
        case 4:
            result += "-Top2.5%";
            break;
        case 5:
            result += "-Top5%";
            break;
        case 6:
            result += "-Top10%";
            break;
        case 7:
            result += "-Top20%";
            break;
        case 8:
            result += "-Top35%";
            break;
        case 9:
            result += "-Top50%";
            break;
        case 10:
            result += "-Top5";
            break;
        case 11:
            result += "-Top10";
            break;
        case 12:
            result += "-Top20";
            break;
        default:
            result += "-UNKNOWN";
            break;
    }

    return result;
}