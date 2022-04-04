# pragma once
/*
 * Implementation by D. Ha, 2021
 * This file describes the bi convex sphere minus sphere scaled cosine (abs x)
 * Make sure that the parameters passed to the objective functions are within the appropriate range
 */

# include <cmath>

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"


namespace hicam {
    class biConvexSphereMinSphereCosine : public fitness_t
    {
    public:
        // Parameters of objective function
        double f_alpha;
        double f_beta;
        double frequency;
        double exp_lambda;  // Not used

        // Derived variables
        double f_ang;

        /**
         * Constructor that sets the default values
         */
        biConvexSphereMinSphereCosine()
        {
            number_of_objectives = 2; // Fixed value
            number_of_parameters = 2; // Default value

            // Set default values of objective function parameters
            f_alpha = 0.0;
            f_beta = 0.0;
            frequency = 0.00;
            exp_lambda = 0.0;   // Not used but kept in here so it less of a hassle to make exception for other problems

            // Calculate derived parameters
            f_ang = 2.0 * M_PI * frequency;

            partial_evaluations_available = false;
            analytical_gradient_available = true;

            hypervolume_max_f0 = 11;
            hypervolume_max_f1 = 11;

            analytical_gd_avialable = true;
        }

        ~biConvexSphereMinSphereCosine(){}

        /**
        * Sets the number of objectives to 2.
        * @param number_of_objectives The number of objectives
        */
        void set_number_of_objectives(size_t & number_of_objectives)
        {
            this->number_of_objectives = 2;
            number_of_objectives = this->number_of_objectives;
        }

        /**
         * Sets the number of parameters 'n'.
         * @param number_of_parameters The number of parameters.
         */
        void set_number_of_parameters(size_t & number_of_parameters)
        {
            this->number_of_parameters = number_of_parameters;
        }

        /**
         * Sets the value of alpha
         * @param value_alpha The value of alpha
         */
        void set_alpha(double & value_alpha)
        {
            this->f_alpha = value_alpha;
        }

        /**
         * Sets the value of beta
         */
        void set_beta(double & value_beta)
        {
            this->f_beta = value_beta;
        }

        /**
         * Sets the value of lambda
         * @param value_lambda The value of lambda
         */
        void set_lambda(double & value_lambda)
        {
            this->exp_lambda = value_lambda;
        }

        /**
         * Sets the frequency 'f'.
         * @param frequency The frequency
         */
        void set_frequency(double & frequency)
        {
            this->frequency = frequency;
            this->f_ang = 2.0 * M_PI * frequency;
        }

        /**
         * Returns the lower and upper bounds of this problem
         * @param lower A pointer to the vec_t to store the lower bounds
         * @param upper A pointer to the vec_t to store the upper bounds
         */
        void get_param_bounds(vec_t & lower, vec_t & upper) const
        {
            lower.clear();
            lower.resize(number_of_parameters, -1000);

            upper.clear();
            upper.resize(number_of_parameters, 1000);

        }

        /**
         * Calculates the sum square of x_c, i.e. sum((x-center)^2), where the x_c is x centered
         * @param solution The solution to calculate the sum of squares of
         * @param center_index The index of the center (-1 to return sum(x_c^2))
         * @return
         */
        double calculate_sum_square_x_min_c(solution_t & solution, size_t center_index)
        {
            double result = 0.0;
            for(size_t i = 0; i < solution.param.size(); ++i) {
                if ( i == center_index){
                    result += (solution.param[i] - 1.0)*(solution.param[i] - 1.0);
                } else {
                    result += solution.param[i] * solution.param[i];
                }
            }

            return result;
        }

        /**
         * Given the objective number, the pre-computed constant (and center_index) this method assigns the gradient.
         * @param solution The solution to inspect
         * @param const_gradient 2*alpha - beta * (2*cos(f_ang*abs(x_c)) + ss(x_c)*sin(f_ang*abs(x_c)/abs(x_c))
         * @param objective_number The objective to inspect
         * @param center_index The index of the center
         */
        void assign_single_gradient(solution_t & solution, double const_gradient,
                                    size_t objective_number, size_t center_index)
        {
            double x_i;
            for(size_t i = 0; i < solution.param.size(); ++i) {
                // Set the center w.r.t the objective's center
                if ( i == center_index){
                    x_i = (solution.param[i] - 1.0);
                } else {
                    x_i = solution.param[i];
                }

                solution.gradients[objective_number][i] = const_gradient * x_i;
            }
        }

        /**
         * Given a solution, this method calculates the objective values
         * @param sol The solution to inspect
         */
        void define_problem_evaluation(solution_t & sol)
        {
            // Calculate the sums squares
            double ss_xc_0 = calculate_sum_square_x_min_c(sol, -1);
            double ss_xc_1 = calculate_sum_square_x_min_c(sol, 0);

            // Calculate absolute values
            double abs_xc_0 = std::sqrt(ss_xc_0);
            double abs_xc_1 = std::sqrt(ss_xc_1);

            // Calculate cosines
            double cos_xc_0 = std::cos(f_ang * abs_xc_0);
            double cos_xc_1 = std::cos(f_ang * abs_xc_1);

            // Objective values
            sol.obj[0] = (f_alpha * ss_xc_0) - (f_beta * ss_xc_0 * cos_xc_0);
            sol.obj[1] = (f_alpha * ss_xc_1) - (f_beta * ss_xc_1 * cos_xc_1);

            // Constraint value
            sol.constraint = 0;
        }

        /**
         * Given a solution, this method calculates the objective values and the gradient
         * @param sol The solution to inspect
         */
        void define_problem_evaluation_with_gradients(solution_t & sol)
        {
            // Calculate the sums squares
            double ss_xc_0 = calculate_sum_square_x_min_c(sol, -1);
            double ss_xc_1 = calculate_sum_square_x_min_c(sol, 0);

            // Calculate absolute values
            double abs_xc_0 = std::sqrt(ss_xc_0);
            double abs_xc_1 = std::sqrt(ss_xc_1);

            // Calculate cosines
            double cos_xc_0 = std::cos(f_ang * abs_xc_0);
            double cos_xc_1 = std::cos(f_ang * abs_xc_1);

            // Calculate sines
            double sin_xc_0 = std::sin(f_ang * abs_xc_0);
            double sin_xc_1 = std::sin(f_ang * abs_xc_1);

            // Calculate constants * sine
            double c_sin_g0 = ((ss_xc_0 * f_ang) / (2*abs_xc_0)) * sin_xc_0;
            double c_sin_g1 = ((ss_xc_1 * f_ang) / (2*abs_xc_1)) * sin_xc_1;

            // Calculate total scaling factor
            double c_g0 = 2 * (f_alpha - f_beta * (cos_xc_0 - c_sin_g0));
            double c_g1 = 2 * (f_alpha - f_beta * (cos_xc_1 - c_sin_g1));

            // Objective values
            sol.obj[0] = (f_alpha * ss_xc_0) - (f_beta * ss_xc_0 * cos_xc_0);
            sol.obj[1] = (f_alpha * ss_xc_1) - (f_beta * ss_xc_1 * cos_xc_1);

            // Constraint value
            sol.constraint = 0;

            // Initialize gradients
            sol.gradients.resize(number_of_objectives);
            sol.gradients[0].resize(number_of_parameters);
            sol.gradients[1].resize(number_of_parameters);

            // Gradients
            assign_single_gradient(sol, c_g0, 0, -1);
            assign_single_gradient(sol, c_g1, 1, 0);
        }

        /**
         * Given a solution, this method calculates the objective values via a partial evaluation
         * @param sol The solution to perform the partial evaluation on
         * @param touched_parameter_idx The indices of the parameters that were changed
         * @param old_sol The old solution
         */
        void define_partial_problem_evaluation(solution_t & sol, const std::vector<size_t> & touched_parameter_idx, const solution_t & old_sol)
        {
            printf("Partial evaluation not implemented.");
        }

        /**
         * The name of the problem
         * @return The name of the problem as a string
         */
        std::string name() const
        {
            return "bi_convex_sphere_min_sphere_cosine";
        }

        /**
         * Retrieve the pareto set. Todo: Only works for the continuous line
         * @return A boolean if the process has been completed.
         */
        bool get_pareto_set()
        {
            size_t pareto_set_size = 5000;

            // Generate default front
            if (pareto_set.size() != pareto_set_size)
            {
                pareto_set.sols.clear();
                pareto_set.sols.reserve(pareto_set_size);

                // Create the front
                for (size_t i = 0; i < pareto_set_size; ++i)
                {
                    solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);

                    sol->param.fill(0);
                    sol->param[0] = (i / ((double) pareto_set_size ));
                    define_problem_evaluation(*sol); // runs a feval without registering it.

                    pareto_set.sols.push_back(sol);
                }

                igdx_available = true; // Todo: find out what this is
                igd_available = true;  // Todo: find out what this is
                // pareto_set.writeToFile("./genMED.txt");
            }

            return true;
        }

        /**
         * Calculates the distance to the front. Todo: Not sure what happens here
         * @param sol The solution to inspect
         * @return The distance to the front.
         */
        double distance_to_front(const solution_t & sol)
        {
            solution_t ref_sol(sol);
            vec_t obj_ranges(sol.param.size(), 1.0);

            for(size_t i = 1; i < ref_sol.param.size(); ++i) {
                ref_sol.param[i] = 0.0;
            }

            if(ref_sol.param[0] < 0.0) {
                ref_sol.param[0] = 0.0;
            }
            if(ref_sol.param[0] > 1.0) {
                ref_sol.param[0] = 1.0;
            }

            define_problem_evaluation(ref_sol);

            return ref_sol.transformed_objective_distance(sol, obj_ranges);
        }

    };

}