# pragma once
/*
 * Implementation by D. Ha, 2021
 * This file describes the sphere-michalewicz problem
 */

# include <cmath>

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam {
    class SphereMichalewicz : public fitness_t
    {
    public:
        // Parameters of objective function
        double m_value;

        /**
         * Constructor that sets the default values
         */
        SphereMichalewicz()
        {
            number_of_objectives = 2; // Fixed value
            number_of_parameters = 2; // Default value

            // Set recommended values of objective function parameters
            m_value = 10.0;

            partial_evaluations_available = false;
            analytical_gradient_available = true;

            hypervolume_max_f0 = 11;
            hypervolume_max_f1 = 11;

            analytical_gd_avialable = true;
        }

        ~SphereMichalewicz(){}

        /**
         * Returns the optimal fitness value when solving michalewicz dimension wise
         * Optimized for m=10
         * @param dimension The dimension
         * @return The fitness value obtained when solving the fitness dimensionwise
         */
        double optimum_michalewicz(size_t dimension)
        {
            switch (dimension)
            {
                case 1: return -0.8013034100985525208327905368534731107111760870576881;
                case 2: return -0.99999999999999996996486874751876150388019062265359458472282;
                case 3: return -0.95909126989600582640994390439224082644370123561121197;
                case 4: return -0.9384624184720831665605960617364062545680284538548816;
                case 5: return -0.98880108062150403740506951579979516000250470253417585;
                case 6: return -0.99999999999999977749501660411932573507463434690621793678262;
                case 7: return -0.99322713535588154231713836888182876154017872171089700;
                case 8: return -0.98287203627221048586184588350233727301032082308701122;
                case 9: return -0.99639436492510295378148150149773034993391832764411368;
                case 10: return -0.9999999999999996769087101358407148279034046301247881377422;
                default: return 0;
            }
        }

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
         * Returns the lower and upper bounds of this problem
         * @param lower A pointer to the vec_t to store the lower bounds
         * @param upper A pointer to the vec_t to store the upper bounds
         */
        void get_param_bounds(vec_t & lower, vec_t & upper) const
        {
            lower.clear();
            lower.resize(number_of_parameters, 0);

            upper.clear();
            upper.resize(number_of_parameters, M_PI);
        }

        /**
         * Calculates the sum square of x_c, i.e. sum((x-center)^2), where the center is [pi/2, ..., pi/2]
         * @param solution The solution to calculate the sum of squares of
         * @return
         */
        double calculateObjectiveValueSphere(solution_t & solution)
        {
            // Initialize objective parameters
            double centerValue = M_PI_2;

            double result = 0.0;
            for(size_t i = 0; i < solution.param.size(); ++i) {
                result += (solution.param[i]-centerValue) * (solution.param[i] - centerValue);
            }

            return result;
        }

        /**
         * Calculates the gradient of the sphere, which is 2*(x-center), where the center is [pi/2, ..., pi/2]
         * @param solution The solution to calculate the gradient of
         */
        void assignGradientSphere(solution_t & solution)
        {
            // Initialize objective parameters
            double centerValue = M_PI_2;

            for(size_t i = 0; i < solution.param.size(); ++i) {
                // Set the center w.r.t the objective's center
                double x_c = solution.param[i] - centerValue;

                solution.gradients[0][i] = 2*x_c;
            }
        }

        /**
         * Calculates the Michalewicz functions
         * @param solution The solution to calculate the Michalewicz value of
         * @return
         */
        double calculateObjectiveValueMichalewicz(solution_t & solution)
        {
            double result = 0.0;
            for(size_t i = 0; i < solution.param.size(); ++i) {
                size_t dimension = i + 1;
                double x_i = solution.param[i];
                double sin_x_i = std::sin(x_i);
                double sin_right = std::sin((double) (dimension) * x_i * x_i / M_PI);

                result += (-sin_x_i * std::pow(sin_right, 2*m_value)) - optimum_michalewicz(dimension);
            }

            return result;
        }

        /**
         * Calculates the gradient of the Michalewicz function, which is:
         * -(cos(x_j)*sin((j/PI)*x_j^2)^(2*m) + 4*m*j*x_j/PI*sin(x_j)*sin(j/PI*x_j^2)^(2*m-1) *cos((j/PI)*x_j^2)) =
         * -(sin((j/PI)*x_j^2)^(2m-1) * (4*m*j/PI * x_j*sin(x_j)*cos((j/PI)*x_j^2)) + cos(x_j)*sin((j/PI)*x_j^2) ))
         *  for dim j
         * @param solution The solution to calculate the gradient of
         */
        void assignGradientMichalewicz(solution_t & solution)
        {
            for(size_t i = 0; i < solution.param.size(); ++i) {
                // Retrieve variable
                double x = solution.param[i];

                // Determine intermediate values
                double sin_x = std::sin(x);
                double cos_x = std::cos(x);

                double sin_x_adv = std::sin((double) (i+1)*x*x / M_PI);
                double cos_x_adv = std::cos((double) (i+1)*x*x / M_PI);

                double sin_x_adv_pow = std::pow(sin_x_adv, (2*m_value) - 1); // sin(i*x_i/PI)^(2m-1)

                double constant_left = 4*m_value*(double)(i+1) / M_PI;

                solution.gradients[1][i] = -(sin_x_adv_pow*(constant_left * x * sin_x * cos_x_adv + cos_x * sin_x_adv));
            }
        }

        /**
         * Given a solution, this method calculates the objective values
         * @param sol The solution to inspect
         */
        void define_problem_evaluation(solution_t & sol)
        {
            // Objective values
            sol.obj[0] = calculateObjectiveValueSphere(sol);
            sol.obj[1] = calculateObjectiveValueMichalewicz(sol);

            // Constraint value
            sol.constraint = 0;
        }

        /**
         * Given a solution, this method calculates the objective values and the gradient
         * @param sol The solution to inspect
         */
        void define_problem_evaluation_with_gradients(solution_t & sol)
        {
            // Objective values
            sol.obj[0] = calculateObjectiveValueSphere(sol);
            sol.obj[1] = calculateObjectiveValueMichalewicz(sol);

            // Constraint value
            sol.constraint = 0;

            // Initialize gradients
            sol.gradients.resize(number_of_objectives);
            sol.gradients[0].resize(number_of_parameters);
            sol.gradients[1].resize(number_of_parameters);

            // Gradients
            assignGradientSphere(sol);
            assignGradientMichalewicz(sol);
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
            return "sphere_michalewicz";
        }

        /**
         * Retrieve the pareto set.
         * @return A boolean if the process has been completed.
         */
        bool get_pareto_set()
        {
            return false;

//            size_t pareto_set_size = 5000;
//
//            // Generate default front
//            if (pareto_set.size() != pareto_set_size)
//            {
//                pareto_set.sols.clear();
//                pareto_set.sols.reserve(pareto_set_size);
//
//                // Create the front
//                for (size_t i = 0; i < pareto_set_size; ++i)
//                {
//                    solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);
//
//                    sol->param.fill(0);
//                    sol->param[0] = (i / ((double) pareto_set_size ));
//                    define_problem_evaluation(*sol); // runs a feval without registering it.
//
//                    pareto_set.sols.push_back(sol);
//                }
//
//                igdx_available = true; // Todo: find out what this is
//                igd_available = true;  // Todo: find out what this is
//                // pareto_set.writeToFile("./genMED.txt");
//            }
//
//            return true;
        }

        /**
         * Calculates the distance to the front. Todo: Not sure what happens here
         * @param sol The solution to inspect
         * @return The distance to the front.
         */
        double distance_to_front(const solution_t & sol)
        {
            return 0;

//            solution_t ref_sol(sol);
//            vec_t obj_ranges(sol.param.size(), 1.0);
//
//            for(size_t i = 1; i < ref_sol.param.size(); ++i) {
//                ref_sol.param[i] = 0.0;
//            }
//
//            if(ref_sol.param[0] < 0.0) {
//                ref_sol.param[0] = 0.0;
//            }
//            if(ref_sol.param[0] > 1.0) {
//                ref_sol.param[0] = 1.0;
//            }
//
//            define_problem_evaluation(ref_sol);
//
//            return ref_sol.transformed_objective_distance(sol, obj_ranges);
        }

    };

}