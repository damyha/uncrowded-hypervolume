#pragma once

/*

Implementation by Damy Ha

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

    class biRastrigin_t : public fitness_t
    {

    public:

        double A;
        double optimum_x0_shift;
        biRastrigin_t()
        {
            number_of_objectives = 2; // fixed
            number_of_parameters = 2; // default, can be adapted

            partial_evaluations_available = false;

            hypervolume_max_f0 = 11;
            hypervolume_max_f1 = 11;

            analytical_gd_avialable = true;
            analytical_gradient_available = true;

            A = 10;
            optimum_x0_shift = 3.5;
        }
        ~biRastrigin_t() {}

        // number of objectives
        void set_number_of_objectives(size_t & number_of_objectives)
        {
            this->number_of_objectives = 2;
            number_of_objectives = this->number_of_objectives;
        }

        // any positive value
        void set_number_of_parameters(size_t & number_of_parameters)
        {
            this->number_of_parameters = number_of_parameters;
        }


        void get_param_bounds(vec_t & lower, vec_t & upper) const
        {

            lower.clear();
            lower.resize(number_of_parameters, -1000);

            upper.clear();
            upper.resize(number_of_parameters, 1000);

        }

        void define_problem_evaluation(solution_t & sol)
        {
            assert(sol.param.size() >= 1);

            // f1
            sol.obj[0] = A * sol.param.size();

            for(size_t i = 0; i < sol.param.size(); ++i) {
                sol.obj[0] += sol.param[i] * sol.param[i] - A * cos(2*PI*sol.param[i]);
            }
            sol.obj[0] /= (A * sol.param.size());

            // f2
            sol.obj[1] = A * sol.param.size();
            sol.obj[1] += (sol.param[0] - optimum_x0_shift) * (sol.param[0] - optimum_x0_shift) - A * cos(2*PI*(sol.param[0] - optimum_x0_shift));

            for(size_t i = 1; i < sol.param.size(); ++i) {
                sol.obj[1] += sol.param[i] * sol.param[i] - A * cos(2*PI*sol.param[i]);
            }
            sol.obj[1] /= (A * sol.param.size() + optimum_x0_shift*optimum_x0_shift);

            sol.constraint = 0;
        }

        void define_problem_evaluation_with_gradients(solution_t & sol)
        {
            assert(sol.param.size() >= 1);

            // f1
            sol.obj[0] = A * sol.param.size();

            for(size_t i = 0; i < sol.param.size(); ++i) {
                sol.obj[0] += sol.param[i] * sol.param[i] - A * cos(2*PI*sol.param[i]);
            }
            sol.obj[0] /= (A * sol.param.size());

            // f2
            sol.obj[1] = A * sol.param.size();
            sol.obj[1] += (sol.param[0] - optimum_x0_shift) * (sol.param[0] - optimum_x0_shift) - A * cos(2*PI*(sol.param[0] - optimum_x0_shift));

            for(size_t i = 1; i < sol.param.size(); ++i) {
                sol.obj[1] += sol.param[i] * sol.param[i] - A * cos(2*PI*sol.param[i]);
            }
            sol.obj[1] /= (A * sol.param.size() + optimum_x0_shift*optimum_x0_shift);


            sol.constraint = 0;

            // compute gradients
            sol.gradients.resize(number_of_objectives); // 2
            sol.gradients[0].resize(number_of_parameters);
            sol.gradients[1].resize(number_of_parameters);

            // f1
            for(size_t i = 0; i < sol.param.size(); ++i) {
                sol.gradients[0][i] = 2 * sol.param[i] + 2 * PI * A * sin(2*PI*sol.param[i]);
                sol.gradients[0][i] /= (A * sol.param.size());
            }

            // f2
            sol.gradients[1][0] = 2 * (sol.param[0] - optimum_x0_shift) + 2 * PI * A * sin(2*PI*(sol.param[0] - optimum_x0_shift));
            sol.gradients[1][0] /= (A * sol.param.size() + optimum_x0_shift*optimum_x0_shift);

            for(size_t i = 1; i < sol.param.size(); ++i) {
                sol.gradients[1][i] = 2 * sol.param[i] + 2 * PI * A * sin(2*PI*sol.param[i]);
                sol.gradients[1][i] /= (A * sol.param.size() + optimum_x0_shift*optimum_x0_shift);
            }


        }

        std::string name() const
        {
            std::ostringstream ss;
            ss << "biRastrigin_" << A;
            return ss.str();
        }


        bool get_pareto_set()
        {
            // Todo: Incorrect I think
            size_t pareto_set_size = 5000;

            // generate default front
            if (pareto_set.size() == 0)
            {

                rng_pt rng = std::make_shared<rng_t>(100);
                size_t target_size = (size_t) (pareto_set_size*1.25);
                elitist_archive_t elitist_archive(target_size, rng);

                // the front
                for (size_t i = 0; i < pareto_set_size; ++i)
                {
                    solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);

                    sol->param.fill(0);
                    sol->param[0] = optimum_x0_shift * (i / ((double)pareto_set_size - 1.0));
                    define_problem_evaluation(*sol); // runs a feval without registering it.

                    // pareto_set.sols.push_back(sol);
                    elitist_archive.updateArchive(sol);
                }

                elitist_archive.removeSolutionNullptrs();

                // std::cout << "Pareto set archive size = " << elitist_archive.actualSize() << std::endl;

                pareto_set.sols.clear();
                pareto_set.sols.reserve(pareto_set_size);

                for(size_t i = 0; i < elitist_archive.size(); ++i) {
                    if(elitist_archive.sols[i] != nullptr) {
                        pareto_set.sols.push_back(elitist_archive.sols[i]);
                        // std::cout << elitist_archive.sols[i]->obj << " " << elitist_archive.sols[i]->param[0] << std::endl;
                    }
                }

                igdx_available = true;
                igd_available = true;
                // pareto_set.writeToFile("./genMED.txt");
            }

            return true;

        }

        double distance_to_front(const solution_t & sol)
        {
            // Todo: Incorrect I think
            solution_t ref_sol(sol);
            vec_t obj_ranges(sol.param.size(), 1.0);

            for(size_t i = 1; i < ref_sol.param.size(); ++i) {
                ref_sol.param[i] = 0.0;
            }

            if(ref_sol.param[0] < 0) {
                ref_sol.param[0] = 0;
            }
            if(ref_sol.param[0] > optimum_x0_shift) {
                ref_sol.param[0] = optimum_x0_shift;
            }

            define_problem_evaluation(ref_sol);

            return ref_sol.transformed_objective_distance(sol, obj_ranges);
        }

    };
}
