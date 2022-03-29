/*
 * optimization_branch.h
 *
 * How it works:
 * optimal_hybrid_optimizer should initialize the initial optimization_branch
 *  - Creates a optimization_branch object with new UHV-GOMEA and UHV-ADAM optimizers
 *  optimal_hybrid_optimizer executes the optimization
 *  - Prepares this object for the next optimizer step
 *      - optimal_hybrid_optimizer pushes the current branch and the statistics to its history.
 *  - optimal_hybrid_optimizer then clones the optimization_branch object
 *      - The current population is cloned. At this point the current population can be found in:
 *        the cloned optimization_branch objects, the old optimization_branch object and in the history.
 *        Note that only the pointer in the historyParameters point to the original population (and the currentPopulation in the old object).
 *      - Depending on the settings the optimizers are also cloned or reinitialized
 *  - The optimizers are applied to the branches. At this point, the old optimization_branch object should be gone.
 *  - The process repeats until optimal_hybrid_optimizer is done with a generation
 *  - At the end of a generation, the statistics are written
 *  - In the next generation/at the end the statistical data is flushed
 *
 */

#include "../adam_on_population.h"
#include "../gomea.hpp"
#include "../population.hpp"
#include "../solution.hpp"

#include "../../UHV.hpp"

namespace hillvallea {

    class optimization_branch_t {
    public:
        // Constructor and destructor
        explicit optimization_branch_t(population_pt newPopulation);
        optimization_branch_t(population_pt newPopulation,
                              gomea_pt newOptimizerUHV_GOMEA,
                              adam_on_population_pt newOptimizerUHV_ADAM,
                              std::string optimizerUHV_GOMEAName,
                              std::string optimizerUHV_ADAMName);
        ~optimization_branch_t();

        // Copy branch
        optimization_branch_pt cloneOptimizationBranch();   // Copies this object and the optimizers
        optimization_branch_pt cloneOptimizationBranch(gomea_pt newOptimizerUHV_GOMEA,
                                                       adam_on_population_pt newOptimizerUHV_ADAM,
                                                       std::string optimizerUHV_GOMEAName,
                                                       std::string optimizerUHV_ADAMName); // Copies this object and assigns a new optimizer

        // Optimizers
        void applyUHV_GOMEA(size_t numberOfCalls, std::shared_ptr<UHV_t> fitnessFunction);
        void applyUHV_ADAM(size_t numberOfCalls, std::shared_ptr<UHV_t> fitnessFunction);

        // Getters and setters
        const population_pt &getCurrentPopulation() const;
        const std::string &getLastAppliedOptimizerName() const;
        size_t getCurrentNumberOfGenerations() const;
        size_t getCurrentNumberOfCalls() const;
        size_t getCurrentNumberOfSOEvaluations() const;
        double getCurrentNumberOfMOEvaluations() const;

        const std::string &getPathStatisticsFile() const;
        const std::shared_ptr<std::ofstream> &getStatisticsFile() const;
        const optimization_branch_pt &getParentBranch() const;
        void getHistoryParameters(std::vector<population_pt> &historyPopulations,
                                  std::vector<std::string> &historyOptimizerNames,
                                  std::vector<size_t> &historyGenerations,
                                  std::vector<size_t> &historyCalls,
                                  std::vector<size_t> &historySOEvaluations,
                                  std::vector<double> &historyMOEvaluations) const;

        void setCurrentPopulation(const population_pt &currentPopulation);
        void setOptimizerUhvGomea(const gomea_pt &optimizerUhvGomea);
        void setOptimizerUhvAdam(const adam_on_population_pt &optimizerUhvAdam);
        void setOptimizersInitialized(bool optimizersInitialized);
        void setNameOptimizerUhvGomea(const std::string &nameOptimizerUhvGomea);
        void setNameOptimizerUhvAdam(const std::string &nameOptimizerUhvAdam);
        void setLastAppliedOptimizerName(const std::string &newLastAppliedOptimizerName);
        void setCurrentNumberOfGenerations(size_t currentNumberOfGenerations);
        void setCurrentNumberOfCalls(size_t currentNumberOfCalls);
        void setCurrentNumberOfSoEvaluations(size_t currentNumberOfSoEvaluations);
        void setCurrentNumberOfMoEvaluations(double currentNumberOfMoEvaluations);
        void setParentBranch(const optimization_branch_pt &parentBranch);

        void setStatisticsFileParameters(
                std::string newPathStatisticsFile,
                std::shared_ptr<std::ofstream> newStatisticsFile);  // Setter for statistics_file and path_statistics_file


        // Statistics file methods
        void copyStatisticsFile(std::shared_ptr<std::ofstream> targetStatisticsFile);    // Copy the statistics file
        void closeStatisticsFile();     // Close the statistics file
        void deleteStatisticsFile();    // Closes and deletes the statistics file

        const adam_on_population_pt &getOptimizerUhvAdam() const;


    private:
        // Related branches
        optimization_branch_pt parentBranch;    // The parent that created this branch

        // Population
        population_pt currentPopulation;    // The current population

        // Optimizers and variables
        bool optimizersInitialized;                 // Boolean if optimizers are initialized
        gomea_pt optimizerUHV_GOMEA;                // UHV-GOMEA optimizer
        adam_on_population_pt optimizerUHV_ADAM;    // UHV-ADAM optimizer

        std::string nameOptimizerUHV_GOMEA;         // Name of the UHV-GOMEA optimizer
        std::string nameOptimizerUHV_ADAM;          // Name of the UHV-ADAM optimizer

        // Statistics
        std::string pathStatisticsFile;                 // Path of the statistics file
        std::shared_ptr<std::ofstream> statisticsFile;  // The statistics file associated with this population branch

        std::string lastAppliedOptimizerName;   // The last applied algorithm name as a string
        size_t currentNumberOfCalls;            // The last applied number of calls
        size_t currentNumberOfGenerations;      // The total number of generations elapsed
        size_t currentNumberOfSOEvaluations;    // The total number of SO evaluations applied to this population
        double currentNumberOfMOEvaluations;    // The total number of MO evaluations applied to this population













    };

}