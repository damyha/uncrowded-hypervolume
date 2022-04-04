/*
 * population_branch.cpp
 *
 * This class is derived from the population class.
 * This class is a population that keeps track which algorithms were applied on it and which statistics file
 * is associated with it.
 */

#include "optimization_branch.h"

namespace hillvallea {

    /**
     * Constructor that does not initialize the optimizers
     * @param newPopulation The new population of this object
     */
    optimization_branch_t::optimization_branch_t(population_pt newPopulation) {
        // Copy parameters
        this->currentPopulation = newPopulation;

        // Assign default parameters
        this->optimizersInitialized = false;
        this->lastAppliedOptimizerName = "NONE";
        this->currentNumberOfGenerations = 0;
        this->currentNumberOfCalls = 0;
        this->currentNumberOfSOEvaluations = 0;
        this->currentNumberOfMOEvaluations = 0;
    }

    /**
     * Constructor
     * @param newPopulation The population object
     * @param optimizerUHV_GOMEA The UHV-GOMEA optimizer to apply on the current population
     * @param optimizerUHV_ADAM The UHV-ADAM optimizer to apply on the current population
     * @param optimizerUHV_GOMEAName Model name of UHV-GOMEA
     * @param optimizerUHV_ADAMName Model name of UHV-ADAM
     */
    optimization_branch_t::optimization_branch_t(population_pt newPopulation,
                                                 gomea_pt newOptimizerUHV_GOMEA,
                                                 adam_on_population_pt newOptimizerUHV_ADAM,
                                                 std::string optimizerUHV_GOMEAName,
                                                 std::string optimizerUHV_ADAMName) {
        // Copy parameters
        this->currentPopulation = newPopulation;
        this->optimizerUHV_GOMEA = newOptimizerUHV_GOMEA;
        this->optimizerUHV_ADAM = newOptimizerUHV_ADAM;
        this->nameOptimizerUHV_GOMEA = optimizerUHV_GOMEAName;
        this->nameOptimizerUHV_ADAM = optimizerUHV_ADAMName;

        // Assign default parameters
        this->lastAppliedOptimizerName = "NONE";
        this->currentNumberOfGenerations = 0;
        this->currentNumberOfCalls = 0;
        this->currentNumberOfSOEvaluations = 0;
        this->currentNumberOfMOEvaluations = 0;

        // Initialize the optimizers
        this->optimizerUHV_GOMEA->initialize_from_population(this->currentPopulation, this->currentPopulation->size());
        this->optimizerUHV_ADAM->initialize_from_population(this->currentPopulation, this->currentPopulation->size());
        this->optimizersInitialized = true;
    }

    /**
     * Destructor
     */
    optimization_branch_t::~optimization_branch_t() {}

    /**
     * Copies the optimization_branch_t object including the population and optimizers
     * @return A copy of this object with a unique population
     */
    optimization_branch_pt optimization_branch_t::cloneOptimizationBranch() {
        // Deep copy this population
        population_pt copyPopulation = this->currentPopulation->deepCopyPopulation();

        // Deep copy the optimizers and set population pointers
        gomea_pt copyUHV_GOMEA = this->optimizerUHV_GOMEA->clone(copyPopulation);
        adam_on_population_pt copyUHV_ADAM = this->optimizerUHV_ADAM->clone(copyPopulation);

        // Create new optimizer branch and copy over everything. (the other constructor can't be used as it uses 'initialize_from_population')
        optimization_branch_pt newOptimizerBranch = std::make_shared<optimization_branch_t>(copyPopulation);

        // Assign optimizers to new optimizer branch
        newOptimizerBranch->setOptimizerUhvGomea(copyUHV_GOMEA);
        newOptimizerBranch->setOptimizerUhvAdam(copyUHV_ADAM);
        newOptimizerBranch->setNameOptimizerUhvGomea(nameOptimizerUHV_GOMEA);
        newOptimizerBranch->setNameOptimizerUhvAdam(nameOptimizerUHV_ADAM);
        newOptimizerBranch->setOptimizersInitialized(true);

        // Copy other settings
        newOptimizerBranch->setLastAppliedOptimizerName(this->lastAppliedOptimizerName);
        newOptimizerBranch->setCurrentNumberOfGenerations(this->currentNumberOfGenerations);
        newOptimizerBranch->setCurrentNumberOfCalls(this->currentNumberOfCalls);
        newOptimizerBranch->setCurrentNumberOfSoEvaluations(this->currentNumberOfSOEvaluations);
        newOptimizerBranch->setCurrentNumberOfMoEvaluations(this->currentNumberOfMOEvaluations);

        // Copy the statistics files Todo: Check if this also works when multiple trees are explored
        newOptimizerBranch->setStatisticsFileParameters(this->pathStatisticsFile,
                                                        this->statisticsFile);

        return newOptimizerBranch;
    }

    /**
     * Copies this optimization_branch_t object including the population.
     * The optimizers are not copied over, but are supplied from the caller. This is most probably a new optimizer instance.
     * @return A copy of this object with a unique population
     */
    optimization_branch_pt optimization_branch_t::cloneOptimizationBranch(gomea_pt newOptimizerUHV_GOMEA,
                                                                          adam_on_population_pt newOptimizerUHV_ADAM,
                                                                          std::string optimizerUHV_GOMEAName,
                                                                          std::string optimizerUHV_ADAMName) {
        // Deep copy current population
        population_pt copyPopulation = this->currentPopulation->deepCopyPopulation();

        // Create new optimizer branch: (Also reinitializes the optimizers)
        optimization_branch_pt newOptimizerBranch = std::make_shared<optimization_branch_t>(copyPopulation,
                                                                                            newOptimizerUHV_GOMEA,
                                                                                            newOptimizerUHV_ADAM,
                                                                                            optimizerUHV_GOMEAName,
                                                                                            optimizerUHV_ADAMName);

        // Copy other settings
        newOptimizerBranch->setLastAppliedOptimizerName(this->lastAppliedOptimizerName);
        newOptimizerBranch->setCurrentNumberOfGenerations(this->currentNumberOfGenerations);
        newOptimizerBranch->setCurrentNumberOfCalls(this->currentNumberOfCalls);
        newOptimizerBranch->setCurrentNumberOfSoEvaluations(this->currentNumberOfSOEvaluations);
        newOptimizerBranch->setCurrentNumberOfMoEvaluations(this->currentNumberOfMOEvaluations);

        // Copy the statistics files Todo: Check if this also works when multiple trees are explored
        newOptimizerBranch->setStatisticsFileParameters(this->pathStatisticsFile,
                                                        this->statisticsFile);

        return newOptimizerBranch;
    }

    /**
     * Applies UHV-GOMEA on the current population
     * Assumes that prepareToBranchOff has been called before this method is executed.
     * @param numberOfCalls The number of calls to execute
     * @param fitnessFunction The fitness function used to determine resources spent
     */
    void optimization_branch_t::applyUHV_GOMEA(size_t numberOfCalls, std::shared_ptr<UHV_t> fitnessFunction) {
        // Prepare statistics variables
        int numberOfSOEvaluationsSpent = 0;
        double currentNumberOfMOEvaluationsSpent = fitnessFunction->number_of_mo_evaluations;

        // Prepare best solution
        optimizerUHV_GOMEA->best = *optimizerUHV_GOMEA->pop->bestSolution();

        // Execute optimizer
        for (size_t callsExecuted = 0; callsExecuted < numberOfCalls; ++callsExecuted) {
            optimizerUHV_GOMEA->generation(currentPopulation->size(), numberOfSOEvaluationsSpent);
        }

        // Determine statistics
        double numberOfMOEvaluationsSpent =
                fitnessFunction->number_of_mo_evaluations - currentNumberOfMOEvaluationsSpent;

        // Write statistics
        this->lastAppliedOptimizerName = this->nameOptimizerUHV_GOMEA;
        this->currentNumberOfGenerations += 1;
        this->currentNumberOfCalls = numberOfCalls;
        this->currentNumberOfSOEvaluations += numberOfSOEvaluationsSpent;
        this->currentNumberOfMOEvaluations += numberOfMOEvaluationsSpent;

    }

    /**
     * Applies UHV-ADAM on the current population
     * Assumes that prepareToBranchOff has been called before this method is executed.
     * @param numberOfCalls The number of calls to execute
     * @param fitnessFunction The fitness function used to determine resources spent
     */
    void optimization_branch_t::applyUHV_ADAM(size_t numberOfCalls, std::shared_ptr<UHV_t> fitnessFunction) {
        // Prepare statistics variables
        int numberOfSOEvaluationsSpent = 0;
        double currentNumberOfMOEvaluationsSpent = fitnessFunction->number_of_mo_evaluations;

        // Prepare population by computing gradient and resetting values
        std::vector<size_t> solutionIndicesToInspect =
                optimizerUHV_ADAM->determineSolutionIndicesToApplyAlgorithmOn(numberOfCalls);
        for (auto const &solIndex : solutionIndicesToInspect) {
            fitnessFunction->evaluate_with_gradients(optimizerUHV_ADAM->pop->sols[solIndex]);

            optimizerUHV_ADAM->pop->sols[solIndex]->adam_mt.assign(optimizerUHV_ADAM->pop->sols[solIndex]->adam_mt.size(), 0);
            optimizerUHV_ADAM->pop->sols[solIndex]->adam_vt.assign(optimizerUHV_ADAM->pop->sols[solIndex]->adam_vt.size(), 0);
        }

        // Determine if gradient came free or should be counted
        if(!fitnessFunction->use_finite_differences) {
            fitnessFunction->number_of_mo_evaluations = currentNumberOfMOEvaluationsSpent;
        }

        // Execute optimizer
        optimizerUHV_ADAM->generation(currentPopulation->size(), numberOfSOEvaluationsSpent, numberOfCalls);

        // Determine statistics
        double numberOfMOEvaluationsSpent =
                fitnessFunction->number_of_mo_evaluations - currentNumberOfMOEvaluationsSpent;

        // Write statistics
        this->lastAppliedOptimizerName = this->nameOptimizerUHV_ADAM;
        this->currentNumberOfGenerations += 1;
        this->currentNumberOfCalls = numberOfCalls;
        this->currentNumberOfSOEvaluations += numberOfSOEvaluationsSpent;
        this->currentNumberOfMOEvaluations += numberOfMOEvaluationsSpent;
    }

    /**
     * Given a pointer to a ofstream, the contents of the statistics file of this object are copied over.
     * @param targetStatisticsFile The statistics file to write to
     */
    void optimization_branch_t::copyStatisticsFile(std::shared_ptr<std::ofstream> targetStatisticsFile) {
        if (this->pathStatisticsFile.empty()) {
            // Check if this object has a path to the statistics file
            throw std::runtime_error("Tried to copy an population branch without statistics file.");

        } else if (targetStatisticsFile->is_open()) {
            // Check that the output file is open
            throw std::runtime_error("Destination file was not open when copying existing statistics file.");

        } else {
            // Prepare streams
            std::ifstream src(this->pathStatisticsFile);

            // Copy the file
            *targetStatisticsFile << src.rdbuf();

            // Close ifstream
            src.close();
        }
    }

    /**
     * Close the statistics file
     */
    void optimization_branch_t::closeStatisticsFile() {
        if (this->statisticsFile->is_open())
            this->statisticsFile->close();
    }

    /**
     * Deletes the statistics file of this object
     */
    void optimization_branch_t::deleteStatisticsFile() {
        this->closeStatisticsFile();
        std::remove(this->pathStatisticsFile.c_str());  // c_str converts string to *char
    }

    /// Getters and setters
    const optimization_branch_pt &optimization_branch_t::getParentBranch() const {
        return parentBranch;
    }


    const population_pt &optimization_branch_t::getCurrentPopulation() const {
        return this->currentPopulation;
    }

    const std::string &optimization_branch_t::getLastAppliedOptimizerName() const {
        return this->lastAppliedOptimizerName;
    }

    size_t optimization_branch_t::getCurrentNumberOfGenerations() const {
        return this->currentNumberOfGenerations;
    }

    size_t optimization_branch_t::getCurrentNumberOfCalls() const {
        return currentNumberOfCalls;
    }

    size_t optimization_branch_t::getCurrentNumberOfSOEvaluations() const {
        return this->currentNumberOfSOEvaluations;
    }

    double optimization_branch_t::getCurrentNumberOfMOEvaluations() const {
        return this->currentNumberOfMOEvaluations;
    }

    const std::string &optimization_branch_t::getPathStatisticsFile() const {
        return this->pathStatisticsFile;
    }

    const std::shared_ptr<std::ofstream> &optimization_branch_t::getStatisticsFile() const {
        return this->statisticsFile;
    }

    void optimization_branch_t::setCurrentPopulation(const population_pt &currentPopulation) {
        this->currentPopulation = currentPopulation;
    }

    void optimization_branch_t::setOptimizerUhvGomea(const gomea_pt &optimizerUhvGomea) {
        this->optimizerUHV_GOMEA = optimizerUhvGomea;
    }

    void optimization_branch_t::setOptimizersInitialized(bool optimizersInitialized) {
        optimization_branch_t::optimizersInitialized = optimizersInitialized;
    }

    void optimization_branch_t::setOptimizerUhvAdam(const adam_on_population_pt &optimizerUhvAdam) {
        this->optimizerUHV_ADAM = optimizerUhvAdam;
    }

    void optimization_branch_t::setNameOptimizerUhvGomea(const std::string &nameOptimizerUhvGomea) {
        this->nameOptimizerUHV_GOMEA = nameOptimizerUhvGomea;
    }

    void optimization_branch_t::setNameOptimizerUhvAdam(const std::string &nameOptimizerUhvAdam) {
        this->nameOptimizerUHV_ADAM = nameOptimizerUhvAdam;
    }

    void optimization_branch_t::setLastAppliedOptimizerName(const std::string &newLastAppliedOptimizerName) {
        this->lastAppliedOptimizerName = newLastAppliedOptimizerName;
    }

    void optimization_branch_t::setCurrentNumberOfGenerations(size_t currentNumberOfGenerations) {
        this->currentNumberOfGenerations = currentNumberOfGenerations;
    }

    void optimization_branch_t::setCurrentNumberOfCalls(size_t currentNumberOfCalls) {
        optimization_branch_t::currentNumberOfCalls = currentNumberOfCalls;
    }

    void optimization_branch_t::setCurrentNumberOfSoEvaluations(size_t currentNumberOfSoEvaluations) {
        this->currentNumberOfSOEvaluations = currentNumberOfSoEvaluations;
    }

    void optimization_branch_t::setCurrentNumberOfMoEvaluations(double currentNumberOfMoEvaluations) {
        this->currentNumberOfMOEvaluations = currentNumberOfMoEvaluations;
    }

    void optimization_branch_t::setParentBranch(const optimization_branch_pt &parentBranch) {
        this->parentBranch = parentBranch;
    }

    /**
     * Setter for variable pathStatisticsFile and statisticsFile
     * @param newPathStatisticsFile The new path of the statistics file
     * @param newStatisticsFile The new statisticsFile location
     */
    void optimization_branch_t::setStatisticsFileParameters(std::string newPathStatisticsFile,
                                                            std::shared_ptr<std::ofstream> newStatisticsFile) {
        this->pathStatisticsFile = newPathStatisticsFile;
        this->statisticsFile = newStatisticsFile;
    }

    /**
     * Retrieves all historic data of branches.
     * This method goes trough all parent branches and returns lists of data, where the first entry is the eldest branch (after cutting off)
     * @param historyPopulations The parent populations + current population before this branch
     * @param historyOptimizerNames The optimizers names applied to get to this branch
     * @param historyGenerations The number of generations per branch to get to this branch
     * @param historyCalls The number of calls per branch to get to this branch
     * @param historySOEvaluations The number of SO evaluations per branch to get to this branch
     * @param historyMOEvaluations The number of MO evaluations per branch to get to this branch
     */
    void optimization_branch_t::getHistoryParameters(std::vector<population_pt> &historyPopulations,
                                                     std::vector<std::string> &historyOptimizerNames,
                                                     std::vector<size_t> &historyGenerations,
                                                     std::vector<size_t> &historyCalls,
                                                     std::vector<size_t> &historySOEvaluations,
                                                     std::vector<double> &historyMOEvaluations) const {
        // Initialize variables to store results
        std::vector<population_pt> retrievedPopulations;
        std::vector<std::string> retrievedOptimizerNames;
        std::vector<size_t> retrievedGenerations, retrievedCalls, retrievedSOEvaluations;
        std::vector<double> retrievedMOEvaluations;

        // Do Bottom to top data retrieval
        optimization_branch_pt nextParentBranch = this->parentBranch;
        while(nextParentBranch)
        {
            // For clarity
            optimization_branch_pt currentParentBranch = nextParentBranch;

            // Append data
            retrievedPopulations.push_back(currentParentBranch->getCurrentPopulation());
            retrievedOptimizerNames.push_back(currentParentBranch->getLastAppliedOptimizerName());
            retrievedGenerations.push_back(currentParentBranch->getCurrentNumberOfGenerations());
            retrievedCalls.push_back(currentParentBranch->getCurrentNumberOfCalls());
            retrievedSOEvaluations.push_back(currentParentBranch->getCurrentNumberOfSOEvaluations());
            retrievedMOEvaluations.push_back(currentParentBranch->getCurrentNumberOfMOEvaluations());

            // Retrieve next parent branch
            nextParentBranch = currentParentBranch->parentBranch;
        }

        // Reverse the order of the vectors
        std::reverse(retrievedPopulations.begin(), retrievedPopulations.end());
        std::reverse(retrievedOptimizerNames.begin(), retrievedOptimizerNames.end());
        std::reverse(retrievedGenerations.begin(), retrievedGenerations.end());
        std::reverse(retrievedCalls.begin(), retrievedCalls.end());
        std::reverse(retrievedSOEvaluations.begin(), retrievedSOEvaluations.end());
        std::reverse(retrievedMOEvaluations.begin(), retrievedMOEvaluations.end());

        // Assign results
        historyPopulations = retrievedPopulations;
        historyOptimizerNames = retrievedOptimizerNames;
        historyGenerations = retrievedGenerations;
        historyCalls = retrievedCalls;
        historySOEvaluations = retrievedSOEvaluations;
        historyMOEvaluations = retrievedMOEvaluations;
    }

    // Todo: Delete
    const adam_on_population_pt &optimization_branch_t::getOptimizerUhvAdam() const {
        return optimizerUHV_ADAM;
    }
}



