/*
 * GenerationalArchive.h
 * Mimics an elitist archive without any discretization and is only designed to be used for a single generation
 *
 * Implementation by D. Ha
 */

#include "GenerationalArchive.h"

namespace hillvallea{
    /**
     * Constructor
     */
    GenerationalArchive::GenerationalArchive() {}

    /**
     * Destructor
     */
    GenerationalArchive::~GenerationalArchive() { }


    /**
     * Finds the non dominated solutions of the current population and puts them into 'archiveMOSolutions'
     */
    void GenerationalArchive::initializeGenerationalMOSolutionsArchive(const population_pt& population)
    {
        // Reset the archive
        archiveMOSolutions.resize(0);

        // Go over all MO solutions of the population
        for(const auto &solutionOfPopulation : population->sols) {
            for( const auto &moSolutionOfPopulation : solutionOfPopulation->mo_test_sols) {
                addMOSolutionToGenerationalArchive(moSolutionOfPopulation);
            }
        }
    }


    /**
     * Adds an MO solution to the generational MO solutions archive.
     * @param solutionToAdd The MO solution to add to the generational archive
     * @return '0' if it was not added, '1' if it was added.
     */
    int GenerationalArchive::addMOSolutionToGenerationalArchive(hicam::solution_pt moSolutionToAdd)
    {
        bool isDominated = false;

        // Determine if any MO solution in the archive dominates 'solutionToAdd' or is dominated by 'solutionToAdd'
        for( size_t archiveIndex = 0; archiveIndex < archiveMOSolutions.size(); ++archiveIndex) {
            hicam::solution_pt moSolutionOfArchive =  archiveMOSolutions[archiveIndex];

            // Check if MO solution of the archive dominates the solution to add to the archive
            if (solutionDominates(moSolutionOfArchive, moSolutionToAdd)) {
                isDominated = true;
                break;
            }

            //  Check if current MO solution dominates the solution of the archive
            if (solutionDominates(moSolutionToAdd, moSolutionOfArchive)) {
                // Remove dominated solution
                archiveMOSolutions.erase(archiveMOSolutions.begin()+archiveIndex);
            }
        }

        // Append the previous solution if it's not dominated
        if (!isDominated) {
            archiveMOSolutions.push_back(moSolutionToAdd);
        }

        return (int) !isDominated;
    }

    const std::vector<hicam::solution_pt> &GenerationalArchive::getArchiveMoSolutions() const {
        return archiveMOSolutions;
    }


    /**
     * Check if solution A dominates solution B assuming that there are no constraints and only two objectives.
     * @param solutionA Solution that should dominate
     * @param solutionB Solution that should be dominated
     * @return true if solution A dominates solution B else false
     */
    bool GenerationalArchive::solutionDominates(const hicam::solution_pt& solutionA,
                                                const hicam::solution_pt& solutionB)
    {
        hicam::vec_t objectivesA = solutionA->obj;
        hicam::vec_t objectivesB = solutionB->obj;

        return ((objectivesA[0] <  objectivesB[0] || objectivesA[1] <  objectivesB[1]) &&
                (objectivesA[0] <= objectivesB[0] && objectivesA[1] <= objectivesB[1]));
    }
}