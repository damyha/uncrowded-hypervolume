/*
 * GenerationalArchive.h
 * Mimics an elitist archive without any discretization and is only designed to be used for a single generation
 *
 * Implementation by D. Ha
 */

#include "../hillvallea_internal.hpp"
#include "../population.hpp"

namespace hillvallea {
    class GenerationalArchive {
    public:
        // Constructor and destructor
        GenerationalArchive();

        virtual ~GenerationalArchive();

        // Archive methods
        void initializeGenerationalMOSolutionsArchive(const population_pt& population);
        int addMOSolutionToGenerationalArchive(hicam::solution_pt solutionToAdd);

        // Getters
        const std::vector<hicam::solution_pt> &getArchiveMoSolutions() const;

        // Helper methods
        static bool solutionDominates(const hicam::solution_pt& solutionA, const hicam::solution_pt& solutionB);

    private:
        std::vector<hicam::solution_pt> archiveMOSolutions; // The MO solutions in the archive


    };
}
