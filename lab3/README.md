This folder contains parallel qsort (hypersort?) implementations using MPI

There are two implementations: 
* using native python lists for storing array (and using high-level gather for MPI interchange)
* using numpy arrays for the array (and using low-level gatherv for interchange)