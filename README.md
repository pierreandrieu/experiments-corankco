# Reproduce experiments of IJAR research paper on rank aggregation

## How to run it:

1. **Import the project:**

`git pull https://github.com/pierreandrieu/experiments-corankco'`


2. **Navigate to the project folder.**

3. **Download the CPLEX solver binary:**

- Visit the [IBM ILOG CPLEX Optimization Studio website](https://www.ibm.com/products/ilog-cplex-optimization-studio)
- Follow the instructions to download the CPLEX binary suitable for your operating system. Note that there is a free version for academic uses.
- Once downloaded, place the binary file into the project folder.

4. **If needed, adjust the script or configuration files to reflect the name of the downloaded CPLEX binary file.**

5. **Build the Docker container:**

`docker build . --tag name_of_container`


6. **Run Docker:**

`docker container run -it --rm name_of_container args`


## Remarks

- Arguments to use:
- `args` are the arguments for the main script.
- Example: `docker container run -it --rm name_of_container -exp=3,5,6` will run the experiments 3, 5 and 6 of the paper (integers between 1 and 6, separated with a comma, no space).
- To reproduce all the experiments, the argument is "all".
- A user guide is displayed if no arguments are given.

- The experiments are run in increasing order of time computation.
- Experiment 1 needs ~36h, Experiment 2 needs ~ 24h, Experiment 3 and 4 need ~ 90 min, Experiment 5 needs ~ 2h, Experiment 6 needs ~ 30 min

