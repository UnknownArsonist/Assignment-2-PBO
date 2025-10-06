Exercises 1, 2, and 4 can be ran via main.py, which uses a function get_algorithm that expects
the first argument from the command line to be the name of one of the four algorithms (RLS, EA, MMAS, MMAS*). 
For example, "python main.py MMAS" will run our MMAS implementation from exercise 4. 


Exercise 3 is ran separately via uniform_ga_mut.py. 
Running "python uniform_ga_mut.py" will run the algorithm with defaults n=100, runs=10, budget=100000.
You can customise these using args as declared in the file. For example, python uniform_ga_mut.py --functions 1 2 3
would only run the first three PBO problem/functions. 


Exercise 5 is ran separately via run_aco.py (i.e., "python run_aco.py"). 