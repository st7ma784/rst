Hi Claude, 

This repo has gotten away from me a bit! It started out as a fork of the RST SuperDARN code, and a significant effort has gone in to use CUDA natively, but this has been ratehr hampered by the issues in design and thow the original code is compiled. 

I think a total rewrite is in order! 

Please rewrite the tools,scripts and utilities in this repo into python, making use of libraries like CUPy that offer CUDA acceleration with python support.

Begin writing this in a new folder called pythonv2 and include documentations and visualisations  where convenient. 

You may find it useful to condense tools, reuse blocks of code where possible to simplify a complex code base into a very streamlined python package.

Begin by summarising how the code base works, make a plan of the new software design so that data storage can make maximal use of accelerators, so emphasise CUDA and GPU support over memory efficiency. 

Once a software architecture has been drawn out in the python folder, begin implementing it. 
