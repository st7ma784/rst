Hi Claude, 

This repo has gotten away from me a bit! It started out as a fork of the RST SuperDARN code, and a significant effort has gone in to use CUDA natively, but this has been ratehr hampered by the issues in design and thow the original code is compiled. 

I think a total rewrite is in order! 

First, make a summarisation of how this repo builds the modules into the whole toolkit, as the build is rather complicated. Lets first make that plan. 

Secondly, when we've got that plan, I need you to analyse the architecture of the algorithms. 

a number of the modules use linked lists and are rather innefficent as they cant be parrelelised using tools like CUDA. 

As a general rule, the linked lists were used for fast item deletion when possibilities were ruled out. A better datastructure would be a 2d array with a parralel mask of bool values for discarding them down the road. 

The next task is to plan how to migrate the current system module-by-module to cuda. Plan a new architecture that doesnt break the existing compilation structure. 

Next, lets build the CUDA-based approaches systematically module-by-module. 

when done, download some RST fitacf files, and consult the docs for how to process them into a map. 

Systematically run the files through the old system and generate images of the output. Then, run the same data through your cuda implementation and show the results are the same or better than the other visualisations 

The final task is to ensure that it compiles separately into a CUDArst library that has the same interface as the old packages. 
