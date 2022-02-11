# tsp-momentum
Solving a question from stack overflow (https://stackoverflow.com/questions/71072404/travelling-salesman-with-momentum/71073168#71073168).

Here's the node numbers that I use:

![numbering](numbers.png)

The solution is 

6 -> 4 -> 3 -> 0 -> 1 -> 7 -> 5 -> 0 -> 1 -> 2 -> 4 -> 6 -> ...

Note that I had to add one non-smooth edge 6->7->6, otherwise solution doesn't exist.

But if you don't care about having a strict loop (required for "proper" TSP),
you can add it near the start node, and then "break it off" after solution is found.
