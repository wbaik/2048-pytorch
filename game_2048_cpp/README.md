## Generating Replay Memory

Most of the code is from [here](https://github.com/nneonneo/2048-ai). Because `python` is slower than `c++` in 
doing this kind of work, I revised some of the codes from the repository to suite my needs. 
It now runs on the `gym-2048`.

### https://github.com/nneonneo/2048-ai:

AI for the [2048 game](http://gabrielecirulli.github.io/2048/). This uses *expectimax optimization*, 
along with a highly-efficient bitboard representation to search upwards of 10 million moves per second 
on recent hardware. Heuristics used include bonuses for empty squares and bonuses for placing large values 
near edges and corners. Read more about the algorithm on the [StackOverflow answer](https://stackoverflow.com/a/22498940/1204143).

## Building

### Unix/Linux/OS X

Execute

    ./configure
    make

in a terminal. Any relatively recent C++ compiler should be able to build the output.

Note that you don't do `make install`; this program is meant to be run from this directory.

## Running the command-line version

Run `bin/2048` if you want to see the AI by itself in action.

