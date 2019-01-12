# Evolve Snake using Genetic Programming 

This respositry is a python implementation of genetic programming used to solve classic computer game 'snake'.

## How to run

First build a virual enviroment through terminal

```js
virtualenv venv
source venv/bin/activate
```

Then install requirements
```js
pip install -r requirements.txt
```
```js
Python snakePlay.py 
```
This version of the snake game allows you to play the same yourself using the arrow keys.
Be sure to run the game from a terminal, and not within a text editor!

Python snakeProblem.py

This version runs a genetic programming which will evolve a decision function to automatically solve the 'snake' game.
Once the evolution is finished, press any key to see how the AI plays the game.

tree.pdf is a example of generated decision function, and you can how the strategy is made from each game step.
```js
Python Bay-opt.py
```
This script runs a bayesian optimization which can give you the best fit parameters.
```js
Python plots.py
```
This script will show you a set of prerunned results.