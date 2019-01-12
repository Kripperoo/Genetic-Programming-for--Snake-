# This version of the snake game allows you to play the same yourself using the arrow keys.
# Be sure to run the game from a terminal, and not within a text editor!

import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
import random

curses.initscr()
XSIZE,YSIZE = 18,18
NFOOD = 1

win = curses.newwin(YSIZE, XSIZE, 0, 0)
win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)

def placeFood(snake, food):
	for last in food:
		win.addch(last[0], last[1], ' ')
	food = []
	while len(food) < NFOOD:
		potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
		if not (potentialfood in snake) and not (potentialfood in food):
			food.append(potentialfood)
			win.addch(potentialfood[0], potentialfood[1], '*')
	return( food )

def playGame():
	score = 0
	key = KEY_RIGHT
	snake = [[4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]                                     # Initial snake co-ordinates
	food = []
	food = placeFood(snake,food)
	win.timeout(150)
	wasAhead = []
	ahead = []

	A = "NO"
	while True:
		win.border(0)

		prevKey = key                                                  # Previous key pressed
		event = win.getch()
		key = key if event == -1 else event 

		if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:     # If an invalid key is pressed
			key = prevKey

	    # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases
	    # This is taken care of later at [1] (where we pop the tail)
		snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])

	    # Game over if the snake goes through a wall
		if snake[0][0] == 0 or snake[0][0] == (YSIZE-1) or snake[0][1] == 0 or snake[0][1] == (XSIZE-1): break

		ahead = [ snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)] 
		if ahead in snake:
			A = "YES"

	    # Game over if the snake runs over itself
		if snake[0] in snake[1:]: break
		if snake[0] in food:                                            # When snake eats the food
			score += 1
			food = placeFood(snake,food)
		else:    
			last = snake.pop()                                          # [1] If it does not eat the food, it moves forward and so last tail item is removed
			win.addch(last[0], last[1], ' ')

		win.addch(snake[0][0], snake[0][1], '#')
	curses.endwin()
	print(A)
	print("\nFinal score - " + str(score))

	print(wasAhead)

playGame()

