# Connect Four AI

A Connect Four game I built in Python while learning about adversarial search and game-playing AI.

The main part of the project is an AI opponent based on **Minimax with Alpha-Beta pruning**. I also wanted it to feel like an actual game rather than just an AI demo, so I built the interface, animations, difficulty system and sound effects using Pygame.

## AI

The AI searches possible future moves using Minimax and uses Alpha-Beta pruning to avoid exploring branches that won't affect the final decision.

There are three difficulty levels:

- Kid - depth 2
- Teen - depth 3
- Adult - depth 5

The evaluation function doesn't just look for a win. It scores board positions based on things like centre-column control, threats and groups of 2, 3 or 4 pieces.

## Testing it

I played 10 games against each difficulty to see whether increasing the search depth actually made a noticeable difference.

My results were:

- Kid: 10/10 wins
- Teen: 4/10 wins
- Adult: 0-1/10 wins

It was a simple test rather than a formal AI benchmark, but it gave me a useful way of checking that the difficulty levels were behaving differently.

## Other stuff I added

The game also has:

- local player vs player
- animated piece drops
- win animations
- difficulty selection
- restart/menu system
- procedural sound effects generated in code
- NumPy-based 6x7 game board

The sound was something I experimented with as well. Instead of loading separate audio files, the game generates tones programmatically using sine waves.

## Tech

- Python
- NumPy
- Pygame
- Minimax
- Alpha-Beta pruning

## What I learned

This was one of my first larger AI projects and helped me understand game trees much better than just studying Minimax theoretically.

The biggest difference between the difficulty levels comes from how far ahead the AI searches. Increasing the depth makes the AI considerably harder to beat, but also increases the amount of computation required, which is where Alpha-Beta pruning becomes useful.

---

Built by **Saleh Pour**  
Robotics & Artificial Intelligence, University of Hertfordshire
