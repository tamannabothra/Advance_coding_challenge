import random

def choose_random_word():
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    return random.choice(words)

def display_word(word, guessed_letters):
    displayed_word = ""
    for letter in word:
        if letter in guessed_letters:
            displayed_word += letter
        else:
            displayed_word += "_"
    return displayed_word

def display_hangman(incorrect_guesses):
    hangman_pics = [
        """
           +---+
               |
               |
               |
              ===""",
        """
           +---+
           O   |
               |
               |
              ===""",
        """
           +---+
           O   |
           |   |
               |
              ===""",
        """
           +---+
           O   |
          /|   |
               |
              ===""",
        """
           +---+
           O   |
          /|\\  |
               |
              ===""",
        """
           +---+
           O   |
          /|\\  |
          /    |
              ===""",
        """
           +---+
           O   |
          /|\\  |
          / \\  |
              ==="""
    ]
    
    return hangman_pics[incorrect_guesses]

def hangman():
    max_attempts = 6
    word_to_guess = choose_random_word()
    guessed_letters = []
    incorrect_guesses = 0

    print("Welcome to Hangman!")
    print("You have 6 attempts to guess the word.")
    
    while incorrect_guesses < max_attempts:
        current_word = display_word(word_to_guess, guessed_letters)
        print(display_hangman(incorrect_guesses))
        print("Word: " + current_word)
        print("Guessed letters: " + ", ".join(guessed_letters))
        
        if "_" not in current_word:
            print("Congratulations! You've guessed the word: " + word_to_guess)
            break

        guess = input("Guess a letter: ").lower()

        if len(guess) != 1 or not guess.isalpha():
            print("Please enter a single letter.")
            continue

        if guess in guessed_letters:
            print("You've already guessed that letter.")
            continue

        guessed_letters.append(guess)

        if guess not in word_to_guess:
            incorrect_guesses += 1
            print(f"Sorry, '{guess}' is not in the word. You have {max_attempts - incorrect_guesses} attempts left.")
        else:
            print(f"Good guess! '{guess}' is in the word.")

    if "_" in current_word:
        print(f"Sorry, you're out of attempts. The word was: {word_to_guess}")

if __name__ == "__main__":
    hangman()
