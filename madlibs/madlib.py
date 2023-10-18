import random
player_name = input("Please Enter your Name: ")
print(player_name,"Welcome to Madlibs Game")
while True:
    adjective1 = input("Adjective: ")
    adjective2 = input("Adjective: ")
    adjective3 = input("Adjective: ")
    adjective4 = input("Adjective: ")
    adverb1 = input("Adverb: ")
    adverb2 = input("Adverb: ")
    noun1 = input("Noun: ")
    noun2 = input("Noun: ")
    noun3 = input("Noun: ")
    verb1 = input("Verb: ")
    verb2 = input("Verb: ")
    verb3 = input("Verb: ")

    madlib1 = f'''           
                                    A Day At The Zoo!
        Today I went to the zoo. I saw an {adjective1} {noun1} jumping up and down in its tree.
        He {verb1} {adverb1} through the large tunnel that led to its {adjective2} {noun2}. 
        I got some peanuts and passed them through the cage to a gigantic gray {noun3}
        towering above my head. Feeding that animal made me hungry. 
        I went to get a {adjective3} scoop of ice cream. It filled my stomach. 
        Afterwards I had to {verb2} {adverb2} to catch our bus. When I got home I {verb3} my
        mom for a {adjective4} day at the zoo.'''
    
    madlib2 = f'''                             
                                     The Fun Park!
    Today, my fabulous camp group went to a {adjective1} amusement park. It was a
    fun park with lots of cool {noun1} and enjoyable play structures. When we got there, my
    kind counselor shouted loudly, "Everybody off the {noun2}." We all pushed out in a terrible
    hurry. My counselor handed out yellow tickets, and we scurried in. I was so excited! I couldn't figure out
    what exciting thing to do first. I saw a scary roller coaster I really liked so, I {adverb1} ran
    over to get in the long line that had about 10 people in it. When I finally
    got on the roller coaster I was {verb1}. In fact I was so nervous my two knees
    were knocking together. This was the {adjective2} ride I had ever been on! In about two
    minutes I heard the crank and grinding of the gears. That’s when the ride began! When I got to the bottom,
    I was a little  {verb2} but I was proud of myself. The rest of the day went {adverb2}. 
    It was a {adjective3} day at the fun park. 
    '''

    madlib_list = [madlib1,madlib2]
    
    print()
    print(random.choice(madlib_list))
    print("\nGood job! " + player_name)

    another_game = input("\nDo you want to play another game!(y/n): ")

    if another_game == "n":
        break
    else:
        continue

print("Thank you playing!")