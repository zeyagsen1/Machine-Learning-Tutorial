vocab = {}  # maps word to integer representing it
word_encoding = 1

i = 0  # Initialize a global variable 'i' to count something


def bag_of_words(text):
    global word_encoding  # Declare 'word_encoding' as global to modify it within the function
    words = text.lower().split(" ")  # Split the text into individual words and convert them to lowercase
    bag = {}  # A dictionary to store word encodings and their frequency
    print(f'words {words}')  # Print the list of words

    for word in words:  # Iterate through each word in the list
        if word in vocab:  # Check if the word exists in the vocabulary
            encoding = vocab[word]  # Get the word's encoding from the vocab dictionary
        else:
            vocab[word] = word_encoding  # If the word is not in the vocab, assign it a new encoding
            encoding = word_encoding  # Assign the new encoding to the word
            word_encoding += 1  # Increment the word encoding for the next unique word

        if encoding in bag:  # Check if the word's encoding already exists in the bag
            bag[encoding] += 1  # If it exists, increment the frequency count
        else:
            bag[encoding] = 1  # If it doesn't exist, set the frequency count to 1

    return bag  # Return the bag of words with their encodings and frequencies

text = "this is a test to see if this test will work is is test a a"
bag = bag_of_words(text)  # Apply the bag of words technique to the given text
print(bag)  # Print the resulting bag of words (word encodings with their frequencies)
print(vocab)  # Print the vocabulary (word-to-encoding mapping)
