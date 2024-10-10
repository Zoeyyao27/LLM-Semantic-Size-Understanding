import matplotlib.pyplot as plt
import squarify
import csv

def read_csv_to_dict(filename):
    data_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        next(reader)  # Skip the second row
        for row in reader:
            word = row[0]
            if "(" in word:
                continue
            attributes = {}
            attributes["Length"] = int(row[1])
            attributes["AROU"] = float(row[2])  # arousal
            attributes["CNC"] = float(row[3])  # concreteness
            attributes["SIZE"] = float(row[4])  # size

            data_dict[word] = attributes
    return data_dict

path_for_scores_for_words = "translation_tool/construct_mode_dataset/scores_for_words.csv"

data_dict = read_csv_to_dict(path_for_scores_for_words)

words = []
size = []
wordlist = ["shrimp", "whale", "least", "flower", "freedom", "spaceship", "narrow", "tree", "ant", "dust", "hush", "love"]

# Collect word and size data from the dict
for word in wordlist:
    words.append(word)
    size.append(data_dict[word]["SIZE"])


labels = [f"{word}\n{data_dict[word]['SIZE']:.2f}" for word in wordlist]

# Normalize the size data for color mapping
cmap = plt.get_cmap('rainbow') 
norm = plt.Normalize(min(size), max(size))
num_colors = len(words)
colors = [cmap(i / num_colors) for i in range(num_colors)] 

# Plot the treemap with color mapping and larger font size
plt.figure(figsize=(12, 12), dpi=300)  
squarify.plot(
    sizes=size, 
    label=labels, 
    color=colors, 
    alpha=.8, 
    text_kwargs={'fontsize': 27, 'color': 'black'}
)

plt.axis('off') 


plt.savefig('treemap.png')
plt.show()
