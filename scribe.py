### USAGE
# place file in same directory as running script
# initialize class before beginning run
# any information that is print()ed to the console should be written with write_text()
# any data that is useful to know about the model should be written with write_data(), along with a data tag (eg itr, disc. score, etc)
# 'save' will package a .txt and .csv in the source directory
# 'read' will write all data in self.output and self.data to the console

# DATA FIELDS ARE STRICTLY NUMERIC

import numpy as np
import collections.abc

class scribe(): # container class for alphanumeric run-data
    def __init__(self):
        self.data = {}
        self.output = ""

    def save(self, fname_out="out.txt", fname_dat="dat.csv"): # create text and data file
        file = open(fname_out, "w")
        file.write(self.output) # create txt
        file.close()
        
        tags = self.data.keys()

        max_val = 0 # find length of largest sequence to enforce homogeneity
        for t in tags:
            if len(self.data[t]) >= max_val: max_val = len(self.data[t])

        for t in tags: # enforce homogeneity
            a = np.zeros(max_val)
            a[:len(self.data[t])] = self.data[t]
            self.data[t] = a

        np.savetxt(fname_dat, [x for x in zip(*[[t, *self.data[t]] for t in tags])], delimiter=',', fmt="%s") # create csv

    def read(self): # output txt data to console
        if self.output != "":
            print(self.output)
        else:
            print("--- Empty output source ---")

        if self.data != {}:
            print(self.data)
        else:
            print("--- Empty data source ---")
        

    def write_text(self, text): # store txt data
        self.output += text+'\n'
        
    def write_data(self, tag, num): # store csv data
        if tag in self.data: # add value to column if column already exists
            if isinstance(num, (collections.abc.Sequence, np.ndarray)):
                self.data[tag].append(*num)
            else:
                self.data[tag].append(num)
        else: # create column
            if isinstance(num, (collections.abc.Sequence, np.ndarray)): # test for array
                self.data[tag] = list(num)
            else:
                self.data[tag] = [num]