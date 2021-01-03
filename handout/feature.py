import sys

#helper returns a dict of word:index pairings (e.g. films: 0, adapted: 1)
def get_dict(dict_input):
  vocab = {}
  with open(dict_input, "r") as file:
    dict_list = file.readlines()
    for row in dict_list:
      row_stripped = row.strip("\n")
      row_pair = row_stripped.split(" ")
      vocab[row_pair[0]] = row_pair[1]
  return vocab

#helper to check feature_flag model
def formatted_data(inputfile, outputfile, feature_flag, word_dict):
  if feature_flag == 1:
    model_one(inputfile, outputfile, word_dict)
  else:
    model_two(inputfile, outputfile, word_dict)

#bag-of-words
def model_one(inputfile, outputfile, word_dict):
  out_text = ""
  with open(inputfile, "r") as f:
    in_file = f.readlines()
    #create formatted line
    for line in in_file:
      index_list = []
      line = line.strip("\n")
      text = line[2:]
      text_list = text.split(" ") #text_list is list of all words in text
      #create index_list for each line of text
      for word in text_list:
        if word in word_dict.keys():
          #index_list is list of text index, if that word exists in dictionary
          index_list.append(word_dict[word])
      #make counter for each index
      index_counts = dict()
      for i in index_list:
        index_counts[i] = index_counts.get(i, 0) + 1
      #now recreate text but with index:1 pairing
      formatted_line = ""
      for index in index_counts:
        formatted_line += "\t" + index + ":" + "1"
      out_text += line[0] + formatted_line + "\n"
  #write output file
  with open(outputfile, "w") as out_file:
    out_file.write(out_text)

#trimmed bag-of-words
def model_two(inputfile, outputfile, word_dict):
  out_text = ""
  with open(inputfile, "r") as f:
    in_file = f.readlines()
    #create formatted line
    for line in in_file:
      index_list = []
      line = line.strip("\n")
      text = line[2:]
      text_list = text.split(" ") #text_list is list of all words in text
      #create index_list for each line of text
      for word in text_list:
        if word in word_dict.keys():
          #index_list is list of text index, if that word exists in dictionary
          index_list.append(word_dict[word])
      #make counter for each index
      index_counts = dict()
      for i in index_list:
        index_counts[i] = index_counts.get(i, 0) + 1
      #now recreate text but with index:1 pairing
      formatted_line = ""
      for index in index_counts:
        if index_counts[index] < 4:
          formatted_line += "\t" + index + ":" + "1"
      out_text += line[0] + formatted_line + "\n"
  #write output file
  with open(outputfile, "w") as out_file:
    out_file.write(out_text)

def main():
  train_input = sys.argv[1]
  validation_input = sys.argv[2]
  test_input = sys.argv[3]
  dict_input = sys.argv[4]
  formatted_train_out = sys.argv[5]
  formatted_validation_out = sys.argv[6]
  formatted_test_out = sys.argv[7]
  feature_flag = sys.argv[8]
  feature_flag = int(feature_flag)
  #create dictionary of usable words
  word_dict = get_dict(dict_input)
  #create formatted data
  formatted_data(train_input, formatted_train_out, feature_flag, word_dict)
  formatted_data(validation_input, formatted_validation_out, feature_flag, word_dict)
  formatted_data(test_input, formatted_test_out, feature_flag, word_dict)

if __name__ == "__main__":
  main()
