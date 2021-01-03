import sys
import numpy as np
import matplotlib.pyplot as plt

'''
def plotter(avg_train_list, avg_valid_list, num_epoch):
  x = range(0, num_epoch)
  a = plt.plot(x, avg_train_list)
  b = plt.plot(x, avg_valid_list)
  plt.xlabel("Epoch Count")
  plt.ylabel("Average Negative Log Likelihood")
  plt.title("Avg. NLL vs. # of Epochs")
  plt.show()
  return None
'''
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
#helper to get label_list from each file
def get_labels(formatted_input):
  label_list = []
  with open(formatted_input, "r") as file:
    review_list = file.readlines()
    for review in review_list:
      label = int(review[0])
      label_list.append(label)
  return label_list
#helper to get list of dictionaries from each file
def get_dicts(formatted_input):
  dict_list = []
  with open(formatted_input, "r") as file:
    review_list = file.readlines()
    for review in review_list:
      d = {}
      review = review.strip("\n")
      review = review[2:]
      pair_list = review.split("\t")
      for pair in pair_list:
        splitted = pair.split(":")
        index = splitted[0]
        val = splitted[1]
        if index not in d:
          d[index] = int(val)
      dict_list.append(d)
  return dict_list
#main train function
def train(labels, dicts, dict_input, num_epoch):
  labels_list = labels
  bias = 0 #starting bias
  with open(dict_input, "r") as file:
    vocab_list = file.readlines()
  theta_vector = np.zeros((len(vocab_list), 1)) #starting theta vector [[0], [0], ...]
  #avg_vector = []
  N = len(dicts) #total number of reviews
  for step in range(num_epoch): #for each epoch
    for i in range(len(dicts)): #for each review
      #get the gradient values from helper
      sgd_update = sgd_helper(theta_vector, bias, dicts[i], labels_list[i])
      theta_gradient = sgd_update[0]
      bias_gradient =  sgd_update[1]
      #update our theta values by subtracting gradient values
      theta_gradient = (0.1 / N) * theta_gradient #0.1 = alpha
      bias_gradient = (0.1 / N) * bias_gradient #0.1 = alpha
      theta_vector = np.subtract(theta_vector, theta_gradient)
      bias = bias - bias_gradient
      #avg_ll = calc_avg_ll(theta_vector, bias, labels, dicts)
      #avg_vector.append(avg_ll)
  #return our trained parameters
  parameters = [theta_vector, bias]
  #parameters = [theta_vector, bias, avg_vector]
  return parameters
'''
def calc_avg_ll(theta_vector, bias, labels, dicts):
  res_train = 0
  N_train = len(labels)
  tmp_train = 0

  for example, label in zip(dicts, labels):
    dot_product = sparse_dot_product(example, theta_vec)
    tmp_train = (-label * (dot_product)) + math.log(1 + np.exp(dot_product))
    res_train += tmp_train

  return ((1 / N_train) * (res_train))
'''

def sgd_helper(theta_vector, bias, d, label):
  update_values = []
  dot_product = 0
  for index in d:
    prod = theta_vector[int(index)]
    dot_product += prod
  dot_product = dot_product + bias

  theta_update = np.zeros((len(theta_vector), 1))
  #update theta values
  for key in d:
    theta_update[int(key)] = (np.exp(dot_product) / ((np.exp(dot_product)) + 1)) - label
  #update bias
  bias_update = (np.exp(dot_product) / (np.exp(dot_product) + 1)) - label

  update_values.append(theta_update)
  update_values.append(bias_update)
  return update_values

def predict_and_write(labels, dicts, theta_vector, bias, name):
  label_list = []
  for d in dicts:
    dot_product = 0

    for index in d:
      dot_product += theta_vector[int(index)]
    dot_product += bias
    dot_product = np.exp(dot_product) / (np.exp(dot_product) + 1)
    #heck probability, return label 1 if p >= 0.5
    if dot_product >= 0.5:
      label = 1
    else:
      label = 0
    label_list.append(int(label))
  #create our output files
  with open(str(name), "w") as file:
    text = ""
    for label in label_list:
      text += str(label)
      text += "\n"
    file.write(text)
  #return predicted labels to use for metric_out
  print("Predicted label file has been created.\n")
  return label_list

def metric_out(train_labels, predicted_train, test_labels, predicted_test, name):
  train_count = 0
  test_count = 0
  for i in range(len(train_labels)):
    if train_labels[i] != predicted_train[i]:
      train_count += 1
  for i in range(len(test_labels)):
    if test_labels[i] != predicted_test[i]:
      test_count += 1

  train_error = train_count / len(train_labels)
  test_error = test_count / len(test_labels)
  #create metric_out file
  with open(str(name), "w") as file:
    text = ""
    text += "error(train): " + str(train_error) + "\n"
    text += "error(test): " + str(test_error) + "\n"
    file.write(text)
  print("Metrics file is complete.\n")

def main():

  formatted_train_input = sys.argv[1]
  formatted_validation_input = sys.argv[2]
  formatted_test_input = sys.argv[3]
  dict_input = sys.argv[4]
  train_out = sys.argv[5]
  test_out = sys.argv[6]
  metrics_out = sys.argv[7]
  num_epoch = sys.argv[8]
  num_epoch = int(num_epoch)
  #create dictionary of usable words
  #word_dict = get_dict(dict_input)
  #get label list and dictionary for each review
  train_labels = get_labels(formatted_train_input)
  train_dicts = get_dicts(formatted_train_input)
  #get trained_parameters
  trained_parameters = train(train_labels, train_dicts, dict_input, num_epoch)
  #extract info for our predictions
  test_labels = get_labels(formatted_test_input)
  test_dicts = get_dicts(formatted_test_input)

  #valid_labels = get_labels(formatted_validation_input)
  #valid_dicts = get_dicts(formatted_validation_input)

  #trained_valid_parameters = train(valid_labels, vald_dicts, dict_input, num_epoch)
  #get predicted labels and create output file
  predicted_train = predict_and_write(train_labels, train_dicts, trained_parameters[0], trained_parameters[1], train_out)
  predicted_test = predict_and_write(test_labels, test_dicts, trained_parameters[0], trained_parameters[1], test_out)
  #create metric_out file
  metric_out(train_labels, predicted_train, test_labels, predicted_test, metrics_out)
  #plotter(avg_train_list, avg_valid_list, num_epoch)

if __name__ == "__main__":
  main()

  print("Done.")
