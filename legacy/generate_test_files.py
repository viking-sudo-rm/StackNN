import random
import time

from formalisms.depth_generate import random_sentences
from tasks.configs import *


def generate_data_set(task):
    test_x, test_y = task.get_tensors(2000)

    x_sent = [[str(w) for w in s] for s in
              task.one_hot_to_sentences(0, test_x)]
    y_sent = [[str(w) for w in s] for s in task.codes_to_sentences(0, test_y)]

    x_text = task.sentences_to_text(*x_sent)
    y_text = task.sentences_to_text(*y_sent)

    combined = list(zip(x_text, y_text))
    combined = list(set(list(combined)))
    random.shuffle(combined)
    x_text[:], y_text[:] = zip(*combined)

    return x_text[:1000], y_text[:1000]


def save_data_set(x_text, y_text, filename):
    f = open("data/testing/final/" + filename, "w")
    for i in xrange(len(x_text)):
        f.write(",".join([x_text[i], y_text[i]]) + "\n")
    f.close()


""" Reverse Task """

random.seed(8597)

print "Generating testing data for Reverse..."
start_time = time.time()

config = testing_reverse_config
del config["task"]

xs, ys = generate_data_set(ReverseTask(**config))
save_data_set(xs, ys, "reverse.csv")

end_time = time.time()
print "Time elapsed: {:.3f} seconds".format(end_time - start_time)

""" XOR Task """

random.seed(8301)

print "Generating testing data for XOR..."
start_time = time.time()

config = final_parity_config
del config["task"]
config["str_length"] = 24

xs, ys = generate_data_set(XORTask(**config))
save_data_set(xs, ys, "parity.csv")

end_time = time.time()
print "Time elapsed: {:.3f} seconds".format(end_time - start_time)

""" Delayed XOR Task """

random.seed(5212)

print "Generating testing data for Delayed XOR..."
start_time = time.time()

config = final_delayed_parity_config
del config["task"]
config["str_length"] = 24

xs, ys = generate_data_set(DelayedXORTask(**config))
save_data_set(xs, ys, "delayed_parity.csv")

end_time = time.time()
print "Time elapsed: {:.3f} seconds".format(end_time - start_time)

""" Formula Task """

random.seed(4968)

print "Generating testing data for Formula..."
start_time = time.time()
sentences = random_sentences(2000, 6, exp_eval_grammar)
end_time = time.time()

print "Sentence generation time: {:.3f} seconds".format(end_time - start_time)

max_length = max([len(s) for s in sentences])
print "Max length: {}".format(max_length)

sentences = list(set([tuple(s) for s in sentences]))[:1000]

x_raw = [a[::2] for a in sentences]
y_raw = [a[1::2] for a in sentences]

xs = [" ".join(x) for x in x_raw]
ys = [" ".join(y) for y in y_raw]
save_data_set(xs, ys, "formula.csv")

end_time = time.time()
print "Time elapsed: {:.3f} seconds".format(end_time - start_time)
