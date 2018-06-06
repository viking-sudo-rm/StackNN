"""
Unit tests and usage examples for SimpleStructs.
"""
import torch
from torch.autograd import Variable

from simple import Stack, Queue, tensor_to_string


def test_push(s_struct, value, strength):
    v_str = tensor_to_string(value)
    s_str = tensor_to_string(strength.data)

    print "\nPushing {} with strength {}".format(v_str, s_str)
    s_struct.push(value, strength)
    s_struct.log()

    return


def test_pop(s_struct, strength):
    print "\nPopping with strength {:4f}".format(strength)
    s_struct.pop(strength)
    s_struct.log()

    return


def test_read(s_struct, strength):
    s_str = tensor_to_string(strength)

    print "\nReading with strength {}".format(s_str)
    print tensor_to_string(s_struct.read(strength).data[0])

    return


test_stack = True  # Whether we are testing a Stack or a Queue
batch_size = 1  # The size of our mini-batches
embedding_size = 2  # The size of vectors held by the SimpleStruct

# Create a struct
if test_stack:
    struct = Stack(batch_size, embedding_size)
else:
    struct = Queue(batch_size, embedding_size)

# Push something
v1 = torch.randn(embedding_size)
v2 = torch.randn(embedding_size)
v3 = torch.randn(embedding_size)
v4 = torch.randn(embedding_size)
s = Variable(torch.FloatTensor([1.]))

test_push(struct, v1, s)
test_push(struct, v2, s)
test_push(struct, v3, s)
test_push(struct, v4, s)

# Pop something
test_pop(struct, 0.4)
test_pop(struct, 1.7)

# Read something
test_read(struct, torch.FloatTensor([1.]))
