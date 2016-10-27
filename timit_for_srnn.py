# Script for generating TIMIT dataset
# -----------------------------------
#   1) Put TIMIT.zip in the same folder as 'timit_for_vrnn.py'
#   2) Instal librosa from https://github.com/librosa/librosa
#      The conda-forge version works. You might need to install the newest
#      version from github
#   3) From terminal run 'python timit_for_srnn.py'
#      This will save the dataset as a npz file in the data subfolder
import os
import fnmatch
import numpy as np
import librosa
import zipfile

timit_folder = 'TIMIT/timit'
frac_in_validation = 0.05
BATCH_SIZE = 64
SAMPLINGRATE = 16000
SEQ_LEN_IN_SEC = 0.5
OUTDIM = 200
SEQ_LEN = int(SAMPLINGRATE*SEQ_LEN_IN_SEC) / OUTDIM


zip_ref = zipfile.ZipFile('TIMIT.ZIP', 'r')
zip_ref.extractall('')
zip_ref.close()


with open('timit_train_files.txt', 'rb') as f:
    train_control = [l.rstrip() for l in f]
with open('timit_test_files.txt', 'rb') as f:
    test_control = [l.rstrip() for l in f]

assert len(train_control) == 4620
assert len(test_control) == 1680


def reorder(data_in, batch_size, model_seq_len, dtype='float32'):
    last_dim = data_in.shape[-1]
    if data_in.shape[0] % (batch_size * model_seq_len) == 0:
        print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
              "set to x_in = x_in[:-1]")
        data_in = data_in[:-1]

    data_resize = \
        (data_in.shape[0] // (batch_size * model_seq_len)) * model_seq_len * batch_size
    n_samples = data_resize // (model_seq_len)
    n_batches = n_samples // batch_size

    u_out = data_in[:data_resize].reshape(n_samples, model_seq_len, last_dim)
    x_out = data_in[1:data_resize + 1].reshape(n_samples, model_seq_len, last_dim)

    out = np.zeros(n_samples, dtype='int32')
    for i in range(n_batches):
        val = range(i, n_batches * batch_size + i, n_batches)
        out[i * batch_size:(i + 1) * batch_size] = val

    u_out = u_out[out]
    x_out = x_out[out]

    return u_out.astype(dtype), x_out.astype(dtype)


def get_files(folder, file_end):
    # Recursively find files in folder structure with the file ending file_end
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, file_end):
            matches.append(os.path.join(root, filename))
    return matches


def create_test_set(x_lst):
    n = len(x_lst)
    x_lens = np.array(map(len, x_lst))
    max_len = max(map(len, x_lst)) - 1
    u_out = np.zeros((n, max_len, OUTDIM), dtype='float32')*np.nan
    x_out = np.zeros((n, max_len, OUTDIM), dtype='float32')*np.nan
    for row, vec in enumerate(x_lst):
        l = len(vec) - 1
        u = vec[:-1]  # all but last element
        x = vec[1:]   # all but first element

        x_out[row, :l] = x
        u_out[row, :l] = u

    mask = np.invert(np.isnan(x_out))
    x_out[np.isnan(x_out)] = 0
    u_out[np.isnan(u_out)] = 0
    mask = mask[:, :, 0]
    assert np.all((mask.sum(axis=1)+1) == x_lens)
    return u_out, x_out, mask.astype('float32')


def load_wav_files(files):
    wav_files = []
    for i, f in enumerate(files):
        print i, f
        wav_files += [librosa.load(f, sr=SAMPLINGRATE)[0]]
    return wav_files


def make_muliple(x, outdim):
    n = int(len(x) // outdim)
    out_len = n*outdim
    return x[:out_len]


train_files = get_files(os.path.join(timit_folder, 'train'), '*.WAV')
test_split_files = get_files(os.path.join(timit_folder, 'test'), '*.WAV')

assert len(train_files) == len(train_control)
assert len(test_split_files) == len(test_control)

# split train set in train and val set
num_train_files = len(train_files)
num_valid_split_files = int(num_train_files * frac_in_validation)
num_train_split_files = num_train_files - num_valid_split_files

print "NUMBER OF TRAIN+VALID FILES:", len(train_files)
print "NUMBER OF TEST FILES", len(test_split_files)
print "TOTAL NUMBER OF SENTENCES", len(train_files) + len(test_split_files)
print "NUMBER OF VALID FILES", num_valid_split_files
print "NUMBER OF TRAIN FILES", num_train_split_files

assert num_valid_split_files + num_train_split_files == num_train_files

## Shuffle training set and split in validation and training
import random
random.seed(1234)
random.shuffle(train_files)
valid_split_files = train_files[:num_valid_split_files]
train_split_files = train_files[num_valid_split_files:]


### check that there is no overlap
s_train = set(train_split_files)
s_valid = set(valid_split_files)
s_test = set(test_split_files)

# nuerotic asserts....
assert len(s_train & s_test) == 0
assert len(s_valid & s_test) == 0
assert len(set(test_control) & s_train) == 0
assert len(set(test_control) & s_valid) == 0
assert len(set(s_test) & set(s_test)) == len(s_test)
assert sum(['test' in f for f in train_split_files]) == 0
assert sum(['test' in f for f in valid_split_files]) == 0

# load wav files
valid_vector = load_wav_files(valid_split_files)
test_vector_lst = load_wav_files(test_split_files)
train_vector = load_wav_files(train_split_files)

assert len(valid_vector) + len(train_vector) == 4620
assert len(test_vector_lst) == 1680


print "MEAN LEN TRAIN", np.mean(map(len, train_vector))
print "MEAN LEN TEST", np.mean(map(len, test_vector_lst))
print "MEAN LEN VALID", np.mean(map(len, valid_vector))
print "test_vector", len(test_vector_lst), test_vector_lst[0].dtype
print "train_vector", len(train_vector), train_vector[0].dtype
# stack train and valid set into a single vector each
train_vector = np.hstack(train_vector)
valid_vector = np.hstack(valid_vector)

print "train_vector_len after hstack", len(train_vector)
print "valid_vector_len after hstack", len(valid_vector)
assert train_vector.ndim == 1
assert valid_vector.ndim == 1

# calculate normalization constants
m = np.mean(train_vector)
sd = np.std(train_vector)
print "MEAN", m
print "STD ", sd
train_vector = (train_vector - m) / sd
valid_vector = (valid_vector - m) / sd
test_vector_lst = [(vec - m)/sd for vec in test_vector_lst]

# make train and valid vectors a multiple of the output dimension
train_vector = make_muliple(train_vector, OUTDIM).reshape((-1, OUTDIM))
valid_vector = make_muliple(valid_vector, OUTDIM).reshape((-1, OUTDIM))

# make test set a multiple of the output dim. Note that for the test set
# we need to do this for each file because the files in the test set are
# treated individually
test_vector_lst = [make_muliple(
    vec, OUTDIM).reshape((-1, OUTDIM)) for vec in test_vector_lst]

assert len(test_vector_lst) == 1680


# Language model like reordering for valid and train set.
print "reordering train..."
u_train_vector, x_train_vector = reorder(
    train_vector, batch_size=BATCH_SIZE, model_seq_len=SEQ_LEN)

print "reordering valid..."
u_valid_vector, x_valid_vector = reorder(
    valid_vector, batch_size=BATCH_SIZE, model_seq_len=SEQ_LEN)

u_test_vector, x_test_vector, mask_test = create_test_set(test_vector_lst)


assert u_test_vector.shape[0] == 1680
assert x_test_vector.shape[0] == 1680
assert mask_test.shape[0] == 1680

assert np.sum(u_train_vector[:, 1:] - x_train_vector[:, :-1]) == 0.0
assert np.sum(u_valid_vector[:, 1:] - x_valid_vector[:, :-1]) == 0.0

for row in range(u_test_vector.shape[0]):
    l = int(mask_test[row].sum())
    assert np.sum(u_test_vector[row, 1:l] - x_test_vector[row, :l-1]) == 0.0

if not os.path.isdir('data'):
    os.makedirs('data')

print "Saving"
np.savez_compressed(
    "data/timit_raw_batchsize%i_seqlen%i.npz" % (BATCH_SIZE, SEQ_LEN),
    u_valid=u_valid_vector,
    u_train=u_train_vector,
    u_test=u_test_vector,
    x_valid=x_valid_vector,
    x_train=x_train_vector,
    x_test=x_test_vector,
    mask_test=mask_test)


