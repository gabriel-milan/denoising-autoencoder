#
# Importing stuff
#
from utils import *
from models import *
from scaler import *
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

#
# Run configuration
#
# Path to the audio files
DATA_PATH = "cmu_us_awb_arctic/wav"
# Random seed for results replication
RANDOM_SEED = 8080
# Scale for noise (5000 is like a lot, 200 is like low background noise)
NOISE_SCALE = 4000
# Number of folds for cross-validation
FOLDS = 10
# Weights for each fold when using the ensemble of folds for prediction
FOLD_WEIGHTS = [1.0 / FOLDS] * FOLDS
# Number of maximum epochs I'll train if it keeps getting better
MAX_EPOCHS = 5000
# Optimizer for model training
OPTIMIZER = "adam"
# Loss function for model training
LOSS_FUNC = "mse"
# What am I monitoring while training
MONITOR = "val_loss"
# How should I decide whether it's better or worse
MONITOR_MODE = "min"
# Verbosity
VERBOSE = 1
# Learning rate reduction callback factor
LR_FACTOR = 0.5
# Learning rate reduction callback patience
LR_PATIENCE = 5
# Early stopping callback patience
ES_PATIENCE = 25
# Batch size for training
BATCH_SIZE = 1024

# Loading file list
files = search_wav(DATA_PATH)
test_files = files[-10:]
files = files[:-10]

# Generating datasets
_, X = generate_dataset(files)
_, X_test = generate_dataset(test_files)
preds = np.zeros(X_test.shape)

# Introducing noise to the datasets
noisy_X = introduce_noise(X, scale=NOISE_SCALE)
noisy_X_test = introduce_noise(X_test, scale=NOISE_SCALE)

# Data normalization
scaler = BitScaler(16)
X = scaler.transform(X)
noisy_X = scaler.transform(noisy_X)
X_test = scaler.transform(X_test)
noisy_X_test = scaler.transform(noisy_X_test)

# Example training with K-Fold for cross-validation
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold, (idxT, idxV) in enumerate(kf.split(X)):

    # Print the fold number
    print("#" * 25)
    print(f"## FOLD {fold + 1}")
    print("#" * 25)

    # Free some memory
    K.clear_session()

    # Building model
    model = DenoisingAutoEncoder()
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNC)

    # Saving best weights for each fold on each epoch
    weights_filename = "fold-{}.h5".format(fold + 1)
    mcp = tf.keras.callbacks.ModelCheckpoint(
        weights_filename, monitor=MONITOR, verbose=VERBOSE, save_best_only=True,
        save_weights_only=True, mode=MONITOR_MODE, save_freq="epoch"
    )

    # Reduce learning rate when reaching a plateau
    lrr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=MONITOR, factor=LR_FACTOR, patience=LR_PATIENCE,
        verbose=VERBOSE, mode=MONITOR_MODE
    )

    # Early stopping if model won't get better
    es = tf.keras.callbacks.EarlyStopping(
        monitor=MONITOR, patience=ES_PATIENCE, verbose=VERBOSE, mode=MONITOR_MODE
    )

    # Train!
    print("Training...")
    history = model.fit(
        noisy_X[idxT], X[idxT],
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[mcp, lrr, es],
        validation_data=(noisy_X[idxV], X[idxV]),
        verbose=VERBOSE
    )

    # Adding collaboration to test prediction
    print("Predicting test set...")
    preds += (model.predict([noisy_X_test],
              verbose=VERBOSE) * FOLD_WEIGHTS[fold])

    # Evaluating fold
    print("Evaluating fold...")
    print(model.evaluate(noisy_X[idxV], X[idxV], verbose=VERBOSE))
    print()
