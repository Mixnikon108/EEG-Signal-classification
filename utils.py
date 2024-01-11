"""
###########################################################################
#                            PROJECT HEADER                               #
# ----------------------------------------------------------------------- #
# Author:        Jorge de la Rosa                                         #
# Affiliation:   Data Science and Artificial Intelligence Student         #
#                Universidad Politécnica de Madrid (UPM)                  #
# Last Updated:  25/12/2023                                               #
#                                                                         #
# PROJECT DETAILS:                                                        #
# "Classification of Motor Imagery EEG Signals in Patients with High      #
#  Uncertainty using a Spectral Transformer."                             #
#                                                                         #
# MODULE DESCRIPTION:                                                     #
# This file contains utility functions designed to assist in the          #
# preprocessing, feature extraction, and classification of EEG signals    #
# related to motor imagery as part of the Data Science Project course.    #
#                                                                         #
# CONTACT INFORMATION:                                                    #
# For inquiries or clarifications, please contact Jorge de la Rosa at     #
# jorgechedo@gmail.com.                                                   #
###########################################################################
"""

# Standard Libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.io as sio

# Third-party Libraries
import mne
import seaborn as sns
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)


# MNE (Neuroinformatics Library) Specific Modules
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, events_from_annotations, pick_types

# Set MNE log level to CRITICAL to reduce verbosity
mne.set_log_level('CRITICAL')


#%%
def index_below_threshold(input_list, threshold=0.6):
    """
    Return indices and values from the input list that are below the specified threshold.

    Parameters:
    - input_list (list): Input list of values.
    - threshold (float, optional): Threshold value. Default is 0.6.

    Returns:
    - result (list of tuples): List of tuples containing (index + 1, value) for values below the threshold.
    """
    return [(index + 1, value) for index, value in enumerate(input_list) if value <= threshold]


#%%
def display_descriptive_statistics(data):
    """
    Calculate and display basic descriptive statistics for a given dataset.

    Parameters:
    - data (numpy.ndarray): Input data array.

    Returns:
    None
    """

    # Calculate basic descriptive statistics
    mean_value = np.mean(data)
    variance_value = np.var(data)
    min_value = np.min(data)
    max_value = np.max(data)
    median_value = np.median(data)
    mode_value = stats.mode(data).mode[0]

    # Print the results
    print(f"Mean: {mean_value}")
    print(f"Variance: {variance_value}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}")
    print(f"Median: {median_value}")
    print(f"Mode: {mode_value}")

    # Create a boxplot and histogram
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(data, ax=ax_box, orient="h")
    sns.histplot(data, ax=ax_hist, kde=False, bins='auto', element="step", stat="count")

    ax_hist.set(xlabel='Accuracy', ylabel='Count')
    ax_box.set(yticks=[])
    plt.suptitle('Accuracy for different subjects')

    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)

    plt.show()

    
    
#%%   
def evaluate_and_visualize(model, history, X_test_reshaped, y_test):
    """
    Evaluar el modelo en el conjunto de prueba y visualizar el historial de entrenamiento.

    Parameters:
    - model (keras.Model): Modelo entrenado.
    - history (keras.callbacks.History): Historial de entrenamiento.
    - X_test_reshaped (numpy.ndarray): Conjunto de prueba (X) reshaped.
    - y_test (numpy.ndarray): Etiquetas del conjunto de prueba.

    Returns:
    - test_accuracy (float): Exactitud en el conjunto de prueba.
    """

    # Evaluar el modelo en el conjunto de prueba
    y_pred = model.predict(X_test_reshaped)
    y_pred_binary = (y_pred > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_binary)
    print(f'Accuracy on test set: {test_accuracy}')

    # Visualizar el historial de entrenamiento con Plotly
    fig_loss = go.Figure()

    # Loss
    fig_loss.add_trace(go.Scatter(x=history.epoch, y=history.history['loss'], mode='lines', name='Training Loss'))
    fig_loss.add_trace(go.Scatter(x=history.epoch, y=history.history['val_loss'], mode='lines', name='Validation Loss'))

    fig_loss.update_layout(title='Training and Validation Loss',
                           xaxis=dict(title='Epoch'),
                           yaxis=dict(title='Loss'))

    fig_loss.show()

    # Accuracy
    fig_acc = px.line(x=history.epoch, y=history.history['accuracy'], labels={'x': 'Epoch', 'y': 'Accuracy'},
                      title='Training Accuracy')
    fig_acc.add_shape(
        go.layout.Shape(type='line', x0=0, x1=max(history.epoch), y0=0.5, y1=0.5, line=dict(color='red'))
    )

    fig_acc.show()

    return test_accuracy


#%%
def extract_motor_imagery_data(subject_id):
    '''
    Extracts motor imagery data from EEG recordings for a given subject.

    Parameters:
    - subject_id (int): Subject identifier.

    Returns:
    - epochs_train (mne.Epochs): Extracted epochs for training.
    - labels (numpy.ndarray): Labels corresponding to the epochs.
    '''

    # Define parameters
    tmin, tmax = -1.0, 4.0
    event_id = dict(hands=2, feet=3)  # Motor imagery: hands vs feet
    runs = [6, 10, 14]  # Runs where hands and feet data is available

    # Load data
    raw_fnames = eegbci.load_data(subject_id, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

    # Standardize channels
    eegbci.standardize(raw)

    # Apply standard montage (10-10 system)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # Apply bandpass filter
    # Note: For NeuroImaging functions, the frequency range is typically 7 Hz to 30 Hz
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    # Get events and picks
    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    # Read epochs (training between 1 and 2 seconds after the event)
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    labels = epochs.events[:, -1] - 2

    return epochs_train, labels


#%%
def load_eeg_data(apply_standardization=False):
    """
    Load EEG motor imagery data for all subjects.

    Parameters:
    - apply_standardization (bool, optional): If True, standardizes the EEG data using z-score. Default is False.

    Returns:
    - all_data (numpy.ndarray): Concatenated standardized or raw EEG data for all subjects.
    - all_labels (numpy.ndarray): Concatenated labels for all subjects.
    """

    subject_data = []

    # Iterate over subjects (109 subjects)
    for subject_id in tqdm(range(1, 110)):
        # Load preprocessed data for each subject
        subject_epochs, subject_labels = extract_motor_imagery_data(subject_id)

        # Add subject data and labels to the list
        subject_data.append((subject_epochs.get_data(), subject_labels))

    # Filter data with 161 time points in the EEG signal
    subject_data = [(X, y) for X, y in subject_data if X.shape[2] == 161]

    # Unpack data and labels
    all_data = np.concatenate([data for data, _ in subject_data], axis=0)
    all_labels = np.concatenate([labels for _, labels in subject_data], axis=0)

    if apply_standardization:
        # Reshape the data for standardization
        flattened_data = all_data.reshape((-1, 161))

        # Calculate mean and standard deviation along the rows
        mean_values = np.mean(flattened_data, axis=0)
        std_values = np.std(flattened_data, axis=0)

        # Apply z-score standardization
        standardized_data = (flattened_data - mean_values) / std_values

        # Reshape back to the original shape
        all_data = standardized_data.reshape(all_data.shape)

    return all_data, all_labels


#%%
def split_eeg_data(standardize=False, test_size=0.1, random_state=42):
    """
    Split and reshape EEG motor imagery data for training and testing.

    Parameters:
    - standardize (bool, optional): If True, standardizes the EEG data using z-score. Default is False.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.1.
    - random_state (int, optional): Seed for random state. Default is 42.

    Returns:
    - X_train (numpy.ndarray): Reshaped EEG data for training.
    - y_train (numpy.ndarray): Labels for training data.
    - X_test (numpy.ndarray): Reshaped EEG data for testing.
    - y_test (numpy.ndarray): Labels for testing data.
    """

    # Load EEG data and labels
    all_data, all_labels = load_eeg_data(apply_standardization=standardize)

    # Reshape the data to have shape (4747, -1)
    all_data_reshaped = all_data.reshape((4747, -1))

    # Split into training and testing sets
    X_train_reshaped, X_test_reshaped, y_train, y_test = train_test_split(
        all_data_reshaped, all_labels, test_size=test_size, random_state=random_state
        )
    
    # Reshape de X_train y X_test al formato original
    X_train_reshaped = X_train_reshaped.reshape((X_train_reshaped.shape[0], 64, 161))
    X_test_reshaped = X_test_reshaped.reshape((X_test_reshaped.shape[0], 64, 161))

    return X_train_reshaped, y_train, X_test_reshaped, y_test


#%%
def classify_individual_subjects(use_lda=False):
    """
    Classify individual subjects using either Linear Discriminant Analysis (LDA) or Neural Network (NN).

    Parameters:
    - use_lda (bool, optional): If True, uses LDA for classification. If False, uses NN. Default is False.

    Returns:
    - acc_results (list): List of accuracy scores for each subject.
    """

    acc_results = []
    subjects_processed = 0

    for subject_id in tqdm(range(1, 110)):
        # Load preprocessed data for each subject
        subject_epochs, subject_labels = extract_motor_imagery_data(subject_id)

        # If not Bad Trial
        if subject_epochs.get_data().shape[2] == 161:
            # Train Test Split
            X_raw, y_raw = subject_epochs.get_data(), subject_labels
            X_train_reshaped, X_test_reshaped, y_train, y_test = train_test_split(
                X_raw.reshape((X_raw.shape[0], -1)), y_raw, test_size=0.1, random_state=42
            )
            X_train = X_train_reshaped.reshape((X_train_reshaped.shape[0], 64, 161))
            X_test = X_test_reshaped.reshape((X_test_reshaped.shape[0], 64, 161))

            # Feature extraction using CSP
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
            X_train_csp = csp.fit_transform(X_train, y_train)
            X_test_csp = csp.transform(X_test)

            # Classification using LDA
            if use_lda:
                lda = LinearDiscriminantAnalysis()
                lda.fit(X_train_csp, y_train)
                y_pred = lda.predict(X_test_csp)
                
            # Classification using NN
            else:
                model = Sequential()
                model.add(Dense(20, input_dim=4, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train_csp, y_train, epochs=50, batch_size=1, verbose=0, validation_data=(X_test_csp, y_test))
                y_pred = (model.predict(X_test_csp, verbose=0) > 0.5).astype(int).flatten()

            # Evaluate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            acc_results.append(accuracy)
            subjects_processed += 1

    print(f'# Subjects processed: {subjects_processed} / 110  ({(subjects_processed / 110) * 100:.2f}%)')

    return acc_results



#%%
def visualize_csp_components(eeg_epochs_data, epoch_labels, num_csp_components=4):
    """
    Visualize 2D scatter plots of CSP components.

    Parameters:
    - num_csp_components (int, optional): Number of CSP components. Default is 4.
    - eeg_epochs_data (numpy.ndarray): EEG epochs data.
    - epoch_labels (numpy.ndarray): Labels corresponding to each epoch.

    Returns:
    None
    """
    # Create CSP instance
    csp = CSP(n_components=num_csp_components, reg=None, log=True, norm_trace=False)

    # Transform the data using CSP
    transformed_csp_data = csp.fit_transform(eeg_epochs_data, epoch_labels)

    # Get all possible combinations of two components
    combinations_2d = list(combinations(range(transformed_csp_data.shape[1]), 2))

    # Create subplots
    fig, axes = plt.subplots(nrows=len(combinations_2d) // 2, ncols=2, figsize=(12, 12))
    fig.suptitle('2D Visualization of CSP Components')

    # Visualize each combination in a 2D plot
    for i, (component1, component2) in enumerate(combinations_2d):
        ax = axes[i // 2, i % 2]
        ax.scatter(transformed_csp_data[:, component1], transformed_csp_data[:, component2], c=epoch_labels, cmap='viridis', marker='o', edgecolors='k')
        ax.set_xlabel(f'Component {component1 + 1}')
        ax.set_ylabel(f'Component {component2 + 1}')
        ax.set_title(f'Components {component1 + 1} vs. {component2 + 1}')

    # Adjust layout and display plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#%%
def load_BCI_competition_data(data_path, subject, training, all_trials=True):
    """
    Load and divide the dataset based on the subject-specific (subject-dependent) approach.

    Parameters:
    - data_path (str): Dataset path.
      Dataset BCI Competition IV-2a is available on http://bnci-horizon-2020.eu/database/data-sets
    - subject (int): Number of the subject in [1, .. ,9].
    - training (bool): If True, load training data. If False, load testing data.
    - all_trials (bool): If True, load all trials. If False, ignore trials with artifacts.

    Returns:
    - data_return (numpy.ndarray): Loaded EEG data.
    - class_return (numpy.ndarray): Labels corresponding to the loaded data.
    """
    # Frecuencia muestreo (Hz)
    fs = 250
    # Canales EEG
    # Hay 25 canales pero 3 de ellos son EOG
    n_channels = 22
    # Nº Test útiles (3 primeros de calibración)
    n_tests = 6 * 48
    # 7 segundos por prueba
    window_length = 7 * fs

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_length))
    no_valid_trial = 0
    if training:
        file = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        file = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
        
    raw_data = file['data']

    for i in range(0, raw_data.size):
        cell_data = [raw_data[0, i][0, 0]][0]
        cell_X, cell_trial, cell_y, cell_artifacts = cell_data[0], cell_data[1], cell_data[2], cell_data[5]

        for trial in range(cell_trial.size):
            # Condicion para saber si desechar pruebas con artefactos
            if cell_artifacts[trial] != 0 and not all_trials:
                continue
            data_return[no_valid_trial, :, :] = np.transpose(cell_X[int(cell_trial[trial]):(int(cell_trial[trial]) + window_length), :n_channels])
            class_return[no_valid_trial] = int(cell_y[trial])
            no_valid_trial += 1

    return data_return[0:no_valid_trial, :, :], class_return[0:no_valid_trial]


#%%
def standardize_BCI_competition_data(X_train, X_test, channels):
    """
    Standardize EEG data across channels using a separate StandardScaler for each channel.

    Parameters:
    - X_train (numpy.ndarray): Training EEG data of shape (Trials, MI-tasks, Channels, Time points).
    - X_test (numpy.ndarray): Testing EEG data of shape (Trials, MI-tasks, Channels, Time points).
    - channels (int): Number of EEG channels.

    Returns:
    - X_train (numpy.ndarray): Standardized training EEG data.
    - X_test (numpy.ndarray): Standardized testing EEG data.
    """
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, j, :])
        X_train[:, j, :] = scaler.transform(X_train[:, j, :])
        X_test[:, j, :] = scaler.transform(X_test[:, j, :])

    return X_train, X_test

#%%
def get_BCI_competition_data(path, subject, isStandard=True):
    """
    Load and split the dataset into training and testing based on the specified approach.

    Parameters:
    - path (str): Dataset path.
    - subject (int): Subject number.
    - isStandard (bool): If True, standardize the data.

    Returns:
    - X_train (numpy.ndarray): Training EEG data.
    - y_train (numpy.ndarray): Training labels.
    - y_train_onehot (numpy.ndarray): One-hot encoded training labels.
    - X_test (numpy.ndarray): Testing EEG data.
    - y_test (numpy.ndarray): Testing labels.
    - y_test_onehot (numpy.ndarray): One-hot encoded testing labels.
    """
    # Define dataset parameters
    fs = 250          # Sampling rate
    t1 = int(1.5 * fs)  # Start time_point
    t2 = int(6 * fs)    # End time_point
    T = t2 - t1         # Length of the MI trial (samples or time_points)

    # Load and split the dataset
    path = path + f's{subject + 1}/'
    X_train, y_train = load_BCI_competition_data(path, subject + 1, True)
    X_test, y_test = load_BCI_competition_data(path, subject + 1, False)

    # Prepare training data
    N_tr, N_ch, _ = X_train.shape
    X_train = X_train[:, :, t1:t2]
    y_train_onehot = (y_train-1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)

    # Prepare testing data
    N_test, N_ch, _ = X_test.shape
    X_test = X_test[:, :, t1:t2]
    y_test_onehot = (y_test-1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)	

    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_BCI_competition_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


# %%
def get_all_BCI_competition_data(n_subjects, data_path='./BCI Competition IV-2a/'):
    """
    Obtener datos completos concatenados de varios sujetos.

    Parámetros:
    - n_subjects (int): Número total de sujetos.
    - data_path (str): Ruta base de los datos. Por defecto, './BCI Competition IV-2a/'.

    Retorna:
    - Tuple: Contiene datos concatenados de entrenamiento y prueba para X, y y y_onehot.
    """

    for subject in tqdm(range(n_subjects)):
        # Obtener datos para cada sujeto
        X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot = get_BCI_competition_data(data_path, subject, isStandard=True)
        
        if subject == 0:
            # Para el primer sujeto, inicializar variables de concatenación
            X_train_concat = X_train
            y_train_concat = y_train
            y_train_onehot_concat = y_train_onehot
            X_test_concat = X_test
            y_test_concat = y_test
            y_test_onehot_concat = y_test_onehot
        else:
            # Para los siguientes sujetos, concatenar resultados
            X_train_concat = np.concatenate([X_train_concat, X_train], axis=0)
            y_train_concat = np.concatenate([y_train_concat, y_train], axis=0)
            y_train_onehot_concat = np.concatenate([y_train_onehot_concat, y_train_onehot], axis=0)
            X_test_concat = np.concatenate([X_test_concat, X_test], axis=0)
            y_test_concat = np.concatenate([y_test_concat, y_test], axis=0)
            y_test_onehot_concat = np.concatenate([y_test_onehot_concat, y_test_onehot], axis=0)
            
    # Devolver datos concatenados
    return X_train_concat, y_train_concat, y_train_onehot_concat, X_test_concat, y_test_concat, y_test_onehot_concat

#%%
def evaluate_classification_metrics(y_true_onehot, y_pred_probs, class_labels=["Left hand", "Right hand", "Both feet", "Tongue"]):
    """
    Evaluate classification metrics and plot a confusion matrix.

    Parameters:
    - y_true_onehot (numpy.ndarray): One-hot encoded true labels.
    - y_pred_probs (numpy.ndarray): Predicted probabilities.
    - class_labels (list): List of class labels.

    Returns:
    None
    """
    # Convert probabilities to predicted labels
    y_true_labels = np.argmax(y_true_onehot, axis=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    # Calculate classification metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predictions")
    plt.ylabel("True Values")
    plt.show()
