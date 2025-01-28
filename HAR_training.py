import numpy as np
import torch.nn as nn
import torch.optim as optim
from models import HAR_Transformer
from torch.utils.data import DataLoader
from sequence_dataset import SequenceDatasetNPY
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from utils import EarlyStopper, create_confusion_matrix_w_precision_recall
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

training_output = 'actions' #actions #emotions

do_train = False    

# Seed for reproducibility
np.random.seed(0)

# Initialized later
input_dim = 0
output_dim = 0
max_seq_length = 0
nhead = 16
num_encoder_layers = 2
dim_feedforward = 256

# Training and Evaluation
num_epochs = 200
learning_rate = 0.0005
batch_size = 16
patience = 10

checkpoint_model_name = f'idle_checkpoint_model_IMU_{learning_rate}lr_{batch_size}bs.pt'
confusion_matrix_name = f'idle_confusion_matrix_IIMU_{learning_rate}lr_{batch_size}bs.png'

pixel_dim = 224
patch_size = 56

labelstring = "Timestamp;foot_r_Socket_0;foot_r_Socket_1;foot_r_Socket_2;foot_l_Socket_0;foot_l_Socket_1;foot_l_Socket_2;lowerarm_L_socket_0;lowerarm_L_socket_1;lowerarm_L_socket_2;upperarm_L_socket_0;upperarm_L_socket_1;upperarm_L_socket_2;torso_Socket_0;torso_Socket_1;torso_Socket_2;lowerarm_r_Socket_0;lowerarm_r_Socket_1;lowerarm_r_Socket_2;upperarm_r_Socket_0;upperarm_r_Socket_1;upperarm_r_Socket_2;head_Socket_0;head_Socket_1;head_Socket_2;calf_l_Socket_0;calf_l_Socket_1;calf_l_Socket_2;calf_r_Socket_0;calf_r_Socket_1;calf_r_Socket_2;thigh_r_Socket_0;thigh_r_Socket_1;thigh_r_Socket_2;thigh_l_Socket_0;thigh_l_Socket_1;thigh_l_Socket_2;spine_01_Socket_0;spine_01_Socket_1;spine_01_Socket_2;hand_l_Socket_0;hand_l_Socket_1;hand_l_Socket_2;hand_r_Socket_0;hand_r_Socket_1;hand_r_Socket_2;index_01_l_Socket_0;index_01_l_Socket_1;index_01_l_Socket_2;index_02_l_Socket_0;index_02_l_Socket_1;index_02_l_Socket_2;index_03_l_Socket_0;index_03_l_Socket_1;index_03_l_Socket_2;middle_01_l_Socket_0;middle_01_l_Socket_1;middle_01_l_Socket_2;middle_02_l_Socket_0;middle_02_l_Socket_1;middle_02_l_Socket_2;middle_03_l_Socket_0;middle_03_l_Socket_1;middle_03_l_Socket_2;pinky_01_l_Socket_0;pinky_01_l_Socket_1;pinky_01_l_Socket_2;pinky_02_l_Socket_0;pinky_02_l_Socket_1;pinky_02_l_Socket_2;pinky_03_l_Socket_0;pinky_03_l_Socket_1;pinky_03_l_Socket_2;ring_01_l_Socket_0;ring_01_l_Socket_1;ring_01_l_Socket_2;ring_02_l_Socket_0;ring_02_l_Socket_1;ring_02_l_Socket_2;ring_03_l_Socket_0;ring_03_l_Socket_1;ring_03_l_Socket_2;index_01_r_Socket_0;index_01_r_Socket_1;index_01_r_Socket_2;index_02_r_Socket_0;index_02_r_Socket_1;index_02_r_Socket_2;index_03_r_Socket_0;index_03_r_Socket_1;index_03_r_Socket_2;middle_01_r_Socket_0;middle_01_r_Socket_1;middle_01_r_Socket_2;middle_02_r_Socket_0;middle_02_r_Socket_1;middle_02_r_Socket_2;middle_03_r_Socket_0;middle_03_r_Socket_1;middle_03_r_Socket_2;pinky_01_r_Socket_0;pinky_01_r_Socket_1;pinky_01_r_Socket_2;pinky_02_r_Socket_0;pinky_02_r_Socket_1;pinky_02_r_Socket_2;pinky_03_r_Socket_0;pinky_03_r_Socket_1;pinky_03_r_Socket_2;ring_01_r_Socket_0;ring_01_r_Socket_1;ring_01_r_Socket_2;ring_02_r_Socket_0;ring_02_r_Socket_1;ring_02_r_Socket_2;ring_03_r_Socket_0;ring_03_r_Socket_1;ring_03_r_Socket_2;thumb_01_l_Socket_0;thumb_01_l_Socket_1;thumb_01_l_Socket_2;thumb_02_l_Socket_0;thumb_02_l_Socket_1;thumb_02_l_Socket_2;thumb_03_l_Socket_0;thumb_03_l_Socket_1;thumb_03_l_Socket_2;thumb_03_r_Socket_0;thumb_03_r_Socket_1;thumb_03_r_Socket_2;thumb_02_r_Socket_0;thumb_02_r_Socket_1;thumb_02_r_Socket_2;thumb_01_r_Socket_0;thumb_01_r_Socket_1;thumb_01_r_Socket_2;spine_02Socket_0;spine_02Socket_1;spine_02Socket_2;spine_03Socket_0;spine_03Socket_1;spine_03Socket_2;spine_04Socket_0;spine_04Socket_1;spine_04Socket_2;spine_05Socket_0;spine_05Socket_1;spine_05Socket_2"
labelist = labelstring.split(";")[1:]
# print(len(labelist))
# print(labelist.index("upperarm_L_socket_0"))
# print(labelist.index("upperarm_r_Socket_0"))
# print(labelist.index("hand_l_Socket_0"))
# print(labelist.index("hand_r_Socket_0"))
# print(labelist.index("torso_Socket_0"))

active_sensors =   [labelist.index("upperarm_L_socket_0"),
                    labelist.index("upperarm_r_Socket_0"),
                    labelist.index("hand_l_Socket_0"),
                    labelist.index("hand_r_Socket_0"),
                    labelist.index("torso_Socket_0")]


filenames = [os.path.join('Bandai-Namco1', f) for f in os.listdir('Bandai-Namco1')]
# filenames.extend([os.path.join('Bandai-Nam', f) for f in os.listdir('Bandai-Namco2')])


# Create the labels
action_labels = []
emotion_labels = []
for f in filenames:
    action_labels.append(os.path.basename(f).split('.')[0][10:][:-4].split('_')[0].split('-')[0])
    emotion_labels.append(os.path.basename(f).split('.')[0][10:][:-4].split('_')[1])

# Create unique versions of the lists
unique_action_labels = list(set(action_labels))
unique_emotion_labels = list(set(emotion_labels))

print("Unique action labels:", unique_action_labels)
print("Unique emotion labels:", unique_emotion_labels)

# Encode labels
action_label_encoder = LabelEncoder()
action_label_encoder.fit(action_labels)
action_labels_encoded = action_label_encoder.transform(action_labels)

emotion_label_encoder = LabelEncoder()
emotion_label_encoder.fit(emotion_labels)
emotion_labels_encoded = emotion_label_encoder.transform(emotion_labels)

print(f'{len(filenames)=}')
print(f'{len(action_labels)=}, unique: {len(unique_action_labels)=}')
print(f'{len(emotion_labels)=}, unique: {len(unique_emotion_labels)=}')

# features = ['acceleration', 'gyroscope', 'magnetometer', 'orientation']
features = ['acceleration', 'gyroscope', 'magnetometer']
n_features = len(active_sensors)*3*len(features)

max_len = 4749
print(f'{max_len=}')

# np_files = []
# for f in filenames:
#     tmp = np.load(f)
#     for feat in features:
#         # print(tmp[feat].shape)
#         if tmp[feat].shape[0] > max_len:
#             max_len = tmp[feat].shape[0]
#     np_files.append(np.load(f))
#     break

# print(f'{len(np_files)}')

indices = [x for x in range(len(filenames))]

X_train_idxs, X_test_idxs = train_test_split(
                                            indices, 
                                            test_size=0.3, 
                                            random_state=0
                                            )

X_train_imu = [filenames[i] for i in X_train_idxs]
X_test_imu = [filenames[i] for i in X_test_idxs]

Y_train_action_labels = [action_labels_encoded[i] for i in X_train_idxs]
Y_test_action_labels = [action_labels_encoded[i] for i in X_test_idxs]

Y_train_emotion_labels = [emotion_labels_encoded[i] for i in X_train_idxs]
Y_test_emotion_labels = [emotion_labels_encoded[i] for i in X_test_idxs]

print(f'{len(X_train_imu)=}, {len(X_test_imu)=}')
print(f'{len(Y_train_action_labels)=}, {len(Y_test_action_labels)=}')
print(f'{len(Y_train_emotion_labels)=}, {len(Y_test_emotion_labels)=}\n')

# Convert to tensors
Y_train_action_labels = torch.tensor(Y_train_action_labels)
Y_test_action_labels = torch.tensor(Y_test_action_labels)
Y_train_emotion_labels = torch.tensor(Y_train_emotion_labels)
Y_test_emotion_labels = torch.tensor(Y_test_emotion_labels)

if training_output == 'actions':

    # Create the imu datasets
    train_dataset_imu = SequenceDatasetNPY(X_train_imu, Y_train_action_labels, features, active_sensors, max_len=max_len)
    test_dataset_imu = SequenceDatasetNPY(X_test_imu, Y_test_action_labels, features, active_sensors, max_len=max_len)
    unique_labels = unique_action_labels

elif training_output == 'emotions':

    # Create the imu datasets
    train_dataset_imu = SequenceDatasetNPY(X_train_imu, Y_train_emotion_labels, max_len=max_len)
    test_dataset_imu = SequenceDatasetNPY(X_test_imu, Y_test_emotion_labels, max_len=max_len)
    unique_labels = unique_emotion_labels

print('Datasets Initialized\n')


# Create the training data loaders for the imu and video datasets
train_loader_imu = DataLoader(train_dataset_imu, batch_size=batch_size, shuffle=True)

# Create the test data loaders for the imu and video datasets
test_loader_imu = DataLoader( test_dataset_imu,  batch_size=batch_size, shuffle=True)
print('Data Loaders Initialized\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HAR_Transformer(n_features, nhead, num_encoder_layers, dim_feedforward, len(unique_labels), max_len).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f'Model initialized on {device}\n')

# Initialize early stopping
early_stopping = EarlyStopper(saving_path=os.path.join('_new_imu_results', checkpoint_model_name), patience=patience)

best_model = None
train_losses = []

print('\n\n\n')

if do_train:
    for epoch in range(num_epochs):
        model.train()
        for imu_seqs, labels, batch_lengths in train_loader_imu:

            # # Normalize the imu sequences using the maxes and mins
            # for i in range(imu_seqs.shape[0]):
            #     imu_seqs[i] = (imu_seqs[i] - mins) / (maxes - mins)

            # Standardize the imu sequences
            # for i in range(imu_seqs.shape[0]):
            #     imu_seqs[i] = (imu_seqs[i] - means) / stds

            # print(imu_seqs.dtype)
            # print(labels.dtype)
            # print(labels)
            # print(batch_lengths.dtype)
            # print(batch_lengths)
            imu_seqs = imu_seqs.to(device)

            labels = labels.to(device)
            batch_lengths = batch_lengths.to(device)

            outputs = model(imu_seqs, batch_lengths)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del imu_seqs
            del labels
            del batch_lengths

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Best Loss: {early_stopping.min_validation_loss:.4f}, Counter: {early_stopping.counter}')

        train_losses.append(loss.item())

        if early_stopping.early_stop(loss.item(), model.state_dict()):
            print("Early stopping")

            # Save the best model
            # torch.save(early_stopping.best_model_state_dict, os.path.join('video_results', checkpoint_model_name))
            # print(f'Model saved to {checkpoint_model_name}')

            break   
        else:
            best_model = model.state_dict()
            # torch.save(model.state_dict(), os.path.join('video_results', checkpoint_model_name))

    # Plot train losses
    plt.figure()
    plt.plot(train_losses)
    plt.title("Train Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


# Load the best model
model.load_state_dict(torch.load(os.path.join('_new_imu_results', checkpoint_model_name)))

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for imu_seqs, labels, batch_lengths in test_loader_imu:

        # # Normalize the imu sequences using the maxes and mins
        # for i in range(imu_seqs.shape[0]):
        #     imu_seqs[i] = (imu_seqs[i] - mins) / (maxes - mins)

        # Standardize the imu sequences
        # for i in range(imu_seqs.shape[0]):
        #     imu_seqs[i] = (imu_seqs[i] - means) / stds
        
        imu_seqs = imu_seqs.to(device)
        labels = labels.to(device)
        batch_lengths = batch_lengths.to(device)

        outputs = model(imu_seqs, batch_lengths)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Compute F1 score using scikit-learn
f1 = f1_score(y_true, y_pred, average='macro')
print(f'F1 Score: {f1:.4f}')

# Confusion matrix with precision and recall
conf_matrix_ext = create_confusion_matrix_w_precision_recall(y_true, y_pred, accuracy)

# Plot extended confusion matrix
plt.figure(figsize=(16, 9))
sns.heatmap(conf_matrix_ext, annot=True, fmt='.2f', cmap='Blues', xticklabels= unique_labels + ['Recall'], yticklabels= unique_labels + ['Precision'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Precision and Recall')
plt.savefig(os.path.join('_new_imu_results', confusion_matrix_name[:-4] + f'_f1_{f1:.3f}.png'))
plt.show()
