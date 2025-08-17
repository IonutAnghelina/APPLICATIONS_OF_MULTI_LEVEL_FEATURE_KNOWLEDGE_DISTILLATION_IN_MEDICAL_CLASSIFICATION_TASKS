from MedViT.MedViT import MedViT_small
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from train_functions import train_one_dataset
from torchvision import transforms
import torch.nn as nn
from train_functions import train_joint_teacher, train_student
from models import MultiLevelJointTeacher
import torch
#Hyperparameters

NUM_CLASSES = 4
ATTENTION_DROPOUT = 0.5
DROPOUT = 0.5
NUM_EPOCHS = 0
NUM_TEACHER_EPOCHS = 0
NUM_STUDENT_EPOCHS = 1

#Configure Individual Teachers

#MedViT Small example
individual_teachers = []
individual_teachers.append(MedViT_small(num_classes=NUM_CLASSES, pretrained=True, attn_drop=ATTENTION_DROPOUT))

#EfficientNet B0 example
individual_teachers.append(efficientnet_b0(dropout=0.3, weights = EfficientNet_B0_Weights.IMAGENET1K_V1))
individual_teachers[-1].classifier[1] = nn.Linear(individual_teachers[-1].classifier[1].in_features, NUM_CLASSES)


individual_teachers.append(MedViT_small(num_classes=NUM_CLASSES, pretrained=True, attn_drop=ATTENTION_DROPOUT))

paths = ["./Dataset1","./Dataset2", "./Dataset3"]

individual_teachers[0] = train_one_dataset(
    model=individual_teachers[0],   
    path=paths[0],
    epochs=NUM_EPOCHS,
    load=False,
    bs=8,
    LR=1e-5
)

individual_teachers[1] = train_one_dataset(
    model=individual_teachers[1],
    path=paths[1],
    epochs=NUM_EPOCHS,    
    load=False,
    bs=8,   
    LR=1e-5
)

individual_teachers[2] = train_one_dataset(
    model=individual_teachers[2],
    path=paths[2],
    epochs=NUM_EPOCHS,    
    load=False,
    bs=8,   
    LR=1e-5
)

fuse_indices = [len(teacher.features) for teacher in individual_teachers]

print(fuse_indices)

joint_teacher = train_joint_teacher(
    teachers=individual_teachers,  
    fuse_indices=fuse_indices,
    joint_ch=1024,
    num_heads=8,
    num_attention_layers=3,
    lambda_adv=1.0,
    num_classes=NUM_CLASSES,
    epochs=NUM_TEACHER_EPOCHS,
    load=False,
    bs=1,
    lr=1e-5,
    wd = 1e-2,
    paths=paths
)

individual_students = []

individual_students.append(MedViT_small(num_classes=NUM_CLASSES, pretrained=True, attn_drop=ATTENTION_DROPOUT))
individual_students.append(MedViT_small(num_classes=NUM_CLASSES, pretrained=True, attn_drop=ATTENTION_DROPOUT))
individual_students.append(MedViT_small(num_classes=NUM_CLASSES, pretrained=True, attn_drop=ATTENTION_DROPOUT))

medvit_last_layer = len(individual_students[0].features)

for i, p in enumerate(paths):
    train_student(path = p,joint_teacher=joint_teacher,
                  student=individual_students[i],load = False,
                  fuse_indices = [medvit_last_layer-1], epochs = NUM_STUDENT_EPOCHS,
                  bs=8, lr=1e-5, num_classes = NUM_CLASSES, 
                  lambda_ce = 1.0, lambda_kld = 0.5, lambda_fa = 0.5,
                  lambda_cos = 1.0, lambda_con = 0.5)