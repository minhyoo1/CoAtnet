# 필요한 패키지 import
import os
from tqdm import tqdm, trange
import torch
from torch import nn
from coatnet import coatnet_0

# CUDA 를 활용한 GPU 가속 여부에 따라, 장치를 할당 할 수 있도록 변수로 선언
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare Custom Model
model = coatnet_0()  # model 빌드
model.to(device)     # 모델의 장치를 device 에 할당
model.zero_grad()    # 모델 gradient 초기화
model.train()        # Train 모드로 모델 설정

# Train Start
train_iterator = trange(int(parameters['epoch']), desc="Epoch")  # 학습 상태 출력을 위한 tqdm.trange 초기 세팅
global_step = 0

# Epoch 루프
for epoch in train_iterator:
    epoch_iterator = tqdm(
        train_dataloader, desc='epoch: X/X, global: XXX/XXX, tr_loss: XXX'  # Description 양식 지정
    )
    epoch = epoch + 1

    # Step(batch) 루프
    for step, batch in enumerate(epoch_iterator):
        # 모델이 할당된 device 와 동일한 device 에 연산용 텐서 역시 할당 되어 있어야 함
        image_tensor, tags = map(lambda elm: elm.to(device), batch)  # device 에 연산용 텐서 할당
        out = model(image_tensor)      # Calculate
        loss = criterion(out, tags)    # loss 연산

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Update learning rate schedule
        global_step += 1
        # One train step Done

        # Step Description
        epoch_iterator.set_description(
            'epoch: {}/{}, global: {}/{}, tr_loss: {:.3f}'.format(
                epoch, parameters['epoch'],
                global_step, train_steps * parameters['epoch'],
                loss.item()
            )
        )