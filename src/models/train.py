
import os
import torch
import torch.nn as nn
from models.segformer_aux import SegAuxClassifier
from utils.loss import get_loss

def trainer(model, trainloader, testloader, optimizer,config):
    """train the model based on the config file

    Args:
        model (_type_): pytorch model to train
        trainloader (_type_): pytorch dataloader to train the data
        testloader (_type_): pytorch dataloader to test the model
        optimizer (_type_): adam optimizer
        config (_type_): config yaml file from main
    """
    if config['classification']['state'] == True:
        train_classification(model, trainloader, testloader, optimizer,config)
    elif config['segmentation']['model_name'] == 'segformer':
        train_segformer(model, trainloader, testloader, optimizer,config)
    else:
        train_segmentation(model, trainloader, testloader, optimizer, config)

def train_segmentation(model, trainloader,testloader,optimizer,config):
    """train the model based for segmentation can include auxilary classification
        based on config

    Args:
        model (_type_): pytorch model to train
        trainloader (_type_): pytorch dataloader to train the data
        testloader (_type_): pytorch dataloader to test the model
        optimizer (_type_): adam optimizer
        config (_type_): config yaml file from main
    """

    epochs = config['model']['epoch']
    model_dir =  config['save_path'] + config['segmentation']['model_name'] 
    logger = model_dir + '/log.txt'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    segmentation_classfication = config['segmentation']['classification']
    best_accuracy = 0
    criterion = get_loss(config['segmentation']['loss'])

    if segmentation_classfication:
        classification_criterion = nn.CrossEntropyLoss()

    device = config['device']
    for epoch in range(epochs):
        train_loss_epoch = 0
        test_loss_epoch = 0
        model.train()
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            image, mask, label = data
            image = image.to(device)
            label = label.to(device)
            mask = mask.to(device)

            if segmentation_classfication:                    
                output, output_label = model(image)
                segmentation_loss = criterion(output, mask)
                classification_loss = classification_criterion(output_label, label)
                loss = classification_loss + segmentation_loss
            
            else:
                output = model(image)
                loss = criterion(output, mask)
    
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
        
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(testloader):
                image, mask, label = data
                image = image.to(device)
                label = label.to(device)
                mask = mask.to(device)

                if segmentation_classfication:                    
                    output, output_label = model(image)
                    segmentation_loss = criterion(output, mask)
                    classification_loss = criterion(output_label, label)
                    loss = classification_loss + segmentation_loss

                    _, predicted = torch.max(output_label.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    accuracy = correct / total

                else:
                    output = model(image)
                    loss = criterion(output, mask)
                    accuracy = 0
            
        progress = 'epoch = {} , train_loss = {} , test_loss = {} , accuracy = {}'\
                                .format(epoch, train_loss_epoch, test_loss_epoch, accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path + '/best_model.pth')
        with open(logger , 'a') as f:
            f.write(progress + '\n')


def train_classification(model, trainloader, testloader, optimizer,config):
    """train the model based on classification

    Args:
        model (_type_): pytorch model to train
        trainloader (_type_): pytorch dataloader to train the data
        testloader (_type_): pytorch dataloader to test the model
        optimizer (_type_): adam optimizer
        config (_type_): config yaml file from main
    """
    epochs = config['model']['epoch']
    model_dir =  config['save_path'] + config['classification']['model_name'] 
    logger = model_dir + '/log.txt'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    save_path = config['save_path'] + config['classification']['model_name']
    best_accuracy = float('inf')
    criterion = nn.CrossEntropyLoss()  
    device = config['device']
    for epoch in range(epochs):
        train_loss_epoch = 0
        test_loss_epoch = 0
        model.train()
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            image, _, label = data
            image = image.to(device)
            label = label.to(device).squeeze()
            output = model(image)

            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
        
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(testloader):
                image = image.to(device)
                label = label.to(device).squeeze()
                output = model(image)
                test_loss_epoch = criterion(output, label.long())
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = correct / total
        progress = 'epoch = {} , train_loss = {} , test_loss = {} , accuracy = {}'\
                                .format(epoch, train_loss_epoch, test_loss_epoch, accuracy)
        

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path + '/best_model.pth')
        with open(logger, 'a') as f:
            f.write(progress + '\n')

def train_segformer(model, trainloader,testloader,optimizer,config):
    """train the model based on segmentation can include classification by specifying in config

    Args:
        model (_type_): pytorch model to train
        trainloader (_type_): pytorch dataloader to train the data
        testloader (_type_): pytorch dataloader to test the model
        optimizer (_type_): adam optimizer
        config (_type_): config yaml file from main
    """
    epochs = config['model']['epoch']
    model_dir =  config['save_path'] + config['segmentation']['model_name'] 
    logger = model_dir + '/log.txt'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    segmentation_classfication = config['segmentation']['classification']
    best_accuracy = 0
    criterion = get_loss(config['segmentation']['loss'])
    device = config['device']
    if segmentation_classfication:
        classification_criterion = nn.CrossEntropyLoss()
        aux = SegAuxClassifier().to(device)

    device = config['device']
    for epoch in range(epochs):
        train_loss_epoch = 0
        test_loss_epoch = 0
        model.train()
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            image, mask, label = data
            image = image.to(device)
            label = label.to(device)
            mask = mask.to(device)
            o = model(pixel_values=image, labels=mask)
            loss , logits = o.loss , o.logits
            
            if segmentation_classfication:
                encoder_output = model.segformer.encoder(image)  
                output_label = aux(encoder_output[0])                  
                classification_loss = classification_criterion(output_label, label)
                loss += classification_loss

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
        
        total = 0
        correct = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(testloader):
                image, mask, label = data
                image = image.to(device)
                label = label.to(device)
                mask = mask.to(device)
                o = model(pixel_values=image, labels=mask)
                loss, _ = o.loss , o.logits
                if segmentation_classfication:
                    encoder_output = model.segformer.encoder(image)  
                    output_label = aux(encoder_output[0])                  
                    classification_loss = criterion(output_label, label)
                    loss += classification_loss
                
                    _, predicted = torch.max(output_label.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

        accuracy = correct / total
        progress = 'epoch = {} , train_loss = {} , test_loss = {} , accuracy = {}'\
                                .format(epoch, train_loss_epoch, test_loss_epoch, accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path + '/best_model.pth')
        with open(logger, 'a') as f:
            f.write(progress + '\n')

