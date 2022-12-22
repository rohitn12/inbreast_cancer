from torch import optim
from models.train import trainer
from models.test import eval
from models.get_model import load_model
from utils.utils import get_dataloaders, load_yaml

path_to_config = '/usr/mvl2/knfdt/cancer_detection/src/config/config.yml'
config = load_yaml(path_to_config)

device = config['device']
model = load_model(config).to(device)
if config['model']['train']:    
    trainloader, testloader = get_dataloaders(config)
    lr = config['model']['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999), weight_decay=0)
    trained_model = trainer(model, trainloader, testloader, optimizer, config)

else:
    trainloader, testloader = get_dataloaders(config)
    eval(model, testloader, config)



