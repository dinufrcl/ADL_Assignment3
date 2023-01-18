import torch.nn
from torch.utils.tensorboard import SummaryWriter
from models import *
from dataset import *
from torchsummary import summary
import sys
import torchvision.transforms.functional as TF

# device config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if sys.gettrace() is None:
    num_workers = 16
else:
    num_workers = 0

# reduce non-deterministic behaviour
torch.manual_seed(0)
np.random.seed(0)

# model config
model = ResNet18Model()
model = model.to(device)

# training config
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)
loss_function = torch.nn.MSELoss()
use_random_scale = False
augment_prob = 1
n_epochs = 400

scaler = torch.cuda.amp.GradScaler()

training_history = []
continue_training = False
performed_epochs = 0
save_path = 'data/ResNet18Model_v3' # save best model
save_path2 = 'data/ResNet18Model_v3backup' # save for backup if continue_training==True

# tensorboard config
writer = SummaryWriter("runs/ResNet18Model_v3")

# test model
summary(model, (3, 128, 128))
