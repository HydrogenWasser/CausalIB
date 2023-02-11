from utils import *


"                                                     "
"               Los Geht's!                           "
"                                 /                   "

num_epoch = 50

MLP_0 = MLP(device=device).to(device)
MLP_1 = MLP(device=device).to(device)
MLP_2 = MLP(device=device).to(device)
MLP_3 = MLP(device=device).to(device)

# CBDNet = Network().to(device)
# CBDNet = nn.DataParallel(CBDNet)

MLP_Train(model=MLP_0, num_epoch=num_epoch)
MLP_Adver(model=MLP_0, epsilon=0.2, name="MLP", adver_type="pgd")

MLP_AT_Train(model=MLP_1, num_epoch=num_epoch, epsilon=0.083, name="MLP_AT", adver_type="pgd")
MLP_Adver(model=MLP_1, epsilon=0.2, name="MLP_AT", adver_type="pgd")

InfoAT_Train(model=MLP_2, num_epoch=num_epoch, epsilon=0.083, name="InfoAT", adver_type="pgd")
MLP_Adver(model=MLP_2, epsilon=0.2, name="InfoAT", adver_type="pgd")

