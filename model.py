""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x
    
def l1_loss(model, lambda_l1=1e-5):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return lambda_l1 * l1_norm

def l2_loss(model, lambda_l2=1e-5):
    l2_norm = 0
    for param in model.parameters():
        l2_norm += torch.sum(param ** 2)
    return lambda_l2 * l2_norm


class MMDynamic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout, lambda_Adaptive_weighting):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout
        self.lambda_Adaptive_weighting = lambda_Adaptive_weighting

        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.prediction_layer = nn.ModuleList([LinearLayer(hidden_dim[0], in_dim[view]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim)-1):
            self.MMClasifier.append(LinearLayer(self.views*hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier = nn.Sequential(*self.MMClasifier)
            self.MMClasifierLayer = LinearLayer(hidden_dim[-1], num_class)
            self.MMTCPLayer = LinearLayer(hidden_dim[-1], 1)
        else:
            # self.MMClasifier = nn.Sequential(*self.MMClasifier)
            self.MMClasifierLayer = LinearLayer(self.views*hidden_dim[-1], num_class)
            self.MMTCPLayer = LinearLayer(self.views*hidden_dim[-1], 1)


    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence, input_recunstructed = dict(), dict(), dict(), dict(), dict()
        for view in range(self.views):

            # Add Noise to input -------------------------
            input_corrupted = F.dropout(data_list[view], self.dropout-0.35, training=self.training)
            if self.training:
                input_corrupted = input_corrupted + 0.0001*torch.mean(input_corrupted)*torch.randn_like(input_corrupted)
            # -------------------------------------------
            print("input_corrupted.shape: ",input_corrupted.shape)

            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](input_corrupted))
            feature[view] = input_corrupted * FeatureInfo[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]
            input_recunstructed[view] = self.prediction_layer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        # MMlogit = self.MMClasifier(MMfeature)
        if len(self.MMClasifier):
            MMfeature = self.MMClasifier(MMfeature)
        MMlogit = self.MMClasifierLayer(MMfeature)
        MMTCP = self.MMTCPLayer(MMfeature)
        if infer:
            return MMlogit
        
        MMpred = F.softmax(MMlogit, dim=1)
        MM_p_target = torch.gather(input=MMpred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
        # MMLoss = torch.mean(criterion(MMlogit, label))
        MMLoss = torch.mean(criterion(MMlogit, label) + 0.1*F.mse_loss(MMTCP.view(-1), MM_p_target))

                            
        for view in range(self.views):
            prediction_loss = torch.mean(F.mse_loss(input_recunstructed[view], data_list[view]))
            MMLoss = MMLoss+ 0.1*torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target)+criterion(TCPLogit[view], label))
            MMLoss = MMLoss + 10*confidence_loss
            MMLoss = MMLoss + 0.1*prediction_loss

            # Optional: Apply a weighting scheme to different views, if needed : 
            view_weight = torch.sigmoid(TCPConfidence[view])  # Weights based on confidence of each view
            MMLoss += self.lambda_Adaptive_weighting * torch.mean(view_weight * criterion(TCPLogit[view], label))
            # For example, if the confidence from a view is too low, reduce its contribution

        lambda_l1 = 1e-6  
        lambda_l2 = 1e-5  
        l1_reg = l1_loss(self, lambda_l1)
        l2_reg = l2_loss(self, lambda_l2)

        MMLoss = MMLoss + l1_reg + l2_reg

        return MMLoss, MMlogit
    
    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

