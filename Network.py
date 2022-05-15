from platform import architecture
import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.embedding = nn.Embedding(87, embedding_dim=256)
        self.layer = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=(5,)),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=(5,)),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 256, kernel_size=(5,)),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer(x)
        return x.squeeze()

'''
architecture one 
'''
class CNN_2X(nn.Module):
    def __init__(self):
        super(CNN_2X, self).__init__()
        self.embeddingnet = EmbeddingNet()
        self.Fc = nn.Sequential(
           
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, pivot, statement):
        embedded_pivot = self.embeddingnet(pivot)
        embedded_statement = self.embeddingnet(statement)
        x = torch.cat([embedded_pivot, embedded_statement], dim=1)
        x = self.Fc(x)
        return(x)




# '''
# architecture two  
# '''
# class CNN_2X(nn.Module):
#     def __init__(self):
#         super(CNN_2X, self).__init__()
#         self.embeddingnet = EmbeddingNet()
#         self.Fc = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, pivot, statement):
#         embedded_pivot = self.embeddingnet(pivot)
#         embedded_statement = self.embeddingnet(statement)
#         x = torch.cat([embedded_pivot, embedded_statement], dim=1)
#         x = self.Fc(x)
#         return(x)


#  architecture Three
# class CNN_2X(nn.Module):
#     def __init__(self):
#         super(CNN_2X, self).__init__()
#         self.embeddingnet = EmbeddingNet()
#         self.Fc = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, pivot, statement):
#         embedded_pivot = self.embeddingnet(pivot)
#         embedded_statement = self.embeddingnet(statement)
#         x = torch.cat([embedded_pivot, embedded_statement], dim=1)
#         x = self.Fc(x)
#         return(x)


# '''
# architecture Four  
# '''
# class CNN_2X(nn.Module):
#     def __init__(self):
#         super(CNN_2X, self).__init__()
#         self.embeddingnet = EmbeddingNet()
#         self.Fc = nn.Sequential(
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, pivot, statement):
#         embedded_pivot = self.embeddingnet(pivot)
#         embedded_statement = self.embeddingnet(statement)
#         x = torch.cat([embedded_pivot, embedded_statement], dim=1)
#         x = self.Fc(x)
#         return(x)


# architecture 3
# class EmbeddingNet(nn.Module):
#     def __init__(self):
#         super(EmbeddingNet, self).__init__()

#         self.embedding = nn.Embedding(87, embedding_dim=256)
#         self.layer = nn.Sequential(
#             nn.Conv1d(512, 256, kernel_size=(5,)),
#             nn.ReLU(),
#             nn.MaxPool1d(3),
#             nn.Conv1d(256, 256, kernel_size=(5,)),
#             nn.ReLU(),
#             nn.MaxPool1d(3),
#             nn.Conv1d(256, 256, kernel_size=(5,)),
#             nn.ReLU()
#             # nn.AdaptiveMaxPool1d(output_size=1)
#         )
#         self.lstm = nn.LSTM(22, 1, batch_first=True)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.layer(x)
#         x, (hidden, c) = self.lstm(x)
#         return x.squeeze()


# class CNN_2X(nn.Module):
#     def __init__(self):
#         super(CNN_2X, self).__init__()
#         self.embeddingnet = EmbeddingNet()
#         self.Fc = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, pivot, statement):
#         embedded_pivot = self.embeddingnet(pivot)
#         embedded_statement = self.embeddingnet(statement)
#         x = torch.cat([embedded_pivot, embedded_statement], dim=1)
#         x = self.Fc(x)
#         return(x)