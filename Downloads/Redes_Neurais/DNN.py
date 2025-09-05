import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import itertools

# ----------------------------
# 1️⃣ Detectar dispositivo
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ----------------------------
# 2️⃣ Definir caminhos
# ----------------------------
caminho_1 = r"C:\Users\Dell\Desktop\modelos_colisao\cologne\cologne_colisao"
caminho_2 = r"C:\Users\Dell\Desktop\modelos_colisao\cologne\cologne"
caminho_resultados = r"C:\Users\Dell\Desktop\Modelos Redes Neurais\DNN"
os.makedirs(caminho_resultados, exist_ok=True)

pastas_colisao = [f"seed_{i}" for i in range(1, 75)]
pastas_sem_colisao = [f"seed_{i}" for i in range(1, 75)]
caminhos_completos = [os.path.join(caminho_1, p) for p in pastas_colisao] + \
                     [os.path.join(caminho_2, p) for p in pastas_sem_colisao]

# ----------------------------
# 3️⃣ Hiperparâmetros
# ----------------------------
num_layers_list = [1, 2]
hidden_sizes_list = [50, 100, 200, 400]
epochs_list = [100, 200, 300]
batch_sizes = [16, 32, 64]
test_steps = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60]

# ----------------------------
# 4️⃣ Função para carregar dados em lotes
# ----------------------------
def carregar_dados_em_batches(caminhos, batch_size=100):
    for i in range(0, len(caminhos), batch_size):
        dados_batch = []
        for caminho in caminhos[i:i + batch_size]:
            for arquivo in os.listdir(caminho):
                if arquivo.endswith(".xlsx"):
                    try:
                        df = pd.read_excel(os.path.join(caminho, arquivo))
                        dados_batch.append(df)
                    except Exception as e:
                        print(f"Erro ao ler {arquivo}: {e}")
        if dados_batch:
            yield pd.concat(dados_batch, ignore_index=True)

# ----------------------------
# 5️⃣ Carregar e pré-processar dados
# ----------------------------
print("Carregando dados...")
dados_completos = []

for dados in carregar_dados_em_batches(caminhos_completos, batch_size=100):
    dados = dados.replace(['colisao', 'Colisao'], 2)
    for h in range(1, 61):
        for col in ['Posicao X', 'Posicao Y', 'Direcao', 'Distancia', 'Time Buffer', 'Status']:
            dados[f'{col} +{h}'] = dados[col].shift(-h)
    dados.dropna(inplace=True)
    dados_completos.append(dados)

dados = pd.concat(dados_completos, ignore_index=True)

# ----------------------------
# 6️⃣ Normalização dos dados
# ----------------------------
scaler_entrada = MinMaxScaler()
scaler_saida = MinMaxScaler()
colunas_entrada = ['Posicao X', 'Posicao Y', 'Velocidade', 'Aceleracao', 'Direcao', 'Distancia']

colunas_saida = []
for h in range(1, 61):
    for col in ['Posicao X', 'Posicao Y', 'Direcao', 'Distancia', 'Time Buffer', 'Status']:
        colunas_saida.append(f'{col} +{h}')

dados_entrada = scaler_entrada.fit_transform(dados[colunas_entrada].astype(np.float32))
dados_saida = scaler_saida.fit_transform(dados[colunas_saida].astype(np.float32))

# ----------------------------
# 7️⃣ Definir modelo DNN
# ----------------------------
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DNNModel, self).__init__()
        layers = []
        in_features = input_size

        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 8️⃣ Criar arquivo de log
# ----------------------------
log_path = os.path.join(caminho_resultados, "resultados_busca.txt")
with open(log_path, "w") as f:
    f.write("Teste de Hiperparâmetros - DNN\n\n")

# ----------------------------
# 9️⃣ Loop principal de treino
# ----------------------------
output_size = len(range(1, 61)) * 6
melhor_mse = float('inf')
melhor_config = None
melhor_modelo = None

param_grid = list(itertools.product(num_layers_list, hidden_sizes_list, epochs_list, batch_sizes, test_steps))

for num_layers, hidden_size, epochs, batch_size, passos in param_grid:
    print(f"\nTreinando: num_layers={num_layers}, hidden_size={hidden_size}, epochs={epochs}, batch_size={batch_size}, passos={passos}")

    # Criar sequências para DNN
    X, y = [], []
    for i in range(passos, len(dados_entrada)):
        X.append(dados_entrada[i - passos:i].flatten())
        y.append(dados_saida[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    modelo = DNNModel(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

    # Treinamento
    modelo.train()
    for epoca in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            otimizador.zero_grad()
            saida = modelo(X_batch)
            perda = criterio(saida, y_batch)
            perda.backward()
            otimizador.step()

    # Avaliação
    modelo.eval()
    previsoes, reais = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            saida = modelo(X_batch)
            previsoes.append(saida.cpu().numpy())
            reais.append(y_batch.numpy())

    previsoes_np = np.concatenate(previsoes)
    reais_np = np.concatenate(reais)

    mae = mean_absolute_error(reais_np, previsoes_np)
    rmse = math.sqrt(mean_squared_error(reais_np, previsoes_np))
    r2 = r2_score(reais_np, previsoes_np)
    mse_val = mean_squared_error(reais_np, previsoes_np)

    # Salvar resultados no log
    with open(log_path, "a") as f:
        f.write(f"Passos: {passos}, Camadas: {num_layers}, Hidden_size: {hidden_size}, Batch: {batch_size}, Epocas: {epochs}\n")
        f.write(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}, MSE: {mse_val:.6f}\n\n")

    # Atualizar melhor modelo
    if mse_val < melhor_mse:
        melhor_mse = mse_val
        melhor_config = (passos, num_layers, hidden_size, batch_size, epochs)
        melhor_modelo = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
        melhor_previsoes = previsoes_np
        melhor_reais = reais_np

    torch.cuda.empty_cache()

# ----------------------------
# 🔟 Salvar CSV com previsões detalhadas
# ----------------------------
colunas_csv = []
for h in range(1, 61):
    for var in ['Posicao X', 'Posicao Y', 'Direcao', 'Distancia', 'Time Buffer', 'Status']:
        colunas_csv.append(f"{var} real +{h}")
    for var in ['Posicao X', 'Posicao Y', 'Direcao', 'Distancia', 'Time Buffer', 'Status']:
        colunas_csv.append(f"{var} prevista +{h}")

reais_desnorm = scaler_saida.inverse_transform(melhor_reais)
previsoes_desnorm = scaler_saida.inverse_transform(melhor_previsoes)

dados_csv = np.hstack([reais_desnorm, previsoes_desnorm])
df_resultados = pd.DataFrame(dados_csv, columns=colunas_csv)
csv_path = os.path.join(caminho_resultados, "resultados_dnn.csv")
df_resultados.to_csv(csv_path, index=False)

# ----------------------------
# 11️⃣ Mostrar melhor configuração
# ----------------------------
print(f"\n✅ Melhor configuração: Passos {melhor_config[0]}, Camadas {melhor_config[1]}, Hidden_size {melhor_config[2]}, Batch {melhor_config[3]}, Epocas {melhor_config[4]}")
print(f"MSE: {melhor_mse:.6f}")
print(f"Resultados detalhados salvos em: {log_path}")
print(f"CSV com previsões salvo em: {csv_path}")
