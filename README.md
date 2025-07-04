# Contador de Alunos em Sala de Aula

Este projeto utiliza técnicas de **visão computacional** com **Python** e **OpenCV** para detectar e contar automaticamente o número de pessoas em uma imagem de uma sala de aula. A solução é útil para controle de presença, gerenciamento de ocupação e estatísticas em ambientes educacionais.

## Exemplo

> Imagem de entrada → Detecção de pessoas → Exibição com caixas delimitadoras e contagem total  
> **Resultado esperado:** `Detectadas 12 pessoas`

---

## Tecnologias Utilizadas

- Python 3.x  
- OpenCV  
- Modelo pré-treinado: **MobileNetSSD**  
- Ambiente virtual: `venv`

---

## Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/SeuUsuario/Contador-Pessoas.git
cd Contador-Pessoas

### 2. Crie e ative o ambiente virtual
Windows:
- python -m venv venv
- venv\Scripts\activate

Linux/macOS:
- python3 -m venv venv
- source venv/bin/activate

### 3. Instale as dependências
- pip install opencv-python

### 4. Adicione a imagem da sala
Adicione sua imagem com o nome sala.jpg na raiz do projeto, ou modifique o nome no script contador_pessoas.py.

## Executando o projeto
bash
Copiar
Editar
python contador_pessoas.py
O script abrirá a imagem com caixas ao redor das pessoas e exibirá a contagem no terminal.

## Como Funciona
-  Usa cv2.dnn.readNetFromCaffe para carregar o modelo.

-  Detecta objetos na imagem.

-  Filtra os objetos com a classe "person".

-  Desenha caixas delimitadoras nas pessoas.

-  Imprime no terminal a contagem total.

## Estrutura de Arquivos
Copiar
Editar
ContadorPessoas/
├── contador_pessoas.py
├── sala.jpg
├── MobileNetSSD_deploy.prototxt.txt
├── MobileNetSSD_deploy.caffemodel
├── venv/
└── requirements.txt (opcional)

## Desenvolvedora
Bruna Conjunscki
GitHub: @BrunaConjunscki
LinkedIn: linkedin.com/in/BrunaConjunscki

