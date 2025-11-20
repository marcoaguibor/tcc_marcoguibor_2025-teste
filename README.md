# TCC Marco Guibor
## Sobre
Esse repositório contém os códigos utilizados no meu Trabalho de Conclusão de Curso (TCC) em Estatística e Ciência de Dados pela UFPR.

O objetivo é compreender o NICE como um método de estimação de densidades e aprendizado generativo, e comparar com KDE e GANs.

## Pré-requisitos
- Python 3.x
- torch==2.9.0
- numpy==2.3.4
- matplotlib==3.10.7
- scikit-learn==1.7.2
- normflows==1.7.3

Para instalar as dependências em um ambiente virtual, utilize:

```bash
python -m venv .venv
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Para instalar a versão com suporte a GPU (CUDA), utilize:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Os códigos desenvolvidos estão na pasta
## 1 - Estimação de densidades

Para aplicar a estimação de densidades em two_moons, utilizando o NICE e KDE:

```bash
python main/density_estimation/density_estimation.py
```
## 2 - Aprendizado generativo

Para gerar a amostra original das bases MNIST e Fashion-MNIST:

```bash
python main/generative_learning/original_sampling.py
```

<p align="center">
  <img src="plots/imgs_mnist_sample.png" alt="" width = 300>

<p align="center">
  <img src="plots/imgs_fmnist_sample.png" alt="" width = 300>
</p>
