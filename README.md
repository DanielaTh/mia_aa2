# MIA - Notebooks de aprendizaje automático

Este proyecto contiene notebooks de Jupyter con ejemplos de perceptrón, Word2Vec y redes neuronales convolucionales.

## Requisitos previos

- Python 3.10 o superior.
- `pip` actualizado.

## Instalación

1. Abrí una terminal en la carpeta del proyecto.

2. Creá un entorno virtual:

   ```powershell
   python -m venv .venv
   ```

3. Activá el entorno virtual:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Si usás CMD en lugar de PowerShell:

   ```bat
   .venv\Scripts\activate.bat
   ```

4. Instalá las dependencias:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Instalá Jupyter para ejecutar los notebooks:

   ```powershell
   python -m pip install jupyter
   ```

6. Iniciá Jupyter:

   ```powershell
   jupyter notebook
   ```

   Después abrí el notebook que quieras ejecutar:

   - `1_Perceptron.ipynb`
   - `2_word2vec.ipynb`
   - `3_CNN.ipynb`

## Notas

- `3_CNN.ipynb` descarga automáticamente el dataset MNIST en la carpeta `dataset/` durante su primera ejecución.
- Para desactivar el entorno virtual al terminar, ejecutá `deactivate`.
