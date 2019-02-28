# Pytorch Workshop
This repo contains materials for use in a Pytorch workshop at [CIDAS 2019](https://cidas.iasbs.ac.ir/) .

**Requirements**
We only need torch, torchvision, numpy, matplotlib, and jupyter notebook, preferably with python3. Tough GPUs are very important for deep learning, We work with simple models and toy data and it is not necessary to have GPU!

**Installing Requirements**
-  Install python3 and pip
- Recommended: virtualenv with virtualenvwrapper (to create isolated environment with python packages for this tutorial).
  -  sudo pip install virtualenv virtualenvwrapper
  -  add the following lines to your ~/.bashrc or ~/.zshrc or ~/.bash_profile (depends what you are using)
    ```
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
    export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
    export VIRTUALENVWRAPPER_VIRTUALENV_ARGS='--no-site-packages'
    ```
  -  mkvirtualenv pytorch_ws --python=python3
  -  workon pytorch_ws
-  pip install matplotlib numpy jupyter notebook
-  install pytorch ccording to the guidelines in https://pytorch.org/

**Alternative - Execute on google colab:**
You can run your code on some google machines for free.

Go to https://colab.research.google.com and sign in with your google account (you need one to use colab)

File --> open notebook --> https://github.com/hadiasheri/cidas2019_pytorch.git

**Note**

Most of the materials are borrowed from popular publicly available resources:
-  Python-numpy: http://cs231n.github.io/python-numpy-tutorial/
-  Pytorch: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
