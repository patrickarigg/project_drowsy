# Project Drowsy
This project outlines a model to predict driver drowsiness based on two key characteristics - eyes open/closed and yawning/not-yawning.

The model uses [mediapipe](https://google.github.io/mediapipe/) for face/eye detection and a custom trained CNN model to predict eyes open/closed and yawning/not-yawning. The model was then packaged to

Try it out for yourself here:
https://share.streamlit.io/patrickarigg/project_drowsy/cloud-app

This project was created as part of the Le Wagon Data Science Bootcamp by 
[Patrick Rigg](https://github.com/patrickarigg), 
[Will Graham](https://github.com/willgraham29), 
[Kai Majerus](https://github.com/kai-majerus) and 
[Julien Festou](https://github.com/JulienFest).


- Document here the project: project_drowsy
- Description: Project Description
- Data Source: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for project_drowsy in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/project_drowsy`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "project_drowsy"
git remote add origin git@github.com:{group}/project_drowsy.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
project_drowsy-run
```

# Install

Go to `https://github.com/{group}/project_drowsy` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/project_drowsy.git
cd project_drowsy
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
project_drowsy-run
```
