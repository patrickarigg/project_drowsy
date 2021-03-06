# Project Drowsy 🥱
Project Drowsy is a computer vision project to predict driver drowsiness based on two key characteristics - eyes open/closed and yawning/not-yawning.

<div class="example_pics">
    <img src="https://github.com/patrickarigg/project_drowsy/blob/master/drowsy_example.png" alt="drowsy example" width="400"/>
    <img src="https://github.com/patrickarigg/project_drowsy/blob/master/alert_example.png" alt="drowsy example" width="400"/>
</div>

The model uses [mediapipe](https://google.github.io/mediapipe/) for face/eye detection and a custom trained CNN model to predict eyes open/closed and yawning/not-yawning. The CNN model was trained using a [kaggle dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
containing
- 726 images of eyes closed
- 726 images of eyes open
- 725 images of drivers yawning
- 723 images of drivers not yawning

Try it out F.R.I.D.A for yourself [here](https://share.streamlit.io/patrickarigg/project_drowsy/cloud-app)!

This project was created as part of the [Le Wagon Data Science Bootcamp](https://www.lewagon.com/data-science-course/part-time) (batch #753) by 
[Patrick Rigg](https://github.com/patrickarigg), 
[Will Graham](https://github.com/willgraham29), 
[Kai Majerus](https://github.com/kai-majerus) and 
[Julien Festou](https://github.com/JulienFest).




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
