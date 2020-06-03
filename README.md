Data and software used to produce all leaf shape analysis in Zhang et al.

File inventory:
1) LeafInterrogator: python software used for leaf shape analysis
2) Leafi_user_guide_minimal.docx
3) Data: LeafI project files


Installation:

Step 1, download:

    • Go to the directory where you want LeafI to be installed

    • Clone from repository (you need an account and access first):

        git clone https://gitlab.mpcdf.mpg.de/g-adamrunions/leafinterrogator_zhang_et_al.git

Step 2, create the python environment:

    • First, install python virtual environment package, and then upgrade pip:

        apt-get install python-virtualenv

        apt-get install python3-venv

    • Create a virtual environment for LeafI (local directory):

        python3 -m venv leafiEnv

Step 3, enable and configure environment:

    • Activate the environment, and upgrade pip:

        source leafiEnv/bin/activate

        pip3 install --upgrade pip

    • Installed required packages using (you can find minimal_requirements.txt in the main directory for LeafI) :

        python3 -m pip install -r minimal_requirements.txt
	
    • If downloading times-out, rerun the preceding command

    • If there are missing packages, these can be installed individually using:

        python3 –m pip install <package_name>==<version-number>

	Note that “==<version-number>” can be omitted for all packages except opencv-python

    • The complete set of packages to install are:

        freetype-py

	    imutils

	    mahotas

	    matplotlib

	    networkx

	    numpy

	    pandas

	    Pillow

	    PyQt5

	    PyOpenGL

	    pyrr

	    QtAwesome

	    scikit-learn

	    scipy

	    opencv-python==3.4.9.31

Step 4, run LeafI:

    • make sure your environment is activated (source leafiEnv/bin/activate)

    • Run: python3 main.py

    • Note: the first time you run LeafI it will be slow to start
    