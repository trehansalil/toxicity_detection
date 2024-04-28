sudo apt -y update
sudo apt -y upgrade
sudo apt -y install python3.10-venv
python3 -m venv env
sudo apt -y install python3-pip
source env/bin/activate
pip install -r requirements.txt
