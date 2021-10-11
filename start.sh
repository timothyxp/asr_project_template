apt update
apt install -y git
apt-get install libsndfile1-dev

git clone https://github.com/timothyxp/asr_project_template.git
cd asr_project_template

git checkout timothyxp

pip install -r requirements.txt

pip install ruamel_yaml --ignore-installed

apt install htop
