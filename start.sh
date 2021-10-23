apt update && \
apt install -y git && \
apt-get install -y libsndfile1-dev && \
git clone https://github.com/timothyxp/asr_project_template.git && \
cd asr_project_template || return

git checkout timothyxp_work && \
pip install -r requirements.txt && \
pip install ruamel_yaml --ignore-installed && \
apt install -y htop

mkdir best_model && \
gdown https://drive.google.com/uc?id=1jkkcVDbOiypn05AMbbDanEB5RZQxGOIN --output best_model/best_model.pth && \
gdown https://drive.google.com/uc\?id\=1lv2CvNx98h0qKhhIsG2v8pJzl4FG_WYW --output best_model/config.json && \
echo "success loading" && \
python test.py -c hw_asr/config_test.json -r best_model/best_model.pth

