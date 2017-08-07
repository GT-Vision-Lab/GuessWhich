cd chatbot
mkdir data && cd data

base_url="https://filebox.ece.vt.edu/~deshraj/guesswhich_github/"

wget "${base_url}chat_processed_params.json"
wget "${base_url}qbot_hre_qih_sl.t7"
wget "${base_url}abot_hre_qih_sl.t7"
wget "${base_url}all_pools_vgg16_features.t7"
wget "${base_url}final_vgg16_pool_features.t7"
wget "${base_url}qbot_rl.t7"
wget "${base_url}abot_rl.t7"

cd ../..
