data_url=www.openslr.org/resources/60
data_dir=/group_share/tts/openslr/libritts

echo "Data Download"
for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  /root/code/LinguaWave/data/libritts/download_and_untar.sh ${data_dir} ${data_url} ${part}
done