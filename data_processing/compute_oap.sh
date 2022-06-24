for r in $(seq -f "%03g" 1 160); do
  for d in /YOUR_PATH_HERE/binaural-sound-perception/dataset_public/scene0$r/split_videoframes; do
    python test_simple.py --image_path $d  --model_name mono+stereo_1024x320 --ext png
    # echo $d
  done
done