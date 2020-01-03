!nvidia-smi
import os
os.getcwd()
os.makedirs("out_transformers", exist_ok=True)
os.system("git clone https://github.com/simonepreite/transformers.git")
os.chdir("/content/transformers/")
%pip install  -e .  
os.chdir("./examples/")
%pip install -r requirements.txt
os.kill(os.getpid(), 9)
import os
os.chdir("/content/transformers/examples/distillation/")
%run run_squad_w_distillation.py --train_file ../tests_samples/SQUAD/train-v2.0.json --predict_file ../tests_samples/SQUAD/dev-v2.0.json --model_type distilbert --model_name_or_path distilbert-base-uncased --teacher_type bert --teacher_name_or_path bert-base-uncased --version_2_with_negative --do_train --do_eval --do_lower_case --gradient_accumulation_steps 32 --per_gpu_train_batch_size 24  --num_train_epochs 2 --output_dir "/content/out_transformers" --overwrite_output_dir
