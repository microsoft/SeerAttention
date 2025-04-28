model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_dir="./dense_eval"
limit=-1
num_gpus=8

# use_batch_exist_options=("true" "false")
use_batch_exist_options=("true")
# tasks=("aime" "math" "gpqa" "olympiadbench")
tasks=("math")
# attention_implementations=("oracle_sparse" "fa2" "seer_dense")
attention_implementations=("oracle_sparse")

for task in "${tasks[@]}"; do
  for use_batch_exist in "${use_batch_exist_options[@]}"; do
    for attention_implementation in "${attention_implementations[@]}"; do
      
      case $task in
          aime)
              bs=30
              repeat=16
              ;;
          math)
              bs=250
              repeat=1
              ;;
          gpqa)
              bs=100   
              repeat=2
              ;;
          olympiadbench)
              bs=60  
              repeat=1
              ;;
          *)
              echo "Error: Unknown task '$task'"
              exit 1
              ;;
      esac

      if [ "$use_batch_exist" = "false" ]; then
          bs=20
      fi


      echo "Starting task: $task | use_batch_exist: $use_batch_exist | attention: $attention_implementation"
      echo "Batch size: $bs | Repeat: $repeat "

      for ((gpu=0; gpu<num_gpus; gpu++)); do
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          python eval.py \
              --model_name_or_path $model_dir \
              --data_name $task \
              --batch_size $bs \
              --limit $limit \
              --repeat $repeat \
              --output_dir $output_dir \
              --attention_implementation $attention_implementation \
              --use_batch_exist $use_batch_exist \
              --surround_with_messages \
              --rank $gpu &
      done
      wait

      python get_results.py \
          --model_name_or_path $model_dir \
          --data_name $task \
          --batch_size $bs \
          --limit $limit \
          --repeat $repeat \
          --output_dir $output_dir \
          --attention_implementation $attention_implementation \
          --use_batch_exist $use_batch_exist \
          --num_gpus $num_gpus

      echo "Completed: $task-$use_batch_exist-$attention_implementation"
      
    done
  done
done

echo "All tasks and configurations completed!"