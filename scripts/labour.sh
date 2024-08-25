# Set the desired prediction length

echo "HEre"
python -u Labour.py \
    --test_len 8 \
    --p 0.1 \
    --data_path datasets/Labour.csv \
    --input_len 2 \
    --output_len 2 \
    --base_model "DLinear" \
    --epochs 100 \
    --batch_size 32 \
