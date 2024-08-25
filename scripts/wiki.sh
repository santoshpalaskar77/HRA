# Set the desired prediction length

echo "HEre"
python -u Wiki.py \
    --test_len 8 \
    --p 0.1 \
    --data_path datasets/wiki.csv \
    --input_len 4 \
    --output_len 4 \
    --base_model "DLinear" \
    --epochs 100 \
    --batch_size 32 \
