# Set the desired prediction length

echo "HEre"
python -u simulated.py \
    --test_len 10 \
    --p 0.1 \
    --input_len 10 \
    --output_len 5 \
    --base_model "DLinear" \
    --epochs 100 \
    --batch_size 20 \
