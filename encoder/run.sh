export CUDA_VISIBLE_DEVICES=0

python -u run.py --root_path weather_ny --model_id weather_ny --data weather --n_heads 16

python -u run.py --root_path weather_sf --model_id weather_ny --data weather --n_heads 16

python -u run.py --root_path weather_hs --model_id weather_ny --data weather --n_heads 16

python -u run.py --root_path finance_sp500 --model_id finance_sp500 --data finance --n_heads 4

python -u run.py --root_path finance_nikkei --model_id finance_nikkei --data finance --n_heads 4

python -u run.py --root_path healthcare_mortality --model_id healthcare_mortality --data healthcare --n_heads 4

python -u run.py --root_path healthcare_positive --model_id healthcare_positive --data healthcare --n_heads 8
