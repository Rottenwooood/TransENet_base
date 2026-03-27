while kill -0 38867 2>/dev/null; do
  sleep 60
done
bash train_rala.sh | tee train_rala.log && bash train_mlla.sh | tee train_mlla.log