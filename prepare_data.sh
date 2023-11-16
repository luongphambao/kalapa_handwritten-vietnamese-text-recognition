wget https://bytebattles2023data.s3.ap-southeast-1.amazonaws.com/KALAPA_ByteBattles_2023_OCR_Set1.zip
unzip KALAPA_ByteBattles_2023_OCR_Set1.zip
unzip OCR/training_data.zip -d OCR/
unzip OCR/public_test.zip -d OCR/
rm -rf KALAPA_ByteBattles_2023_OCR_Set1.zip
rm -rf OCR/training_data.zip
rm -rf OCR/public_test.zip
python3 merge.py