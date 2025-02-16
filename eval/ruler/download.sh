cd data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
python -c "import nltk; nltk.download('punkt')"