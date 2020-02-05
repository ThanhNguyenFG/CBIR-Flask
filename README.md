# CBIR-Flask
Local web search for CBIR on Oxford 5K dataset
# 1. Install Flask
```sh
pip install flask
```
# 2. Run web
```sh
python app.py
```
Link http://127.0.0.1:5000/

You can watch demo in [here](https://drive.google.com/drive/folders/1RqAn0tLvplwuQilzzOyXtjASC6aqhrHh?usp=sharing)
# 3. Reference
https://github.com/jorjasso/VLAD
# 4. Note
I just upload file with ORB extractor feature and number of visual word = 2 because SIFT, SURF and bigger visual word is lagger than capacity of git allowed. So if you want to get them plese connect me or go to reference and see how to get these file.

Please install dataset Oxford 5K in /src/dataset/img and ground truth file in /src/dataset/gt if you want to get mAP.

My web just allow you to query img in dir /src/dataset/img
