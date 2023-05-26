# Word2Vec Implementation with Negative Sampling
### Introduction
Word2Vec model generates word embeddings ที่สามารถจับความหมายคำที่ใกล้เคียงกันจาก context รอบๆคำ การสร้างนี้ใช้โมเดล Skip-Gram ร่วมกับ Negative Sampling
### Dependencies
Python packages:
- numpy
- pandas
- sklearn
- matplotlib
### Features
- นับจำนวนคำแต่ละคำที่เกิดขึ้นใน corpus
- Generates training data โดยจับคู่ (center_word, context_word)
- Trains the Word2Vec model โดยใช้ Stochastic Gradient Descent (SGD).
### Usage
ขั้นตอนการใช้ เปิดไฟล์ word2vec.ipynb
1. เตรียมไฟล์ .txt ที่ต้องการใช้ ex book.txt
```python
file_path = 'book.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    corpus = file.read()

```
2. ปรับ parameters ตามต้องการ
```python
np.random.seed(0)
window_size = 2
embedding_size = 50
learning_rate = 0.01
epochs = 150
negative_samples = 5
random_state = 1234
model = Word2Vec(window_size, embedding_size, learning_rate, epochs, negative_samples, random_state)

```
3. Train the model 
```python
model.fit(corpus)

```
4. สุ่มคำมาดู word vector ขอคำที่สุ่มได้
```python
random_word = random.choice(list(model.word2id.keys()))
word_vector = model.get_word_vector(random_word)
print("Random Word:", random_word)
print("Word Vector:", word_vector)

```
5. สร้าง dataframe word vector ของ word ทั้งหมดใน corpus
 ```python
df_word_vectors = model.create_word_vector_dataframe()
display(df_word_vectors)

```
### Output Example
![image](https://github.com/kittipat7/Word2Vec_from_scratch/assets/97491541/649f21ad-50aa-4a56-8540-f5fd95621f3e)
